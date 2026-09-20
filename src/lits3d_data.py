import json
import os
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def volume_ids(folder):
    ids = []
    for f in glob(os.path.join(folder, "image_*.npy")):
        ids.append(int(os.path.basename(f).replace("image_", "").replace(".npy", "")))
    return sorted(ids)


def load_case(folder, vid, mmap=True):
    kw = {"mmap_mode": "r"} if mmap else {}
    img = np.load(os.path.join(folder, f"image_{vid}.npy"), **kw)
    msk = np.load(os.path.join(folder, f"mask_{vid}.npy"), **kw)
    return img, msk


class LiTS3DPatchDataset(Dataset):

    def __init__(self, folder, patch_size=(128, 128, 128), samples_per_epoch=250,
                 fg_fraction=0.66, tumour_fraction=0.33, augment=True, seed=0,
                 cache_index=True):
        self.folder = folder
        self.patch = tuple(patch_size)
        self.n = samples_per_epoch
        self.fg_fraction = fg_fraction
        self.tumour_fraction = tumour_fraction
        self.augment = augment
        self.seed = seed
        self.ids = volume_ids(folder)
        if not self.ids:
            raise SystemExit(f"No image_*.npy in {folder}")
        self._rng = None
        self._vols = {}

        self.index = self._build_index() if cache_index else {}
        n_tum = sum(1 for v in self.index.values() if len(v["tumour"]) > 0)
        print(f"{folder}: {len(self.ids)} volumes, {n_tum} contain tumour, "
              f"patch {self.patch}, {self.n} samples/epoch")

    def _build_index(self):
        cache = os.path.join(self.folder, "fg_index.npz")
        if os.path.exists(cache):
            z = np.load(cache, allow_pickle=True)
            return {int(k): v.item() for k, v in z.items()}

        print(f"Indexing foreground voxels in {self.folder} (one time)...")
        index = {}
        for vid in self.ids:
            _, msk = load_case(self.folder, vid)
            msk = np.asarray(msk)
            out = {}
            for name, sel in (("liver", msk == 1), ("tumour", msk == 2)):
                coords = np.argwhere(sel)
                if len(coords) > 10000:
                    step = len(coords) // 10000
                    coords = coords[::step]
                out[name] = coords.astype(np.int32)
            index[vid] = out
            print(f"  volume {vid}: liver {len(out['liver'])}, "
                  f"tumour {len(out['tumour'])} candidate centres", flush=True)
        np.savez_compressed(cache, **{str(k): v for k, v in index.items()})
        return index

    def _ensure(self):
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else 0
            self._rng = np.random.RandomState(self.seed + 1013 * wid)

    def _get(self, vid):
        if vid not in self._vols:
            self._vols[vid] = load_case(self.folder, vid)
        return self._vols[vid]

    def __len__(self):
        return self.n

    def _sample_centre(self, vid, shape):
        rng = self._rng
        r = rng.rand()
        entry = self.index.get(vid, {})
        pool = None
        if r < self.tumour_fraction and len(entry.get("tumour", [])) > 0:
            pool = entry["tumour"]
        elif r < self.fg_fraction and len(entry.get("liver", [])) > 0:
            pool = entry["liver"]

        if pool is not None and len(pool):
            c = pool[rng.randint(len(pool))]
        else:
            c = [rng.randint(s) for s in shape]

        # Clamp so the patch stays inside the volume
        start = []
        for axis in range(3):
            half = self.patch[axis] // 2
            lo = 0
            hi = max(0, shape[axis] - self.patch[axis])
            start.append(int(np.clip(c[axis] - half, lo, hi)))
        return start

    def __getitem__(self, i):
        self._ensure()
        rng = self._rng
        vid = self.ids[rng.randint(len(self.ids))]
        img, msk = self._get(vid)
        shape = img.shape

        s = self._sample_centre(vid, shape)
        sl = tuple(slice(s[a], s[a] + self.patch[a]) for a in range(3))
        patch_img = np.asarray(img[sl]).astype(np.float32)
        patch_msk = np.asarray(msk[sl]).astype(np.int64)

        # Pad if the volume is smaller than the patch on some axis
        pad = [(0, max(0, self.patch[a] - patch_img.shape[a])) for a in range(3)]
        if any(p[1] for p in pad):
            patch_img = np.pad(patch_img, pad, mode="constant", constant_values=0)
            patch_msk = np.pad(patch_msk, pad, mode="constant", constant_values=0)

        if self.augment:
            for axis in range(3):
                if rng.rand() < 0.5:
                    patch_img = np.flip(patch_img, axis)
                    patch_msk = np.flip(patch_msk, axis)
            if rng.rand() < 0.3:                      # intensity jitter
                patch_img = patch_img * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)

        patch_img = np.ascontiguousarray(patch_img)[None]      # (1, D, H, W)
        patch_msk = np.ascontiguousarray(patch_msk)
        return torch.from_numpy(patch_img).float(), torch.from_numpy(patch_msk).long()


def gaussian_weight(patch_size, sigma_scale=0.125, device="cpu"):
    grids = []
    for s in patch_size:
        coords = torch.arange(s, dtype=torch.float32) - (s - 1) / 2
        sigma = s * sigma_scale
        grids.append(torch.exp(-(coords ** 2) / (2 * sigma ** 2)))
    w = grids[0][:, None, None] * grids[1][None, :, None] * grids[2][None, None, :]
    return (w / w.max()).clamp_min(1e-4).to(device)


@torch.no_grad()
def sliding_window_predict(model, volume, patch_size=(128, 128, 128), overlap=0.5,
                           n_classes=3, device="cuda", batch_size=2, amp=True,
                           activation="softmax", return_prob=False):
    
    model.eval()
    D, H, W = volume.shape
    pd, ph, pw = patch_size
    # Pad up to at least one patch in every axis
    pad = [(0, max(0, p - s)) for s, p in zip((D, H, W), patch_size)]
    if any(p[1] for p in pad):
        volume = np.pad(volume, pad, mode="constant", constant_values=0)
    Dp, Hp, Wp = volume.shape

    step = [max(1, int(p * (1 - overlap))) for p in patch_size]

    def starts(total, p, st):
        if total <= p:
            return [0]
        xs = list(range(0, total - p + 1, st))
        if xs[-1] != total - p:
            xs.append(total - p)
        return xs

    zs = starts(Dp, pd, step[0])
    ys = starts(Hp, ph, step[1])
    xs = starts(Wp, pw, step[2])

    logits = torch.zeros((n_classes, Dp, Hp, Wp), dtype=torch.float32, device=device)
    counts = torch.zeros((1, Dp, Hp, Wp), dtype=torch.float32, device=device)
    gw = gaussian_weight(patch_size, device=device)

    coords = [(z, y, x) for z in zs for y in ys for x in xs]
    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i + batch_size]
        batch = np.stack([volume[z:z + pd, y:y + ph, x:x + pw] for z, y, x in chunk])
        t = torch.from_numpy(batch).float().unsqueeze(1).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device == "cuda"):
            out = model(t).float()
        out = torch.softmax(out, dim=1) if activation == "softmax" else torch.sigmoid(out)
        for j, (z, y, x) in enumerate(chunk):
            logits[:, z:z + pd, y:y + ph, x:x + pw] += out[j] * gw
            counts[:, z:z + pd, y:y + ph, x:x + pw] += gw

    logits /= counts.clamp_min(1e-6)
    if return_prob:
        return logits[:, :D, :H, :W].cpu().numpy()
    if activation == "softmax":
        pred = torch.argmax(logits, dim=0).cpu().numpy().astype(np.uint8)
    else:
        p = (logits > 0.5).cpu().numpy()
        pred = np.zeros(p.shape[1:], dtype=np.uint8)
        pred[p[0]] = 1
        pred[p[1]] = 2
    return pred[:D, :H, :W]


def dice_per_volume(pred, gt, num_classes=3):
    scores = {}
    for cls in range(1, num_classes):
        name = "liver" if cls == 1 else "tumour" if cls == 2 else f"class_{cls}"
        p, g = (pred == cls), (gt == cls)
        np_, ng = int(p.sum()), int(g.sum())
        ov = int((p & g).sum())
        if ng == 0 and np_ == 0:
            d, case = 1.0, "empty_correct"
        elif ng == 0:
            d, case = 0.0, "false_positive"
        elif np_ == 0:
            d, case = 0.0, "missed"
        elif ov == 0:
            d, case = 0.0, "no_overlap"
        else:
            d, case = 2.0 * ov / (np_ + ng), "detected"
        scores[name] = float(d)
        scores[f"{name}_case"] = case
        scores[f"{name}_gt"] = ng
        scores[f"{name}_pred"] = np_
    return scores


# ---------------------------------------------------------------------------
# Post-processing

def keep_largest_component(mask):
    from scipy import ndimage
    if not mask.any():
        return mask
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def remove_small_components(mask, min_size):
    from scipy import ndimage
    if min_size <= 0 or not mask.any():
        return mask
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    out = mask.copy()
    for i, sz in enumerate(sizes, start=1):
        if sz < min_size:
            out[lab == i] = False
    return out


def postprocess(pred, keep_largest_liver=True, fill_liver_holes=True,
                min_tumour_size=0, constrain_tumour=True):
    from scipy import ndimage
    out = pred.copy()
    liver = (out == 1) | (out == 2)

    if keep_largest_liver and liver.any():
        liver = keep_largest_component(liver)
    if fill_liver_holes and liver.any():
        liver = ndimage.binary_fill_holes(liver)

    tumour = (out == 2)
    if constrain_tumour:
        tumour = tumour & liver
    if min_tumour_size > 0:
        tumour = remove_small_components(tumour, min_tumour_size)

    out = np.zeros_like(pred)
    out[liver] = 1
    out[tumour] = 2
    return out
