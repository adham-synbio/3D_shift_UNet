import argparse
import json
import os
import random
import shutil
import tarfile
import tempfile
from glob import glob

import numpy as np


class TarCaseReader:
    """Reads one case at a time straight out of the .tar
    """
    def __init__(self, tar_path):
        self.tar_path = tar_path
        self.tar = tarfile.open(tar_path, "r")
        self.images, self.labels = {}, {}
        for m in self.tar.getmembers():
            if not m.isfile():
                continue
            base = os.path.basename(m.name)
            if base.startswith("._") or not base.endswith((".nii", ".nii.gz")):
                continue
            cid = case_id(base)
            if cid is None:
                continue
            if "/imagesTr/" in m.name:
                self.images[cid] = m
            elif "/labelsTr/" in m.name:
                self.labels[cid] = m

    def case_ids(self):
        return sorted(set(self.images) & set(self.labels))

    def _extract_to_temp(self, member):
        suffix = ".nii.gz" if member.name.endswith(".gz") else ".nii"
        fh = self.tar.extractfile(member)
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            while True:
                chunk = fh.read(8 * 1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
            tmp.close()
            return tmp.name
        except Exception:
            tmp.close()
            os.unlink(tmp.name)
            raise

    def read(self, cid):
        paths = []
        try:
            for member in (self.images[cid], self.labels[cid]):
                paths.append(self._extract_to_temp(member))
            return load_nifti(paths[0]), load_nifti(paths[1])
        finally:
            for p in paths:
                if os.path.exists(p):
                    os.unlink(p)

    def close(self):
        self.tar.close()


def load_nifti(path):
    import nibabel as nib
    img = nib.load(path)
    data = np.asanyarray(img.dataobj)
    # nibabel gives (x, y, z); move to (z, y, x) so patches index depth first
    data = np.transpose(data, (2, 1, 0))
    zooms = img.header.get_zooms()[:3]          # (x, y, z)
    spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))   # (z, y, x)
    return data, spacing


def resample(volume, spacing, target, is_label):
    from scipy import ndimage
    factors = [s / t for s, t in zip(spacing, target)]
    if all(abs(f - 1.0) < 1e-3 for f in factors):
        return volume
    order = 0 if is_label else 1
    out = ndimage.zoom(volume.astype(np.float32 if not is_label else np.uint8),
                       factors, order=order, mode="nearest")
    return out.astype(np.uint8) if is_label else out


def normalize(vol, clip_lo, clip_hi, mean, std):
    v = np.clip(vol.astype(np.float32), clip_lo, clip_hi)
    return (v - mean) / std


def case_id(path):
    base = os.path.basename(path).replace(".nii.gz", "").replace(".nii", "")
    digits = "".join(ch for ch in base if ch.isdigit())
    return int(digits) if digits else None


def _case_exists(cid, out_dir, archive_dir, split_hint=None):
    """A case counts as done if its image lands in either location, under any
    split folder, so resuming never reprocesses work already archived."""
    for root in filter(None, (out_dir, archive_dir)):
        for name in ("train", "val", "test"):
            if os.path.exists(os.path.join(root, name, f"image_{cid}.npy")):
                return True
    return False


def _migrate_existing(out_dir, archive_dir):
    moved = 0
    for name in ("train", "val", "test"):
        srcd = os.path.join(out_dir, name)
        if not os.path.isdir(srcd):
            continue
        dstd = os.path.join(archive_dir, name)
        os.makedirs(dstd, exist_ok=True)
        for f in sorted(glob(os.path.join(srcd, "*.npy"))):
            dst = os.path.join(dstd, os.path.basename(f))
            if os.path.exists(dst):
                os.remove(f)
            else:
                shutil.move(f, dst)
            moved += 1
    if moved:
        print(f"Moved {moved} existing files from {out_dir} to {archive_dir}")
    return moved


def main(args):
    reader = None
    if os.path.isfile(args.src) and args.src.endswith(".tar"):
        reader = TarCaseReader(args.src)
        ids = reader.case_ids()
        cases = [(c, None, None) for c in ids]
        print(f"Reading directly from {args.src} - the archive is never extracted.")
    else:
        img_dir = os.path.join(args.src, "imagesTr")
        lbl_dir = os.path.join(args.src, "labelsTr")
        images = sorted(f for f in glob(os.path.join(img_dir, "*.nii*"))
                        if not os.path.basename(f).startswith("._"))
        if not images:
            raise SystemExit(f"No NIfTI files in {img_dir}, and {args.src} is not a .tar")
        cases = []
        for ip in images:
            lp = os.path.join(lbl_dir, os.path.basename(ip))
            cid = case_id(ip)
            if os.path.exists(lp) and cid is not None:
                cases.append((cid, ip, lp))
        cases.sort()
    if args.only_ids:
        wanted = set(int(v) for v in args.only_ids)
        cases = [c for c in cases if c[0] in wanted]
    elif args.max_cases > 0:
        step = max(1, len(cases) // args.max_cases)
        cases = cases[::step][:args.max_cases]
    if not cases:
        raise SystemExit("No labelled volumes found")
    print(f"{len(cases)} labelled volumes, ids {cases[0][0]}..{cases[-1][0]}")

    
    if args.split_json and os.path.exists(args.split_json):
        with open(args.split_json) as f:
            sp = json.load(f)
        split_of = {}
        for name in ("train", "val", "test"):
            for v in sp[f"{name}_ids"]:
                split_of[int(v)] = name
        missing = [c for c, _, _ in cases if c not in split_of]
        print(f"Reusing split from {args.split_json}"
              + (f" ({len(missing)} ids not listed, assigned to train)" if missing else ""))
        for c in missing:
            split_of[c] = "train"
    else:
        ids = [c for c, _, _ in cases]
        random.Random(args.split_seed).shuffle(ids)
        n_test = max(1, int(len(ids) * args.test_frac))
        n_val = max(1, int(len(ids) * args.val_frac))
        split_of = {}
        for v in ids[:n_test]:
            split_of[v] = "test"
        for v in ids[n_test:n_test + n_val]:
            split_of[v] = "val"
        for v in ids[n_test + n_val:]:
            split_of[v] = "train"
        print(f"New volume-level split (seed {args.split_seed})")

    target = tuple(args.target_spacing)
    manifest = {"target_spacing": list(target),
                "intensity": {"clip": [args.clip_lo, args.clip_hi],
                              "mean": args.mean, "std": args.std},
                "dtype": args.dtype, "cases": {}}

    for name in ("train", "val", "test"):
        os.makedirs(os.path.join(args.out_dir, name), exist_ok=True)
        if args.archive_dir:
            os.makedirs(os.path.join(args.archive_dir, name), exist_ok=True)

    if args.archive_dir:
        _migrate_existing(args.out_dir, args.archive_dir)

    total_bytes = 0
    for cid, ip, lp in cases:
        split = split_of.get(cid, "train")
        out_img = os.path.join(args.out_dir, split, f"image_{cid}.npy")
        if _case_exists(cid, args.out_dir, args.archive_dir) and not args.overwrite:
            print(f"  case {cid}: exists, skipping")
            continue

        if reader is not None:
            (vol, spacing), (lab, _) = reader.read(cid)
        else:
            vol, spacing = load_nifti(ip)
            lab, _ = load_nifti(lp)

        vol = resample(vol, spacing, target, is_label=False)
        lab = resample(lab, spacing, target, is_label=True)
        vol = normalize(vol, args.clip_lo, args.clip_hi, args.mean, args.std)

        vol = vol.astype(np.float16 if args.dtype == "float16" else np.float32)
        lab = lab.astype(np.uint8)

        out_msk = os.path.join(args.out_dir, split, f"mask_{cid}.npy")
        np.save(out_img, vol)
        np.save(out_msk, lab)

        nb = vol.nbytes + lab.nbytes

        if args.archive_dir:
            # Move straight to the archive so local disk never accumulates.
            for f in (out_img, out_msk):
                shutil.move(f, os.path.join(args.archive_dir, split, os.path.basename(f)))
        total_bytes += nb
        manifest["cases"][str(cid)] = {
            "split": split, "original_spacing": list(spacing),
            "shape": list(vol.shape),
            "liver_voxels": int((lab == 1).sum()), "tumour_voxels": int((lab == 2).sum()),
        }
        print(f"  case {cid:>3} [{split:>5}] spacing {tuple(round(s,2) for s in spacing)} "
              f"-> shape {vol.shape}  tumour {int((lab==2).sum()):>8}  ({nb/1e6:.0f} MB)",
              flush=True)

    if reader is not None:
        reader.close()

    with open(os.path.join(args.out_dir, "manifest_3d.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    counts = {}
    for v in manifest["cases"].values():
        counts[v["split"]] = counts.get(v["split"], 0) + 1
    print(f"\nVolumes per split: {counts}")
    print(f"Total written: {total_bytes/1e9:.2f} GB")
    n_done = len(manifest["cases"])
    if n_done:
        per = total_bytes / n_done
        print(f"Mean per case: {per/1e6:.0f} MB")
        print(f"PROJECTED FOR ALL 131 CASES: {per*131/1e9:.1f} GB")
    print(f"Manifest: {os.path.join(args.out_dir, 'manifest_3d.json')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=str, required=True,
                   help="Either the Task03_Liver.tar file (read case by case, never "
                        "extracted) or an already-extracted Task03_Liver directory.")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--target_spacing", type=float, nargs=3, default=[1.0, 0.77, 0.77],
                   help="(z, y, x) in mm. nnU-Net's LiTS default. 1.5 1.5 1.5 is "
                        "about 6x smaller on disk.")
    p.add_argument("--clip_lo", type=float, default=-17.0)
    p.add_argument("--clip_hi", type=float, default=201.0)
    p.add_argument("--mean", type=float, default=99.40)
    p.add_argument("--std", type=float, default=39.39)
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    p.add_argument("--split_json", type=str, default="",
                   help="Reuse an existing volume-level split so 3D results stay "
                        "comparable with the 2D ones.")
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--archive_dir", type=str, default="",
                   help="Move each case here immediately after writing it, e.g. a Drive "
                        "path. Local disk then never accumulates output. Any files "
                        "already in --out_dir are migrated at startup, so a run that "
                        "ran out of space can be resumed without redoing it.")
    p.add_argument("--max_cases", type=int, default=0,
                   help="Process only this many cases, sampled evenly across the id "
                        "range. Use it to measure disk cost before committing.")
    p.add_argument("--only_ids", type=int, nargs="+", default=None,
                   help="Process only these case ids.")
    p.add_argument("--overwrite", action="store_true")
    main(p.parse_args())
