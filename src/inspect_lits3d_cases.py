import argparse
import json
import os

import numpy as np


def find_case(data_dir, vid):
    for split in ("train", "val", "test"):
        p = os.path.join(data_dir, split, f"image_{vid}.npy")
        if os.path.exists(p):
            return split, p, os.path.join(data_dir, split, f"mask_{vid}.npy")
    return None, None, None


def components(mask, min_size=1):
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n == 0:
        return []
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return sorted([int(s) for s in sizes if s >= min_size], reverse=True)


def main(args):
    manifest = {}
    mpath = os.path.join(args.data_dir, "manifest_3d.json")
    if not os.path.exists(mpath) and args.manifest:
        mpath = args.manifest
    if os.path.exists(mpath):
        manifest = json.load(open(mpath)).get("cases", {})

    model = None
    if args.checkpoint:
        import torch
        from model3d import UNet3D
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = UNet3D(**ck["config"])
        if ck.get("frozen_blocks"):
            from model3d import freeze_blocks_to_shift_3d
            blocks = tuple(b for b in ck["frozen_blocks"].split(",") if b)
            freeze_blocks_to_shift_3d(model, blocks or ("*",), verbose=False)
        model.load_state_dict(ck["model"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        print(f"Loaded {args.checkpoint} (epoch {ck.get('epoch')})\n")

    for vid in args.ids:
        split, ip, mp = find_case(args.data_dir, vid)
        if ip is None:
            print(f"volume {vid}: not found\n")
            continue

        img = np.load(ip).astype(np.float32)
        msk = np.load(mp)
        info = manifest.get(str(vid), {})
        spacing = info.get("original_spacing")

        liver = (msk == 1) | (msk == 2)
        tumour = (msk == 2)
        tum_comp = components(tumour)

        print(f"=== volume {vid}  [{split}] " + "=" * 40)
        if spacing:
            print(f"  original spacing (z,y,x) : {[round(s,2) for s in spacing]} mm"
                  f"   {'THICK SLICES' if spacing[0] >= 3 else ''}")
        print(f"  resampled shape          : {img.shape}")
        print(f"  intensity                : mean {img.mean():+.3f}  sd {img.std():.3f}  "
              f"range [{img.min():+.2f}, {img.max():+.2f}]")
        print(f"  liver voxels             : {int(liver.sum()):,}  "
              f"({100*liver.mean():.2f}% of volume)")
        print(f"  tumour voxels            : {int(tumour.sum()):,}  "
              f"({100*tumour.sum()/max(1,liver.sum()):.2f}% of liver)")
        print(f"  tumour components        : {len(tum_comp)}  "
              f"largest {tum_comp[:5] if tum_comp else '-'}")

        if tumour.any():
            t_mean = img[tumour].mean()
            l_only = liver & ~tumour
            l_mean = img[l_only].mean() if l_only.any() else float("nan")
            print(f"  mean intensity tumour    : {t_mean:+.3f}")
            print(f"  mean intensity liver     : {l_mean:+.3f}")
            print(f"  CONTRAST (liver-tumour)  : {l_mean - t_mean:+.3f}  "
                  f"(sd of liver {img[l_only].std():.3f})")

        if model is not None:
            import torch
            from lits3d_data import sliding_window_predict, dice_per_volume
            pred = sliding_window_predict(model, img, patch_size=tuple(args.patch_size),
                                          overlap=0.5, device=device, amp=False)
            s = dice_per_volume(pred, msk)
            pt = (pred == 2)
            pl = (pred == 1) | (pred == 2)
            print(f"  --- prediction ---")
            print(f"  liver dice {s['liver']:.4f}   tumour dice {s['tumour']:.4f} "
                  f"[{s['tumour_case']}]")
            print(f"  predicted tumour comps   : {len(components(pt))}  "
                  f"largest {components(pt)[:5]}")
            if pt.any():
                inside = (pt & pl).sum() / pt.sum()
                print(f"  predicted tumour inside predicted liver : {100*inside:.1f}%")
            if tumour.any() and pt.any():
                recall = (pt & tumour).sum() / tumour.sum()
                prec = (pt & tumour).sum() / pt.sum()
                print(f"  recall {recall:.3f}   precision {prec:.3f}")
            # did the liver segmentation cover the tumour at all?
            if tumour.any():
                cov = (pl & tumour).sum() / tumour.sum()
                print(f"  ground-truth tumour covered by PREDICTED liver : {100*cov:.1f}%"
                      f"{'   <-- liver miss explains the tumour miss' if cov < 0.5 else ''}")
        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--ids", type=int, nargs="+", required=True)
    p.add_argument("--manifest", type=str, default="")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])
    main(p.parse_args())
