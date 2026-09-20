import argparse
import csv
import json
import os
import random
import subprocess
import sys
from glob import glob


def all_volume_ids(data_dir):
    ids = {}
    for split in ("train", "val", "test"):
        folder = os.path.join(data_dir, split)
        for f in glob(os.path.join(folder, "image_*.npy")):
            vid = int(os.path.basename(f).replace("image_", "").replace(".npy", ""))
            ids[vid] = folder
    return ids


def make_folds(data_dir, n_folds, seed, out_json):
    ids = all_volume_ids(data_dir)
    vids = sorted(ids)
    print(f"{len(vids)} volumes found across train/val/test")

    shuffled = vids[:]
    random.Random(seed).shuffle(shuffled)
    folds = [sorted(shuffled[i::n_folds]) for i in range(n_folds)]

    spec = {"seed": seed, "n_folds": n_folds,
            "source": {str(k): v for k, v in ids.items()},
            "folds": {str(i): f for i, f in enumerate(folds)}}
    with open(out_json, "w") as f:
        json.dump(spec, f, indent=2)

    for i, f in enumerate(folds):
        print(f"  fold {i}: {len(f)} held-out volumes  {f}")
    # every volume held out exactly once
    seen = sorted(v for f in folds for v in f)
    assert seen == vids, "fold assignment lost or duplicated volumes"
    print(f"\nEvery volume held out exactly once. Wrote {out_json}")
    return spec


def build_fold_dirs(spec, fold, root):
    held = set(spec["folds"][str(fold)])
    src = {int(k): v for k, v in spec["source"].items()}
    base = os.path.join(root, f"fold_{fold}")
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(base, split), exist_ok=True)

    n_tr = n_ho = 0
    for vid, folder in src.items():
        target = "train" if vid not in held else "val"
        for kind in ("image", "mask"):
            s = os.path.abspath(os.path.join(folder, f"{kind}_{vid}.npy"))
            for split in ([target] if target == "train" else ["val", "test"]):
                d = os.path.join(base, split, f"{kind}_{vid}.npy")
                if not os.path.exists(d):
                    os.symlink(s, d)
        n_tr += vid not in held
        n_ho += vid in held
    # a stale foreground index from a different fold would be wrong
    for split in ("train", "val", "test"):
        idx = os.path.join(base, split, "fg_index.npz")
        if os.path.exists(idx):
            os.remove(idx)
    print(f"fold {fold}: {n_tr} train / {n_ho} held out (used as both val and test)")
    return base


def _done_marker(args, fold):
    root = args.checkpoint_dir or os.path.dirname(os.path.abspath(args.results_csv))
    tag = f"{args.arm}{'_' + args.frozen_blocks.replace(',', '-') if args.frozen_blocks else ''}"
    return os.path.join(root, f".done_{tag}_seed{args.seed}_fold{fold}")


def run_folds(args, spec):
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_lits3d.py")
    for fold in range(spec["n_folds"]):
        if args.only_fold >= 0 and fold != args.only_fold:
            continue

        marker = _done_marker(args, fold)
        if os.path.exists(marker) and not args.rerun_done:
            print(f"\nFOLD {fold}: already complete ({marker}), skipping. "
                  f"Use --rerun_done to force.")
            continue

        base = build_fold_dirs(spec, fold, args.fold_root)
        cmd = [sys.executable, script,
               "--data_dir", base,
               "--arm", args.arm,
               "--base_channels", str(args.base_channels),
               "--patch_size", *[str(p) for p in args.patch_size],
               "--epochs", str(args.epochs),
               "--samples_per_epoch", str(args.samples_per_epoch),
               "--batch_size", str(args.batch_size),
               "--val_every", str(args.val_every),
               "--seed", str(args.seed),
               "--select_metric", args.select_metric,
               "--results_csv", args.results_csv]
        if args.frozen_blocks:
            cmd += ["--frozen_blocks", args.frozen_blocks]
        if args.checkpoint_dir:
            cmd += ["--checkpoint_dir", os.path.join(args.checkpoint_dir, f"fold_{fold}")]
        if args.postprocess:
            cmd += ["--postprocess", "--min_tumour_size", str(args.min_tumour_size)]
        print("\n" + "=" * 70)
        print(f"FOLD {fold}/{spec['n_folds'] - 1}")
        print("=" * 70, flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise SystemExit(
                f"\nFold {fold} exited with code {r.returncode}. Nothing was marked "
                f"complete, so rerunning the same command resumes this fold from its "
                f"last checkpoint and leaves earlier folds untouched.")
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        with open(marker, "w") as f:
            f.write("complete\n")
        print(f"Fold {fold} complete.")


def summarize(results_csv, arm_filter=""):
    import numpy as np
    rows = [r for r in csv.DictReader(open(results_csv))
            if not arm_filter or r["arm"].startswith(arm_filter)]
    if not rows:
        print("no matching rows"); return
    liver = np.array([float(r["test_liver"]) for r in rows])
    t_all = np.array([float(r["test_tumour_all"]) for r in rows])
    t_case = np.array([float(r["test_tumour_cases"]) for r in rows])
    n_bear = np.array([int(r["n_tumour_bearing"]) for r in rows])

    def ms(x):
        x = x[~np.isnan(x)]
        if len(x) == 0:
            return "n/a"
        sd = np.std(x, ddof=1) if len(x) > 1 else 0.0
        return f"{np.mean(x):.4f} +/- {sd:.4f}"

    print(f"{len(rows)} folds, {n_bear.sum()} tumour-bearing volumes in total")
    print(f"  liver              {ms(liver)}")
    print(f"  tumour (all vols)  {ms(t_all)}   <- headline, LiTS convention")
    print(f"  tumour (bearing)   {ms(t_case)}")
    print(f"  per fold (all):    {[round(float(x), 4) for x in t_all]}")
    print(f"  fold sizes (tumour-bearing): {n_bear.tolist()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--fold_json", type=str, default="")
    p.add_argument("--fold_root", type=str, default="/content/lits3d_folds")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--fold_seed", type=int, default=42)
    p.add_argument("--make_folds", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--only_fold", type=int, default=-1)
    p.add_argument("--rerun_done", action="store_true",
                   help="Rerun folds already marked complete.")
    # passed straight through to train_lits3d.py
    p.add_argument("--arm", type=str, default="hybrid")
    p.add_argument("--frozen_blocks", type=str, default="up1")
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--samples_per_epoch", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--select_metric", type=str, default="tumour_mean_all")
    p.add_argument("--postprocess", action="store_true")
    p.add_argument("--min_tumour_size", type=int, default=50)
    p.add_argument("--checkpoint_dir", type=str, default="")
    p.add_argument("--results_csv", type=str, default="lits3d_cv.csv")
    args = p.parse_args()

    fold_json = args.fold_json or os.path.join(args.data_dir, "folds_5.json")
    if args.make_folds:
        spec = make_folds(args.data_dir, args.n_folds, args.fold_seed, fold_json)
    else:
        if not os.path.exists(fold_json):
            raise SystemExit(f"{fold_json} not found. Run with --make_folds first.")
        spec = json.load(open(fold_json))

    if args.run:
        run_folds(args, spec)
    if args.summarize:
        summarize(args.results_csv, args.arm)
