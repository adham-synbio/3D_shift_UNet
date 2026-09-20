import argparse
import csv
import json
import os
import time

import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model3d import UNet3D, count_parameters, freeze_blocks_to_shift_3d
from lits3d_data import (LiTS3DPatchDataset, volume_ids, load_case,
                         sliding_window_predict, dice_per_volume, postprocess)
from torch.utils.data import DataLoader


class RegionDiceBCELoss(nn.Module):
    
    def __init__(self, pos_weight=(1.0, 4.0), eps=1.0):
        super().__init__()
        self.eps = eps
        self.register_buffer("pw", torch.tensor(pos_weight, dtype=torch.float32))

    def forward(self, logits, target):
        # target holds class indices; build the two nested binary targets
        t = torch.stack([(target >= 1).float(), (target == 2).float()], dim=1)
        bce = F.binary_cross_entropy_with_logits(
            logits, t, pos_weight=self.pw.view(1, -1, 1, 1, 1))
        p = torch.sigmoid(logits)
        dice = 0.0
        for c in range(t.shape[1]):
            inter = (p[:, c] * t[:, c]).sum()
            dice = dice + (1 - (2 * inter + self.eps) /
                           (p[:, c].sum() + t[:, c].sum() + self.eps))
        return bce + dice / t.shape[1]


class DiceCELoss(nn.Module):
    
    def __init__(self, n_classes=3, class_weights=(0.2, 1.0, 4.0), eps=1.0):
        super().__init__()
        self.n_classes = n_classes
        self.eps = eps
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        p = torch.softmax(logits, dim=1)
        dice = 0.0
        for c in range(1, self.n_classes):
            pc = p[:, c]
            tc = (target == c).float()
            inter = (pc * tc).sum()
            dice = dice + (1 - (2 * inter + self.eps) / (pc.sum() + tc.sum() + self.eps))
        return ce + dice / (self.n_classes - 1)


@torch.no_grad()
def validate(model, folder, patch, device, overlap=0.5, max_cases=0, amp=True,
             activation="softmax", n_classes=3, post=None):
    ids = volume_ids(folder)
    if max_cases:
        ids = ids[:max_cases]
    rows = []
    for vid in ids:
        img, msk = load_case(folder, vid, mmap=False)
        pred = sliding_window_predict(model, np.asarray(img).astype(np.float32),
                                      patch_size=patch, overlap=overlap,
                                      device=device, amp=amp,
                                      activation=activation, n_classes=n_classes)
        if post:
            pred = postprocess(pred, **post)
        s = dice_per_volume(pred, np.asarray(msk))
        s["volume_id"] = vid
        rows.append(s)
    liver = [r["liver"] for r in rows]
    tum_all = [r["tumour"] for r in rows]
    tum_case = [r["tumour"] for r in rows if r["tumour_gt"] > 0]
    return rows, {
        "liver_mean": float(np.mean(liver)),
        "tumour_mean_cases": float(np.mean(tum_case)) if tum_case else float("nan"),
        "tumour_mean_all": float(np.mean(tum_all)),
        "n_tumour_bearing": len(tum_case),
        "n_volumes": len(rows),
    }


class Tee:
    
    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.f = open(path, "a", buffering=1)
        self.stdout = sys.stdout
        self.f.write(f"\n{'='*70}\n{datetime.now():%Y-%m-%d %H:%M:%S}  "
                     f"{' '.join(sys.argv)}\n{'='*70}\n")

    def write(self, msg):
        self.stdout.write(msg)
        self.f.write(msg)

    def flush(self):
        self.stdout.flush()
        self.f.flush()


def main(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    patch = tuple(args.patch_size)

    if args.log_file or args.checkpoint_dir:
        log_path = args.log_file
        if not log_path:
            frozen_tag = args.frozen_blocks.replace(",", "-") if args.frozen_blocks else ""
            log_path = os.path.join(
                args.checkpoint_dir,
                f"log_{args.arm}{'_' + frozen_tag if frozen_tag else ''}"
                f"_b{args.base_channels}_p{args.patch_size[0]}_seed{args.seed}.txt")
        sys.stdout = Tee(log_path)
        print(f"Logging to {log_path}")

    region = args.label_mode == "region"
    cfg = dict(n_channels=1, n_classes=2 if region else 3,
               base_channels=args.base_channels, norm=args.norm)
    if args.arm == "depthwise":
        cfg.update(use_learnable_shift=True)
    elif args.arm in ("fair_shift", "hybrid"):
        cfg.update(use_learnable_shift=True)
    elif args.arm == "frozen_shift":
        cfg.update(use_shift_conv=True)
    # "standard" leaves both off
    model = UNet3D(**cfg).to(device)

    frozen = ""
    if args.arm == "fair_shift":
        freeze_blocks_to_shift_3d(model, ("*",)); frozen = "*"
    elif args.arm == "hybrid":
        blocks = tuple(b.strip() for b in args.frozen_blocks.split(",") if b.strip())
        if not blocks:
            raise SystemExit("--arm hybrid needs --frozen_blocks")
        freeze_blocks_to_shift_3d(model, blocks); frozen = args.frozen_blocks

    n_params = count_parameters(model)
    print(f"arm={args.arm} base={args.base_channels} patch={patch} "
          f"params={n_params:,}")

    train_ds = LiTS3DPatchDataset(os.path.join(args.data_dir, "train"), patch,
                                  samples_per_epoch=args.samples_per_epoch,
                                  fg_fraction=args.fg_fraction,
                                  tumour_fraction=args.tumour_fraction,
                                  augment=True, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=torch.cuda.is_available(),
                              persistent_workers=args.num_workers > 0)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=args.min_lr)
    if region:
        criterion = RegionDiceBCELoss(pos_weight=tuple(args.region_pos_weight)).to(device)
    else:
        criterion = DiceCELoss(class_weights=tuple(args.class_weights)).to(device)
    activation = "sigmoid" if region else "softmax"

    ckpt_path, start_epoch, best = None, 0, -1.0
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        tag = (f"{args.arm}{'_' + frozen.replace(',', '-') if frozen else ''}"
               f"_b{args.base_channels}_p{patch[0]}_seed{args.seed}")
        ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_{tag}.pth")
        best_path = os.path.join(args.checkpoint_dir, f"best_{tag}.pth")
        if os.path.exists(ckpt_path) and not args.no_resume:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            if ck["n_params"] != n_params:
                raise SystemExit(f"{ckpt_path} has {ck['n_params']:,} params, "
                                 f"this run builds {n_params:,}. Use --no_resume.")
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["scheduler"])
            start_epoch, best = ck["epoch"] + 1, ck["best"]
            print(f"Resumed at epoch {start_epoch}/{args.epochs} (best tumour {best:.4f})")
    else:
        best_path = None

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        tot = 0.0
        inter = torch.zeros(3, device=device)
        psum = torch.zeros(3, device=device)
        tsum = torch.zeros(3, device=device)
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=bool(args.amp) and device == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward(); opt.step()
            tot += loss.item()
            with torch.no_grad():
                lf = logits.float()
                if region:
                    # nested channels: 0 = liver-or-tumour, 1 = tumour
                    p = torch.sigmoid(lf) > 0.5
                    pairs = [(p[:, 0], y >= 1), (p[:, 1], y == 2)]
                else:
                    am = torch.argmax(lf, dim=1)
                    pairs = [(am == 1, y == 1), (am == 2, y == 2)]
                for c, (pc, tc) in enumerate(pairs, start=1):
                    inter[c] += (pc & tc).sum()
                    psum[c] += pc.sum()
                    tsum[c] += tc.sum()
        sched.step()
        avg = tot / max(1, len(train_loader))
        patch_dice = (2 * inter / (psum + tsum).clamp_min(1)).tolist()

        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            rows, summ = validate(model, os.path.join(args.data_dir, "val"), patch,
                                  device, overlap=args.overlap,
                                  max_cases=args.val_max_cases, amp=bool(args.amp),
                                  activation=activation, n_classes=cfg["n_classes"])
            print(f"epoch {epoch+1:>3}/{args.epochs} loss={avg:.4f}  "
                  f"patchDice L={patch_dice[1]:.3f} T={patch_dice[2]:.3f}  |  "
                  f"per-case liver={summ['liver_mean']:.4f}  "
                  f"tumour(all)={summ['tumour_mean_all']:.4f} "
                  f"tumour(bearing)={summ['tumour_mean_cases']:.4f} "
                  f"({summ['n_tumour_bearing']}/{summ['n_volumes']})  "
                  f"[{(time.time()-t0)/60:.0f} min]", flush=True)
            
            sel = summ[args.select_metric]
            if sel > best:
                best = sel
                if best_path:
                    torch.save({"model": model.state_dict(), "config": cfg,
                                "frozen_blocks": frozen, "epoch": epoch,
                                "summary": summ, "n_params": n_params}, best_path)
                print(f"    new best, saved")
        else:
            print(f"epoch {epoch+1:>3}/{args.epochs} loss={avg:.4f}  "
                  f"patchDice liver={patch_dice[1]:.4f} tumour={patch_dice[2]:.4f}"
                  f"  [{(time.time()-t0)/60:.0f} min]", flush=True)

        if ckpt_path and ((epoch + 1) % args.checkpoint_every == 0
                          or epoch == args.epochs - 1):
            tmp = ckpt_path + ".tmp"
            torch.save({"epoch": epoch, "best": best, "model": model.state_dict(),
                        "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                        "n_params": n_params, "arm": args.arm}, tmp)
            os.replace(tmp, ckpt_path)

    # ---- final test evaluation with the best checkpoint ----
    if best_path and os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device,
                                         weights_only=False)["model"])
        print("\nEvaluating best checkpoint on the test split")
    post = None
    if args.postprocess:
        post = dict(keep_largest_liver=True, fill_liver_holes=True,
                    min_tumour_size=args.min_tumour_size, constrain_tumour=True)
    # eval_split exists so post-processing can be tuned on validation. Choosing
    # a threshold by comparing test results and then reporting that test number
    # is selection on the reported quantity.
    print(f"\nEvaluating on the {args.eval_split} split"
          + (f" (min_tumour_size={args.min_tumour_size})" if args.postprocess else
             " (no post-processing)"))
    rows, summ = validate(model, os.path.join(args.data_dir, args.eval_split), patch,
                          device, overlap=args.overlap, amp=bool(args.amp),
                          activation=activation, n_classes=cfg["n_classes"], post=post)
    for r in rows:
        print(f"  volume {r['volume_id']:>3}: liver={r['liver']:.4f} "
              f"tumour={r['tumour']:.4f}  gt={r['tumour_gt']} pred={r['tumour_pred']} "
              f"[{r['tumour_case']}]")
    print(f"\n{args.eval_split.upper()}  liver={summ['liver_mean']:.4f}  "
          f"tumour={summ['tumour_mean_all']:.4f} over all {summ['n_volumes']} volumes "
          f"(LiTS convention)  "
          f"[tumour-bearing only: {summ['tumour_mean_cases']:.4f} over "
          f"{summ['n_tumour_bearing']}]")

    if args.results_csv:
        new = not os.path.exists(args.results_csv)
        with open(args.results_csv, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["arm", "frozen_blocks", "base_channels", "patch", "seed",
                            "params", "val_best_tumour", "test_liver",
                            "test_tumour_cases", "test_tumour_all",
                            "n_tumour_bearing", "epochs", "minutes",
                            "eval_split", "min_tumour_size", "postprocess"])
            w.writerow([args.arm, frozen, args.base_channels, patch[0], args.seed,
                        n_params, round(best, 4), round(summ["liver_mean"], 4),
                        round(summ["tumour_mean_cases"], 4),
                        round(summ["tumour_mean_all"], 4),
                        summ["n_tumour_bearing"], args.epochs,
                        round((time.time() - t0) / 60, 1),
                        args.eval_split,
                        args.min_tumour_size if args.postprocess else "",
                        int(bool(args.postprocess))])
        print(f"Appended to {args.results_csv}")

    if ckpt_path and os.path.exists(ckpt_path) and not args.keep_checkpoint:
        os.remove(ckpt_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--arm", type=str, default="depthwise",
                   choices=["standard", "frozen_shift", "fair_shift", "depthwise", "hybrid"])
    p.add_argument("--frozen_blocks", type=str, default="")
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--norm", type=str, default="instance",
                   choices=["instance", "batch", "group"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--samples_per_epoch", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--class_weights", type=float, nargs=3, default=[0.2, 1.0, 4.0])
    p.add_argument("--label_mode", type=str, default="argmax", choices=["argmax", "region"],
                   help="'region' predicts nested liver-or-tumour and tumour channels "
                        "with sigmoid, so a voxel can be both. Stops the network "
                        "resolving hypodense lesions by calling them background.")
    p.add_argument("--region_pos_weight", type=float, nargs=2, default=[1.0, 4.0])
    p.add_argument("--postprocess", action="store_true",
                   help="Largest-component liver, hole filling, tumour constrained to "
                        "liver, small-component removal, applied at test time.")
    p.add_argument("--min_tumour_size", type=int, default=0)
    p.add_argument("--log_file", type=str, default="",
                   help="Append all output here. Defaults to a file in "
                        "--checkpoint_dir, so trajectories survive a disconnect.")
    p.add_argument("--eval_split", type=str, default="test", choices=["val", "test"],
                   help="Which split the final evaluation reports. Use 'val' to tune "
                        "post-processing without touching test.")
    p.add_argument("--select_metric", type=str, default="tumour_mean_all",
                   choices=["tumour_mean_all", "tumour_mean_cases", "liver_mean"],
                   help="Which validation metric picks the best epoch. Keep it the "
                        "same as the metric you report.")
    p.add_argument("--fg_fraction", type=float, default=0.66)
    p.add_argument("--tumour_fraction", type=float, default=0.33)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--val_every", type=int, default=10)
    p.add_argument("--val_max_cases", type=int, default=0,
                   help="Validate on only the first N volumes to save time; 0 = all")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--checkpoint_dir", type=str, default="")
    p.add_argument("--checkpoint_every", type=int, default=5)
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--keep_checkpoint", action="store_true")
    p.add_argument("--results_csv", type=str, default="")
    main(p.parse_args())
