import argparse
import csv
import os
import random
import time
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from model import (UNet, count_parameters, freeze_blocks_to_shift,
                   spatial_tap_summary)

try:
    from l0_depthwise import collect_l0_penalty, tap_report, print_tap_report
except ImportError:
    collect_l0_penalty = tap_report = print_tap_report = None

try:
    from augmentation import SliceAugmenter
except ImportError:
    SliceAugmenter = None


class ImageMaskDataset(Dataset):
    def __init__(self, pairs, img_size=256, is_train=False, seed=0):
        self.pairs = pairs
        self.img_size = img_size
        self.is_train = is_train
        self.aug = (SliceAugmenter(seed=seed, brightness=0.10)
                    if (is_train and SliceAugmenter is not None) else None)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = Image.open(ip).convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        msk = Image.open(mp).convert("L").resize((self.img_size, self.img_size), Image.NEAREST)

        image = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mask = (np.asarray(msk, dtype=np.uint8) > 127).astype(np.int64)

        if self.aug is not None:
            image, mask = self.aug(image, mask)

        return torch.from_numpy(np.ascontiguousarray(image)).float(), \
               torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)


def build_presplit(data_dir):
    out = []
    for split in ("train", "val", "test"):
        folder = os.path.join(data_dir, split)
        imgs = sorted(glob(os.path.join(folder, "images", "*")))
        pairs = [(ip, os.path.join(folder, "masks", os.path.basename(ip)))
                 for ip in imgs
                 if os.path.exists(os.path.join(folder, "masks", os.path.basename(ip)))]
        if not pairs:
            raise SystemExit(f"No image/mask pairs in {folder}. "
                             f"--presplit expects {data_dir}/{{train,val,test}}/{{images,masks}}")
        out.append(pairs)
    return tuple(out)


def build_splits(data_dir, split_seed=0, val_frac=0.1, test_frac=0.1):
    """Split is controlled by split_seed ONLY, never by the run seed."""
    imgs = sorted(glob(os.path.join(data_dir, "images", "*")))
    if not imgs:
        raise SystemExit(f"No files in {os.path.join(data_dir, 'images')}. "
                         f"Expected {data_dir}/images and {data_dir}/masks.")
    pairs = []
    for ip in imgs:
        mp = os.path.join(data_dir, "masks", os.path.basename(ip))
        if os.path.exists(mp):
            pairs.append((ip, mp))
    if not pairs:
        raise SystemExit("Found images but no matching masks (filenames must match).")

    rng = random.Random(split_seed)
    rng.shuffle(pairs)
    n = len(pairs)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    return pairs[n_test + n_val:], pairs[n_test:n_test + n_val], pairs[:n_test]


def dice_iou(logits, target, eps=1e-6):
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    psum = pred.sum(dim=(1, 2, 3))
    tsum = target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (psum + tsum + eps)
    iou = (inter + eps) / (psum + tsum - inter + eps)
    return dice.sum().item(), iou.sum().item()


class DiceBCELoss(nn.Module):
    def __init__(self, eps=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.eps = eps

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        inter = (p * target).sum()
        d = 1 - (2 * inter + self.eps) / (p.sum() + target.sum() + self.eps)
        return self.bce(logits, target) + d


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    d = i = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        bd, bi = dice_iou(out, y)
        d += bd; i += bi; n += x.shape[0]
    return d / n, i / n


def main(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.presplit:
        train_p, val_p, test_p = build_presplit(args.data_dir)
        print(f"{len(train_p)} train / {len(val_p)} val / {len(test_p)} test images "
              f"(pre-split on disk, run seed={args.seed})")
    else:
        train_p, val_p, test_p = build_splits(args.data_dir, split_seed=args.split_seed)
        print(f"{len(train_p)} train / {len(val_p)} val / {len(test_p)} test images "
              f"(split_seed={args.split_seed}, run seed={args.seed})")

    loaders = {}
    for name, pairs, train in [("train", train_p, True), ("val", val_p, False), ("test", test_p, False)]:
        ds = ImageMaskDataset(pairs, args.img_size, is_train=train, seed=args.seed)
        loaders[name] = DataLoader(ds, batch_size=args.batch_size, shuffle=train,
                                   num_workers=args.num_workers, drop_last=train)

    cfg = dict(n_channels=3, n_classes=1, bilinear=True)
    if args.arm == "frozen_shift":
        cfg.update(use_shift_conv=True)
    elif args.arm in ("depthwise", "fair_shift", "hybrid"):
        cfg.update(use_learnable_shift=True)
    elif args.arm == "s2_shift":
        
        cfg.update(use_s2_shift=True)
    elif args.arm == "tok_shift":
        
        cfg.update(use_tok_shift=True)
    elif args.arm == "l0":
        cfg.update(use_learnable_shift=True, dw_l0=True)
    # "standard" leaves both flags off -> plain convolutional U-Net
    cfg.update(dw_kernel_size=args.dw_kernel_size, dw_mid_norm=args.dw_mid_norm)
    model = UNet(**cfg).to(device)

    frozen_blocks = ""
    if args.arm == "fair_shift":
        
        freeze_blocks_to_shift(model, ("*",))
        frozen_blocks = "*"
    elif args.arm == "hybrid":
        blocks = tuple(b.strip() for b in args.frozen_blocks.split(",") if b.strip())
        if not blocks:
            raise SystemExit("--arm hybrid needs --frozen_blocks, e.g. up1,up2,up3")
        freeze_blocks_to_shift(model, blocks)
        frozen_blocks = args.frozen_blocks

    
    gate_params = [p for n, p in model.named_parameters()
                   if "log_alpha" in n and p.requires_grad]
    other = [p for n, p in model.named_parameters()
             if "log_alpha" not in n and p.requires_grad]
    groups = [{"params": other, "lr": args.lr}]
    if gate_params:
        
        groups.append({"params": gate_params, "lr": args.gate_lr, "weight_decay": 0.0})
    if args.optimizer == "adam":
        opt = torch.optim.Adam(groups, lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=args.min_lr)
    criterion = DiceBCELoss()

    n_params = count_parameters(model)
    print(f"arm={args.arm} lambda={args.l0_lambda:g} seed={args.seed} "
          f"params={n_params:,}")

    # ---- resume support -------------------------------------------------
    
    ckpt_path = None
    start_epoch, best_val, best_state = 0, -1.0, None
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        tag = (f"{args.arm}{'_' + frozen_blocks.replace(',', '-') if frozen_blocks else ''}"
               f"_lam{args.l0_lambda:g}_seed{args.seed}_split{args.split_seed}"
               f"_k{args.dw_kernel_size}_sz{args.img_size}_ep{args.epochs}")
        ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_{tag}.pth")

        if os.path.exists(ckpt_path) and not args.no_resume:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            if ck.get("n_params") != n_params:
                raise SystemExit(
                    f"\n{ckpt_path} holds a model with {ck.get('n_params'):,} parameters "
                    f"but this run builds {n_params:,}.\nMove it aside or pass --no_resume.")
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["scheduler"])
            start_epoch = ck["epoch"] + 1
            best_val = ck["best_val"]
            best_state = ck["best_state"]
            print(f"Resumed from {ckpt_path} at epoch {start_epoch}/{args.epochs} "
                  f"(best val_dice so far {best_val:.4f})")

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        tot = 0.0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            if args.arm == "l0" and args.l0_lambda > 0:
                loss = loss + args.l0_lambda * collect_l0_penalty(model)
            loss.backward(); opt.step()
            tot += loss.item()
        sched.step()

        vd, vi = evaluate(model, loaders["val"], device)
        if vd > best_val:
            best_val = vd
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            print(f"  epoch {epoch+1:>3}/{args.epochs}  loss={tot/len(loaders['train']):.4f}  "
                  f"val_dice={vd:.4f}  best={best_val:.4f}", flush=True)

        if ckpt_path and ((epoch + 1) % args.checkpoint_every == 0
                          or epoch == args.epochs - 1):
            # Written to a temporary file then renamed, so a disconnect during
            # the write cannot leave a truncated checkpoint behind.
            tmp = ckpt_path + ".tmp"
            torch.save({"epoch": epoch, "best_val": best_val, "best_state": best_state,
                        "model": model.state_dict(), "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(), "n_params": n_params,
                        "arm": args.arm, "frozen_blocks": frozen_blocks}, tmp)
            os.replace(tmp, ckpt_path)

    if best_state is None:
        raise SystemExit("No epochs ran. Is --epochs smaller than the resumed epoch count?")
    model.load_state_dict(best_state)
    td, ti = evaluate(model, loaders["test"], device)
    mins = (time.time() - t0) / 60

    mean_taps = max_taps = float("nan")
    frac_shift = float("nan")
    if args.arm == "l0" and tap_report is not None:
        rows = tap_report(model)
        print_tap_report(rows)
        ch = sum(r["channels"] for r in rows)
        mean_taps = sum(r["mean_taps"] * r["channels"] for r in rows) / ch
        max_taps = rows[0]["max_taps"]
        frac_shift = sum(r["frac_pure_shift"] * r["channels"] for r in rows) / ch
    elif args.arm == "frozen_shift":
        mean_taps, max_taps, frac_shift = 1.0, args.dw_kernel_size ** 2, 1.0
    elif args.arm in ("depthwise", "fair_shift", "hybrid"):
        k2 = args.dw_kernel_size ** 2
        mean_taps, max_taps = spatial_tap_summary(model), k2
        frac_shift = (k2 - mean_taps) / (k2 - 1) if k2 > 1 else 0.0
    elif args.arm == "s2_shift":
        mean_taps, max_taps, frac_shift = 1.0, args.dw_kernel_size ** 2, 1.0
    elif args.arm == "tok_shift":
        mean_taps = max_taps = args.dw_kernel_size ** 2
        frac_shift = 0.0
    elif args.arm == "standard":
        mean_taps = max_taps = args.dw_kernel_size ** 2
        frac_shift = 0.0

    print(f"\nTEST  dice={td:.4f}  iou={ti:.4f}  "
          f"mean_taps={mean_taps:.2f}  pure_shift={100*frac_shift:.1f}%  ({mins:.1f} min)")

    new = not os.path.exists(args.results_csv)
    with open(args.results_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["arm", "l0_lambda", "seed", "split_seed", "dw_kernel_size", "dw_mid_norm",
                        "params", "mean_taps", "max_taps", "frac_pure_shift",
                        "val_dice", "test_dice", "test_iou", "epochs", "minutes"])
        w.writerow([args.arm + ("" if not frozen_blocks else f"[{frozen_blocks}]"),
                    args.l0_lambda, args.seed, args.split_seed,
                    args.dw_kernel_size, args.dw_mid_norm,
                    n_params, round(mean_taps, 4), max_taps, round(frac_shift, 4),
                    round(best_val, 4), round(td, 4), round(ti, 4), args.epochs, round(mins, 2)])
    print(f"Appended to {args.results_csv}")
    if ckpt_path and os.path.exists(ckpt_path) and not args.keep_checkpoint:
        os.remove(ckpt_path)
        print("Removed the resume checkpoint (run complete).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory containing images/ and masks/ with matching filenames")
    p.add_argument("--arm", type=str, required=True,
                   choices=["standard", "frozen_shift", "fair_shift", "s2_shift", "tok_shift", "depthwise", "hybrid", "l0"],
                   help="standard=plain conv U-Net; frozen_shift=original ShiftConv2d; "
                        "fair_shift=shift without the adjust_input projection; "
                        "s2_shift=the S2-MLPv2 shift operator alone; "
                        "tok_shift=LSU-Net's full Tokenized Shift Block, Eq.8-10, "
                        "NOT parameter-matched (~8.3M at these widths); "
                        "depthwise=fully learnable; hybrid=learnable except --frozen_blocks; "
                        "l0=learned per-channel tap budget")
    p.add_argument("--frozen_blocks", type=str, default="",
                   help="For --arm hybrid: comma-separated block names to pin to fixed "
                        "shifts, e.g. up1,up2,up3. Blocks are inc,down1..down4,up1..up4.")
    p.add_argument("--l0_lambda", type=float, default=0.0)
    p.add_argument("--gate_lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0,
                   help="Initialization, augmentation and shuffling. Vary this across runs.")
    p.add_argument("--checkpoint_dir", type=str, default="",
                   help="Enables resume. Save on Drive so a disconnect costs at most "
                        "--checkpoint_every epochs. Deleted automatically on completion.")
    p.add_argument("--checkpoint_every", type=int, default=5,
                   help="Epochs between checkpoint writes. Writing every epoch is slow "
                        "over a mounted Drive.")
    p.add_argument("--no_resume", action="store_true",
                   help="Ignore an existing checkpoint and start over.")
    p.add_argument("--keep_checkpoint", action="store_true",
                   help="Keep the resume checkpoint after the run finishes.")
    p.add_argument("--presplit", action="store_true",
                   help="Read <data_dir>/{train,val,test}/{images,masks} instead of "
                        "splitting files. Required when the split must respect volume "
                        "or patient boundaries.")
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"],
                   help="Use 'adam' to match published protocols that specify it.")
    p.add_argument("--min_lr", type=float, default=0.0,
                   help="eta_min for CosineAnnealingLR.")
    p.add_argument("--split_seed", type=int, default=0,
                   help="Controls the train/val/test split ONLY. Keep fixed across every "
                        "arm and seed so paired comparisons measure the architecture, not "
                        "which images landed in the test set.")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--dw_kernel_size", type=int, default=3)
    p.add_argument("--dw_mid_norm", action="store_true")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--results_csv", type=str, default="sparse_shift_results.csv")
    main(p.parse_args())
