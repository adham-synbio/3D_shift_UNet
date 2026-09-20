import argparse
import csv
import inspect
import os

import numpy as np
import torch

from model import UNet, LearnableShiftConv2d, count_parameters


def peakiness(w):
    a = w.abs()
    s = a.sum()
    return (a.max() / s).item() if s > 0 else 0.0


def centre_of_mass_offset(w):
    a = w.abs().numpy()
    k = a.shape[-1]
    c = (k - 1) / 2.0
    tot = a.sum()
    if tot == 0:
        return 0.0, 0.0
    rows = np.arange(k)[:, None] * np.ones((1, k))
    cols = np.ones((k, 1)) * np.arange(k)[None, :]
    return float((a * rows).sum() / tot - c), float((a * cols).sum() / tot - c)


def dc_gain(w):
    return w.sum().item()


def quantize_to_shift(w):
    q = torch.zeros_like(w)
    idx = w.abs().argmax()
    r, c = divmod(idx.item(), w.shape[-1])
    q[r, c] = w.sum()
    return q


def analyse(model):
    rows = []
    for name, module in model.named_modules():
        if not isinstance(module, LearnableShiftConv2d):
            continue
        w = module.depthwise.weight.detach().cpu()  # (C, 1, k, k)
        C, _, k, _ = w.shape

        peaks, dcs, offs, cos_to_shift = [], [], [], []
        for c in range(C):
            f = w[c, 0]
            peaks.append(peakiness(f))
            dcs.append(dc_gain(f))
            dr, dc_ = centre_of_mass_offset(f)
            offs.append((dr ** 2 + dc_ ** 2) ** 0.5)
            q = quantize_to_shift(f)
            denom = f.norm() * q.norm()
            cos_to_shift.append((f * q).sum().item() / denom.item() if denom > 0 else 0.0)

        rows.append({
            "layer": name,
            "channels": C,
            "kernel": k,
            "peakiness_mean": float(np.mean(peaks)),
            "peakiness_std": float(np.std(peaks)),
            "cos_to_nearest_shift": float(np.mean(cos_to_shift)),
            "offset_magnitude_mean": float(np.mean(offs)),
            "dc_gain_mean": float(np.mean(dcs)),
            "frac_highpass": float(np.mean([abs(d) < 0.1 for d in dcs])),
        })
    return rows


def main(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if "model_config" in ckpt:
        cfg = dict(ckpt["model_config"])
        print("Rebuilding from the checkpoint's stored model_config.")
    else:
        cfg = dict(n_channels=args.n_channels, n_classes=3, bilinear=True,
                   use_shift_conv=args.use_shift_conv,
                   use_learnable_shift=args.use_learnable_shift)
        print("No model_config in checkpoint; using command-line flags.")

    supported = inspect.signature(UNet.__init__).parameters
    cfg = {k: v for k, v in cfg.items() if k in supported}
    model = UNet(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])

    rows = analyse(model)
    if not rows:
        raise SystemExit("No LearnableShiftConv2d layers found. This checkpoint is "
                         "a frozen-shift or standard-conv model; there is nothing to measure.")

    k = rows[0]["kernel"]
    flat = 1.0 / (k * k)
    print(f"\n{'layer':<34}{'ch':>5}{'peak':>8}{'cos':>7}{'offset':>8}{'DC':>7}{'HP%':>6}")
    print("-" * 75)
    for r in rows:
        print(f"{r['layer']:<34}{r['channels']:>5}{r['peakiness_mean']:>8.3f}"
              f"{r['cos_to_nearest_shift']:>7.3f}{r['offset_magnitude_mean']:>8.3f}"
              f"{r['dc_gain_mean']:>7.3f}{100*r['frac_highpass']:>5.0f}%")

    peaks = np.array([r["peakiness_mean"] for r in rows])
    cos = np.array([r["cos_to_nearest_shift"] for r in rows])

    print(f"\nReference points for 'peak': 1.000 = exact shift, {flat:.3f} = flat blur")
    print(f"Network mean peakiness : {peaks.mean():.3f}")
    print(f"Network mean cosine to nearest shift : {cos.mean():.3f}")

    n_enc = min(5, len(rows))
    print(f"\nEncoder (first {n_enc} blocks) peakiness : {peaks[:n_enc].mean():.3f}")
    print(f"Decoder (remaining blocks) peakiness   : {peaks[n_enc:].mean():.3f}")

    print("\nInterpretation:")
    if cos.mean() > 0.90:
        print("  Filters stayed close to single-tap shifts. Snapping them back to exact")
    elif cos.mean() > 0.70:
        print("  Partial drift. Some layers are shift-like and some are not; the")
    else:
        print("  Filters drifted well away from single-tap shifts. This is direct")
        

    
    tap = None
    n_ch = 0
    for module in model.modules():
        if isinstance(module, LearnableShiftConv2d):
            w = module.depthwise.weight.detach().cpu().abs()
            w = w / (w.sum(dim=(2, 3), keepdim=True) + 1e-12)   # scale-normalise per filter
            s = w.sum(dim=(0, 1))
            tap = s if tap is None else tap + s
            n_ch += w.shape[0]
    tap = (tap / n_ch).numpy()
    for r in range(tap.shape[0]):
        print("   " + "  ".join(f"{v:.4f}" for v in tap[r]))
    flat_val = 1.0 / tap.size
    dev = float(np.abs(tap - flat_val).max() / flat_val)
    print(f"   flat reference = {flat_val:.4f}; max deviation = {100*dev:.1f}%")

    out_csv = os.path.join(os.path.dirname(args.checkpoint) or ".", "depthwise_filter_analysis.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved per-layer measurements to {out_csv}")

    if args.export_shuffled:
        g = torch.Generator().manual_seed(0)
        n = 0
        for module in model.modules():
            if isinstance(module, LearnableShiftConv2d):
                w = module.depthwise.weight.data
                C, _, kh, kw = w.shape
                flat = w.view(C, kh * kw)
                for c in range(C):
                    flat[c] = flat[c][torch.randperm(kh * kw, generator=g)]
                w.copy_(flat.view(C, 1, kh, kw))
                n += C
        ckpt["model_state_dict"] = model.state_dict()
        ckpt["depthwise_shuffled"] = True
        torch.save(ckpt, args.export_shuffled)
        print(f"\nShuffled taps within {n} depthwise filters.")
        print(f"Wrote {args.export_shuffled}")

    if args.export_randomized:
        torch.manual_seed(0)
        n = 0
        for module in model.modules():
            if isinstance(module, LearnableShiftConv2d):
                torch.nn.init.kaiming_uniform_(module.depthwise.weight, a=5 ** 0.5)
                n += module.depthwise.weight.shape[0]
        ckpt["model_state_dict"] = model.state_dict()
        ckpt["depthwise_randomized"] = True
        torch.save(ckpt, args.export_randomized)
        print(f"\nRandomized {n} depthwise filters; all other weights untouched.")
        print(f"Wrote {args.export_randomized}")
        
    if args.export_quantized:
        n_filters = 0
        for module in model.modules():
            if isinstance(module, LearnableShiftConv2d):
                w = module.depthwise.weight.data
                for c in range(w.shape[0]):
                    w[c, 0] = quantize_to_shift(w[c, 0])
                    n_filters += 1
        ckpt["model_state_dict"] = model.state_dict()
        ckpt["shift_quantized"] = True
        torch.save(ckpt, args.export_quantized)
        print(f"\nQuantized {n_filters} depthwise filters to exact single-tap shifts.")
        print(f"Wrote {args.export_quantized}")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_channels", type=int, default=3)
    parser.add_argument("--use_shift_conv", action="store_true")
    parser.add_argument("--use_learnable_shift", action="store_true")
    parser.add_argument("--export_shuffled", type=str, default=None,
                        help="Permute taps within each filter. Norm-preserving, so it "
                             "isolates spatial structure from BatchNorm mismatch.")
    parser.add_argument("--export_randomized", type=str, default=None,
                        help="Write a copy with ONLY the depthwise filters re-randomized, "
                             "as a control: if accuracy survives, they were not trained.")
    parser.add_argument("--export_quantized", type=str, default=None,
                        help="Write a copy with every depthwise filter snapped to its "
                             "nearest exact shift, to measure what freezing costs.")
    args = parser.parse_args()
    main(args)
