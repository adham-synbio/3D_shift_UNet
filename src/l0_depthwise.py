import math

import torch
import torch.nn as nn

BETA = 2.0 / 3.0      # temperature
GAMMA = -0.1          # stretch below zero
ZETA = 1.1            # stretch above one
# Offset relating log_alpha to the probability a gate is open after stretching
_OFFSET = BETA * math.log(-GAMMA / ZETA)


class HardConcreteGate(nn.Module):

    def __init__(self, shape, init_open_prob=0.95, min_open=1):
        super().__init__()
        self.min_open = min_open
        logit = math.log(init_open_prob / (1 - init_open_prob))
        init = logit + _OFFSET
        self.log_alpha = nn.Parameter(torch.full(shape, init) + 0.01 * torch.randn(shape))

    def forward(self):
        if self.training:
            u = torch.rand_like(self.log_alpha).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + self.log_alpha) / BETA)
        else:
            s = torch.sigmoid(self.log_alpha / BETA)
        z = torch.clamp(s * (ZETA - GAMMA) + GAMMA, 0.0, 1.0)
        return self._apply_floor(z)

    def _apply_floor(self, z):
        if not self.min_open:
            return z
        C = z.shape[0]
        flat = self.log_alpha.detach().view(C, -1)
        keep = torch.zeros_like(flat)
        idx = flat.topk(self.min_open, dim=1).indices
        keep.scatter_(1, idx, 1.0)
        return torch.maximum(z, keep.view_as(z))

    def open_prob(self):
        return torch.sigmoid(self.log_alpha - _OFFSET)

    def expected_l0(self):
        return self.open_prob().sum()

    def hard_mask(self):
        return self._apply_floor((self.open_prob() > 0.5).float())


def collect_l0_penalty(model):
    total = None
    for module in model.modules():
        if isinstance(module, HardConcreteGate):
            e = module.expected_l0()
            total = e if total is None else total + e
    if total is None:
        return torch.zeros((), device=next(model.parameters()).device)
    return total


def tap_report(model):
    rows = []
    for name, module in model.named_modules():
        gate = getattr(module, "l0_gate", None)
        if not isinstance(gate, HardConcreteGate):
            continue
        hard = gate.hard_mask()
        C = hard.shape[0]
        k2 = hard[0].numel()
        per_filter = hard.view(C, -1).sum(dim=1)
        rows.append({
            "layer": name,
            "channels": C,
            "max_taps": k2,
            "mean_taps": per_filter.mean().item(),
            "frac_pure_shift": (per_filter == 1).float().mean().item(),
            "frac_dead": (per_filter == 0).float().mean().item(),
            "expected_taps": (gate.open_prob().view(C, -1).sum(dim=1)).mean().item(),
        })
    return rows


def print_tap_report(rows):
    if not rows:
        print("  (no L0 gates in this model)")
        return
    print(f"\n{'layer':<34}{'ch':>6}{'taps':>7}{'of':>4}{'shift%':>8}{'dead%':>7}")
    print("-" * 68)
    for r in rows:
        print(f"{r['layer']:<34}{r['channels']:>6}{r['mean_taps']:>7.2f}{r['max_taps']:>4}"
              f"{100*r['frac_pure_shift']:>7.0f}%{100*r['frac_dead']:>6.0f}%")
    total_taps = sum(r["mean_taps"] * r["channels"] for r in rows)
    total_max = sum(r["max_taps"] * r["channels"] for r in rows)
    print(f"\n  Spatial parameters kept: {total_taps:,.0f} of {total_max:,} "
          f"({100*total_taps/total_max:.1f}%)")
    print(f"  Filters collapsed to a pure shift: "
          f"{100*sum(r['frac_pure_shift']*r['channels'] for r in rows)/sum(r['channels'] for r in rows):.1f}%")
