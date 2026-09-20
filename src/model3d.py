import torch
import torch.nn as nn
import torch.nn.functional as F

SHIFT_OFFSETS_3D = [(dz, dy, dx)
                    for dz in (-1, 0, 1)
                    for dy in (-1, 0, 1)
                    for dx in (-1, 0, 1)
                    if not (dz == 0 and dy == 0 and dx == 0)]
assert len(SHIFT_OFFSETS_3D) == 26


class ShiftConv3d(nn.Module):
    """Fixed 3D shift: channels split into 27 groups
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.n_div = 27
        if in_channels % self.n_div != 0:
            padded = ((in_channels + self.n_div - 1) // self.n_div) * self.n_div
            self.adjust_input = nn.Conv3d(in_channels, padded, kernel_size=1, bias=False)
            self.adjusted_in_channels = padded
        else:
            self.adjust_input = nn.Identity()
            self.adjusted_in_channels = in_channels

        per_group = self.adjusted_in_channels // self.n_div
        self.per_group = per_group
        self.fold = per_group * (self.n_div - 1)

        self.shift_kernel = nn.Parameter(torch.zeros(self.fold, 1, 3, 3, 3),
                                         requires_grad=False)
        self.pointwise_conv = nn.Conv3d(self.adjusted_in_channels, out_channels,
                                        kernel_size=1, bias=bias)
        self._init_shift_kernel()

    def _init_shift_kernel(self):
        mask = torch.zeros(self.fold, 1, 3, 3, 3)
        for i, (dz, dy, dx) in enumerate(SHIFT_OFFSETS_3D):
            s, e = i * self.per_group, (i + 1) * self.per_group
            if s >= self.fold:
                break
            mask[s:e, 0, 1 + dz, 1 + dy, 1 + dx] = 1.0
        self.shift_kernel.data = mask

    def forward(self, x):
        x = self.adjust_input(x)
        shifted = F.conv3d(x[:, :self.fold], self.shift_kernel, stride=1,
                           padding=1, groups=self.fold)
        out = torch.cat([shifted, x[:, self.fold:]], dim=1)
        return self.pointwise_conv(out)


class LearnableShiftConv3d(nn.Module):
    """Learnable depthwise 3x3x3 then pointwise 1x1x1
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=None, bias=True, mid_norm=False, shift_init=False):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if padding is None:
            padding = kernel_size // 2
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding,
                                   groups=in_channels, bias=False)
        self.mid = (nn.Sequential(nn.BatchNorm3d(in_channels), nn.ReLU(inplace=True))
                    if mid_norm else nn.Identity())
        self.pointwise_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1,
                                        stride=1, bias=bias)
        if shift_init:
            self._initialize_from_shift_pattern()

    def _initialize_from_shift_pattern(self):
        k = self.kernel_size
        c0 = k // 2
        w = torch.zeros(self.in_channels, 1, k, k, k)
        group = max(1, (self.in_channels + 26) // 27)
        for c in range(self.in_channels):
            g = min(c // group, 26)
            if g == 26:
                dz = dy = dx = 0
            else:
                dz, dy, dx = SHIFT_OFFSETS_3D[g]
            w[c, 0, c0 + dz, c0 + dy, c0 + dx] = 1.0
        self.depthwise.weight.data.copy_(w)

    def forward(self, x):
        return self.pointwise_conv(self.mid(self.depthwise(x)))


class DoubleConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 use_shift_conv=False, use_learnable_shift=False,
                 dw_kernel_size=3, dw_mid_norm=False, dw_shift_init=False,
                 norm="instance"):
        super().__init__()
        mid_channels = mid_channels or out_channels

        if use_learnable_shift:
            kw = dict(kernel_size=dw_kernel_size, mid_norm=dw_mid_norm,
                      shift_init=dw_shift_init)
            self.conv1 = LearnableShiftConv3d(in_channels, mid_channels, **kw)
            self.conv2 = LearnableShiftConv3d(mid_channels, out_channels, **kw)
        elif use_shift_conv:
            self.conv1 = ShiftConv3d(in_channels, mid_channels)
            self.conv2 = ShiftConv3d(mid_channels, out_channels)
        else:
            self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1)

        
        def make_norm(c):
            if norm == "batch":
                return nn.BatchNorm3d(c)
            if norm == "group":
                return nn.GroupNorm(min(8, c), c)
            return nn.InstanceNorm3d(c, affine=True)

        self.norm1, self.norm2 = make_norm(mid_channels), make_norm(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        return self.relu(self.norm2(self.conv2(x)))


class Down3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kw):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool3d(2), DoubleConv3d(in_channels, out_channels, **kw))

    def forward(self, x):
        return self.block(x)


class Up3d(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=True, **kw):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            self.conv = DoubleConv3d(in_channels, out_channels, in_channels // 2, **kw)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3d(in_channels, out_channels, **kw)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        d = [x2.size(i) - x1.size(i) for i in (2, 3, 4)]
        x1 = F.pad(x1, [d[2] // 2, d[2] - d[2] // 2,
                        d[1] // 2, d[1] - d[1] // 2,
                        d[0] // 2, d[0] - d[0] // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet3D(nn.Module):
    
    def __init__(self, n_channels=1, n_classes=3, trilinear=True, base_channels=32,
                 use_shift_conv=False, use_learnable_shift=False,
                 dw_kernel_size=3, dw_mid_norm=False, dw_shift_init=False,
                 norm="instance"):
        super().__init__()
        self.n_channels, self.n_classes = n_channels, n_classes
        self.trilinear, self.base_channels = trilinear, base_channels
        self.use_shift_conv, self.use_learnable_shift = use_shift_conv, use_learnable_shift
        self.dw_kernel_size, self.dw_mid_norm = dw_kernel_size, dw_mid_norm
        self.dw_shift_init, self.norm = dw_shift_init, norm

        kw = dict(use_shift_conv=use_shift_conv, use_learnable_shift=use_learnable_shift,
                  dw_kernel_size=dw_kernel_size, dw_mid_norm=dw_mid_norm,
                  dw_shift_init=dw_shift_init, norm=norm)
        c = base_channels
        self.inc = DoubleConv3d(n_channels, c, **kw)
        self.down1 = Down3d(c, c * 2, **kw)
        self.down2 = Down3d(c * 2, c * 4, **kw)
        self.down3 = Down3d(c * 4, c * 8, **kw)
        factor = 2 if trilinear else 1
        self.down4 = Down3d(c * 8, c * 16 // factor, **kw)

        self.up1 = Up3d(c * 16, c * 8 // factor, trilinear, **kw)
        self.up2 = Up3d(c * 8, c * 4 // factor, trilinear, **kw)
        self.up3 = Up3d(c * 4, c * 2 // factor, trilinear, **kw)
        self.up4 = Up3d(c * 2, c, trilinear, **kw)
        self.outc = nn.Conv3d(c, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def freeze_blocks_to_shift_3d(model, prefixes, verbose=True):
    """Pin selected blocks to fixed one-hot shifts and stop training them."""
    frozen = []
    for name, m in model.named_modules():
        if isinstance(m, LearnableShiftConv3d):
            if "*" in prefixes or name.split(".")[0] in prefixes:
                m._initialize_from_shift_pattern()
                m.depthwise.weight.requires_grad_(False)
                frozen.append(name)
    if verbose:
        print(f"Froze {len(frozen)} depthwise layers to fixed 3D shifts "
              f"(blocks: {sorted(set(n.split('.')[0] for n in frozen))})")
    return frozen


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 1, 32, 32, 32).to(device)

    print(f"{'variant':<34}{'base=32':>12}{'base=64':>12}")
    print("-" * 58)
    variants = [("standard conv 3D", dict()),
                ("frozen ShiftConv3d", dict(use_shift_conv=True)),
                ("learnable depthwise 3D", dict(use_learnable_shift=True))]
    for name, cfg in variants:
        row = []
        for bc in (32, 64):
            m = UNet3D(1, 3, base_channels=bc, **cfg)
            row.append(count_parameters(m))
        print(f"{name:<34}{row[0]:>12,}{row[1]:>12,}")

    m = UNet3D(1, 3, base_channels=32, use_learnable_shift=True).to(device).eval()
    with torch.no_grad():
        print(f"\nforward: {tuple(x.shape)} -> {tuple(m(x).shape)}")
    fair = UNet3D(1, 3, base_channels=32, use_learnable_shift=True).to(device)
    freeze_blocks_to_shift_3d(fair, ("*",))
    print(f"fair 3D shift (all frozen): {count_parameters(fair):,}")
