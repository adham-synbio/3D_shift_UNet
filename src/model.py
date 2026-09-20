import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from l0_depthwise import HardConcreteGate
except ImportError:      
    HardConcreteGate = None


class ShiftConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(ShiftConv2d, self).__init__()

        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

        # For kernel_size=3, we have 9 groups (8 shifts + 1 center)
        self.n_div = 9

        # Adjust input channels to be divisible by 9
        if in_channels % self.n_div != 0:
            self.adjust_input = nn.Conv2d(in_channels, ((in_channels + self.n_div - 1) // self.n_div) * self.n_div,
                                          kernel_size=1, stride=1, bias=False)
            self.adjusted_in_channels = ((in_channels + self.n_div - 1) // self.n_div) * self.n_div
        else:
            self.adjust_input = nn.Identity()
            self.adjusted_in_channels = in_channels

        in_channels_per_group = self.adjusted_in_channels // self.n_div
        self.in_channels_per_group = in_channels_per_group
        self.fold = in_channels_per_group * (self.n_div - 1)  # Channels to be shifted

        # Fixed shift kernel (non-learnable)
        self.shift_kernel = nn.Parameter(torch.zeros(size=[self.fold, 1, 3, 3]),
                                         requires_grad=False)

        # Learnable 1x1 convolution
        self.pointwise_conv = nn.Conv2d(self.adjusted_in_channels, out_channels,
                                        kernel_size=1, stride=1,
                                        bias=bias)

        self._initialize_shift_kernel()

    def _initialize_shift_kernel(self):
        shift_mask = torch.zeros(size=[self.fold, 1, 3, 3])
        group_size = self.in_channels_per_group

        for i in range(8):  # 8 shift directions
            start = i * group_size
            end = (i + 1) * group_size

            shift_positions = [
                (0, 0), (0, 1), (0, 2),  # top row
                (1, 0),         (1, 2),  # middle row (center is 1,1)
                (2, 0), (2, 1), (2, 2)   # bottom row
            ]

            row, col = shift_positions[i]
            shift_mask[start:end, 0, row, col] = 1.0

        self.shift_kernel.data = shift_mask

    def _shift_features(self, x):
        batch, channels, height, width = x.shape

        x_shift = x[:, :self.fold, :, :]   # Channels to shift
        x_center = x[:, self.fold:, :, :]  # Center channels (no shift)

        x_shifted = F.conv2d(x_shift, self.shift_kernel, stride=1,
                             padding=1, groups=self.fold)

        out = torch.cat([x_shifted, x_center], dim=1)
        return out

    def forward(self, x):
        x = self.adjust_input(x)  # Adjust input channels if needed
        x_shifted = self._shift_features(x)
        out = self.pointwise_conv(x_shifted)
        return out


SHIFT_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]


class ChannelAttention(nn.Module):
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, mid, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = self.sigmoid(self.fc2(self.relu(self.fc1(w))))
        return x * w


class LearnableShiftConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None,
                 bias=True, mid_norm=False, shift_init=False, dilations=(1,), l0=False):
        super(LearnableShiftConv2d, self).__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        if padding is None:
            padding = kernel_size // 2

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dilations = tuple(dilations)

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)

        
        if l0:
            if HardConcreteGate is None:
                raise ImportError("dw_l0=True requires l0_depthwise.py on the path")
            self.l0_gate = HardConcreteGate((in_channels, 1, kernel_size, kernel_size))
        else:
            self.l0_gate = None

        self.extra_branches = nn.ModuleList()
        for d in self.dilations:
            if d == 1:
                continue
            branch = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                               stride=stride, padding=d * (kernel_size // 2),
                               dilation=d, groups=in_channels, bias=False)
            nn.init.zeros_(branch.weight)
            self.extra_branches.append(branch)

        if mid_norm:
            self.mid = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))
        else:
            self.mid = nn.Identity()

        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)

        if shift_init:
            self._initialize_from_shift_pattern()

    def _initialize_from_shift_pattern(self):
        k = self.kernel_size
        centre = k // 2
        weight = torch.zeros(self.in_channels, 1, k, k)

        
        group_size = max(1, (self.in_channels + 8) // 9)
        for c in range(self.in_channels):
            group = min(c // group_size, 8)
            if group == 8:
                row, col = centre, centre  # identity / no shift
            else:
                dr, dc = SHIFT_OFFSETS[group]
                row, col = centre + dr, centre + dc
            weight[c, 0, row, col] = 1.0

        self.depthwise.weight.data.copy_(weight)

    def forward(self, x):
        if self.l0_gate is not None:
            w = self.depthwise.weight * self.l0_gate()
            spatial = F.conv2d(x, w, None, self.depthwise.stride,
                               self.depthwise.padding, self.depthwise.dilation,
                               self.depthwise.groups)
        else:
            spatial = self.depthwise(x)
        for branch in self.extra_branches:
            spatial = spatial + branch(x)
        spatial = self.mid(spatial)
        return self.pointwise_conv(spatial)



class SpatialShiftS2MLP(nn.Module):
    """The Spatial Shift Block of S2-MLPv2 (Yu et al., 2021)
    """
    def __init__(self, mode="hw"):
        super().__init__()
        if mode not in ("hw", "wh"):
            raise ValueError("mode must be 'hw' (Eq. 6) or 'wh' (Eq. 7)")
        self.mode = mode

    def forward(self, x):
        _, c, _, _ = x.shape
        q = c // 4
        if q == 0:
            return x
        out = torch.zeros_like(x)
        a, b, d = q, 2 * q, 3 * q

        if self.mode == "hw":
            out[:, :a, 1:, :] = x[:, :a, :-1, :]      # shift down
            out[:, a:b, :-1, :] = x[:, a:b, 1:, :]    # shift up
            out[:, b:d, :, 1:] = x[:, b:d, :, :-1]    # shift right
            out[:, d:, :, :-1] = x[:, d:, :, 1:]      # shift left
        else:
            out[:, :a, :, 1:] = x[:, :a, :, :-1]
            out[:, a:b, :, :-1] = x[:, a:b, :, 1:]
            out[:, b:d, 1:, :] = x[:, b:d, :-1, :]
            out[:, d:, :-1, :] = x[:, d:, 1:, :]
        return out


class S2ShiftConv2d(nn.Module):
    """LSU-Net's Tokenized Shift arrangement
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=None, bias=True, mode="hw", residual=True):
        super().__init__()
        self.shift = SpatialShiftS2MLP(mode=mode)
        self.residual = residual
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                        stride=stride, bias=bias)

    def forward(self, x):
        s = self.shift(x)
        if self.residual:
            s = x + s          # Eq. 8
        return self.pointwise_conv(s)



class TokenizedShiftBlock(nn.Module):
    """LSU-Net's Tokenized Shift Block
    """
    def __init__(self, in_channels, out_channels, gn_groups=8, mode="hw"):
        super().__init__()
        self.shift = SpatialShiftS2MLP(mode=mode)

        # DWSConv: depthwise 3x3 then pointwise 1x1
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                            groups=in_channels, bias=False)
        self.pw = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        g = max(1, min(gn_groups, in_channels))
        while in_channels % g != 0:
            g -= 1
        self.gnorm = nn.GroupNorm(g, in_channels)

        self.conv_a = nn.Conv2d(in_channels, out_channels, kernel_size=1)   # Eq. 9
        self.conv_b = nn.Conv2d(out_channels, out_channels, kernel_size=1)  # Eq. 10
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) # Eq. 10
        self.gelu = nn.GELU()

    def forward(self, x):
        x1 = x + self.shift(x)                       # Eq. 8
        x2 = self.conv_a(self.gnorm(self.pw(self.dw(x1))))   # Eq. 9
        return self.conv_b(self.gelu(x2)) + self.shortcut(x)  # Eq. 10


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, use_shift_conv=False,
                 use_learnable_shift=False, dw_kernel_size=3, dw_mid_norm=False,
                 dw_shift_init=False, dw_dilations=(1,), use_se=False, se_reduction=16,
                 dw_l0=False, use_s2_shift=False, use_tok_shift=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.use_shift_conv = use_shift_conv
        self.use_learnable_shift = use_learnable_shift

        if use_tok_shift:
            
            self.conv1 = TokenizedShiftBlock(in_channels, mid_channels)
            self.conv2 = TokenizedShiftBlock(mid_channels, out_channels)
        elif use_s2_shift:
            
            self.conv1 = S2ShiftConv2d(in_channels, mid_channels)
            self.conv2 = S2ShiftConv2d(mid_channels, out_channels)
        elif use_learnable_shift:
            
            dw_kwargs = dict(kernel_size=dw_kernel_size, mid_norm=dw_mid_norm,
                             shift_init=dw_shift_init, dilations=dw_dilations, l0=dw_l0)
            self.conv1 = LearnableShiftConv2d(in_channels, mid_channels, **dw_kwargs)
            self.conv2 = LearnableShiftConv2d(mid_channels, out_channels, **dw_kwargs)
        elif use_shift_conv:
            
            self.conv1 = ShiftConv2d(in_channels, mid_channels, kernel_size=3, padding=1)
            self.conv2 = ShiftConv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        else:
            
            self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)

        self.norm1 = nn.BatchNorm2d(mid_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.se = ChannelAttention(out_channels, se_reduction) if use_se else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        return self.se(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, **kwargs)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, **kwargs):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, **kwargs)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, **kwargs)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, use_shift_conv=False,
                 use_learnable_shift=False, dw_kernel_size=3, dw_mid_norm=False,
                 dw_shift_init=False, dw_dilations=(1,), use_se=False, se_reduction=16,
                 dw_l0=False, use_s2_shift=False, use_tok_shift=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_shift_conv = use_shift_conv
        self.use_learnable_shift = use_learnable_shift
        self.dw_kernel_size = dw_kernel_size
        self.dw_mid_norm = dw_mid_norm
        self.dw_shift_init = dw_shift_init
        self.dw_dilations = tuple(dw_dilations)
        self.use_se = use_se
        self.dw_l0 = dw_l0
        self.use_s2_shift = use_s2_shift
        self.use_tok_shift = use_tok_shift

        kwargs = dict(use_shift_conv=use_shift_conv, use_learnable_shift=use_learnable_shift,
                      dw_kernel_size=dw_kernel_size, dw_mid_norm=dw_mid_norm,
                      dw_shift_init=dw_shift_init, dw_dilations=dw_dilations,
                      use_se=use_se, se_reduction=se_reduction, dw_l0=dw_l0,
                      use_s2_shift=use_s2_shift, use_tok_shift=use_tok_shift)

        # Encoder
        self.inc = DoubleConv(n_channels, 64, **kwargs)
        self.down1 = Down(64, 128, **kwargs)
        self.down2 = Down(128, 256, **kwargs)
        self.down3 = Down(256, 512, **kwargs)

        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, **kwargs)

        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear, **kwargs)
        self.up2 = Up(512, 256 // factor, bilinear, **kwargs)
        self.up3 = Up(256, 128 // factor, bilinear, **kwargs)
        self.up4 = Up(128, 64, bilinear, **kwargs)

        self.outc = OutConv(64, n_classes)

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

        logits = self.outc(x)
        return logits



def freeze_blocks_to_shift(model, prefixes, verbose=True):
    """Turn selected blocks into TRUE fixed shifts
    """
    frozen = []
    for name, module in model.named_modules():
        if not isinstance(module, LearnableShiftConv2d):
            continue
        block = name.split(".")[0]
        if "*" in prefixes or block in prefixes:
            module._initialize_from_shift_pattern()
            module.depthwise.weight.requires_grad_(False)
            for branch in module.extra_branches:
                branch.weight.requires_grad_(False)
            frozen.append(name)
    if verbose:
        print(f"Froze {len(frozen)} depthwise layers to fixed shifts "
              f"(blocks: {sorted(set(n.split('.')[0] for n in frozen))})")
    return frozen


def spatial_tap_summary(model):
    """Mean taps per filter, treating a frozen one-hot layer as 1 tap."""
    import numpy as _np
    tot_ch = tot_taps = 0
    for module in model.modules():
        if isinstance(module, LearnableShiftConv2d):
            w = module.depthwise.weight
            C = w.shape[0]
            taps = 1.0 if not w.requires_grad else float(w[0, 0].numel())
            tot_ch += C
            tot_taps += C * taps
    return (tot_taps / tot_ch) if tot_ch else float("nan")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Testing ShiftConv2d layer...")
    shift_conv = ShiftConv2d(9, 64).to(device)
    test_input = torch.randn(1, 9, 32, 32).to(device)
    print(f"  input {tuple(test_input.shape)} -> output {tuple(shift_conv(test_input).shape)}")

    print("Testing LearnableShiftConv2d layer...")
    for k in (3, 5, 7):
        layer = LearnableShiftConv2d(9, 64, kernel_size=k, mid_norm=True, shift_init=True).to(device)
        print(f"  k={k}: input {tuple(test_input.shape)} -> output {tuple(layer(test_input).shape)}")

    variants = {
        "Standard U-Net":                 dict(),
        "Shift-U-Net (frozen)":           dict(use_shift_conv=True),
        "Learnable-Shift 3x3":            dict(use_learnable_shift=True),
        "Learnable-Shift 3x3 + mid_norm": dict(use_learnable_shift=True, dw_mid_norm=True),
        "Learnable-Shift 7x7 + mid_norm": dict(use_learnable_shift=True, dw_kernel_size=7,
                                               dw_mid_norm=True),
    }

    x = torch.randn(1, 3, 128, 128).to(device)
    for name, cfg in variants.items():
        model = UNet(n_channels=3, n_classes=3, **cfg).to(device)
        with torch.no_grad():
            out = model(x)
        print(f"{name:32s} {count_parameters(model):>11,} params, output {tuple(out.shape)}")

    print("Models working correctly!")
