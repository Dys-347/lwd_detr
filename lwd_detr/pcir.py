import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv


class PConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, partial_ratio=1 / 4):
        super().__init__()
        self.partial_channels = int(in_c * partial_ratio)
        self.remaining_channels = in_c - self.partial_channels
        self.stride = stride

        self.conv = nn.Conv2d(
            self.partial_channels,
            self.partial_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.pool = (
            nn.AvgPool2d(kernel_size=stride, stride=stride)
            if (stride != 1 and self.remaining_channels > 0)
            else nn.Identity()
        )
        self.bn = nn.BatchNorm2d(in_c)

        if in_c != out_c:
            self.proj = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_c),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        x1 = x[:, : self.partial_channels, :, :]
        x2 = x[:, self.partial_channels :, :, :]
        x1 = self.conv(x1)
        x2 = self.pool(x2)
        out = torch.cat([x1, x2], dim=1)
        out = self.bn(out)
        out = self.proj(out)
        return out


class InvertedResidual(nn.Module):
    def __init__(self, in_c, out_c, expand_ratio=4, stride=1):
        super().__init__()
        hidden_dim = int(in_c * expand_ratio)
        self.use_res = stride == 1 and in_c == out_c

        layers = [
            nn.Conv2d(in_c, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)


class PCIRBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, expand_ratio=4):
        super().__init__()
        self.pconv = PConv(in_c, out_c, stride=stride)
        self.ir = InvertedResidual(out_c, out_c, expand_ratio=expand_ratio, stride=1)

    def forward(self, x):
        x = self.pconv(x)
        x = self.ir(x)
        return x


class PCIRLayer(nn.Module):
    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        super().__init__()
        self.is_first = is_first
        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=3, s=2, p=1, act=nn.SiLU()),
                Conv(c2, c2, k=3, s=2, p=1, act=nn.SiLU()),
            )
        else:
            blocks = [PCIRBlock(c1, c2, stride=s, expand_ratio=e)]
            blocks.extend([PCIRBlock(c2, c2, stride=1, expand_ratio=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        return self.layer(x)
