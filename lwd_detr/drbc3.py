import torch
import torch.nn as nn


def _fuse_conv_bn(conv, bn):
    if bn is None:
        return conv.weight.data, conv.bias.data if conv.bias is not None else torch.zeros(conv.out_channels, device=conv.weight.device)
    std = (bn.running_var + bn.eps).sqrt()
    t = bn.weight / std
    t = t.reshape(-1, 1, 1, 1)
    w = conv.weight * t
    b = bn.bias - bn.running_mean * bn.weight / std
    if conv.bias is not None:
        b = b + conv.bias
    return w, b


class DRBC3Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.hidden = out_c

        self.branch_5x5 = nn.Conv2d(in_c, self.hidden, 5, 1, 2, bias=False)
        self.bn_5x5 = nn.BatchNorm2d(self.hidden)

        self.branch_3x3_d1 = nn.Conv2d(in_c, self.hidden, 3, 1, 1, dilation=1, bias=False)
        self.bn_3x3_d1 = nn.BatchNorm2d(self.hidden)

        self.branch_3x3_d2 = nn.Conv2d(in_c, self.hidden, 3, 1, 2, dilation=2, bias=False)
        self.bn_3x3_d2 = nn.BatchNorm2d(self.hidden)

        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_c),
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.SiLU(inplace=True)
        self.fused = False

    def forward(self, x):
        if self.fused:
            return self.act(self.fused_conv(x) + self.shortcut(x))

        out = self.bn_5x5(self.branch_5x5(x))
        out = out + self.bn_3x3_d1(self.branch_3x3_d1(x))
        out = out + self.bn_3x3_d2(self.branch_3x3_d2(x))

        return self.act(out + self.shortcut(x))

    def switch_to_deploy(self):
        if self.fused:
            return

        device = self.branch_5x5.weight.device
        fused_conv = nn.Conv2d(self.in_c, self.hidden, 5, 1, 2, bias=True).to(device)
        fused_w = torch.zeros(self.hidden, self.in_c, 5, 5, device=device)
        fused_b = torch.zeros(self.hidden, device=device)

        w, b = _fuse_conv_bn(self.branch_5x5, self.bn_5x5)
        fused_w += w
        fused_b += b

        w, b = _fuse_conv_bn(self.branch_3x3_d1, self.bn_3x3_d1)
        fused_w[:, :, 1:4, 1:4] += w
        fused_b += b

        w, b = _fuse_conv_bn(self.branch_3x3_d2, self.bn_3x3_d2)
        fused_w[:, :, ::2, ::2] += w
        fused_b += b

        fused_conv.weight.data = fused_w
        fused_conv.bias.data = fused_b
        self.fused_conv = fused_conv
        self.fused = True

        del self.branch_5x5, self.bn_5x5, self.branch_3x3_d1, self.bn_3x3_d1, self.branch_3x3_d2, self.bn_3x3_d2

    def fuse(self):
        self.switch_to_deploy()


class DRBC3(nn.Module):
    def __init__(self, c1, c2, n=3, e=1.0):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, 0, bias=False)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, 0, bias=False)
        self.m = nn.Sequential(*[DRBC3Block(c_, c_) for _ in range(n)])
        self.cv3 = nn.Conv2d(c_, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x1 = self.m(x1)
        out = self.act(self.bn(self.cv3(x1 + x2)))
        return out

    def switch_to_deploy(self):
        for m in self.m:
            m.switch_to_deploy()
