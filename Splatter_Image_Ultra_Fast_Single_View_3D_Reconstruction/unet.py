import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        g1 = min(groups, out_ch)
        g2 = min(groups, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(g1, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return F.silu(h + self.skip(x))


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch)
        self.res2 = ResBlock(out_ch, out_ch)
        self.down = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.res1 = ResBlock(out_ch + skip_ch, out_ch)
        self.res2 = ResBlock(out_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x)
        x = self.res2(x)
        return x


class SplatterImageUNet(nn.Module):

    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()

        self.stem = ResBlock(in_ch, base_ch)
        self.down1 = DownBlock(base_ch, base_ch * 2)
        self.down2 = DownBlock(base_ch * 2, base_ch * 4)
        self.down3 = DownBlock(base_ch * 4, base_ch * 8)

        self.mid1 = ResBlock(base_ch * 8, base_ch * 8)
        self.mid2 = ResBlock(base_ch * 8, base_ch * 8)

        self.up3 = UpBlock(base_ch * 8, base_ch * 8, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 4, base_ch * 2)
        self.up1 = UpBlock(base_ch * 2, base_ch * 2, base_ch)

        self.pre_head = ResBlock(base_ch, base_ch)
        self.head = nn.Conv2d(base_ch, 15, kernel_size=1)

    def forward(self, x):
        x0 = self.stem(x)
        x1, s1 = self.down1(x0)
        x2, s2 = self.down2(x1)
        x3, s3 = self.down3(x2)

        h = self.mid1(x3)
        h = self.mid2(h)

        h = self.up3(h, s3)
        h = self.up2(h, s2)
        h = self.up1(h, s1)

        h = self.pre_head(h)
        return self.head(h)  # [B, 15, H, W]
