"""U-Net for forest / non-forest segmentation.

Class convention (use the same one when training):
    0 = non-forest, 1 = forest
Input: (B, in_channels, H, W) with H and W divisible by 2**depth (256 works).
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 9, num_classes: int = 2,
                 base: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth
        chans = [base * 2 ** i for i in range(depth + 1)]

        self.encoders = nn.ModuleList()
        cin = in_channels
        for c in chans[:-1]:
            self.encoders.append(DoubleConv(cin, c))
            cin = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(chans[-2], chans[-1])

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for c in reversed(chans[:-1]):
            self.upconvs.append(nn.ConvTranspose2d(c * 2, c, 2, stride=2))
            self.decoders.append(DoubleConv(c * 2, c))
        self.head = nn.Conv2d(chans[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([skip, x], dim=1))
        return self.head(x)  # raw logits