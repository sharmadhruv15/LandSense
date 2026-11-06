import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two Conv2d-BN-ReLU layers. Optional Dropout2d at the end."""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """
    U-Net with 4 downsampling levels trained on Landslide4Sense (14-band input).
    Layer names MUST match the checkpoint saved by improved_training.ipynb:
      enc1, enc2, enc3, enc4  — encoder blocks
      bot                     — bottleneck (MaxPool + DoubleConv with dropout)
      dec1..dec4              — decoder DoubleConv blocks
      up1..up4                — ConvTranspose2d upsamplers
      out                     — final 1x1 conv
    """
    def __init__(self, n_channels=14, n_classes=2):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────
        self.enc1 = DoubleConv(n_channels, 64)
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64,  128))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.enc4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        # ── Bottleneck (dropout=0.3 to reduce overfitting) ───────
        self.bot  = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024, dropout=0.3))

        # ── Decoder ──────────────────────────────────────────────
        self.up1  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec1 = DoubleConv(1024, 512)   # 512 up + 512 skip

        self.up2  = nn.ConvTranspose2d(512,  256, 2, stride=2)
        self.dec2 = DoubleConv(512,  256)   # 256 up + 256 skip

        self.up3  = nn.ConvTranspose2d(256,  128, 2, stride=2)
        self.dec3 = DoubleConv(256,  128)   # 128 up + 128 skip

        self.up4  = nn.ConvTranspose2d(128,   64, 2, stride=2)
        self.dec4 = DoubleConv(128,   64)   # 64 up + 64 skip

        # ── Output ───────────────────────────────────────────────
        self.out  = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.bot(e4)

        d  = self.dec1(torch.cat([self.up1(b),  e4], dim=1))
        d  = self.dec2(torch.cat([self.up2(d),  e3], dim=1))
        d  = self.dec3(torch.cat([self.up3(d),  e2], dim=1))
        d  = self.dec4(torch.cat([self.up4(d),  e1], dim=1))

        return self.out(d)
