"""Dual-branch gated-fusion U-Net (MODELING_REPORT.md §5.2, eq. 5.1) and the loss (5.2)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Encoder(nn.Module):
    def __init__(self, cin, widths):
        super().__init__()
        self.stages = nn.ModuleList()
        c = cin
        for w in widths:
            self.stages.append(block(c, w))
            c = w

    def forward(self, x):
        feats = []
        for i, s in enumerate(self.stages):
            x = s(x if i == 0 else F.max_pool2d(x, 2))
            feats.append(x)
        return feats            # pre-pool features at 4 scales


class GatedFusionUNet(nn.Module):
    """SAR branch + optional optical branch; at each scale s the fused feature is
    h = g*h_S + (1-g)*h_O with g = sigmoid(W_g * [h_S; h_O]) (1x1 conv)."""

    def __init__(self, c_sar=3, c_opt=7, widths=(16, 32, 64, 128), bottleneck=256, p_drop=0.2):
        super().__init__()
        self.use_opt = c_opt > 0
        self.enc_s = Encoder(c_sar, widths)
        if self.use_opt:
            self.enc_o = Encoder(c_opt, widths)
            self.gates = nn.ModuleList([nn.Conv2d(2 * w, w, 1) for w in widths])
        self.bott = block(widths[-1], bottleneck)
        self.drop = nn.Dropout2d(p_drop)
        self.dec = nn.ModuleList()
        c = bottleneck
        for w in reversed(widths):
            self.dec.append(block(c + w, w))
            c = w
        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x_sar, x_opt=None, return_gates=False):
        hs = self.enc_s(x_sar)
        if self.use_opt:
            ho = self.enc_o(x_opt)
            skips, gvals = [], []
            for s, gate in enumerate(self.gates):
                g = torch.sigmoid(gate(torch.cat([hs[s], ho[s]], 1)))
                skips.append(g * hs[s] + (1 - g) * ho[s])
                gvals.append(g.mean())
        else:
            skips, gvals = hs, []
        x = self.drop(self.bott(F.max_pool2d(skips[-1], 2)))
        for d, skip in zip(self.dec, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.drop(d(torch.cat([x, skip], 1)))
        out = self.head(x)[:, 0]
        return (out, gvals) if return_gates else out


def flood_loss(logit, y, valid, lam_area=1.0, gamma=2.0, alpha1=0.75):
    """(5.2) Focal (gamma=2, alpha1=0.75) + soft Dice + lambda_A * area-consistency, valid pixels only."""
    v = valid.float()
    p = torch.sigmoid(logit)
    yf = y.float()
    pt = torch.where(yf > 0.5, p, 1 - p).clamp(1e-6, 1 - 1e-6)
    at = torch.where(yf > 0.5, torch.full_like(p, alpha1), torch.full_like(p, 1 - alpha1))
    focal = (-(at * (1 - pt) ** gamma * torch.log(pt)) * v).sum() / v.sum().clamp(min=1)
    inter = (p * yf * v).sum()
    dice = 1 - (2 * inter + 1) / ((p * v).sum() + (yf * v).sum() + 1)
    nb = v.sum((1, 2)).clamp(min=1)
    area = (((p - yf) * v).sum((1, 2)).abs() / nb).mean()
    return focal + dice + lam_area * area, dict(focal=focal.item(), dice=dice.item(), area=area.item())


def enable_mc_dropout(model):
    """MC dropout: BatchNorm in eval mode, dropout layers active."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()
