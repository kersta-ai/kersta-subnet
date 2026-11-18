"""Kersta starter image detector.

A small CNN with two heads: one logit for "is it fake", and a coarse mask of
where it is fake, upsampled to the input grid. It is deliberately modest; it
exists so nobody starts from zero and there is a score to beat.

Interface (see docs/package_spec.md):
    load_model(weights_path, config) -> detector
    detector.predict(batch) -> [{"p_fake": float, "mask": HxW array}, ...]
batch["inputs"] is uint8 [B, 3, H, W] (scale: uint8, layout: nchw).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def conv(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.body = nn.Sequential(conv(cin, cout, stride), conv(cout, cout))
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return F.relu(self.body(x) + self.skip(x))


class Detector(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        w = width
        self.stem = conv(3, w, stride=2)  # /2 ; stride 2 early keeps it fast
        self.s1 = Block(w, w, 1)
        self.s2 = Block(w, 2 * w, 2)  # /4
        self.s3 = Block(2 * w, 4 * w, 2)  # /8
        self.s4 = Block(4 * w, 8 * w, 2)  # /16
        self.head = nn.Linear(8 * w, 1)
        self.mask_head = nn.Sequential(conv(4 * w + 2 * w, 2 * w), nn.Conv2d(2 * w, 1, 1))
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, pixels):
        """uint8 [B, 3, H, W] -> (logit [B], mask logits [B, 1, H, W])."""
        dtype = self.head.weight.dtype
        x = (pixels.to(dtype) / 255.0 - self.mean.to(dtype)) / self.std.to(dtype)
        f1 = self.s1(self.stem(x))
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        f4 = self.s4(f3)
        logit = self.head(f4.mean(dim=(2, 3))).squeeze(1)
        up = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        mask = self.mask_head(torch.cat([up, f2], dim=1))
        mask = F.interpolate(mask, size=pixels.shape[-2:], mode="bilinear", align_corners=False)
        return logit, mask


class StarterImageDetector:
    def __init__(self, net, device):
        self.net = net
        self.device = device

    def predict(self, batch):
        pixels = torch.from_numpy(np.ascontiguousarray(batch["inputs"])).to(self.device)
        with torch.inference_mode():
            logit, mask = self.net(pixels)
        p_fake = torch.sigmoid(logit.float()).cpu().numpy()
        masks = torch.sigmoid(mask.float())[:, 0].cpu().numpy()
        return [{"p_fake": float(p_fake[i]), "mask": masks[i]} for i in range(len(p_fake))]


def load_model(weights_path, config):
    width = int((config.get("starter") or {}).get("width", 32))
    net = Detector(width)
    net.load_state_dict(load_file(weights_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(
        str(config.get("dtype")), torch.float32)
    if device == "cpu":
        dtype = torch.float32  # half precision convolutions are slow or missing on CPU
    net = net.to(device=device, dtype=dtype).eval()
    return StarterImageDetector(net, device)
