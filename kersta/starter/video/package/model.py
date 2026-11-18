"""Kersta starter video detector.

Scores every sampled frame with a small CNN, smooths the frame scores over
time, and pools them into a clip score. Frames called fake become the time
segments it returns.

batch["inputs"]: uint8 [B, F, 3, H, W], zero frames after meta["frames"];
meta["frame_times"] gives each real frame's time in seconds.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def conv(cin, cout, stride):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Detector(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        w = width
        self.frames = nn.Sequential(
            conv(3, w, 2), conv(w, w, 1), conv(w, 2 * w, 2), conv(2 * w, 2 * w, 1),
            conv(2 * w, 4 * w, 2), conv(4 * w, 8 * w, 2),
        )
        self.temporal = nn.Conv1d(8 * w, 8 * w, 3, padding=1)
        self.frame_head = nn.Conv1d(8 * w, 1, 1)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, video, valid_frames):
        """uint8 [B, F, 3, H, W] -> (clip logit [B], frame logits [B, F])."""
        b, f = video.shape[:2]
        dtype = self.frame_head.weight.dtype
        x = video.reshape(b * f, *video.shape[2:]).to(dtype) / 255.0
        x = (x - self.mean.to(dtype)) / self.std.to(dtype)
        feats = self.frames(x).mean(dim=(2, 3)).reshape(b, f, -1).transpose(1, 2)  # [B, C, F]
        feats = F.relu(self.temporal(feats)) + feats
        frame_logits = self.frame_head(feats)[:, 0]
        steps = torch.arange(f, device=video.device)[None, :]
        valid = (steps < valid_frames[:, None]).to(frame_logits.dtype)
        clip = (frame_logits * valid).sum(dim=-1) / valid.sum(dim=-1).clamp(min=1)
        return clip, frame_logits


def segments(probs, times, count):
    if count == 0:
        return []
    step = times[1] - times[0] if count > 1 else 0.25
    out, start = [], None
    for i in range(count):
        if probs[i] >= 0.5 and start is None:
            start = times[i]
        elif probs[i] < 0.5 and start is not None:
            out.append([start, times[i]])
            start = None
    if start is not None:
        out.append([start, times[count - 1] + step])
    return out[:64]


class StarterVideoDetector:
    def __init__(self, net, device):
        self.net = net
        self.device = device

    def predict(self, batch):
        video = torch.from_numpy(np.ascontiguousarray(batch["inputs"])).to(self.device)
        counts = [int(m.get("frames", video.shape[1])) for m in batch["meta"]]
        valid = torch.tensor([max(1, c) for c in counts], device=self.device)
        with torch.inference_mode():
            clip, frames = self.net(video, valid)
        p_fake = torch.sigmoid(clip.float()).cpu().numpy()
        frame_probs = torch.sigmoid(frames.float()).cpu().numpy()
        out = []
        for i, meta in enumerate(batch["meta"]):
            answer = {"p_fake": float(p_fake[i])}
            if p_fake[i] >= 0.5:
                times = list(meta.get("frame_times") or [])
                answer["segments"] = segments(frame_probs[i], times, min(counts[i], len(times)))
            out.append(answer)
        return out


def load_model(weights_path, config):
    net = Detector(int((config.get("starter") or {}).get("width", 24)))
    net.load_state_dict(load_file(weights_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(
        str(config.get("dtype")), torch.float32)
    if device == "cpu":
        dtype = torch.float32
    return StarterVideoDetector(net.to(device=device, dtype=dtype).eval(), device)
