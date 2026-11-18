"""Kersta starter audio detector.

Log-mel front end, a small 2D CNN that keeps time resolution, a per frame
fake logit, and attention pooling over valid frames for the clip logit. Frames
the model calls fake become the time segments it returns, which the
location score compares with spliced in parts.

batch["inputs"]: float32 [B, T] in [-1, 1] at 16 kHz, zero padded to 10 s;
batch["meta"][i]["samples"] says how much is real audio.
"""

import math

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

SAMPLE_RATE = 16000
N_FFT = 512
HOP = 160
TIME_POOL = 4  # the CNN halves time twice
FRAME_SECONDS = HOP * TIME_POOL / SAMPLE_RATE  # 0.04 s per output frame


def mel_filters(n_mels, fmin=20.0, fmax=SAMPLE_RATE / 2):
    def to_mel(f):
        return 2595.0 * math.log10(1.0 + f / 700.0)

    points = torch.linspace(to_mel(fmin), to_mel(fmax), n_mels + 2)
    hz = 700.0 * (10.0 ** (points / 2595.0) - 1.0)
    bins = torch.floor((N_FFT + 1) * hz / SAMPLE_RATE).long().tolist()
    bank = torch.zeros(n_mels, N_FFT // 2 + 1)
    for i in range(n_mels):
        left, centre, right = bins[i], bins[i + 1], bins[i + 2]
        for k in range(left, centre):
            bank[i, k] = (k - left) / max(1, centre - left)
        for k in range(centre, right):
            bank[i, k] = (right - k) / max(1, right - centre)
    return bank


def conv(cin, cout, pool):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(pool),
    )


class Detector(nn.Module):
    def __init__(self, n_mels=64, width=32):
        super().__init__()
        w = width
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)
        self.register_buffer("bank", mel_filters(n_mels), persistent=False)
        self.cnn = nn.Sequential(
            conv(1, w, (2, 2)), conv(w, 2 * w, (2, 2)),  # time /4
            conv(2 * w, 4 * w, (2, 1)), conv(4 * w, 4 * w, (2, 1)),  # frequency only
        )
        self.frame = nn.Conv1d(4 * w, 1, 1)
        self.attend = nn.Conv1d(4 * w, 1, 1)

    def features(self, wave):
        # The front end always runs in float32; STFT in half precision is unreliable.
        spec = torch.stft(wave.float(), N_FFT, HOP, window=self.window, return_complex=True)
        mel = torch.log(self.bank @ spec.abs().pow(2) + 1e-6)
        mel = (mel - mel.mean(dim=(1, 2), keepdim=True)) / (mel.std(dim=(1, 2), keepdim=True) + 1e-5)
        return mel.unsqueeze(1)

    def forward(self, wave, valid_frames=None):
        """[B, T] -> (clip logit [B], frame logits [B, F])."""
        x = self.cnn(self.features(wave).to(self.frame.weight.dtype))
        x = x.mean(dim=2)  # pool frequency -> [B, C, F]
        frame_logits = self.frame(x)[:, 0]
        attention = self.attend(x)[:, 0]
        if valid_frames is not None:
            steps = torch.arange(attention.shape[-1], device=attention.device)[None, :]
            attention = attention.masked_fill(steps >= valid_frames[:, None], -1e4)
        weights = torch.softmax(attention.float(), dim=-1).to(frame_logits.dtype)
        return (weights * frame_logits).sum(dim=-1), frame_logits


def segments(probs, offset, frames):
    out, start = [], None
    for i in range(frames):
        if probs[i] >= 0.5 and start is None:
            start = i
        elif probs[i] < 0.5 and start is not None:
            out.append([offset + start * FRAME_SECONDS, offset + i * FRAME_SECONDS])
            start = None
    if start is not None:
        out.append([offset + start * FRAME_SECONDS, offset + frames * FRAME_SECONDS])
    return out[:64]


class StarterAudioDetector:
    def __init__(self, net, device):
        self.net = net
        self.device = device

    def predict(self, batch):
        wave = torch.from_numpy(np.ascontiguousarray(batch["inputs"], dtype=np.float32))
        metas = batch["meta"]
        samples = [int(m.get("samples", wave.shape[-1])) for m in metas]
        frames = torch.tensor([max(1, s // (HOP * TIME_POOL)) for s in samples])
        with torch.inference_mode():
            clip, frame_logits = self.net(wave.to(self.device), frames.to(self.device))
        p_fake = torch.sigmoid(clip.float()).cpu().numpy()
        frame_probs = torch.sigmoid(frame_logits.float()).cpu().numpy()
        out = []
        for i, meta in enumerate(metas):
            answer = {"p_fake": float(p_fake[i])}
            if p_fake[i] >= 0.5:
                count = min(int(frames[i]), frame_probs.shape[-1])
                answer["segments"] = segments(frame_probs[i], float(meta.get("offset", 0.0)), count)
            out.append(answer)
        return out


def load_model(weights_path, config):
    starter = config.get("starter") or {}
    net = Detector(int(starter.get("n_mels", 64)), int(starter.get("width", 32)))
    net.load_state_dict(load_file(weights_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return StarterAudioDetector(net.to(device).eval(), device)
