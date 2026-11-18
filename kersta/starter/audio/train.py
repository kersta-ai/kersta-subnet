"""Train the starter audio detector.

    python -m kersta.starter.audio.train --sources data/ --out submission/audio/model.safetensors

Clips are cut to random 3-10 s windows (the same way the test set factory
cuts them, so length tells the model nothing), resampled to 16 kHz and, 70%
of the time, damaged with the validators' own audio damage (phone line,
codecs, re-recording, noise). Partial spoofs train the per frame head on
their segments.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from kersta.core.lanes import Lane
from kersta.media.damage import damage_audio, sample_chain
from kersta.media.preprocess import AudioPrep, read_audio
from kersta.starter.common import fit_temperature, gather, model_module, save_weights, split

CONFIG = Path(__file__).resolve().parent / "package" / "model_config.yaml"


class Clips(torch.utils.data.Dataset):
    def __init__(self, items, prep: AudioPrep, damage: float, seed: int, frame_seconds: float):
        self.items, self.prep, self.damage, self.seed = items, prep, damage, seed
        self.frame_seconds = frame_seconds
        self.length = int(prep.seconds * prep.sample_rate)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        rng = random.Random(self.seed * 1_000_003 + index)
        samples, rate = read_audio(item.path, self.prep.sample_rate)
        want = rng.uniform(3.0, self.prep.seconds)
        span = int(want * rate)
        start = 0 if len(samples) <= span else rng.randrange(len(samples) - span)
        clip = samples[start:start + span]
        if rng.random() < self.damage:
            clip = damage_audio(clip, rate, sample_chain(Lane.AUDIO, rng))
        wave = np.zeros(self.length, dtype=np.float32)
        wave[: len(clip)] = clip[: self.length]
        frames = self.length // 640
        target = np.full(frames, float(item.label), dtype=np.float32)
        if item.segments:
            target[:] = 0.0
            for a, b in item.segments:
                lo, hi = (a - start / rate) / self.frame_seconds, (b - start / rate) / self.frame_seconds
                target[max(0, int(lo)): max(0, min(frames, int(np.ceil(hi))))] = 1.0
        valid = max(1, len(clip) // 640)
        return (torch.from_numpy(wave), torch.tensor(float(item.label)), torch.from_numpy(target),
                torch.tensor(valid))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--sources", action="append", default=[])
    parser.add_argument("--testset", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--damage", type=float, default=0.7)
    parser.add_argument("--frame-weight", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    config = yaml.safe_load(CONFIG.read_text("utf-8"))
    prep = AudioPrep.parse(config["preprocessing"])
    module = model_module(Lane.AUDIO)
    starter = config["starter"]
    train, held = split(gather(Lane.AUDIO, args.sources, args.testset), 0.1, args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = module.Detector(int(starter["n_mels"]), int(starter["width"])).to(device)
    dataset = Clips(train, prep, args.damage, args.seed, module.FRAME_SECONDS)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True,
                                         num_workers=args.workers, drop_last=True)
    optimiser = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, args.lr,
                                                   total_steps=max(1, args.epochs * len(loader)))
    for epoch in range(args.epochs):
        net.train()
        total = 0.0
        for wave, label, target, valid in loader:
            wave, label, target, valid = (t.to(device) for t in (wave, label, target, valid))
            clip, frames = net(wave, valid)
            n = min(frames.shape[-1], target.shape[-1])
            steps = torch.arange(n, device=device)[None, :]
            keep = (steps < valid[:, None]).float()
            frame_loss = (F.binary_cross_entropy_with_logits(frames[:, :n], target[:, :n],
                                                             reduction="none") * keep).sum()
            loss = F.binary_cross_entropy_with_logits(clip, label)
            loss = loss + args.frame_weight * frame_loss / keep.sum().clamp(min=1)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            schedule.step()
            total += float(loss)
        print(f"epoch {epoch + 1}/{args.epochs} loss {total / max(1, len(loader)):.4f}")

    net.eval()
    logits, labels = [], []
    with torch.inference_mode():
        held_set = Clips(held, prep, args.damage, 99, module.FRAME_SECONDS)
        for wave, label, _, valid in torch.utils.data.DataLoader(held_set, batch_size=args.batch):
            logits += net(wave.to(device), valid.to(device))[0].float().cpu().tolist()
            labels += [int(v) for v in label.tolist()]
    temperature = fit_temperature(logits, labels)
    with torch.no_grad():  # the clip logit is a weighted sum of frame logits
        net.frame.weight /= temperature
        net.frame.bias /= temperature
    print(f"temperature {temperature:.2f} folded into the frame head")
    save_weights(net.state_dict(), Path(args.out))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
