"""Train the starter video detector (needs ffmpeg for decoding).

    python -m kersta.starter.video.train --sources data/ --out submission/video/model.safetensors

Videos are decoded with the validators' own preprocessing (fps, frame count,
size from package/model_config.yaml). Damage for video is an ffmpeg re-encode,
so it is applied offline: run the test set factory over your sources first
and train on its output with --testset, which also gives you damaged copies
labelled exactly as validators will see them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from kersta.core.lanes import Lane
from kersta.media.preprocess import VideoPrep, load_video
from kersta.starter.common import fit_temperature, gather, model_module, save_weights, split

CONFIG = Path(__file__).resolve().parent / "package" / "model_config.yaml"


class Videos(torch.utils.data.Dataset):
    def __init__(self, items, prep: VideoPrep):
        self.items, self.prep = items, prep

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        arrays, meta = load_video(item.path, self.prep)
        frames = int(meta["frames"])
        target = np.full(self.prep.max_frames, float(item.label), dtype=np.float32)
        if item.segments:
            times = np.asarray(meta["frame_times"] + [0.0] * (self.prep.max_frames - frames))
            target[:] = 0.0
            for a, b in item.segments:
                target[(times >= a) & (times < b)] = 1.0
        return (torch.from_numpy(arrays["inputs"]), torch.tensor(float(item.label)),
                torch.from_numpy(target), torch.tensor(max(1, frames)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--sources", action="append", default=[])
    parser.add_argument("--testset", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    config = yaml.safe_load(CONFIG.read_text("utf-8"))
    prep = VideoPrep.parse(config["preprocessing"])
    train, held = split(gather(Lane.VIDEO, args.sources, args.testset), 0.1, args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = model_module(Lane.VIDEO).Detector(int(config["starter"]["width"])).to(device)
    loader = torch.utils.data.DataLoader(Videos(train, prep), batch_size=args.batch, shuffle=True,
                                         num_workers=args.workers, drop_last=True)
    optimiser = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(args.epochs):
        net.train()
        total = 0.0
        for video, label, target, valid in loader:
            video, label, target, valid = (t.to(device) for t in (video, label, target, valid))
            clip, frames = net(video, valid)
            keep = (torch.arange(frames.shape[-1], device=device)[None, :] < valid[:, None]).float()
            frame_loss = (F.binary_cross_entropy_with_logits(frames, target, reduction="none")
                          * keep).sum() / keep.sum().clamp(min=1)
            loss = F.binary_cross_entropy_with_logits(clip, label) + 0.5 * frame_loss
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += float(loss)
        print(f"epoch {epoch + 1}/{args.epochs} loss {total / max(1, len(loader)):.4f}")

    net.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for video, label, _, valid in torch.utils.data.DataLoader(Videos(held, prep), batch_size=args.batch):
            logits += net(video.to(device), valid.to(device))[0].float().cpu().tolist()
            labels += [int(v) for v in label.tolist()]
    temperature = fit_temperature(logits, labels)
    with torch.no_grad():
        net.frame_head.weight /= temperature
        net.frame_head.bias /= temperature
    print(f"temperature {temperature:.2f} folded into the frame head")
    save_weights(net.state_dict(), Path(args.out))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
