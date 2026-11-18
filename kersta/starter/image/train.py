"""Train the starter image detector.

    python -m kersta.starter.image.train --sources data/ --out submission/image/model.safetensors
    python -m kersta.starter.image.train --testset competition-3/ --testset competition-4/ --out ...

--sources takes the test set factory layout (real/, fake/<gen>/, partial/<gen>/ with
.mask.png sidecars); --testset takes released competitions. Training damages 70% of
items with the same damage tool validators use, real and fake alike.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from kersta.core.lanes import Lane
from kersta.media.damage import damage_image, sample_chain
from kersta.media.preprocess import ImagePrep, load_mask, shape_picture, transform_mask
from kersta.starter.common import fit_temperature, gather, model_module, save_weights, split

CONFIG = Path(__file__).resolve().parent / "package" / "model_config.yaml"


class Images(torch.utils.data.Dataset):
    def __init__(self, items, prep: ImagePrep, damage: float, seed: int):
        self.items, self.prep, self.damage, self.seed = items, prep, damage, seed

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        rng = random.Random(self.seed * 1_000_003 + index)
        with Image.open(item.path) as opened:
            img = opened.convert("RGB")
        mask = load_mask(item.mask) if item.mask else None
        if mask is not None and mask.shape != (img.height, img.width):
            mask = None
        if rng.random() < self.damage:
            img, mask = damage_image(img, mask, sample_chain(Lane.IMAGE, rng))
        pixels = shape_picture(img, self.prep)  # exactly what a validator would send
        h, w = self.prep.size
        if mask is not None:
            target = transform_mask(mask, self.prep.size, self.prep.fit).astype(np.float32)
        else:  # real: nothing is fake; a full fake: everything is
            target = np.full((h, w), float(item.label), dtype=np.float32)
        return torch.from_numpy(pixels), torch.tensor(float(item.label)), torch.from_numpy(target)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--sources", action="append", default=[])
    parser.add_argument("--testset", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--damage", type=float, default=0.7)
    parser.add_argument("--mask-weight", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    config = yaml.safe_load(CONFIG.read_text("utf-8"))
    prep = ImagePrep.parse(config["preprocessing"])
    train, held = split(gather(Lane.IMAGE, args.sources, args.testset), 0.1, args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = model_module(Lane.IMAGE).Detector(int(config["starter"]["width"])).to(device)
    loader = torch.utils.data.DataLoader(Images(train, prep, args.damage, args.seed),
                                         batch_size=args.batch, shuffle=True,
                                         num_workers=args.workers, drop_last=True)
    optimiser = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, args.lr,
                                                   total_steps=max(1, args.epochs * len(loader)))
    for epoch in range(args.epochs):
        net.train()
        total = 0.0
        for pixels, labels, masks in loader:
            pixels, labels, masks = pixels.to(device), labels.to(device), masks.to(device)
            logit, mask_logits = net(pixels)
            loss = F.binary_cross_entropy_with_logits(logit, labels)
            loss = loss + args.mask_weight * F.binary_cross_entropy_with_logits(
                mask_logits[:, 0], masks)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            schedule.step()
            total += float(loss)
        print(f"epoch {epoch + 1}/{args.epochs} loss {total / max(1, len(loader)):.4f}")

    # Fold a fitted temperature into the head so p_fake comes out calibrated.
    net.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for pixels, label, _ in torch.utils.data.DataLoader(Images(held, prep, args.damage, 99),
                                                           batch_size=args.batch):
            logits += net(pixels.to(device))[0].float().cpu().tolist()
            labels += [int(v) for v in label.tolist()]
    temperature = fit_temperature(logits, labels)
    with torch.no_grad():
        net.head.weight /= temperature
        net.head.bias /= temperature
    print(f"temperature {temperature:.2f} folded into the head")
    save_weights(net.state_dict(), Path(args.out))
    print(f"saved {args.out}; next: kersta miner check {Path(args.out).parent.parent}")


if __name__ == "__main__":
    main()
