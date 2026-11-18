"""Small builders shared by the tests: numpy only models and synthetic media."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.numpy import save_file

# A policy compliant, numpy only detector: brighter pictures are "fake", and the
# brightest pixels are its location mask. Good enough to exercise every path.
BRIGHTNESS_MODEL = '''
import numpy as np
from safetensors.numpy import load_file


class Detector:
    def __init__(self, centre, sharpness, use_mask):
        self.centre = centre
        self.sharpness = sharpness
        self.use_mask = use_mask

    def predict(self, batch):
        out = []
        for item in batch["inputs"]:
            pixels = item.astype(np.float32) / 255.0
            level = float(pixels.mean())
            p = 1.0 / (1.0 + np.exp(-(level - self.centre) * self.sharpness))
            answer = {"p_fake": p}
            if self.use_mask:
                answer["mask"] = (pixels.mean(axis=0) > 0.8).astype(np.float32)
            out.append(answer)
        return out


def load_model(weights_path, config):
    weights = load_file(weights_path)
    return Detector(float(weights["centre"][0]), float(weights["sharpness"][0]),
                    bool(weights["use_mask"][0] > 0))
'''

CONFIG = """name: {name}
version: "1.0.0"
lane: image
dtype: float32
preprocessing:
  size: [64, 64]
  fit: stretch
"""


def make_image_package(root: Path, *, name: str = "bright", centre: float = 0.5,
                       sharpness: float = 25.0, use_mask: bool = True,
                       code: str = BRIGHTNESS_MODEL) -> Path:
    lane = root / "image"
    lane.mkdir(parents=True, exist_ok=True)
    (lane / "model.py").write_text(code, encoding="utf-8")
    (lane / "model_config.yaml").write_text(CONFIG.format(name=name), encoding="utf-8")
    save_file({"centre": np.array([centre], dtype=np.float32),
               "sharpness": np.array([sharpness], dtype=np.float32),
               "use_mask": np.array([1.0 if use_mask else 0.0], dtype=np.float32)},
              str(lane / "model.safetensors"))
    return root


def _texture(rng: np.random.Generator, level: float, size: int = 96) -> np.ndarray:
    base = rng.normal(level * 255, 18, size=(size, size, 3))
    return np.clip(base, 0, 255).astype(np.uint8)


def make_image_sources(root: Path, *, per_pool: int = 30, seed: int = 0) -> Path:
    """Reals are dark, fakes bright, partial edits dark with a bright square."""
    rng = np.random.default_rng(seed)
    lane = root / "image"
    for i in range(per_pool):
        path = lane / "real" / f"r{i}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(_texture(rng, 0.3)).save(path)
    for gen in ("gen_a", "gen_b", "gen_new"):
        for i in range(per_pool // 2):
            path = lane / "fake" / gen / f"f{i}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(_texture(rng, 0.75)).save(path)
            picture = _texture(rng, 0.45)
            mask = np.zeros(picture.shape[:2], dtype=bool)
            mask[20:70, 20:70] = True
            picture[mask] = 250
            part = lane / "partial" / gen / f"p{i}.png"
            part.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(picture).save(part)
            Image.fromarray(mask.astype(np.uint8) * 255).save(part.with_name(f"p{i}.mask.png"))
    (root / "generators.json").write_text(json.dumps({"image": ["gen_a", "gen_b"]}), "utf-8")
    return root


def random_hex(rng: random.Random, chars: int) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(chars))
