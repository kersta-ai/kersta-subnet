"""Training helpers shared by the starter train.py scripts (needs torch)."""

from __future__ import annotations

import importlib.util
import math
import random
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from kersta.core.lanes import Lane
from kersta.core.testset import MANIFEST_FILE, Manifest, lane_answers
from kersta.owner.factory import collect

STARTER = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class TrainItem:
    path: Path
    label: int
    mask: Path | None = None
    segments: tuple[tuple[float, float], ...] | None = None


def model_module(lane: Lane) -> ModuleType:
    """Import the lane's package/model.py, so training and the sandbox share one definition."""
    path = STARTER / lane.value / "package" / "model.py"
    spec = importlib.util.spec_from_file_location(f"kersta_starter_{lane.value}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def items_from_sources(folder: Path, lane: Lane) -> list[TrainItem]:
    """The test set factory's sources layout (real/, fake/<gen>/, partial/<gen>/)."""
    reals, fakes = collect(folder, lane)
    out = [TrainItem(s.path, 0) for s in reals]
    for sources in fakes.values():
        out += [TrainItem(s.path, 1, s.mask, s.segments) for s in sources]
    return out


def items_from_testset(folder: Path, lane: Lane) -> list[TrainItem]:
    """A released competition folder: every past test set is free training data."""
    manifest = Manifest.load(folder / MANIFEST_FILE)
    return [
        TrainItem(folder / a.file, a.label, folder / a.mask if a.mask else None, a.segments)
        for a in lane_answers(folder, manifest, lane)
    ]


def gather(lane: Lane, sources: list[str], testsets: list[str]) -> list[TrainItem]:
    items: list[TrainItem] = []
    for folder in sources:
        items += items_from_sources(Path(folder) / lane.value, lane)
    for folder in testsets:
        items += items_from_testset(Path(folder), lane)
    if not items:
        raise SystemExit("no training items: pass --sources and/or --testset")
    return items


def split(items: list[TrainItem], holdout: float, seed: int) -> tuple[list[TrainItem], list[TrainItem]]:
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    cut = max(1, int(len(shuffled) * holdout))
    return shuffled[cut:], shuffled[:cut]


def fit_temperature(logits: list[float], labels: list[int]) -> float:
    """The temperature that minimises log loss on held out data.

    Calibration is half of every slice score, so this one scalar is worth
    real points. Grid search keeps it dependency free and deterministic.
    """
    def nll(t: float) -> float:
        total = 0.0
        for z, y in zip(logits, labels, strict=True):
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z / t))))
            p = min(1 - 1e-7, max(1e-7, p))
            total -= y * math.log(p) + (1 - y) * math.log(1 - p)
        return total / max(1, len(logits))

    grid = [0.25 + 0.05 * i for i in range(96)]  # 0.25 .. 5.0
    return min(grid, key=nll)


def save_weights(state: dict[str, Any], path: Path) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, str(path))
