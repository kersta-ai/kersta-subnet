"""Decoded test inputs, cached per preprocessing so models that ask for the
same input share one decode (video decoding is the expensive part)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from kersta.core.lanes import Lane
from kersta.core.testset import Answer
from kersta.media.preprocess import (
    ImagePrep,
    Prep,
    load_item,
    load_mask,
    prep_key,
    transform_mask,
)


class InputCache:
    def __init__(self, testset_root: Path, cache_root: Path, lane: Lane) -> None:
        self.testset_root = testset_root
        self.cache_root = cache_root
        self.lane = lane

    def _folder(self, prep: Prep) -> Path:
        folder = self.cache_root / self.lane.value / prep_key(prep)
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def load(self, answer: Answer, prep: Prep) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        folder = self._folder(prep)
        meta_path = folder / f"{answer.id}.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            arrays = {name: np.load(folder / f"{answer.id}.{name}.npy", allow_pickle=False)
                      for name in meta.pop("_arrays")}
            return arrays, meta
        arrays, meta = load_item(self.testset_root / answer.file, self.lane, prep)
        for name, array in arrays.items():
            np.save(folder / f"{answer.id}.{name}.npy", array, allow_pickle=False)
        meta_path.write_text(json.dumps({**meta, "_arrays": list(arrays)}), encoding="utf-8")
        return arrays, meta

    def loader(self, answer: Answer, prep: Prep):  # noqa: ANN201 - a zero arg callable
        return lambda: self.load(answer, prep)

    def true_mask(self, answer: Answer, prep: ImagePrep) -> np.ndarray:
        """The answer mask on the model's input grid (see transform_mask)."""
        if not answer.mask:
            raise ValueError(f"{answer.id} has no mask")
        folder = self._folder(prep)
        path = folder / f"{answer.id}.truemask.npy"
        if path.is_file():
            return np.load(path, allow_pickle=False)
        mask = transform_mask(load_mask(self.testset_root / answer.mask), prep.size, prep.fit)
        np.save(path, mask, allow_pickle=False)
        return mask
