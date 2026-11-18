"""Run one lane package over one lane's answer key and score it."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from kersta.core.lanes import Lane
from kersta.core.package import LanePackage
from kersta.core.testset import Answer
from kersta.media.preprocess import ImagePrep
from kersta.sandbox.runner import Launcher, RunOutcome, run_model
from kersta.scoring.lane import LaneScore, aligned_probabilities, score_lane
from kersta.validator.inputs import InputCache


@dataclass(slots=True)
class Evaluation:
    hotkey: str
    lane: Lane
    commit_block: int
    lane_digest: str
    score: LaneScore
    run: RunOutcome
    p_fake: np.ndarray

    @property
    def ran(self) -> bool:
        return not self.run.error.startswith(("load", "warmup")) and bool(self.run.results)


def evaluate(
    launcher: Launcher,
    package: LanePackage,
    answers: Sequence[Answer],
    inputs: InputCache,
    *,
    hotkey: str,
    commit_block: int,
) -> Evaluation:
    lane = package.lane
    prep = package.config.preprocessing
    items = [(a.id, inputs.loader(a, prep)) for a in answers]
    run = run_model(launcher, package.root, lane, items, total=len(answers),
                    layout=getattr(prep, "layout", "nchw"))
    true_mask = None
    if lane is Lane.IMAGE and isinstance(prep, ImagePrep):
        true_mask = lambda answer: inputs.true_mask(answer, prep)  # noqa: E731
    score = score_lane(lane, answers, run.results, hotkey=hotkey, true_mask=true_mask)
    if run.error:
        score.reasons.append(f"run: {run.error[:300]}")
        if run.error.startswith(("load", "warmup")):
            score.zero("the model did not load")
    if run.stopped_early:
        score.reasons.append(f"stopped early: {run.stopped_early}")
    return Evaluation(hotkey, lane, commit_block, package.digest, score, run,
                      aligned_probabilities(answers, run.results))


def item_rows(evaluation: Evaluation, answers: Sequence[Answer]) -> list[dict[str, Any]]:
    """Per item record kept for audits, appeals and the miner's failure report."""
    rows = []
    for answer in answers:
        result = evaluation.run.results.get(answer.id)
        rows.append({
            "id": answer.id,
            "label": answer.label,
            "slices": list(answer.slices),
            "damage": list(answer.damage),
            "generator": answer.generator,
            "p_fake": None if result is None else result.p_fake,
            "seconds": None if result is None else result.seconds,
            "segments": None if result is None or result.segments is None
            else [list(s) for s in result.segments],
            "mask": None if result is None else result.mask is not None,
            "source": None if result is None else result.source,
            "error": None if result is None else result.error,
        })
    return rows
