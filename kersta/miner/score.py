"""Score a submission locally on a released test set, exactly as validators do.

Uses the same preprocessing, worker protocol, scoring and copy free formula
as the validator. Two differences: your model runs as a plain subprocess
(not in the sandbox image), and speed is measured on your GPU, not the
reference one.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from kersta.core.lanes import Lane
from kersta.core.package import scan_submission
from kersta.core.testset import MANIFEST_FILE, Manifest, lane_answers
from kersta.sandbox.runner import LocalLauncher
from kersta.scoring.lane import LaneScore
from kersta.validator.evaluate import evaluate
from kersta.validator.inputs import InputCache


def score_local(
    root: Path,
    testset: Path,
    *,
    lanes: list[Lane] | None = None,
    limit: int | None = None,
    cache: Path | None = None,
) -> dict[Lane, LaneScore]:
    submission = scan_submission(root)
    manifest = Manifest.load(testset / MANIFEST_FILE)
    wanted = [lane for lane in submission.lanes if lane in manifest.lane_ids]
    if lanes:
        wanted = [lane for lane in wanted if lane in lanes]
    if not wanted:
        raise ValueError("the submission and the test set share no lane")
    out: dict[Lane, LaneScore] = {}
    with tempfile.TemporaryDirectory() as tmp:
        cache_root = cache or Path(tmp)
        for lane in wanted:
            answers = lane_answers(testset, manifest, lane)[: limit or None]
            inputs = InputCache(testset, cache_root / "inputs", lane)
            evaluation = evaluate(LocalLauncher(), submission.lanes[lane], answers, inputs,
                                  hotkey="local", commit_block=0)
            out[lane] = evaluation.score
    return out


def render(scores: dict[Lane, LaneScore]) -> str:
    lines = []
    for lane, s in scores.items():
        lines.append(f"== {lane.value}: S = {s.score:.4f}  (D {s.detection:.4f}, L {s.location:.4f})")
        for name, sl in s.slices.items():
            lines.append(f"   {name:<8} n={sl.n:<5} MCC={sl.mcc:+.3f} cal={sl.calibration:.3f} "
                         f"slice={sl.score:.4f}")
        lines.append(f"   false alarms {s.false_alarm_rate:.2%}   p95 {s.p95_ms:.1f} ms "
                     f"(limit {s.speed_limit_ms:.0f})   failures {s.failures}/{s.items}")
        for reason in s.reasons:
            lines.append(f"   ! {reason}")
        weakest = sorted(s.breakdown.get("by_damage", {}).items(), key=lambda kv: kv[1]["accuracy"])
        if weakest:
            lines.append("   weakest damage: " + ", ".join(
                f"{k} {v['accuracy']:.0%} (n={v['n']})" for k, v in weakest[:4]))
        gens = sorted(s.breakdown.get("by_generator", {}).items(), key=lambda kv: kv[1]["accuracy"])
        if gens:
            lines.append("   weakest generators: " + ", ".join(
                f"{k} {v['accuracy']:.0%} (n={v['n']})" for k, v in gens[:4]))
    return "\n".join(lines)
