"""Test set factory v1 (spec: "The test set factory").

Input folder, per lane::

    sources/<lane>/real/**/<file>                       real media
    sources/<lane>/fake/<generator>/**/<file>           fully generated media
    sources/<lane>/partial/<generator>/<name>.<ext>     partial edits, with a sidecar:
        <name>.mask.png        image: white where it was edited
        <name>.segments.json   audio/video: [[start_s, end_s], ...]

plus ``generators.json`` listing, per lane, the generators earlier competitions used
(everything else counts as unseen in this one).

What it guarantees (each one closes a shortcut a model could learn instead
of detecting fakes):
  * half real, half fake; 20% of fakes from unseen generators; 40% of fakes
    partial edits with an exact answer key; generators drawn evenly;
  * 70% of reals AND 70% of fakes damaged, each by a chain drawn from the
    same distribution without looking at the label;
  * every file re-encoded to one container per lane (PNG / 16 kHz WAV /
    H.264 MP4), metadata stripped, renamed to a random id;
  * images capped to the same long edge; clip lengths drawn the same way for
    real and fake, and the per label means written to notes so a leak shows;
  * the secret seed never leaves the owner; only its hash goes in the notes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from kersta.core.constants import (
    DAMAGE_SHARE,
    ITEMS_PER_LANE,
    PARTIAL_SHARE,
    REAL_SHARE,
    UNSEEN_SHARE,
)
from kersta.core.lanes import Lane, spec_for
from kersta.core.testset import (
    ANSWERS_FILE,
    CLEAN,
    DAMAGED,
    FAKE,
    REAL,
    UNSEEN,
    Answer,
    write_answers,
)
from kersta.media.damage import (
    Op,
    damage_audio,
    damage_image,
    damage_video,
    has_ffmpeg,
    sample_chain,
)
from kersta.media.preprocess import read_audio, write_wav

log = logging.getLogger("kersta.owner.factory")
MASK_SUFFIX = ".mask.png"
SEGMENTS_SUFFIX = ".segments.json"


class FactoryError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FactoryConfig:
    competition: int
    seed: str  # secret; keep it off every machine but the owner's
    items: int = ITEMS_PER_LANE
    real_share: float = REAL_SHARE
    unseen_share: float = UNSEEN_SHARE
    partial_share: float = PARTIAL_SHARE
    damage_share: float = DAMAGE_SHARE
    max_side: int = 1536
    sample_rate: int = 16000
    min_seconds: float = 3.0
    max_seconds: float = 10.0
    allow_reuse: bool = False  # small dev sets only


@dataclass(frozen=True, slots=True)
class Source:
    path: Path
    label: int
    generator: str | None = None
    partial: bool = False
    mask: Path | None = None
    segments: tuple[tuple[float, float], ...] | None = None


@dataclass(slots=True)
class Planned:
    source: Source
    seen: bool = True
    damaged: bool = False
    in_unseen_slice: bool = False


@dataclass(slots=True)
class LaneBuild:
    lane: Lane
    answers: list[Answer] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)


def _media(folder: Path, lane: Lane) -> list[Path]:
    if not folder.is_dir():
        return []
    suffixes = spec_for(lane).media_suffixes
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in suffixes and not p.name.endswith(MASK_SUFFIX)
    )


def collect(lane_dir: Path, lane: Lane) -> tuple[list[Source], dict[str, list[Source]]]:
    """(reals, fakes by generator)."""
    reals = [Source(p, REAL) for p in _media(lane_dir / "real", lane)]
    fakes: dict[str, list[Source]] = {}
    for root, partial in ((lane_dir / "fake", False), (lane_dir / "partial", True)):
        if not root.is_dir():
            continue
        for gen_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for path in _media(gen_dir, lane):
                mask = segments = None
                if partial:
                    if lane is Lane.IMAGE:
                        mask = path.with_name(path.stem + MASK_SUFFIX)
                        if not mask.is_file():
                            raise FactoryError(f"{path} has no {mask.name}")
                    else:
                        sidecar = path.with_name(path.stem + SEGMENTS_SUFFIX)
                        if not sidecar.is_file():
                            raise FactoryError(f"{path} has no {sidecar.name}")
                        segments = tuple(
                            (float(a), float(b)) for a, b in json.loads(sidecar.read_text("utf-8"))
                        )
                fakes.setdefault(gen_dir.name, []).append(
                    Source(path, FAKE, gen_dir.name, partial, mask, segments))
    return reals, fakes


def _draw(rng: random.Random, pools: dict[str, list[Source]], k: int, reuse: bool,
          what: str) -> list[Source]:
    """k sources spread evenly over generators."""
    if k <= 0:
        return []
    names = [n for n in sorted(pools) if pools[n]]
    if not names:
        raise FactoryError(f"no sources for {what}")
    rng.shuffle(names)
    quota = {n: k // len(names) + (1 if i < k % len(names) else 0) for i, n in enumerate(names)}
    picked: list[Source] = []
    spill = 0
    for name in names:
        pool, want = pools[name], quota[name] + spill
        take = min(want, len(pool))
        picked += rng.sample(pool, take)
        spill = want - take
    if spill:
        everything = [s for n in names for s in pools[n] if s not in picked]
        if len(everything) >= spill:
            picked += rng.sample(everything, spill)
        elif reuse:
            picked += rng.choices([s for n in names for s in pools[n]], k=spill)
        else:
            raise FactoryError(f"{what}: need {k}, have {k - spill}; add sources or allow reuse")
    return picked


def plan(lane: Lane, cfg: FactoryConfig, reals: list[Source], fakes: dict[str, list[Source]],
         seen_generators: Sequence[str], rng: random.Random) -> list[Planned]:
    n_real = round(cfg.items * cfg.real_share)
    n_fake = cfg.items - n_real
    n_unseen = round(n_fake * cfg.unseen_share)
    seen_set = set(seen_generators)

    def split(pool_seen: bool, partial: bool) -> dict[str, list[Source]]:
        return {g: [s for s in v if s.partial == partial] for g, v in fakes.items()
                if (g in seen_set) == pool_seen}

    def available(pools: dict[str, list[Source]]) -> int:
        return sum(len(v) for v in pools.values())

    pools = {(seen, partial): split(seen, partial) for seen in (True, False)
             for partial in (True, False)}
    if n_unseen and not available(pools[False, False]) and not available(pools[False, True]):
        raise FactoryError(f"{lane.value}: no unseen generator in the sources; "
                           "every generator listed in generators.json counts as seen")
    seen_count = n_fake - n_unseen
    wanted = {False: n_unseen, True: seen_count}
    chosen_fakes: list[tuple[Source, bool]] = []
    for seen in (False, True):
        label = "seen" if seen else "unseen"
        partial = round(wanted[seen] * cfg.partial_share) if available(pools[seen, True]) else 0
        if wanted[seen] and not partial:
            log.warning("%s: no %s partial edits in the sources; location gets fewer items",
                        lane.value, label)
        full_pool = pools[seen, False] if available(pools[seen, False]) else pools[seen, True]
        chosen_fakes += [(s, seen) for s in _draw(rng, pools[seen, True], partial,
                                                   cfg.allow_reuse, f"{label} partial edits")]
        chosen_fakes += [(s, seen) for s in _draw(rng, full_pool, wanted[seen] - partial,
                                                   cfg.allow_reuse, f"{label} fakes")]
    if len(reals) < n_real and not cfg.allow_reuse:
        raise FactoryError(f"{lane.value}: need {n_real} real items, have {len(reals)}")
    chosen_reals = (rng.sample(reals, n_real) if len(reals) >= n_real
                    else rng.choices(reals, k=n_real))

    items = [Planned(s) for s in chosen_reals] + [Planned(s, seen=seen) for s, seen in chosen_fakes]
    for label in (REAL, FAKE):
        group = [p for p in items if p.source.label == label]
        for p in rng.sample(group, round(len(group) * cfg.damage_share)):
            p.damaged = True
    real_items = [p for p in items if p.source.label == REAL]
    for p in rng.sample(real_items, min(n_unseen, len(real_items))):
        p.in_unseen_slice = True
    rng.shuffle(items)
    return items


def _slices(p: Planned) -> tuple[str, ...]:
    if p.source.label == FAKE and not p.seen:
        return (UNSEEN,)
    base = (DAMAGED,) if p.damaged else (CLEAN,)
    return (*base, UNSEEN) if p.in_unseen_slice else base


def _cap(img: Image.Image, mask: np.ndarray | None, max_side: int):
    scale = max_side / max(img.size)
    if scale >= 1.0:
        return img, mask
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    small_mask = None
    if mask is not None:
        small_mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
            size, Image.Resampling.NEAREST)) > 127
    return img.resize(size, Image.Resampling.LANCZOS), small_mask


def _window(length: float, want: float, segments, rng: random.Random) -> float:
    """Start of a ``want`` second window; partial edits keep part of their first edit."""
    if length <= want:
        return 0.0
    if segments:
        first_start, first_end = segments[0]
        low = max(0.0, first_start - want + 0.25)
        high = min(length - want, first_end - 0.25)
        if high >= low:
            return rng.uniform(low, high)
    return rng.uniform(0.0, length - want)


def _shift(segments, start: float, seconds: float):
    if segments is None:
        return None
    moved = [(max(0.0, a - start), min(seconds, b - start)) for a, b in segments]
    return tuple((round(a, 3), round(b, 3)) for a, b in moved if b > a)


def build_lane(lane: Lane, lane_dir: Path, out_root: Path, cfg: FactoryConfig,
               seen_generators: Sequence[str]) -> LaneBuild:
    rng = random.Random(hashlib.sha256(f"{cfg.seed}|{cfg.competition}|{lane.value}".encode()).digest())
    reals, fakes = collect(lane_dir, lane)
    planned = plan(lane, cfg, reals, fakes, seen_generators, rng)
    if lane is Lane.VIDEO and not has_ffmpeg():
        raise FactoryError("the video lane needs ffmpeg on PATH")
    media_dir = out_root / lane.value / "media"
    mask_dir = out_root / lane.value / "masks"
    media_dir.mkdir(parents=True, exist_ok=True)
    build = LaneBuild(lane)
    durations: dict[int, list[float]] = {REAL: [], FAKE: []}
    sizes: dict[int, list[int]] = {REAL: [], FAKE: []}
    short = 0

    for p in planned:
        item_id = f"{rng.getrandbits(64):016x}"
        chain: list[Op] = sample_chain(lane, rng) if p.damaged else []
        src = p.source
        mask_rel = segments = seconds = None

        if lane is Lane.IMAGE:
            with Image.open(src.path) as opened:
                img = opened.convert("RGB")
            mask = None
            if src.mask is not None:
                with Image.open(src.mask) as opened_mask:
                    mask = np.asarray(opened_mask.convert("L")) > 127
                if mask.shape != (img.height, img.width):
                    raise FactoryError(f"{src.mask} does not match {src.path} in size")
            img, mask = _cap(img, mask, cfg.max_side)
            if chain:
                img, mask = damage_image(img, mask, chain)
            rel = f"{lane.value}/media/{item_id}.png"
            img.save(out_root / rel, format="PNG")
            sizes[src.label].append(max(img.size))
            if mask is not None:
                mask_dir.mkdir(parents=True, exist_ok=True)
                mask_rel = f"{lane.value}/masks/{item_id}.png"
                Image.fromarray(mask.astype(np.uint8) * 255).save(out_root / mask_rel)

        elif lane is Lane.AUDIO:
            samples, rate = read_audio(src.path, cfg.sample_rate)
            length = len(samples) / rate
            want = rng.uniform(cfg.min_seconds, cfg.max_seconds)
            if length < want:
                short += 1
            start = _window(length, want, src.segments, rng)
            clip = samples[int(start * rate): int((start + want) * rate)]
            seconds = round(len(clip) / rate, 3)
            segments = _shift(src.segments, start, seconds)
            if src.partial and not segments:
                raise FactoryError(f"{src.path}: no edited part inside the clip window")
            if chain:
                clip = damage_audio(clip, rate, chain)
            rel = f"{lane.value}/media/{item_id}.wav"
            write_wav(out_root / rel, clip, rate)
            durations[src.label].append(seconds)

        else:
            want = rng.uniform(cfg.min_seconds, cfg.max_seconds)
            rel = f"{lane.value}/media/{item_id}.mp4"
            with tempfile.TemporaryDirectory() as tmp:
                trimmed = Path(tmp) / "trim.mp4"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src.path), "-t",
                                f"{want:.3f}", "-map_metadata", "-1", "-c:v", "libx264", "-crf",
                                "18", "-pix_fmt", "yuv420p", "-c:a", "aac", str(trimmed)],
                               check=True, capture_output=True)
                if chain:
                    damage_video(trimmed, out_root / rel, chain)
                else:
                    shutil.copyfile(trimmed, out_root / rel)
            seconds = round(want, 3)
            segments = _shift(src.segments, 0.0, seconds)
            durations[src.label].append(seconds)

        build.answers.append(Answer(
            id=item_id, file=rel, label=src.label, slices=_slices(p),
            generator=src.generator, seen=p.seen, partial=src.partial, mask=mask_rel,
            segments=segments if src.partial else None,
            damage=tuple(op.tag() for op in chain), seconds=seconds,
        ))

    write_answers(out_root / lane.value / ANSWERS_FILE, build.answers)
    generators = sorted({a.generator for a in build.answers if a.generator})
    build.notes = {
        "items": len(build.answers),
        "generators": generators,
        "unseen_generators": sorted({a.generator for a in build.answers
                                     if a.generator and not a.seen}),
        "leak_check": {
            "mean_seconds": {k: round(float(np.mean(v)), 3) for k, v in durations.items() if v},
            "mean_long_edge": {k: round(float(np.mean(v)), 1) for k, v in sizes.items() if v},
            "sources_shorter_than_window": short,
        },
    }
    means = build.notes["leak_check"]["mean_seconds"]
    if len(means) == 2 and abs(means[REAL] - means[FAKE]) > 0.5:
        log.warning("%s: mean clip length differs by label (%s); a model could learn it",
                    lane.value, means)
    return build


def build(sources: Path, out_dir: Path, cfg: FactoryConfig,
          lanes: Sequence[Lane] | None = None) -> tuple[Path, dict[str, Any]]:
    """Build ``out_dir/competition-<n>/`` (unsigned). Returns the folder and the notes."""
    registry_path = sources / "generators.json"
    registry = json.loads(registry_path.read_text("utf-8")) if registry_path.is_file() else {}
    root = out_dir / f"competition-{cfg.competition}"
    if root.exists():
        raise FactoryError(f"{root} exists; refusing to overwrite a competition")
    root.mkdir(parents=True)
    notes: dict[str, Any] = {"seed_sha256": hashlib.sha256(cfg.seed.encode()).hexdigest(),
                             "lanes": {}}
    for lane in lanes or [lane for lane in Lane if (sources / lane.value).is_dir()]:
        result = build_lane(lane, sources / lane.value, root, cfg, registry.get(lane.value, []))
        notes["lanes"][lane.value] = result.notes
        registry[lane.value] = sorted(set(registry.get(lane.value, [])) | set(result.notes["generators"]))
    (out_dir / f"generators.after-competition-{cfg.competition}.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    return root, notes
