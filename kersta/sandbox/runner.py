"""The validator side of a model run: start the process, feed items, time answers.

``DockerLauncher`` is what validators use: no network, read only package and
root filesystem, every capability dropped, an unprivileged user, process and
memory limits. ``LocalLauncher`` runs the same worker as a plain subprocess,
for a miner scoring their own model; it is not a sandbox.

Timing is taken here, around each request and reply, so the model process
cannot influence it. Items go one at a time (batch of one), which is also
the latency a phone call or meeting plugin needs.
"""

from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np

from kersta.core.constants import MAX_FAILURE_SHARE
from kersta.core.lanes import Lane, spec_for
from kersta.core.outputs import OutputError, validate
from kersta.sandbox.protocol import read_frame, write_frame
from kersta.scoring.lane import ItemResult

log = logging.getLogger("kersta.sandbox")
LOAD_TIMEOUT_SECONDS: Final[float] = 900.0
MIN_ITEM_TIMEOUT_SECONDS: Final[float] = 10.0
ITEM_TIMEOUT_FACTOR: Final[float] = 20.0
SLOW_SHARE: Final[float] = 0.05

Loader = Callable[[], tuple[dict[str, np.ndarray], dict[str, Any]]]


class RunError(RuntimeError):
    pass


class Launcher(Protocol):
    def command(self, package_dir: Path, lane: Lane, name: str) -> list[str]: ...

    def kill(self, name: str) -> None: ...


@dataclass(frozen=True, slots=True)
class LocalLauncher:
    """Plain subprocess. Fine for your own model; never for someone else's."""

    guard: bool = True

    def command(self, package_dir: Path, lane: Lane, name: str) -> list[str]:
        command = [sys.executable, "-m", "kersta.sandbox.worker", "--package", str(package_dir),
                   "--lane", lane.value]
        return command if self.guard else [*command, "--no-guard"]

    def kill(self, name: str) -> None:
        return None


@dataclass(frozen=True, slots=True)
class DockerLauncher:
    image: str
    gpu: str | None = "0"  # device index, "all", or None for CPU only
    memory: str = "48g"
    cpus: str = "8"
    pids: int = 512
    tmpfs: str = "/tmp:rw,size=4g"

    def command(self, package_dir: Path, lane: Lane, name: str) -> list[str]:
        if shutil.which("docker") is None:
            raise RunError("docker is not on PATH")
        command = [
            "docker", "run", "--rm", "-i", "--name", name,
            "--network", "none",
            "--read-only", "--tmpfs", self.tmpfs,
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", str(self.pids), "--memory", self.memory, "--cpus", self.cpus,
            "--user", "10001:10001", "-e", "HOME=/tmp", "-e", "HF_HUB_OFFLINE=1",
            "-v", f"{package_dir.resolve()}:/package:ro",
        ]
        if self.gpu is not None:
            command += ["--gpus", "all" if self.gpu == "all" else f"device={self.gpu}"]
        return [*command, self.image, "python", "-m", "kersta.sandbox.worker",
                "--package", "/package", "--lane", lane.value]

    def kill(self, name: str) -> None:
        subprocess.run(["docker", "kill", name], capture_output=True, check=False)


class ModelProcess:
    def __init__(self, launcher: Launcher, package_dir: Path, lane: Lane) -> None:
        self.launcher = launcher
        self.lane = lane
        self.name = f"kersta-{uuid.uuid4().hex[:12]}"
        command = launcher.command(package_dir, lane, self.name)
        self.proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self._frames: queue.Queue[Any] = queue.Queue()
        self._stderr: list[bytes] = []
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._drain, daemon=True).start()
        self.ready: dict[str, Any] = {}

    def _read(self) -> None:
        assert self.proc.stdout is not None
        while True:
            try:
                self._frames.put(read_frame(self.proc.stdout))
            except Exception as exc:  # noqa: BLE001 - EOF or garbage ends the stream
                self._frames.put(exc)
                return

    def _drain(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            if len(self._stderr) < 400:
                self._stderr.append(line)

    def logs(self) -> str:
        return b"".join(self._stderr[-60:]).decode("utf-8", errors="replace")

    def _next(self, timeout: float) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        try:
            frame = self._frames.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError(f"no answer within {timeout:.0f} s") from exc
        if isinstance(frame, Exception):
            raise RunError(f"model process ended: {frame}; logs: {self.logs()[-1500:]}")
        return frame

    def start(self, timeout: float = LOAD_TIMEOUT_SECONDS) -> dict[str, Any]:
        header, _ = self._next(timeout)
        if header.get("type") != "ready":
            raise RunError(str(header.get("error") or "the model did not load"))
        self.ready = header
        return header

    def predict(
        self, item_id: str, arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any],
        timeout: float,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray], float]:
        assert self.proc.stdin is not None
        started = time.perf_counter()
        write_frame(self.proc.stdin, {"type": "predict", "id": item_id, "meta": dict(meta)},
                    dict(arrays))
        header, blobs = self._next(timeout)
        seconds = time.perf_counter() - started
        if header.get("id") != item_id:
            raise RunError(f"answer for {header.get('id')!r} arrived for item {item_id!r}")
        return header, blobs, seconds

    def close(self) -> None:
        try:
            if self.proc.poll() is None and self.proc.stdin is not None:
                write_frame(self.proc.stdin, {"type": "stop"})
                self.proc.stdin.close()
                self.proc.wait(timeout=20)
        except Exception:  # noqa: BLE001
            pass
        if self.proc.poll() is None:
            self.launcher.kill(self.name)
            self.proc.kill()


@dataclass(slots=True)
class RunOutcome:
    results: dict[str, ItemResult] = field(default_factory=dict)
    load_seconds: float = 0.0
    interface: str = ""
    error: str = ""
    stopped_early: str = ""


def _grid(arrays: Mapping[str, np.ndarray], lane: Lane, layout: str) -> tuple[int, int] | None:
    if lane is not Lane.IMAGE:
        return None
    shape = arrays["inputs"].shape
    return (shape[-2], shape[-1]) if layout == "nchw" else (shape[-3], shape[-2])


def run_model(
    launcher: Launcher,
    package_dir: Path,
    lane: Lane,
    items: Iterable[tuple[str, Loader]],
    *,
    total: int,
    layout: str = "nchw",
    speed_factor: float = 1.0,
    item_timeout: float | None = None,
) -> RunOutcome:
    """Run every item through one model and collect validated results.

    Stops early only when the zero is already certain for the whole set:
    more failures than 1% of ``total``, or more items over the speed limit
    than 5% of ``total`` (which puts the p95 over the limit). The remaining
    items are recorded as not run.
    """
    limit_ms = spec_for(lane).speed_limit_ms
    timeout = item_timeout or max(MIN_ITEM_TIMEOUT_SECONDS, ITEM_TIMEOUT_FACTOR * limit_ms / 1000)
    outcome = RunOutcome()
    failures = slow = 0
    process = ModelProcess(launcher, package_dir, lane)
    pending = list(items)
    try:
        try:
            ready = process.start()
        except (RunError, TimeoutError) as exc:
            outcome.error = f"load: {exc}"
            return outcome
        outcome.load_seconds = float(ready.get("load_seconds") or 0.0)
        outcome.interface = str(ready.get("interface") or "")

        if pending:  # warm up on the first item; this answer is not scored or timed
            arrays, meta = pending[0][1]()
            try:
                process.predict("warmup", arrays, meta, timeout=max(timeout, 300.0))
            except (RunError, TimeoutError) as exc:
                outcome.error = f"warmup: {exc}"
                return outcome

        for index, (item_id, loader) in enumerate(pending):
            arrays, meta = loader()
            try:
                header, blobs, seconds = process.predict(item_id, arrays, meta, timeout)
            except TimeoutError as exc:
                outcome.error = f"item {item_id}: {exc}"
                break
            except RunError as exc:
                outcome.error = str(exc)
                break
            result = _result(item_id, header, blobs, seconds, lane, _grid(arrays, lane, layout),
                             meta)
            outcome.results[item_id] = result
            if result.error is not None:
                failures += 1
            elif seconds * 1000 / max(speed_factor, 1e-9) > limit_ms:
                slow += 1
            if failures > MAX_FAILURE_SHARE * total:
                outcome.stopped_early = f"more than {MAX_FAILURE_SHARE:.0%} of items failed"
            elif slow > SLOW_SHARE * total + 1:
                outcome.stopped_early = f"more than {SLOW_SHARE:.0%} of items over the speed limit"
            if outcome.stopped_early:
                log.info("stopping after %d items: %s", index + 1, outcome.stopped_early)
                break
    finally:
        process.close()

    for item_id, _ in pending:
        if item_id not in outcome.results:
            reason = outcome.stopped_early or outcome.error or "not run"
            outcome.results[item_id] = ItemResult(item_id, None, error=f"not run: {reason}")
    return outcome


def _result(
    item_id: str,
    header: Mapping[str, Any],
    blobs: Mapping[str, np.ndarray],
    seconds: float,
    lane: Lane,
    grid: tuple[int, int] | None,
    meta: Mapping[str, Any],
) -> ItemResult:
    if header.get("error"):
        return ItemResult(item_id, None, seconds=seconds, error=str(header["error"]))
    raw = dict(header)
    if "mask" in blobs:
        raw["mask"] = blobs["mask"]
    try:
        prediction = validate(raw, lane, grid=grid, duration=meta.get("seconds"))
    except (OutputError, TypeError, ValueError) as exc:
        return ItemResult(item_id, None, seconds=seconds, error=f"invalid output: {exc}")
    return ItemResult(
        item_id,
        prediction.p_fake,
        mask=prediction.mask,
        segments=prediction.segments,
        source=prediction.source,
        seconds=seconds,
    )
