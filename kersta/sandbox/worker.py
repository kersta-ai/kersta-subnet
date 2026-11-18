"""The process that hosts one miner model. Runs inside the sandbox container.

    python -m kersta.sandbox.worker --package /package --lane image

It loads the package, says "ready", then answers one predict frame at a
time on stdin/stdout. The validator times each answer from outside, so
nothing in here can change the timing; labels never enter this process.

Two package interfaces are accepted:
  * Kersta: ``load_model(weights_path, config)`` returns an object with
    ``predict(batch) -> list[dict]`` (see kersta.core.outputs).
  * Plain classifier: ``load_model(weights_path, **model_kwargs)`` returns a
    torch module producing class logits with class 0 = real. It is wrapped
    so p_fake = 1 - p(real). Such a model gives no location.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import inspect
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import yaml

from kersta.sandbox.policy import install_import_guard
from kersta.sandbox.protocol import read_frame, write_frame

# Network and process creation. File writes are checked separately below; in
# the container the root filesystem is read only anyway.
BLOCKED_EVENTS = (
    "socket.", "subprocess.", "os.system", "os.exec", "os.posix_spawn", "os.spawn",
    "os.fork", "os.forkpty", "os.kill", "pty.", "webbrowser.", "urllib.", "http.",
    "ftplib.", "smtplib.", "telnetlib.",
)
WRITE_MODES = ("w", "a", "x", "+")


def install_audit_hook() -> None:
    """Refuse network, processes and file writes outside the temp dir.

    Audit hooks cannot be removed once added, so model code cannot undo this.
    """
    scratch = os.path.realpath(tempfile.gettempdir())

    def hook(event: str, args: tuple) -> None:
        if event.startswith(BLOCKED_EVENTS):
            raise RuntimeError(f"kersta sandbox: {event} is not allowed")
        if event == "open" and len(args) >= 2:
            path, mode = args[0], args[1]
            if isinstance(mode, str) and any(m in mode for m in WRITE_MODES):
                where = os.path.realpath(str(path)) if isinstance(path, (str, bytes, os.PathLike)) else ""
                if not where.startswith(scratch):
                    raise RuntimeError(f"kersta sandbox: writing {path!r} is not allowed")

    sys.addaudithook(hook)


def _preload() -> None:
    """Import heavy libraries before the guards so their own internals are untouched."""
    for name in ("torch", "torchvision", "torchaudio", "safetensors", "safetensors.torch",
                 "safetensors.numpy", "scipy", "PIL.Image", "timm", "einops", "transformers"):
        with contextlib.suppress(Exception):  # optional libraries
            importlib.import_module(name)


def _plain(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    if isinstance(value, np.generic):
        return value.item()
    return value


class ClassifierAdapter:
    """Wrap a plain torch classifier so it answers like a Kersta detector."""

    def __init__(self, module: Any, dtype: str) -> None:
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cast = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype, torch.float32)
        self.module = module.to(self.device, dtype=cast).eval()

    def predict(self, batch: dict[str, Any]) -> list[dict[str, Any]]:
        torch = self.torch
        with torch.inference_mode():
            out = self.module(torch.from_numpy(np.array(batch["inputs"])).to(self.device))
        out = getattr(out, "logits", out)
        if isinstance(out, dict):
            out = out.get("logits", out.get("output", next(iter(out.values()))))
        if isinstance(out, (tuple, list)) and len(out) == 1:
            out = out[0]
        logits = out.float()
        if logits.ndim == 1:
            logits = logits[:, None]
        if logits.shape[-1] == 1:
            p_fake = torch.sigmoid(logits[:, 0])
        else:
            p_fake = 1.0 - torch.softmax(logits, dim=-1)[:, 0]
        return [{"p_fake": float(p)} for p in p_fake.cpu().tolist()]


def load_detector(package: Path) -> tuple[Any, str]:
    config = yaml.safe_load((package / "model_config.yaml").read_text(encoding="utf-8")) or {}
    spec = importlib.util.spec_from_file_location("kersta_model", package / "model.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("model.py could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules["kersta_model"] = module
    spec.loader.exec_module(module)
    loader = getattr(module, "load_model", None)
    if not callable(loader):
        raise RuntimeError("model.py defines no load_model()")

    classifier = isinstance(config.get("model"), dict)
    if classifier:
        kwargs = {k: v for k, v in config["model"].items() if k != "weights_file"}
        weights = package / str(config["model"].get("weights_file") or "model.safetensors")
        try:
            inspect.signature(loader).bind(str(weights), **kwargs)
            loaded = loader(str(weights), **kwargs)
        except TypeError:
            loaded = loader(str(weights), num_classes=int(kwargs.get("num_classes", 2)))
    else:
        loaded = loader(str(package / "model.safetensors"), config)

    if callable(getattr(loaded, "predict", None)):
        return loaded, "kersta"
    if callable(loaded):
        return ClassifierAdapter(loaded, str(config.get("dtype") or "float32")), "classifier"
    raise RuntimeError("load_model() returned neither a detector with predict() nor a module")


def _answer(item_id: str, output: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not isinstance(output, dict):
        return {"type": "result", "id": item_id, "error": "prediction is not a dict"}, {}
    header: dict[str, Any] = {"type": "result", "id": item_id}
    arrays: dict[str, np.ndarray] = {}
    p = _plain(output.get("p_fake"))
    header["p_fake"] = float(p) if isinstance(p, (int, float)) and np.isfinite(p) else None
    if output.get("mask") is not None:
        mask = np.asarray(_plain(output["mask"]))
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        elif mask.dtype != np.uint8:
            mask = np.round(np.clip(np.nan_to_num(mask.astype(np.float32)), 0, 1) * 255)
        arrays["mask"] = mask.astype(np.uint8)
    if output.get("segments") is not None:
        header["segments"] = [[float(_plain(a)), float(_plain(b))] for a, b in output["segments"]]
    if output.get("source") is not None:
        header["source"] = str(output["source"])[:64]
    return header, arrays


def serve(package: Path, lane: str, proto_in: BinaryIO, proto_out: BinaryIO) -> int:
    started = time.perf_counter()
    try:
        detector, interface = load_detector(package)
    except Exception as exc:  # noqa: BLE001 - report any load failure to the validator
        write_frame(proto_out, {"type": "error", "error": f"load failed: {exc!r}"[:2000],
                                "trace": traceback.format_exc()[-4000:]})
        return 1
    write_frame(proto_out, {"type": "ready", "interface": interface, "lane": lane,
                            "load_seconds": round(time.perf_counter() - started, 3)})
    while True:
        try:
            header, arrays = read_frame(proto_in)
        except EOFError:
            return 0
        if header.get("type") == "stop":
            return 0
        item_id = str(header.get("id", ""))
        batch: dict[str, Any] = {k: v[None, ...] for k, v in arrays.items()}
        batch["meta"] = [header.get("meta") or {}]
        try:
            outputs = detector.predict(batch)
            if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
                raise ValueError("predict(batch) must return one dict per item")
            reply, blobs = _answer(item_id, outputs[0])
        except Exception as exc:  # noqa: BLE001 - an item failure is data, not a crash
            reply, blobs = {"type": "result", "id": item_id, "error": repr(exc)[:500]}, {}
        write_frame(proto_out, reply, blobs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kersta-worker")
    parser.add_argument("--package", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--no-guard", action="store_true", help="tests only")
    args = parser.parse_args(argv)

    # The protocol owns the real stdout; anything the model prints goes to stderr.
    proto_out = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    proto_in = sys.stdin.buffer

    package = Path(args.package).resolve()
    _preload()
    if not args.no_guard:
        install_import_guard(package)
        install_audit_hook()
    sys.path.insert(0, str(package))
    return serve(package, args.lane, proto_in, proto_out)


if __name__ == "__main__":
    raise SystemExit(main())
