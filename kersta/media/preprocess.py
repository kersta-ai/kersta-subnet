"""Decoding and preprocessing: media file -> the array a model sees.

The validator does all decoding, so every model gets exactly the same input
and model code never parses an untrusted file. Each package's
model_config.yaml carries a ``preprocessing`` block that this module parses
and applies. Miners train with these same functions.

Test set audio is stored as PCM WAV (codec damage is applied, then decoded
back), so audio decoding needs no ffmpeg and is identical on every host.
Video always needs ffmpeg; the validator image pins its version.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

import numpy as np
from PIL import Image

from kersta.core.hashing import canonical_json, sha256_bytes
from kersta.core.lanes import Lane

FITS: Final[tuple[str, ...]] = ("stretch", "center_crop", "letterbox")
LAYOUTS: Final[tuple[str, ...]] = ("nchw", "nhwc")
# "uint8" (the default) hands the model raw 0..255 pixels and lets it cast and
# normalise on the GPU; it also moves a quarter of the bytes into the sandbox.
# "unit" (float 0..1) and "byte" (float 0..255) are for models that want the
# validator to normalise.
SCALES: Final[tuple[str, ...]] = ("uint8", "unit", "byte")
AUDIO_CROPS: Final[tuple[str, ...]] = ("start", "center")
INTERPOLATIONS: Final[dict[str, int]] = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}
FFMPEG_FILTERS: Final[dict[str, str]] = {
    "nearest": "neighbor",
    "bilinear": "bilinear",
    "bicubic": "bicubic",
    "lanczos": "lanczos",
}
MIN_SIDE: Final[int] = 16
MAX_SIDE: Final[int] = 2048


class PreprocessError(ValueError):
    pass


# --------------------------------------------------------------------------- specs


def _size(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        value = [value, value]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise PreprocessError("size must be [height, width]")
    h, w = (int(v) for v in value)
    if not (MIN_SIDE <= h <= MAX_SIDE and MIN_SIDE <= w <= MAX_SIDE):
        raise PreprocessError(f"size sides must be between {MIN_SIDE} and {MAX_SIDE}")
    return h, w


def _choice(block: dict[str, Any], key: str, options: tuple[str, ...], default: str) -> str:
    value = str(block.get(key, default)).strip().lower()
    if value not in options:
        raise PreprocessError(f"{key} must be one of {', '.join(options)}, got {value!r}")
    return value


def _triple(value: Any, key: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise PreprocessError(f"{key} must list three numbers (R, G, B)")
    out = tuple(float(v) for v in value)
    if key == "std" and any(v <= 0 for v in out):
        raise PreprocessError("std values must be positive")
    return out  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ImagePrep:
    size: tuple[int, int] = (224, 224)
    fit: str = "stretch"
    interpolation: str = "bilinear"
    layout: str = "nchw"
    scale: str = "uint8"
    mean: tuple[float, float, float] | None = None
    std: tuple[float, float, float] | None = None

    @classmethod
    def parse(cls, block: dict[str, Any]) -> ImagePrep:
        block = alias_keys(block)
        prep = cls(
            size=_size(block.get("size"), (224, 224)),
            fit=_choice(block, "fit", FITS, "stretch"),
            interpolation=_choice(block, "interpolation", tuple(INTERPOLATIONS), "bilinear"),
            layout=_choice(block, "layout", LAYOUTS, "nchw"),
            scale=_choice(block, "scale", SCALES, "uint8"),
            mean=_triple(block.get("mean"), "mean"),
            std=_triple(block.get("std"), "std"),
        )
        if prep.scale == "uint8" and (prep.mean is not None or prep.std is not None):
            raise PreprocessError("mean and std need scale 'unit' or 'byte', not 'uint8'")
        return prep


@dataclass(frozen=True, slots=True)
class AudioPrep:
    sample_rate: int = 16000
    seconds: float = 10.0
    crop: str = "start"

    @classmethod
    def parse(cls, block: dict[str, Any]) -> AudioPrep:
        block = alias_keys(block)
        rate = int(block.get("sample_rate", 16000))
        seconds = float(block.get("seconds", 10.0))
        if not 8000 <= rate <= 48000:
            raise PreprocessError("sample_rate must be between 8000 and 48000")
        if not 0.5 <= seconds <= 10.0:
            raise PreprocessError("seconds must be between 0.5 and 10")
        if block.get("mono", True) is not True:
            raise PreprocessError("only mono audio is supported")
        return cls(sample_rate=rate, seconds=seconds, crop=_choice(block, "crop", AUDIO_CROPS, "start"))


def alias_keys(block: dict[str, Any]) -> dict[str, Any]:
    """Accept the alternative key names plain classifier packages use.

    ``resize`` for the image size (centre cropped to the aspect first),
    ``num_frames`` and ``frame_rate`` for video, ``duration_seconds`` for
    audio (centre cropped). Kersta's own keys win when both are present.
    """
    out = dict(block)
    if "resize" in out and "size" not in out:
        out["size"] = out["resize"]
        out.setdefault("fit", "center_crop")
    if "num_frames" in out and "max_frames" not in out:
        out["max_frames"] = out["num_frames"]
    if "frame_rate" in out and "fps" not in out:
        out["fps"] = out["frame_rate"]
    if "duration_seconds" in out and "seconds" not in out:
        out["seconds"] = out["duration_seconds"]
        out.setdefault("crop", "center")
    normalize = out.get("normalize")
    if isinstance(normalize, dict) and "mean" not in out:
        # A normalize block is ignored: such models cast and normalise internally.
        out.pop("normalize", None)
    return out


@dataclass(frozen=True, slots=True)
class VideoPrep:
    fps: float = 4.0
    max_frames: int = 40
    size: tuple[int, int] = (224, 224)
    fit: str = "stretch"
    interpolation: str = "bilinear"
    layout: str = "nchw"
    scale: str = "unit"
    mean: tuple[float, float, float] | None = None
    std: tuple[float, float, float] | None = None
    audio: bool = False
    sample_rate: int = 16000

    @classmethod
    def parse(cls, block: dict[str, Any]) -> VideoPrep:
        block = alias_keys(block)
        fps = float(block.get("fps", 4.0))
        if not 0.5 <= fps <= 30.0:
            raise PreprocessError("fps must be between 0.5 and 30")
        frames = int(block.get("max_frames", math.ceil(fps * 10)))
        if not 1 <= frames <= 300:
            raise PreprocessError("max_frames must be between 1 and 300")
        image = ImagePrep.parse(block)
        rate = int(block.get("sample_rate", 16000))
        if not 8000 <= rate <= 48000:
            raise PreprocessError("sample_rate must be between 8000 and 48000")
        return cls(
            fps=fps,
            max_frames=frames,
            size=image.size,
            fit=image.fit,
            interpolation=image.interpolation,
            layout=image.layout,
            scale=image.scale,
            mean=image.mean,
            std=image.std,
            audio=bool(block.get("audio", False)),
            sample_rate=rate,
        )

    @property
    def image(self) -> ImagePrep:
        return ImagePrep(
            self.size, self.fit, self.interpolation, self.layout, self.scale, self.mean, self.std
        )


Prep = ImagePrep | AudioPrep | VideoPrep


def parse_preprocessing(lane: Lane | str, block: dict[str, Any] | None) -> Prep:
    lane = Lane.parse(lane)
    block = dict(block or {})
    if lane is Lane.IMAGE:
        return ImagePrep.parse(block)
    if lane is Lane.AUDIO:
        return AudioPrep.parse(block)
    return VideoPrep.parse(block)


def prep_key(prep: Prep) -> str:
    """Cache key: two models with the same preprocessing share decoded inputs."""
    body = {"kind": type(prep).__name__, **asdict(prep)}
    return sha256_bytes(canonical_json(body))[:16]


# --------------------------------------------------------------------------- geometry


@dataclass(frozen=True, slots=True)
class Geometry:
    """How a (src_w, src_h) picture maps onto the model's (w, h) input."""

    resize: tuple[int, int]  # (w, h) after scaling
    crop: tuple[int, int, int, int] | None  # box inside the resized picture
    paste: tuple[int, int] | None  # offset on the output canvas (letterbox)
    out: tuple[int, int]  # (w, h)


def geometry(src_w: int, src_h: int, size: tuple[int, int], fit: str) -> Geometry:
    out_h, out_w = size
    if fit == "stretch":
        return Geometry((out_w, out_h), None, None, (out_w, out_h))
    if fit == "center_crop":
        scale = max(out_w / src_w, out_h / src_h)
        rw, rh = max(out_w, round(src_w * scale)), max(out_h, round(src_h * scale))
        left, top = (rw - out_w) // 2, (rh - out_h) // 2
        return Geometry((rw, rh), (left, top, left + out_w, top + out_h), None, (out_w, out_h))
    scale = min(out_w / src_w, out_h / src_h)
    rw, rh = max(1, min(out_w, round(src_w * scale))), max(1, min(out_h, round(src_h * scale)))
    return Geometry((rw, rh), None, ((out_w - rw) // 2, (out_h - rh) // 2), (out_w, out_h))


def _apply(picture: Image.Image, geo: Geometry, resample: int, fill: int | tuple) -> Image.Image:
    resized = picture.resize(geo.resize, resample=resample)
    if geo.crop is not None:
        return resized.crop(geo.crop)
    if geo.paste is not None:
        canvas = Image.new(picture.mode, geo.out, fill)
        canvas.paste(resized, geo.paste)
        return canvas
    return resized


def _finish(pixels: np.ndarray, scale: str, mean, std, layout: str) -> np.ndarray:
    """uint8 HWC (or FHWC) -> the model's dtype and layout."""
    if scale == "uint8":
        out_u8 = np.moveaxis(pixels, -1, -3) if layout == "nchw" else pixels
        return np.ascontiguousarray(out_u8, dtype=np.uint8)
    out = pixels.astype(np.float32)
    if scale == "unit":
        out /= 255.0
    if mean is not None:
        out -= np.asarray(mean, dtype=np.float32)
    if std is not None:
        out /= np.asarray(std, dtype=np.float32)
    if layout == "nchw":
        out = np.moveaxis(out, -1, -3)
    return np.ascontiguousarray(out, dtype=np.float32)


# --------------------------------------------------------------------------- image


def shape_picture(picture: Image.Image, prep: ImagePrep) -> np.ndarray:
    """An RGB picture -> the model's input array (resize, crop, scale, layout)."""
    geo = geometry(picture.width, picture.height, prep.size, prep.fit)
    shaped = _apply(picture.convert("RGB"), geo, INTERPOLATIONS[prep.interpolation], (0, 0, 0))
    return _finish(np.asarray(shaped, dtype=np.uint8), prep.scale, prep.mean, prep.std, prep.layout)


def load_image(path: str | Path, prep: ImagePrep) -> tuple[np.ndarray, dict[str, Any]]:
    with Image.open(path) as opened:
        picture = opened.convert("RGB")
    return shape_picture(picture, prep), {"width": picture.width, "height": picture.height}


def load_mask(path: str | Path) -> np.ndarray:
    """A stored answer mask: white (>127) marks the fake region."""
    with Image.open(path) as opened:
        return np.asarray(opened.convert("L"), dtype=np.uint8) > 127


def transform_mask(mask: np.ndarray, size: tuple[int, int], fit: str) -> np.ndarray:
    """Map a full resolution answer mask onto a model's input grid.

    Location is scored on the model's own grid, so the true mask goes through
    the same resize and crop the picture did, with nearest neighbour so it
    stays binary.
    """
    src_h, src_w = mask.shape
    geo = geometry(src_w, src_h, size, fit)
    picture = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    shaped = _apply(picture, geo, Image.Resampling.NEAREST, 0)
    return np.asarray(shaped, dtype=np.uint8) > 127


# --------------------------------------------------------------------------- audio


def _ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if not found:
        raise PreprocessError("ffmpeg is required for this file type and is not on PATH")
    return found


def read_audio(path: str | Path, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    """Mono float32 samples in [-1, 1] and their rate.

    WAV is read without ffmpeg; anything else is decoded by ffmpeg.
    """
    path = Path(path)
    if path.suffix.lower() == ".wav":
        from scipy.io import wavfile

        rate, data = wavfile.read(str(path))
        samples = _to_float(data)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        if sample_rate is not None and rate != sample_rate:
            samples = resample(samples, rate, sample_rate)
            rate = sample_rate
        return samples.astype(np.float32), int(rate)

    rate = sample_rate or 48000
    command = [_ffmpeg(), "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1"]
    command += ["-ar", str(rate), "-"]
    raw = subprocess.run(command, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<f4").astype(np.float32), rate


def _to_float(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.uint8:
        return (data.astype(np.float32) - 128.0) / 128.0
    if np.issubdtype(data.dtype, np.integer):
        return data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    return np.clip(data.astype(np.float32), -1.0, 1.0)


def resample(samples: np.ndarray, source: int, target: int) -> np.ndarray:
    if source == target:
        return samples
    from scipy.signal import resample_poly

    ratio = Fraction(target, source)
    return resample_poly(samples, ratio.numerator, ratio.denominator).astype(np.float32)


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    from scipy.io import wavfile

    pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    wavfile.write(str(path), int(sample_rate), (pcm * 32767.0).round().astype(np.int16))


def load_audio(path: str | Path, prep: AudioPrep) -> tuple[np.ndarray, dict[str, Any]]:
    """Fixed length mono float32; ``offset`` is where the window starts, in seconds.

    Segments a model returns are always in seconds of the original clip, so a
    centre cropped model adds ``offset`` back to its frame times.
    """
    samples, rate = read_audio(path, prep.sample_rate)
    length = int(round(prep.seconds * rate))
    start = max(0, (len(samples) - length) // 2) if prep.crop == "center" else 0
    window = samples[start : start + length]
    out = np.zeros(length, dtype=np.float32)
    out[: len(window)] = window
    return out, {
        "sample_rate": rate,
        "samples": len(window),
        "seconds": len(samples) / rate,
        "offset": start / rate,
    }


# --------------------------------------------------------------------------- video


def probe(path: str | Path) -> dict[str, float]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise PreprocessError("ffprobe is required for video and is not on PATH")
    command = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries"]
    command += ["stream=width,height:format=duration", "-of", "json", str(path)]
    found = json.loads(subprocess.run(command, capture_output=True, check=True).stdout)
    stream = (found.get("streams") or [{}])[0]
    return {
        "width": float(stream.get("width") or 0),
        "height": float(stream.get("height") or 0),
        "duration": float((found.get("format") or {}).get("duration") or 0.0),
    }


def _video_filter(geo: Geometry, interpolation: str, fps: float) -> str:
    flags = FFMPEG_FILTERS[interpolation]
    parts = [f"fps={fps}", f"scale={geo.resize[0]}:{geo.resize[1]}:flags={flags}"]
    if geo.crop is not None:
        left, top, right, bottom = geo.crop
        parts.append(f"crop={right - left}:{bottom - top}:{left}:{top}")
    if geo.paste is not None:
        parts.append(f"pad={geo.out[0]}:{geo.out[1]}:{geo.paste[0]}:{geo.paste[1]}:black")
    return ",".join(parts)


def load_video(path: str | Path, prep: VideoPrep) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    info = probe(path)
    src_w, src_h = int(info["width"]), int(info["height"])
    if src_w <= 0 or src_h <= 0:
        raise PreprocessError(f"{path} has no readable video stream")
    geo = geometry(src_w, src_h, prep.size, prep.fit)
    seconds = min(10.0, info["duration"] or 10.0)
    command = [_ffmpeg(), "-v", "error", "-i", str(path), "-t", f"{seconds:.3f}"]
    command += ["-vf", _video_filter(geo, prep.interpolation, prep.fps)]
    command += ["-frames:v", str(prep.max_frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(command, capture_output=True, check=True).stdout
    out_w, out_h = geo.out
    count = len(raw) // (out_w * out_h * 3)
    frames = np.zeros((prep.max_frames, out_h, out_w, 3), dtype=np.uint8)
    if count:
        decoded = np.frombuffer(raw[: count * out_w * out_h * 3], dtype=np.uint8)
        frames[:count] = decoded.reshape(count, out_h, out_w, 3)
    arrays = {"inputs": _finish(frames, prep.scale, prep.mean, prep.std, prep.layout)}
    if prep.audio:
        try:
            sound, _ = read_audio(path, prep.sample_rate)
        except subprocess.CalledProcessError:
            sound = np.zeros(0, dtype=np.float32)
        length = int(round(10.0 * prep.sample_rate))
        track = np.zeros(length, dtype=np.float32)
        track[: min(length, len(sound))] = sound[:length]
        arrays["audio"] = track
    meta = {
        "width": src_w,
        "height": src_h,
        "seconds": seconds,
        "frames": count,
        "frame_times": [round(i / prep.fps, 4) for i in range(count)],
    }
    return arrays, meta


# --------------------------------------------------------------------------- dispatch


def load_item(path: str | Path, lane: Lane | str, prep: Prep) -> tuple[dict[str, np.ndarray], dict]:
    """Decode one test file into named arrays plus per item metadata."""
    lane = Lane.parse(lane)
    if lane is Lane.IMAGE:
        assert isinstance(prep, ImagePrep)
        array, meta = load_image(path, prep)
        return {"inputs": array}, meta
    if lane is Lane.AUDIO:
        assert isinstance(prep, AudioPrep)
        array, meta = load_audio(path, prep)
        return {"inputs": array}, meta
    assert isinstance(prep, VideoPrep)
    return load_video(path, prep)
