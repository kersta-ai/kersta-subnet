"""The damage tool: the real world things that happen to media before a
detector sees it (spec: "70% of every test set is damaged media").

The same code builds test sets and is shipped to miners so they can train
for it. Fixed parameters are not the defence (one fixed chain is something
anyone can train against); the defence is that each week's chains are drawn
from wide ranges with a secret seed, and real and fake items are damaged by
exactly the same draw.

Image ops carry the answer mask through every geometric change. Audio and
video ops never shift time, so answer segments stay valid.
"""

from __future__ import annotations

import io
import math
import random
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from kersta.core.lanes import Lane
from kersta.media.preprocess import read_audio, resample, write_wav


@dataclass(frozen=True, slots=True)
class Op:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def tag(self) -> str:
        if not self.params:
            return self.name
        inner = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}:{inner}"


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


# --------------------------------------------------------------------------- image ops

Picture = tuple[Image.Image, np.ndarray | None]


def _mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L")


def _mask_array(picture: Image.Image) -> np.ndarray:
    return np.asarray(picture.convert("L"), dtype=np.uint8) > 127


def _encode(img: Image.Image, fmt: str, **kwargs: Any) -> Image.Image:
    buffer = io.BytesIO()
    img.save(buffer, format=fmt, **kwargs)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def img_jpeg(p: Picture, quality: int, **_: Any) -> Picture:
    return _encode(p[0], "JPEG", quality=quality, subsampling=2), p[1]


def img_webp(p: Picture, quality: int, **_: Any) -> Picture:
    return _encode(p[0], "WEBP", quality=quality), p[1]


def img_resize(p: Picture, scale: float, **_: Any) -> Picture:
    img, mask = p
    size = (max(16, round(img.width * scale)), max(16, round(img.height * scale)))
    resized_mask = None if mask is None else _mask_array(
        _mask_image(mask).resize(size, Image.Resampling.NEAREST))
    return img.resize(size, Image.Resampling.BILINEAR), resized_mask


def img_crop(p: Picture, keep: float, x: float, y: float, **_: Any) -> Picture:
    img, mask = p
    w, h = max(16, round(img.width * keep)), max(16, round(img.height * keep))
    left, top = round((img.width - w) * x), round((img.height - h) * y)
    box = (left, top, left + w, top + h)
    return img.crop(box), None if mask is None else mask[top : top + h, left : left + w]


def img_blur(p: Picture, radius: float, **_: Any) -> Picture:
    return p[0].filter(ImageFilter.GaussianBlur(radius)), p[1]


def img_noise(p: Picture, sigma: float, seed: int, **_: Any) -> Picture:
    pixels = np.asarray(p[0], dtype=np.float32)
    noisy = pixels + np.random.default_rng(seed).normal(0.0, sigma, pixels.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8)), p[1]


def img_social(p: Picture, long_edge: int, quality: int, **_: Any) -> Picture:
    """A social media upload: cap the long edge, sharpen, recompress."""
    img, mask = p
    scale = min(1.0, long_edge / max(img.width, img.height))
    if scale < 1.0:
        img, mask = img_resize((img, mask), scale)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=60, threshold=2))
    return img_jpeg((img, mask), quality)


def img_screenshot(p: Picture, scale: float, bar: int, shade: int, **_: Any) -> Picture:
    """The picture shown in an app and screenshotted: rescaled, framed by UI bars."""
    img, mask = img_resize(p, scale)
    canvas = Image.new("RGB", (img.width, img.height + 2 * bar), (shade, shade, shade))
    draw = ImageDraw.Draw(canvas)
    for i in range(3):  # a few UI glyphs so the bars are not flat colour
        x = 12 + i * max(16, bar)
        draw.rectangle((x, bar // 4, x + bar // 2, bar - bar // 4), fill=(255 - shade,) * 3)
    canvas.paste(img, (0, bar))
    out_mask = None
    if mask is not None:
        out_mask = np.zeros((canvas.height, canvas.width), dtype=bool)
        out_mask[bar : bar + mask.shape[0], : mask.shape[1]] = mask
    return canvas, out_mask


def img_recapture(p: Picture, gamma: float, moire: float, seed: int, **_: Any) -> Picture:
    """A photo of a screen: gamma shift, faint moire, softening, sensor noise."""
    img, mask = p
    pixels = np.asarray(img, dtype=np.float32) / 255.0
    pixels = pixels ** gamma
    h, w = pixels.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    angle = (seed % 90) * math.pi / 180
    pattern = np.sin((xx * math.cos(angle) + yy * math.sin(angle)) * 1.7)
    pixels = pixels + moire * pattern[..., None]
    out = Image.fromarray(np.clip(pixels * 255, 0, 255).astype(np.uint8))
    out = out.filter(ImageFilter.GaussianBlur(0.6))
    return img_noise((out, mask), 3.0, seed)


IMAGE_OPS: Final[dict[str, Callable[..., Picture]]] = {
    "jpeg": img_jpeg, "webp": img_webp, "resize": img_resize, "crop": img_crop,
    "blur": img_blur, "noise": img_noise, "social": img_social,
    "screenshot": img_screenshot, "recapture": img_recapture,
}


def _image_params(name: str, rng: random.Random) -> dict[str, Any]:
    if name == "jpeg":
        return {"quality": rng.randint(30, 90)}
    if name == "webp":
        return {"quality": rng.randint(40, 90)}
    if name == "resize":
        return {"scale": round(rng.uniform(0.25, 0.9), 3)}
    if name == "crop":
        return {"keep": round(rng.uniform(0.55, 0.95), 3), "x": round(rng.random(), 3),
                "y": round(rng.random(), 3)}
    if name == "blur":
        return {"radius": round(rng.uniform(0.4, 1.8), 2)}
    if name == "noise":
        return {"sigma": round(rng.uniform(2.0, 9.0), 2), "seed": rng.randrange(2**31)}
    if name == "social":
        return {"long_edge": rng.choice([720, 1080, 1280]), "quality": rng.randint(68, 88)}
    if name == "screenshot":
        return {"scale": round(rng.uniform(0.5, 1.0), 3), "bar": rng.randint(24, 72),
                "shade": rng.choice([18, 32, 240, 250])}
    return {"gamma": round(rng.uniform(0.8, 1.25), 3), "moire": round(rng.uniform(0.01, 0.05), 3),
            "seed": rng.randrange(2**31)}


def damage_image(img: Image.Image, mask: np.ndarray | None, chain: Sequence[Op]) -> Picture:
    picture: Picture = (img.convert("RGB"), mask)
    for op in chain:
        picture = IMAGE_OPS[op.name](picture, **op.params)
    return picture


# --------------------------------------------------------------------------- audio ops


def _bandpass(samples: np.ndarray, rate: int, low: float, high: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    high = min(high, rate / 2 * 0.95)
    sos = butter(4, [low, high], btype="bandpass", fs=rate, output="sos")
    return sosfiltfilt(sos, samples).astype(np.float32)


def mu_law(samples: np.ndarray, mu: int = 255) -> np.ndarray:
    """G.711 style 8 bit companding round trip."""
    x = np.clip(samples, -1.0, 1.0)
    encoded = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    quantised = np.round((encoded + 1) / 2 * mu) / mu * 2 - 1
    return (np.sign(quantised) * np.expm1(np.abs(quantised) * np.log1p(mu)) / mu).astype(np.float32)


def aud_phone(samples: np.ndarray, rate: int, **_: Any) -> np.ndarray:
    """A phone line: 300-3400 Hz, 8 kHz, mu-law, back to the original rate."""
    narrow = resample(_bandpass(samples, rate, 300.0, 3400.0), rate, 8000)
    return resample(mu_law(narrow), 8000, rate)[: len(samples)]


def aud_resample(samples: np.ndarray, rate: int, low_rate: int, **_: Any) -> np.ndarray:
    return resample(resample(samples, rate, low_rate), low_rate, rate)[: len(samples)]


def aud_noise(samples: np.ndarray, rate: int, snr_db: float, seed: int, **_: Any) -> np.ndarray:
    power = float(np.mean(samples**2)) or 1e-8
    noise = np.random.default_rng(seed).normal(0.0, math.sqrt(power / 10 ** (snr_db / 10)),
                                               samples.shape)
    return np.clip(samples + noise, -1.0, 1.0).astype(np.float32)


def aud_gain(samples: np.ndarray, rate: int, db: float, **_: Any) -> np.ndarray:
    return np.clip(samples * 10 ** (db / 20), -1.0, 1.0).astype(np.float32)


def aud_rerecord(samples: np.ndarray, rate: int, rt60: float, seed: int, **_: Any) -> np.ndarray:
    """Played on a speaker and recorded again: room reverb, band limit, hiss."""
    from scipy.signal import fftconvolve

    rng = np.random.default_rng(seed)
    length = int(rate * rt60)
    decay = np.exp(-6.9 * np.arange(length) / max(1, length))
    impulse = rng.normal(0, 1, length) * decay
    impulse[0] = 1.0
    impulse /= np.sqrt(np.sum(impulse**2))
    wet = fftconvolve(samples, impulse)[: len(samples)]
    wet = _bandpass(wet.astype(np.float32), rate, 120.0, 7000.0)
    peak = float(np.max(np.abs(wet))) or 1.0
    return aud_noise((wet / peak * 0.9).astype(np.float32), rate, 35.0, seed + 1)


def aud_codec(samples: np.ndarray, rate: int, codec: str, bitrate: str, **_: Any) -> np.ndarray:
    """A lossy codec round trip through ffmpeg (mp3, opus, aac)."""
    if not has_ffmpeg():
        raise RuntimeError("the codec op needs ffmpeg on PATH")
    suffix = {"mp3": ".mp3", "opus": ".ogg", "aac": ".m4a"}[codec]
    encoder = {"mp3": "libmp3lame", "opus": "libopus", "aac": "aac"}[codec]
    with tempfile.TemporaryDirectory() as tmp:
        src, mid = Path(tmp) / "in.wav", Path(tmp) / f"mid{suffix}"
        write_wav(src, samples, rate)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:a", encoder,
                        "-b:a", bitrate, str(mid)], check=True, capture_output=True)
        decoded, _ = read_audio(mid, rate)
    out = np.zeros_like(samples)
    out[: min(len(out), len(decoded))] = decoded[: len(out)]
    return out


AUDIO_OPS: Final[dict[str, Callable[..., np.ndarray]]] = {
    "phone": aud_phone, "resample": aud_resample, "noise": aud_noise, "gain": aud_gain,
    "rerecord": aud_rerecord, "codec": aud_codec,
}


def _audio_params(name: str, rng: random.Random) -> dict[str, Any]:
    if name == "resample":
        return {"low_rate": rng.choice([8000, 11025, 12000, 16000])}
    if name == "noise":
        return {"snr_db": round(rng.uniform(8.0, 30.0), 2), "seed": rng.randrange(2**31)}
    if name == "gain":
        return {"db": round(rng.uniform(-14.0, 6.0), 2)}
    if name == "rerecord":
        return {"rt60": round(rng.uniform(0.15, 0.7), 3), "seed": rng.randrange(2**31)}
    if name == "codec":
        codec = rng.choice(["mp3", "opus", "aac"])
        return {"codec": codec, "bitrate": rng.choice(["24k", "32k", "48k", "64k"])}
    return {}


def damage_audio(samples: np.ndarray, rate: int, chain: Sequence[Op]) -> np.ndarray:
    out = np.asarray(samples, dtype=np.float32)
    for op in chain:
        out = AUDIO_OPS[op.name](out, rate, **op.params)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- video ops

VIDEO_OPS: Final[tuple[str, ...]] = ("reencode", "scale", "crop", "fps")


def _video_params(name: str, rng: random.Random) -> dict[str, Any]:
    if name == "reencode":
        return {"crf": rng.randint(23, 38), "preset": rng.choice(["veryfast", "medium"])}
    if name == "scale":
        return {"scale": round(rng.uniform(0.35, 0.9), 3)}
    if name == "crop":
        return {"keep": round(rng.uniform(0.6, 0.95), 3)}
    return {"fps": rng.choice([12, 15, 24])}


def damage_video(src: Path, dst: Path, chain: Sequence[Op]) -> None:
    """Apply a chain with one ffmpeg pass (filters) plus the final encode settings."""
    if not has_ffmpeg():
        raise RuntimeError("video damage needs ffmpeg on PATH")
    filters, crf, preset = [], 23, "medium"
    for op in chain:
        if op.name == "scale":
            s = op.params["scale"]
            filters.append(f"scale=trunc(iw*{s}/2)*2:trunc(ih*{s}/2)*2")
        elif op.name == "crop":
            k = op.params["keep"]
            filters.append(f"crop=trunc(iw*{k}/2)*2:trunc(ih*{k}/2)*2")
        elif op.name == "fps":
            filters.append(f"fps={op.params['fps']}")
        elif op.name == "reencode":
            crf, preset = op.params["crf"], op.params["preset"]
    command = ["ffmpeg", "-v", "error", "-y", "-i", str(src)]
    if filters:
        command += ["-vf", ",".join(filters)]
    command += ["-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "64k", str(dst)]
    subprocess.run(command, check=True, capture_output=True)


# --------------------------------------------------------------------------- chains

FAMILIES: Final[dict[Lane, dict[str, float]]] = {
    Lane.IMAGE: {"jpeg": 3, "resize": 3, "social": 3, "crop": 2, "screenshot": 2, "webp": 1,
                 "blur": 1, "noise": 1, "recapture": 1},
    Lane.AUDIO: {"phone": 3, "codec": 3, "noise": 2, "resample": 2, "rerecord": 2, "gain": 1},
    Lane.VIDEO: {"reencode": 3, "scale": 3, "crop": 1, "fps": 1},
}
NEEDS_FFMPEG: Final[frozenset[tuple[Lane, str]]] = frozenset({(Lane.AUDIO, "codec")})


def sample_chain(lane: Lane, rng: random.Random, *, max_ops: int = 3) -> list[Op]:
    """Draw one damage chain. Never looks at the label: real and fake share it."""
    lane = Lane.parse(lane)
    weights = dict(FAMILIES[lane])
    if not has_ffmpeg():
        weights = {k: v for k, v in weights.items() if (lane, k) not in NEEDS_FFMPEG}
    count = min(max_ops, rng.choices([1, 2, 3], weights=[2, 2, 1])[0])
    names: list[str] = []
    for _ in range(count):
        options = [n for n in weights if n not in names]
        if not options:
            break
        names.append(rng.choices(options, weights=[weights[n] for n in options])[0])
    params = {Lane.IMAGE: _image_params, Lane.AUDIO: _audio_params, Lane.VIDEO: _video_params}[lane]
    return [Op(name, params(name, rng)) for name in names]


def parse_chain(text: str) -> list[Op]:
    """'jpeg:quality=40;resize:scale=0.5' -> ops, for the CLI."""
    ops = []
    for part in filter(None, (p.strip() for p in text.split(";"))):
        name, _, rest = part.partition(":")
        params: dict[str, Any] = {}
        for pair in filter(None, rest.split(",")):
            key, _, value = pair.partition("=")
            try:
                params[key] = int(value)
            except ValueError:
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value
        ops.append(Op(name.strip(), params))
    return ops


def apply_to_file(lane: Lane, src: Path, dst: Path, chain: Sequence[Op]) -> None:
    """Damage one file (the miner facing `kersta damage` command)."""
    lane = Lane.parse(lane)
    if lane is Lane.IMAGE:
        with Image.open(src) as opened:
            img, _ = damage_image(opened, None, chain)
        img.save(dst)
    elif lane is Lane.AUDIO:
        samples, rate = read_audio(src)
        write_wav(dst, damage_audio(samples, rate, chain), rate)
    else:
        damage_video(src, dst, chain)
