import random

import numpy as np
import pytest
from PIL import Image

from kersta.core.lanes import Lane
from kersta.core.outputs import OutputError, validate
from kersta.media import damage
from kersta.media.preprocess import (
    AudioPrep,
    ImagePrep,
    PreprocessError,
    geometry,
    load_audio,
    load_image,
    parse_preprocessing,
    transform_mask,
    write_wav,
)


def _picture(path, w=120, h=80):
    pixels = np.zeros((h, w, 3), dtype=np.uint8)
    pixels[:, : w // 2] = 255
    Image.fromarray(pixels).save(path)
    return path


def test_image_default_is_uint8_nchw(tmp_path):
    array, meta = load_image(_picture(tmp_path / "a.png"), ImagePrep(size=(32, 48)))
    assert array.dtype == np.uint8 and array.shape == (3, 32, 48)
    assert meta == {"width": 120, "height": 80}


def test_image_unit_scale_with_normalisation(tmp_path):
    prep = ImagePrep.parse({"size": [16, 16], "scale": "unit", "layout": "nhwc",
                            "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]})
    array, _ = load_image(_picture(tmp_path / "a.png"), prep)
    assert array.dtype == np.float32 and array.shape == (16, 16, 3)
    assert array.min() >= -1.0 and array.max() <= 1.0


def test_uint8_scale_refuses_mean_and_std():
    with pytest.raises(PreprocessError):
        ImagePrep.parse({"mean": [0.5, 0.5, 0.5]})


@pytest.mark.parametrize("fit", ["stretch", "center_crop", "letterbox"])
def test_geometry_always_lands_on_the_model_grid(fit):
    geo = geometry(1000, 300, (224, 160), fit)
    assert geo.out == (160, 224)


def test_true_mask_follows_the_same_crop_as_the_picture():
    mask = np.zeros((100, 200), bool)
    mask[:, :100] = True  # left half
    moved = transform_mask(mask, (100, 100), "center_crop")  # keeps the middle square
    assert moved.shape == (100, 100)
    assert moved[:, :50].all() and not moved[:, 50:].any()


def test_audio_window_and_centre_crop(tmp_path):
    rate = 16000
    samples = np.linspace(-0.5, 0.5, rate * 4, dtype=np.float32)
    write_wav(tmp_path / "a.wav", samples, rate)
    start, meta = load_audio(tmp_path / "a.wav", AudioPrep(rate, 10.0))
    assert start.shape == (rate * 10,) and meta["samples"] == rate * 4
    centre, meta = load_audio(tmp_path / "a.wav", AudioPrep(rate, 2.0, "center"))
    assert centre.shape == (rate * 2,) and meta["offset"] == pytest.approx(1.0)


def test_alternative_audio_keys_map_to_a_centre_crop():
    prep = parse_preprocessing("audio", {"sample_rate": 16000, "duration_seconds": 6.0})
    assert prep == AudioPrep(16000, 6.0, "center")


def test_outputs_are_validated():
    assert validate({"p_fake": 0.3}, Lane.IMAGE).p_fake == 0.3
    for bad in ({"p_fake": 1.5}, {"p_fake": float("nan")}, {"p_fake": True}, {}):
        with pytest.raises(OutputError):
            validate(bad, Lane.IMAGE)
    mask = validate({"p_fake": 1.0, "mask": np.ones((8, 8), np.float32)}, Lane.IMAGE,
                    grid=(16, 16)).mask
    assert mask.shape == (16, 16) and mask.all()
    segs = validate({"p_fake": 1.0, "segments": [[-1, 2], [3, 50]]}, Lane.AUDIO,
                    duration=10.0).segments
    assert segs == ((0.0, 2.0), (3.0, 10.0))


def test_image_damage_keeps_the_mask_aligned():
    img = Image.new("RGB", (200, 100), (10, 10, 10))
    mask = np.zeros((100, 200), bool)
    mask[10:40, 10:40] = True
    for name in damage.IMAGE_OPS:
        op = damage.Op(name, damage._image_params(name, random.Random(3)))
        out, out_mask = damage.damage_image(img, mask, [op])
        assert out_mask.shape == (out.height, out.width), name


def test_damage_chains_are_deterministic_and_label_free():
    a = damage.sample_chain(Lane.IMAGE, random.Random(42))
    b = damage.sample_chain(Lane.IMAGE, random.Random(42))
    assert a == b and 1 <= len(a) <= 3


def test_audio_damage_keeps_length_and_phone_cuts_high_frequencies():
    rate = 16000
    t = np.arange(rate) / rate
    tone = (0.3 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)  # above the 3.4 kHz phone band
    out = damage.aud_phone(tone, rate)
    assert out.shape == tone.shape
    assert float(np.sqrt(np.mean(out**2))) < 0.1 * float(np.sqrt(np.mean(tone**2)))
    for name in ("noise", "gain", "resample", "rerecord"):
        op = damage.Op(name, damage._audio_params(name, random.Random(1)))
        assert damage.damage_audio(tone, rate, [op]).shape == tone.shape
