import io

import numpy as np

from kersta.core.lanes import Lane
from kersta.miner.package import check
from kersta.sandbox.protocol import read_frame, write_frame
from kersta.sandbox.runner import LocalLauncher, run_model
from tests.helpers import make_image_package


def test_frames_round_trip_arrays_without_pickle():
    stream = io.BytesIO()
    image = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    audio = np.linspace(-1, 1, 5, dtype=np.float32)
    write_frame(stream, {"type": "predict", "id": "x"}, {"inputs": image, "audio": audio})
    stream.seek(0)
    header, arrays = read_frame(stream)
    assert header == {"type": "predict", "id": "x"}
    assert np.array_equal(arrays["inputs"], image) and np.array_equal(arrays["audio"], audio)


def _items(count, level):
    def loader(i):
        pixels = np.full((3, 64, 64), int(level * 255), dtype=np.uint8)
        return {"inputs": pixels}, {"width": 64, "height": 64}
    return [(f"i{i}", lambda i=i: loader(i)) for i in range(count)]


def test_worker_runs_a_model_and_returns_masks(tmp_path):
    root = make_image_package(tmp_path / "sub")
    run = run_model(LocalLauncher(), root / "image", Lane.IMAGE, _items(5, 0.9), total=5)
    assert run.error == "" and run.interface == "kersta"
    first = run.results["i0"]
    assert first.p_fake > 0.9 and first.mask is not None and first.mask.shape == (64, 64)
    assert all(r.seconds is not None and r.seconds > 0 for r in run.results.values())


def test_a_model_that_raises_fails_items_not_the_validator(tmp_path):
    code = ("def load_model(p, c):\n    return Bad()\n\nclass Bad:\n"
            "    def predict(self, batch):\n        raise ValueError('boom')\n")
    root = make_image_package(tmp_path / "sub", code=code)
    run = run_model(LocalLauncher(), root / "image", Lane.IMAGE, _items(300, 0.5), total=300)
    assert all(r.p_fake is None for r in run.results.values())
    assert "failed" in run.stopped_early  # certain zero: stopped after 1% failed


def test_a_model_that_cannot_load_is_reported(tmp_path):
    code = "def load_model(p, c):\n    raise RuntimeError('no weights')\n"
    root = make_image_package(tmp_path / "sub", code=code)
    run = run_model(LocalLauncher(), root / "image", Lane.IMAGE, _items(2, 0.5), total=2)
    assert run.error.startswith("load") and "no weights" in run.error


def test_the_audit_hook_blocks_writes_outside_the_temp_dir(tmp_path):
    # np.save passes the static checks; the audit hook inside the worker still
    # refuses to let model code write into the working directory.
    probe = "kersta_sandbox_probe.npy"
    code = (
        "import numpy as np\n"
        "def load_model(p, c):\n    return Writer()\n\n"
        "class Writer:\n"
        "    def predict(self, batch):\n"
        f"        np.save('{probe}', batch['inputs'])\n"
        "        return [{'p_fake': 0.5}]\n"
    )
    root = make_image_package(tmp_path / "sub", code=code)
    try:
        run = run_model(LocalLauncher(), root / "image", Lane.IMAGE, _items(1, 0.5), total=1)
        result = run.results["i0"]
        assert result.p_fake is None and "not allowed" in (result.error or "")
    finally:
        import pathlib

        pathlib.Path(probe).unlink(missing_ok=True)


def test_miner_check_smoke_runs_the_package(tmp_path):
    report = check(make_image_package(tmp_path / "sub"), items=4)
    assert report.ok, report.to_dict()
    assert report.lanes[0].interface == "kersta"
