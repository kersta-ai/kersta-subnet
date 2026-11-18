import numpy as np
import pytest
from safetensors.numpy import save_file

from kersta.core.lanes import Lane
from kersta.core.package import PackageError, check_remote_listing, scan_submission
from kersta.sandbox.policy import check_source
from tests.helpers import make_image_package


def test_a_valid_package_scans_and_hashes_stably(tmp_path):
    root = make_image_package(tmp_path / "sub")
    first = scan_submission(root)
    second = scan_submission(root)
    assert set(first.lanes) == {Lane.IMAGE}
    assert first.digest == second.digest and len(first.digest) == 64
    assert set(first.files) == {"image/model.py", "image/model_config.yaml",
                                "image/model.safetensors"}


def test_files_outside_lane_folders_and_hidden_files_are_ignored(tmp_path):
    root = make_image_package(tmp_path / "sub")
    digest = scan_submission(root).digest
    (root / "README.md").write_text("hello")
    (root / "image" / ".DS_Store").write_text("junk")
    assert scan_submission(root).digest == digest


def test_changing_any_byte_changes_the_digest(tmp_path):
    root = make_image_package(tmp_path / "sub")
    digest = scan_submission(root).digest
    (root / "image" / "model.py").write_text((root / "image" / "model.py").read_text() + "\n")
    assert scan_submission(root).digest != digest


def test_pickle_weights_are_refused(tmp_path):
    root = make_image_package(tmp_path / "sub")
    (root / "image" / "extra.bin").write_bytes(b"\x80\x04")
    with pytest.raises(PackageError, match="not allowed"):
        scan_submission(root)


def test_missing_weights_are_refused(tmp_path):
    root = make_image_package(tmp_path / "sub")
    (root / "image" / "model.safetensors").unlink()
    with pytest.raises(PackageError, match="model.safetensors is missing"):
        scan_submission(root)


def test_a_config_in_the_wrong_lane_folder_is_refused(tmp_path):
    root = make_image_package(tmp_path / "sub")
    (root / "audio").mkdir()
    for name in ("model.py", "model_config.yaml", "model.safetensors"):
        (root / "audio" / name).write_bytes((root / "image" / name).read_bytes())
    with pytest.raises(PackageError, match="lane"):
        scan_submission(root)


def test_forbidden_code_is_refused(tmp_path):
    root = make_image_package(tmp_path / "sub", code="import os\nos.system('x')\n"
                              "def load_model(p, c):\n    return None\n")
    with pytest.raises(PackageError, match="not allowed"):
        scan_submission(root)


def test_a_plain_classifier_config_is_accepted(tmp_path):
    lane = tmp_path / "sub" / "image"
    lane.mkdir(parents=True)
    (lane / "model.py").write_text("def load_model(weights_path, num_classes):\n    return None\n")
    (lane / "model_config.yaml").write_text(
        'name: ported\nversion: "1.0"\nmodality: image\ndtype: bf16\n'
        "preprocessing:\n  resize: [224, 224]\nmodel:\n  num_classes: 3\n")
    save_file({"w": np.zeros(2, dtype=np.float32)}, str(lane / "model.safetensors"))
    package = scan_submission(tmp_path / "sub").lanes[Lane.IMAGE]
    assert package.config.dtype == "bfloat16"
    assert package.config.preprocessing.size == (224, 224)
    assert package.config.preprocessing.fit == "center_crop"


def test_remote_listing_is_checked_before_download():
    check_remote_listing(["image/model.py", "README.md"], {"image/model.py": 10})
    with pytest.raises(PackageError, match="not allowed"):
        check_remote_listing(["image/model.pt"])
    with pytest.raises(PackageError, match="limit"):
        check_remote_listing(["audio/model.safetensors"],
                             {"audio/model.safetensors": 600 * 1024**2})


@pytest.mark.parametrize("source", [
    "import socket",
    "import subprocess",
    "from os import system",
    "x = eval('1')",
    "open('/etc/passwd')",
    "import torch\ntorch.load('w.pt')",
    "().__class__.__base__.__subclasses__()",
    "import numpy as np\nnp.load('a', allow_pickle=True)",
    "getattr(object, '__subclasses__')",
    "from transformers import AutoModel\nAutoModel.from_pretrained('x', trust_remote_code=True)",
])
def test_policy_refuses(source):
    assert check_source(source, "model.py", frozenset())


@pytest.mark.parametrize("source", [
    "import numpy as np\nimport torch.nn as nn\nfrom safetensors.torch import load_file",
    "import os.path\nfrom os import path",
    "from . import blocks",
    "import blocks",
    "x = getattr(nn, 'ReLU')",
])
def test_policy_allows(source):
    assert check_source(source, "model.py", frozenset({"blocks"})) == []
