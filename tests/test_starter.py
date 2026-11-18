"""The starter packages must pass the same checks every miner package faces.

Running them needs torch; these checks do not, so they run everywhere.
"""

import pytest
import yaml

from kersta.core.lanes import Lane
from kersta.core.package import ModelConfig
from kersta.miner.package import STARTER_ROOT, init
from kersta.sandbox.policy import check_package_code


@pytest.mark.parametrize("lane", list(Lane))
def test_starter_code_passes_the_sandbox_policy(lane):
    assert check_package_code(STARTER_ROOT / lane.value / "package") == []


@pytest.mark.parametrize("lane", list(Lane))
def test_starter_config_parses(lane):
    raw = yaml.safe_load((STARTER_ROOT / lane.value / "package" / "model_config.yaml").read_text())
    config = ModelConfig.parse(raw, lane)
    assert config.lane is lane


def test_init_copies_the_starter_package(tmp_path):
    target = init("audio", tmp_path / "sub")
    assert {p.name for p in target.iterdir()} == {"model.py", "model_config.yaml"}
    with pytest.raises(FileExistsError):
        init("audio", tmp_path / "sub")
