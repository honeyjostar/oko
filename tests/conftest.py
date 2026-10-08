import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oko.dataset import generate, load  # noqa: E402


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory):
    """Маленький набор на 4 машины — генерируется один раз на все тесты."""
    out = tmp_path_factory.mktemp("data")
    generate(out, n_vehicles=4, seed=7, test_share=0.5)
    return out


@pytest.fixture(scope="session")
def tiny_items(tiny_data):
    return load(tiny_data)
