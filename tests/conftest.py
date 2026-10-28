import sys, warnings, pathlib
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pytest
from m5fc.features import build_all


@pytest.fixture(scope="session")
def data():
    bt, at, S, h = build_all()
    return bt, at, S, h
