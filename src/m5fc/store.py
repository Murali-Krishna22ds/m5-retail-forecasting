"""Uniform forecast storage: one parquet per model, long format.
columns: window ('cal'|'test'), d, node, point, q0.025 ... q0.975 (quantile cols optional)."""
from __future__ import annotations
import pandas as pd
from .config import RES, QUANTILES, CAL_START, CAL_END, TEST_START, TEST_END

FC = RES / "forecasts"; FC.mkdir(exist_ok=True, parents=True)
WINDOWS = {"cal": (CAL_START, CAL_END), "test": (TEST_START, TEST_END)}


def save(model: str, df: pd.DataFrame):
    df.to_parquet(FC / f"{model}.parquet")


def load(model: str, window: str | None = None) -> pd.DataFrame:
    df = pd.read_parquet(FC / f"{model}.parquet")
    return df if window is None else df[df.window == window]


def to_wide(df: pd.DataFrame, col: str = "point") -> pd.DataFrame:
    return df.pivot(index="d", columns="node", values=col).sort_index()


def qcols():
    return [f"q{q}" for q in QUANTILES]
