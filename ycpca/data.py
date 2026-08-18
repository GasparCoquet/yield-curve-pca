"""US Treasury par yield curve: download, cache, and load as a rectangular panel."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

TREASURY_URL: str = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all"
    "?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"
)

# Tenors quoted without structural gaps since 2010. "2 Mo" (starts 2018-10) and
# "4 Mo" (starts 2022-10) are excluded so the panel needs no gap filling.
TENORS: list[str] = [
    "1 Mo", "3 Mo", "6 Mo", "1 Yr", "2 Yr", "3 Yr",
    "5 Yr", "7 Yr", "10 Yr", "20 Yr", "30 Yr",
]

TENOR_YEARS: dict[str, float] = {
    "1 Mo": 1 / 12, "3 Mo": 0.25, "6 Mo": 0.5, "1 Yr": 1.0, "2 Yr": 2.0,
    "3 Yr": 3.0, "5 Yr": 5.0, "7 Yr": 7.0, "10 Yr": 10.0, "20 Yr": 20.0, "30 Yr": 30.0,
}

CACHE: Path = Path(__file__).resolve().parent.parent / "data" / "ust_par_yields.csv"


def download(start_year: int = 2015, end_year: int = 2026, timeout: int = 30) -> pd.DataFrame:
    """Fetch one CSV per year from Treasury and stack them into a date-indexed panel."""
    frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        response: requests.Response = requests.get(TREASURY_URL.format(year=year), timeout=timeout)
        response.raise_for_status()
        frame: pd.DataFrame = pd.read_csv(io.StringIO(response.text))
        frames.append(frame)

    panel: pd.DataFrame = pd.concat(frames, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"], format="%m/%d/%Y")
    panel = panel.set_index("Date").sort_index()
    return panel[TENORS]


def save(panel: pd.DataFrame, path: Path = CACHE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(path, float_format="%.2f")
    return path


def load(path: Path = CACHE, dropna: bool = True) -> pd.DataFrame:
    """Load the cached panel. Rows with any missing tenor are dropped by default."""
    panel: pd.DataFrame = pd.read_csv(path, index_col=0, parse_dates=True)
    if dropna:
        panel = panel.dropna(how="any")
    return panel[TENORS]


def daily_changes(levels: pd.DataFrame, in_bp: bool = True) -> pd.DataFrame:
    """First differences of the curve. Treasury quotes percent, so 1 unit = 100 bp."""
    changes: pd.DataFrame = levels.diff().dropna(how="any")
    return changes * 100.0 if in_bp else changes


# The coupon curve: the object that "PC1 is about 90% of curve variance" is a statement
# about. Bills (1 Mo, 3 Mo, 6 Mo) trade on policy dates, bill supply and debt-ceiling
# stress rather than on duration, so they dilute the factor structure. Evidence in
# scripts/sensitivity.py: 1 Mo daily changes are 4% correlated with the 10 Yr.
COUPON: list[str] = ["2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "20 Yr", "30 Yr"]
BILLS: list[str] = ["1 Mo", "3 Mo", "6 Mo"]
