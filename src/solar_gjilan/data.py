"""Load the cached raw files into tidy, UTC-indexed DataFrames.

Only parsing and renaming happens here. Cleaning and aligning the two sources
on a common time base is milestone M1.

Time conventions (important when the two sources are joined later):

* PVGIS timestamps look like ``20230615:1010`` and are UTC. SARAH3 values are
  stamped at minute 10 of each hour.
* Open-Meteo is requested with ``timezone=GMT``; timestamps are UTC on the full
  hour. Radiation values are averages over the *preceding* hour.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from solar_gjilan import config as cfg
from solar_gjilan.download import openmeteo_path, pvcalc_path, pvgis_hourly_path

PVGIS_COLUMNS = {
    "P": "P_W",  # PV system power, W (PVGIS model, 5 kWp, 14 % loss)
    "Gb(i)": "Gb",  # beam (direct) irradiance on the plane, W/m²
    "Gd(i)": "Gd",  # diffuse irradiance on the plane, W/m²
    "Gr(i)": "Gr",  # ground-reflected irradiance on the plane, W/m²
    "H_sun": "sun_height",  # sun elevation, degrees
    "T2m": "T2m",  # air temperature at 2 m, °C
    "WS10m": "WS10m",  # wind speed at 10 m, m/s
    "Int": "reconstructed",  # 1 = radiation value reconstructed (gap filled)
}


def parse_pvgis_time(values: pd.Series) -> pd.DatetimeIndex:
    """``'20230615:1010'`` → tz-aware UTC timestamps."""
    return pd.DatetimeIndex(pd.to_datetime(values, format="%Y%m%d:%H%M", utc=True), name="time")


def load_pvgis_hourly(tilt: float = 0, path: Path | None = None) -> pd.DataFrame:
    """Hourly PVGIS series for a south-facing plane with the given tilt.

    Adds ``G_poa = Gb + Gd + Gr``, the total irradiance on the plane. For the
    horizontal series (tilt 0) this is the global horizontal irradiance (GHI).
    """
    path = path or pvgis_hourly_path(tilt)
    raw = pd.read_csv(path)
    df = raw.drop(columns="time").rename(columns=PVGIS_COLUMNS)
    df.index = parse_pvgis_time(raw["time"])
    df["reconstructed"] = df["reconstructed"].astype(bool)
    df["G_poa"] = df["Gb"] + df["Gd"] + df["Gr"]
    return df


def load_openmeteo_hourly(path: Path | None = None) -> pd.DataFrame:
    """Hourly ERA5 weather from Open-Meteo, UTC index."""
    raw = pd.read_csv(path or openmeteo_path())
    df = raw.drop(columns="time")
    df.index = pd.DatetimeIndex(pd.to_datetime(raw["time"], utc=True), name="time")
    return df


def load_pvcalc(name: str = "optimal", path: Path | None = None) -> dict:
    """Official PVGIS long-term estimate for the default system.

    Returns tilt, azimuth, yearly energy ``E_y`` (kWh), mean daily energy ``E_d``,
    yearly in-plane irradiation ``H_y`` (kWh/m²), interannual std ``SD_y`` (kWh)
    and total loss ``l_total`` (%).
    """
    payload = json.loads((path or pvcalc_path(name)).read_text())
    mount = payload["inputs"]["mounting_system"]["fixed"]
    totals = payload["outputs"]["totals"]["fixed"]
    return {
        "tilt": mount["slope"]["value"],
        "azimuth": mount["azimuth"]["value"],
        "E_y": totals["E_y"],
        "E_d": totals["E_d"],
        "H_y": totals["H(i)_y"],
        "SD_y": totals["SD_y"],
        "l_total": totals["l_total"],
    }


def time_index(obj: pd.DataFrame | pd.Series) -> pd.DatetimeIndex:
    """The DatetimeIndex of a time-indexed frame or series.

    pandas types ``.index`` as a plain ``Index``; this checks at runtime that it is a
    DatetimeIndex, so ``.date``, ``.year``, ``.tz`` etc. are valid (and typed).
    """
    idx = obj.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError(f"expected a DatetimeIndex, got {type(idx).__name__}")
    return idx


def argmax_date(s: pd.Series) -> pd.Timestamp:
    """Index label (a date or timestamp) of the largest value of ``s``, ignoring NaN."""
    return pd.Timestamp(s.index[int(np.nanargmax(s.to_numpy(dtype=float)))])


def to_local(df: pd.DataFrame) -> pd.DataFrame:
    """Same rows, index converted to local clock time (Europe/Belgrade)."""
    return df.tz_convert(cfg.TZ)


def daily_energy(df: pd.DataFrame, column: str = "P_W") -> pd.Series:
    """Energy per local calendar day.

    For power ``P_W`` (W) the result is kWh; for irradiance (W/m²) it is kWh/m².

    Each hourly value is treated as constant over its hour (a left Riemann sum,
    ``Σ P·Δt`` with Δt = 1 h). This is the simplest numerical integral; the
    trapezoid and Simpson rules and their errors are studied in milestone M2.

    Local days with fewer than 23 hourly values are dropped. This removes the
    partial edge day created by the UTC→local shift (31 Dec 23:10 UTC is already
    1 Jan of the next year in Gjilan) while keeping the 23-hour DST day.
    """
    hourly = to_local(df)[column]
    grouped = hourly.groupby(hourly.index.date)
    daily = grouped.sum()[grouped.count() >= 23] / 1000
    return daily.rename(f"{column}_kWh_day")


def sunniest_day(df: pd.DataFrame, year: int, month: int, column: str = "G_poa") -> pd.Timestamp:
    """Local date with the highest daily irradiation in a given month."""
    daily = daily_energy(df, column)
    dates = pd.to_datetime(daily.index)
    subset = daily[(dates.year == year) & (dates.month == month)]
    if subset.empty:
        raise ValueError(f"no data for {year}-{month:02d}")
    return pd.Timestamp(subset.idxmax())


def one_day(df: pd.DataFrame, date: pd.Timestamp | str) -> pd.DataFrame:
    """Rows of one local calendar day, local-time index."""
    local = to_local(df)
    return local[local.index.date == pd.Timestamp(date).date()]


def quick_summary(df: pd.DataFrame) -> dict:
    """Basic completeness check: coverage, missing hours, NaNs, reconstructed share."""
    full = pd.date_range(df.index.min().floor("h"), df.index.max().floor("h"), freq="h")
    present = df.index.floor("h")
    summary = {
        "rows": len(df),
        "start": df.index.min(),
        "end": df.index.max(),
        "missing_hours": int((~full.isin(present)).sum()),
        "duplicate_hours": int(present.duplicated().sum()),
        "nan_values": int(df.isna().sum().sum()),
    }
    if "reconstructed" in df:
        summary["reconstructed_pct"] = round(100 * df["reconstructed"].mean(), 3)
    return summary
