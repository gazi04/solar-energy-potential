"""Clean and align the raw hourly data (milestone M1, step 4).

Pipeline (``build``):

1. ``rename_and_convert``  final column names, wind km/h → m/s
2. ``align``               one hourly UTC key for PVGIS and Open-Meteo
3. ``add_geometry``        our sun elevation and extraterrestrial irradiance g0
4. ``value_checks``        physical-limit checks, counted and flagged, nothing dropped
5. ``detect_gaps``         PVGIS zero-filled daylight hours → NaN
6. ``fill_short_gaps``     gaps of ≤ 2 h filled by interpolating the clearness ratio
7. ``build_daily``         one row per local day, ``complete`` flag
8. ``yearly_totals``       yearly kWh three ways, to show the effect of the gaps

Time key: every row is the UTC hour interval [HH:00, HH+1:00).

* PVGIS stamps values at HH:10 (a value near the start of the hour) → key HH.
* Open-Meteo stamps HH+1:00 with the mean of the preceding hour → key HH
  (labels shifted by −1 h).

About 20 minutes of offset remains (PVGIS point at HH:10, ERA5 mean centred at
HH:30). This is accepted for hourly work.

Run ``uv run python -m solar_gjilan.clean`` to rebuild ``data/processed/`` offline.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pvlib.irradiance import get_extra_radiation

from solar_gjilan import config as cfg
from solar_gjilan import data, sun

N_YEARS = cfg.END_YEAR - cfg.START_YEAR + 1

HOURLY_PATH = cfg.PROCESSED_DIR / f"hourly_{cfg.START_YEAR}_{cfg.END_YEAR}.csv"
DAILY_PATH = cfg.PROCESSED_DIR / f"daily_{cfg.START_YEAR}_{cfg.END_YEAR}.csv"
REPORT_PATH = cfg.PROCESSED_DIR / "quality_report.json"

GAP_THRESHOLD = 5.0  # degrees: decision D5
MAX_FILL_RUN = 2  # hours: decision D6

# PVGIS radiation/power columns that are zero in a gap (horizontal + 35° reference).
PVGIS_RAD = ["ghi", "bhi", "dhi", "p_w", "poa35", "p35_w"]
ERA5_RAD = ["ghi_era5", "bhi_era5", "dhi_era5"]

ERA5_COLUMNS = {
    "shortwave_radiation": "ghi_era5",
    "direct_radiation": "bhi_era5",
    "diffuse_radiation": "dhi_era5",
    "temperature_2m": "temp_era5",
    "relative_humidity_2m": "rh_era5",
    "cloud_cover": "cloud_era5",
    "wind_speed_10m": "wind_era5",
}


# --- 1. Names and units ------------------------------------------------------


def rename_and_convert(
    pvgis: pd.DataFrame, pvgis35: pd.DataFrame, era5: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Final column names for each source (inputs as returned by ``data.load_*``).

    * PVGIS horizontal: ``ghi`` (= Gb + Gd + Gr, Gr is 0 on a flat plane), ``bhi``,
      ``dhi``, ``p_w``, ``sun_height_pvgis``, ``t2m_pvgis``, ``ws10m_pvgis``.
    * PVGIS 35°: ``poa35`` (plane-of-array irradiance), ``p35_w`` (reference only).
    * ERA5: names from ``ERA5_COLUMNS``; wind converted km/h → m/s (÷ 3.6).
    """
    pv = pd.DataFrame(
        {
            "ghi": pvgis["G_poa"],
            "bhi": pvgis["Gb"],
            "dhi": pvgis["Gd"],
            "p_w": pvgis["P_W"],
            "sun_height_pvgis": pvgis["sun_height"],
            "t2m_pvgis": pvgis["T2m"],
            "ws10m_pvgis": pvgis["WS10m"],
        },
        index=pvgis.index,
    )
    pv35 = pd.DataFrame({"poa35": pvgis35["G_poa"], "p35_w": pvgis35["P_W"]}, index=pvgis35.index)
    era = era5.rename(columns=ERA5_COLUMNS)[list(ERA5_COLUMNS.values())].copy()
    era["wind_era5"] = era["wind_era5"] / 3.6
    return pv, pv35, era


# --- 2. Time alignment -------------------------------------------------------


def pvgis_key(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """PVGIS ``HH:10`` → hour key ``HH``."""
    return index.floor("h")


def era5_key(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Open-Meteo ``HH+1:00`` (mean over the preceding hour) → hour key ``HH``."""
    return index - pd.Timedelta(hours=1)


def align(pv: pd.DataFrame, pv35: pd.DataFrame, era: pd.DataFrame) -> pd.DataFrame:
    """Join the three sources on the UTC hour key (inner join).

    The result must be a continuous hourly series without duplicates.
    """
    pv = pv.set_axis(pvgis_key(pv.index))
    pv35 = pv35.set_axis(pvgis_key(pv35.index))
    era = era.set_axis(era5_key(era.index))
    df = pv.join(pv35, how="inner").join(era, how="inner")
    df.index.name = "time"
    if df.index.has_duplicates:
        raise ValueError("duplicate hour keys after alignment")
    expected = pd.date_range(df.index[0], df.index[-1], freq="h")
    if len(df) != len(expected):
        raise ValueError(f"{len(expected) - len(df)} hours missing after alignment")
    return df


def lag_correlation(a: pd.Series, b: pd.Series, lags: Iterable[int] = range(-2, 3)) -> dict[int, float]:
    """Pearson r between ``a`` and ``b`` shifted by ``lag`` hours (index-based).

    The lag with the highest r is the time shift that best lines the two up. After
    correct alignment the peak must be at lag 0.
    """
    return {int(lag): float(a.corr(b.shift(lag, freq="h"))) for lag in lags}


def mean_peak_hour(s: pd.Series) -> float:
    """Mean UTC hour (key) of the daily maximum."""
    peak = s.groupby(s.index.date).idxmax()
    return float(pd.DatetimeIndex(peak.to_numpy()).hour.to_numpy().mean())


# --- 3. Geometry -------------------------------------------------------------


def add_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Add our sun position and the extraterrestrial horizontal irradiance ``g0``.

    The sun is evaluated at the real PVGIS stamp (key + 10 min), so ``elevation``
    describes the same moment as the PVGIS values.

    ``g0 = I0(n) · max(sin α, 0)``. pvlib's ``get_extra_radiation`` gives I0, the
    irradiance outside the atmosphere on a plane facing the sun (solar constant
    1366.1 W/m² × Spencer's Earth–Sun distance factor). It agrees with the
    textbook ``1361 · (1 + 0.033 cos(360° n / 365))`` within about 0.5 %.
    """
    stamp = df.index + pd.Timedelta(minutes=10)
    pos = sun.sun_position(stamp)
    out = df.copy()
    out["elevation"] = pos["elevation"].to_numpy()
    out["solar_time"] = pos["solar_time"].to_numpy()
    out["declination"] = pos["declination"].to_numpy()
    i0 = np.asarray(get_extra_radiation(stamp.day_of_year.to_numpy()), dtype=float)
    out["g0"] = i0 * np.clip(np.sin(np.deg2rad(out["elevation"].to_numpy())), 0, None)
    return out


# --- 4. Value checks ---------------------------------------------------------


def value_checks(df: pd.DataFrame) -> tuple[dict[str, int], pd.Series]:
    """Physical-plausibility checks (M1 plan §4.3).

    Returns violation counts per check and a per-hour ``qc_flag`` (any violation).
    Nothing is dropped. Night and limit checks use PVGIS only, because ERA5 hourly
    means legitimately contain light when the sun rises or sets within the hour.
    """
    elev = df["elevation"]
    day = elev > GAP_THRESHOLD
    flags: dict[str, pd.Series] = {
        "negative_radiation": (df[PVGIS_RAD + ERA5_RAD] < 0).any(axis=1),
        "radiation_at_night": (elev < -1) & (df[PVGIS_RAD] > 0).any(axis=1),
        "above_physical_limit": day & (df["ghi"] > 1.1 * df["g0"]),
        "diffuse_above_global": df["dhi"] > df["ghi"] + 1,
        "temperature_out_of_range": ~df["temp_era5"].between(-40, 50) | ~df["t2m_pvgis"].between(-40, 50),
        "humidity_out_of_range": ~df["rh_era5"].between(0, 100),
        "cloud_out_of_range": ~df["cloud_era5"].between(0, 100),
        "negative_wind": (df["wind_era5"] < 0) | (df["ws10m_pvgis"] < 0),
        "stuck_ghi": _stuck(df["ghi"], day & (df["ghi"] > 0)),
    }
    counts = {name: int(f.sum()) for name, f in flags.items()}
    counts["duplicate_hours"] = int(df.index.duplicated().sum())
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    counts["missing_hours"] = len(full) - len(df)
    qc_flag = pd.concat(flags, axis=1).any(axis=1).rename("qc_flag")
    return counts, qc_flag


def _stuck(s: pd.Series, valid: pd.Series, n: int = 3) -> pd.Series:
    """Hours that end a run of ``n`` equal consecutive valid values."""
    same = valid.copy()
    for k in range(1, n):
        same &= (s == s.shift(k)) & valid.shift(k, fill_value=False)
    return same


# --- 5. Gap detection --------------------------------------------------------


def detect_gaps(df: pd.DataFrame, threshold: float = GAP_THRESHOLD) -> pd.Series:
    """PVGIS zero-filled gap: our sun is above ``threshold`` but PVGIS sun height is 0.

    Sun height is geometry, so weather cannot make it zero; these are missing
    satellite data (decision D5).
    """
    return ((df["elevation"] > threshold) & (df["sun_height_pvgis"] == 0)).rename("pvgis_gap")


def gap_sensitivity(df: pd.DataFrame, thresholds: Sequence[float] = (0, 0.5, 1, 2, 3, 5, 10)) -> dict[str, dict[str, float]]:
    """Gap count and mean ERA5 GHI in the gap hours, for several thresholds.

    With low thresholds PVGIS's ordinary zeros at very low sun count as gaps; ERA5
    shows little light there. With the chosen 5° the gap hours are clearly daylight.
    """
    out = {}
    for t in thresholds:
        gap = detect_gaps(df, t)
        out[str(t)] = {"hours": int(gap.sum()), "mean_ghi_era5": round(float(df.loc[gap, "ghi_era5"].mean()), 1)}
    return out


def apply_gaps(df: pd.DataFrame, gap: pd.Series) -> pd.DataFrame:
    """Set PVGIS radiation, power and sun height to NaN in gap hours; add ``pvgis_gap``."""
    out = df.copy()
    out.loc[gap, PVGIS_RAD + ["sun_height_pvgis"]] = np.nan
    out["pvgis_gap"] = gap
    return out


def run_lengths(mask: pd.Series) -> pd.Series:
    """Length of the run of consecutive True values each True row belongs to (0 elsewhere)."""
    run_id = (mask != mask.shift()).cumsum()
    length = mask.groupby(run_id).transform("size")
    return length.where(mask, 0)


# --- 6. Gap filling ----------------------------------------------------------


def fill_short_gaps(df: pd.DataFrame, cols: Sequence[str] = PVGIS_RAD, max_run: int = MAX_FILL_RUN) -> pd.DataFrame:
    """Fill short NaN runs using the clearness ratio k = X / g0 (decision D6).

    k changes slowly through the day, while X itself follows the sun curve. So k is
    interpolated linearly in time between the neighbouring valid hours and turned
    back into X = k · g0.

    A run is filled only if it is at most ``max_run`` hours long. The hours just
    before and just after the run must both be valid (not missing):

    * both are daylight (g0 > 0) → k is interpolated linearly between them;
    * one is night (g0 = 0, k undefined; the gap is the first or last daylight
      hour) → k of the daylight neighbour is held constant.

    Longer runs stay NaN. Filled rows get ``filled = True``.
    """
    out = df.copy()
    g0 = out["g0"]
    missing = out[list(cols)].isna().any(axis=1)
    run_id = (missing != missing.shift()).cumsum()
    length = run_lengths(missing)

    def neighbour(flag: pd.Series, side: int) -> pd.Series:
        """Value of ``flag`` in the row just before (side=1) / after (side=-1) each run."""
        how = "first" if side == 1 else "last"
        return flag.shift(side, fill_value=False).groupby(run_id).transform(how).astype(bool)

    day_ok = ~missing & (g0 > 0)
    night = ~missing & (g0 <= 0)
    day_before, day_after = neighbour(day_ok, 1), neighbour(day_ok, -1)
    night_before, night_after = neighbour(night, 1), neighbour(night, -1)

    short = missing & (length <= max_run) & (g0 > 0)
    both = short & day_before & day_after
    hold_next = short & night_before & day_after  # first daylight hour of the day
    hold_prev = short & day_before & night_after  # last daylight hour of the day
    fillable = both | hold_next | hold_prev

    for col in cols:
        k = (out[col] / g0).where(g0 > 0)
        k_fill = k.interpolate(method="linear", limit_area="inside").where(both)
        k_fill = k_fill.fillna(k.bfill(limit=max_run).where(hold_next))
        k_fill = k_fill.fillna(k.ffill(limit=max_run).where(hold_prev))
        out.loc[fillable, col] = (k_fill * g0)[fillable]

    prev = out["filled"] if "filled" in out else False
    out["filled"] = fillable | prev
    return out


def validate_fill(df: pd.DataFrame, n: int = 500, seed: int = 0, cols: Sequence[str] = ("ghi", "p_w")) -> dict[str, Any]:
    """Honest error bar for the fill: hide ``n`` real single hours, fill them, compare.

    Candidates are valid daylight hours (elevation > 5°) whose neighbours are valid
    too. No two hidden hours are adjacent, so each one is a 1-hour gap like most
    real gaps.
    """
    ok = df[list(cols)].notna().all(axis=1) & (df["g0"] > 0)
    candidate = ok & (df["elevation"] > GAP_THRESHOLD) & ok.shift(1, fill_value=False) & ok.shift(-1, fill_value=False)
    rng = np.random.default_rng(seed)
    pos = np.sort(rng.choice(np.flatnonzero(candidate.to_numpy()), size=n, replace=False))
    pos = pos[np.r_[True, np.diff(pos) > 1]]  # drop a hidden hour next to another one

    hidden = df.copy()
    rows = hidden.index[pos]
    hidden.loc[rows, list(cols)] = np.nan
    filled = fill_short_gaps(hidden, cols)

    result: dict[str, Any] = {"n_hours": len(pos), "seed": seed}
    for col in cols:
        err = filled.loc[rows, col] - df.loc[rows, col]
        mean_true = float(df.loc[rows, col].mean())
        rmse = float(np.sqrt((err**2).mean()))
        result[col] = {
            "rmse": round(rmse, 2),
            "bias": round(float(err.mean()), 2),
            "mean_true": round(mean_true, 2),
            "rmse_pct_of_mean": round(100 * rmse / mean_true, 2),
        }
    return result


# --- 7. Daily table ----------------------------------------------------------


def build_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """One row per local calendar day (Europe/Belgrade).

    Energies are left Riemann sums ``Σ X · 1 h``; M2 compares them with trapezoid
    and Simpson. NaN hours count as 0 in the sums; ``complete`` says whether any
    daylight hour is still NaN (decision D7: M3 uses complete days only).
    Days with fewer than 23 hourly rows (edges of the UTC→local shift) are dropped.
    """
    local = data.to_local(hourly)
    daylight = local["elevation"] > 0
    day = pd.Index(local.index.date, name="date")
    g = local.groupby(day)

    weather = local[["temp_era5", "cloud_era5", "wind_era5", "rh_era5"]].where(daylight, axis=0)
    out = pd.DataFrame(
        {
            "e_kwh": g["p_w"].sum() / 1000,
            "h_kwh_m2": g["ghi"].sum() / 1000,
            "h0_kwh_m2": g["g0"].sum() / 1000,
            "e35_kwh": g["p35_w"].sum() / 1000,
        }
    )
    out["kt_daily"] = out["h_kwh_m2"] / out["h0_kwh_m2"]
    means = weather.groupby(day).mean().add_suffix("_day")
    out = out.join(means)
    out["n_gap_hours"] = g["pvgis_gap"].sum().astype(int)
    out["n_filled_hours"] = g["filled"].sum().astype(int)
    out["complete"] = (local["p_w"].isna() & daylight).groupby(day).sum() == 0

    out = out[g.size() >= 23]
    out.index = pd.DatetimeIndex(out.index, name="date")
    return out


# --- 8. Yearly totals --------------------------------------------------------


def yearly_totals(raw: pd.DataFrame, daily: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Mean yearly kWh, computed three ways to show the effect of the gaps.

    * ``raw_zeros``: PVGIS as delivered (gaps count as 0 kWh).
    * ``gap_filled``: after filling short gaps; long gaps still count as 0.
    * ``complete_day_mean``: for each (year, month), the mean of complete days
      × days in the month. The result is summed per year and averaged.

    All sums are divided by the 19 years, not by the number of distinct years.
    """
    out: dict[str, dict[str, float]] = {}
    for col, dcol in (("p_w", "e_kwh"), ("p35_w", "e35_kwh")):
        raw_daily = data.daily_energy(raw, col)
        comp = daily.loc[daily["complete"], dcol]
        monthly_mean = comp.groupby([comp.index.year, comp.index.month]).mean()
        days = pd.Series(
            [pd.Period(year=y, month=m, freq="M").days_in_month for y, m in monthly_mean.index],
            index=monthly_mean.index,
        )
        out[dcol] = {
            "raw_zeros": round(float(raw_daily.sum()) / N_YEARS, 1),
            "gap_filled": round(float(daily[dcol].sum()) / N_YEARS, 1),
            "complete_day_mean": round(float((monthly_mean * days).sum()) / N_YEARS, 1),
        }
    return out


# --- Pipeline ----------------------------------------------------------------


def build(write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Raw cached files → cleaned hourly table, daily table, quality report."""
    pvgis = data.load_pvgis_hourly(0)
    pvgis35 = data.load_pvgis_hourly(cfg.REF_TILT)
    era5 = data.load_openmeteo_hourly()

    report: dict[str, Any] = {
        "pvgis_gr_all_zero": bool((pvgis["Gr"] == 0).all()),
        "pvgis_reconstructed_hours": int(pvgis["reconstructed"].sum()),
    }

    pv, pv35, era = rename_and_convert(pvgis, pvgis35, era5)
    before = lag_correlation(pv.set_axis(pvgis_key(pv.index))["ghi"], era["ghi_era5"])
    df = align(pv, pv35, era)
    after = lag_correlation(df["ghi"], df["ghi_era5"])
    report["alignment"] = {
        "lag_corr_before": before,
        "lag_corr_after": after,
        "best_lag_before": max(before, key=before.__getitem__),
        "best_lag_after": max(after, key=after.__getitem__),
        "mean_peak_hour_utc": {"pvgis": round(mean_peak_hour(df["ghi"]), 2), "era5": round(mean_peak_hour(df["ghi_era5"]), 2)},
        "rows": len(df),
    }

    df = add_geometry(df)
    counts, qc_flag = value_checks(df)
    df["qc_flag"] = qc_flag
    report["value_checks"] = counts

    raw = df.copy()  # gaps still zero, for the "raw" yearly total
    gap = detect_gaps(df)
    report["gap_sensitivity"] = gap_sensitivity(df)
    df = apply_gaps(df, gap)
    report["fill_validation"] = validate_fill(df)
    df = fill_short_gaps(df)

    lengths = run_lengths(gap)
    run_starts = gap & ~gap.shift(1, fill_value=False)
    run_sizes = lengths[run_starts]
    report["gaps"] = {
        "threshold_deg": GAP_THRESHOLD,
        "hours": int(gap.sum()),
        "days_with_gap": int(pd.Series(data.to_local(df).index.date)[gap.to_numpy()].nunique()),
        "runs_by_length": {str(k): int(v) for k, v in run_sizes.value_counts().sort_index().items()},
        "filled_hours": int(df["filled"].sum()),
        "still_nan_hours": int(df["p_w"].isna().sum()),
        "tilt35_nonzero_in_gap": int((raw.loc[gap, "poa35"] > 0).sum()),
        "per_year": {
            str(y): {"gap_hours": int(gh), "filled_hours": int(fh)}
            for y, gh, fh in zip(
                gap.groupby(gap.index.year).sum().index,
                gap.groupby(gap.index.year).sum(),
                df["filled"].groupby(df.index.year).sum(),
                strict=True,
            )
        },
    }

    daily = build_daily(df)
    incomplete = daily.index[~daily["complete"]]
    report["daily"] = {
        "days": len(daily),
        "incomplete_days": len(incomplete),
        "incomplete_dates": [d.strftime("%Y-%m-%d") for d in incomplete],
    }
    report["yearly_kwh"] = yearly_totals(raw, daily)

    if write:
        cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(HOURLY_PATH, date_format="%Y-%m-%dT%H:%M:%SZ", float_format="%.4f")
        daily.to_csv(DAILY_PATH, date_format="%Y-%m-%d", float_format="%.4f")
        REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    return df, daily, report


def load_hourly(path: Path | None = None) -> pd.DataFrame:
    """Cleaned hourly table, UTC index (start of the hour)."""
    df = pd.read_csv(path or HOURLY_PATH, index_col="time")
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True), name="time")
    return df


def load_daily(path: Path | None = None) -> pd.DataFrame:
    """Daily table, index = local calendar date."""
    df = pd.read_csv(path or DAILY_PATH, index_col="date", parse_dates=["date"])
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build data/processed/ from the cached raw files (offline).")
    parser.add_argument("--no-write", action="store_true", help="run and print the summary without writing files")
    args = parser.parse_args(argv)
    _, _, report = build(write=not args.no_write)
    g = report["gaps"]
    print(f"rows: {report['alignment']['rows']}, best lag after alignment: {report['alignment']['best_lag_after']} h")
    print(f"value checks: {report['value_checks']}")
    print(f"gaps: {g['hours']} h, filled {g['filled_hours']}, still NaN {g['still_nan_hours']}, runs {g['runs_by_length']}")
    print(f"fill validation: {report['fill_validation']}")
    print(f"incomplete days: {report['daily']['incomplete_days']} of {report['daily']['days']}")
    print(f"yearly kWh: {report['yearly_kwh']}")
    if not args.no_write:
        print(f"written: {HOURLY_PATH}, {DAILY_PATH}, {REPORT_PATH}")


if __name__ == "__main__":
    main()
