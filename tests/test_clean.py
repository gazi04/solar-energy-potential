"""Offline tests for the cleaning pipeline (synthetic frames + 48-row real fixtures)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solar_gjilan import clean, data

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_hourly() -> pd.DataFrame:
    """21–22 June 2023 fixtures, renamed, aligned and with geometry (no gaps)."""
    pvgis = data.load_pvgis_hourly(path=FIX / "pvgis_hourly_sample.csv")
    era5 = data.load_openmeteo_hourly(path=FIX / "openmeteo_hourly_sample.csv")
    pv, pv35, era = clean.rename_and_convert(pvgis, pvgis, era5)  # fixture has no 35° file
    return clean.add_geometry(clean.align(pv, pv35, era))


def synthetic(ghi: list[float], g0: float = 500.0) -> pd.DataFrame:
    """Daylight hours with constant g0, for testing the fill arithmetic."""
    idx = pd.date_range("2023-06-21 06:00", periods=len(ghi), freq="h", tz="UTC")
    return pd.DataFrame({"ghi": ghi, "g0": g0}, index=idx)


def test_pvgis_and_era5_stamps_get_same_key():
    pv = clean.pvgis_key(pd.DatetimeIndex([pd.Timestamp("2023-06-21 10:10", tz="UTC")]))
    era = clean.era5_key(pd.DatetimeIndex([pd.Timestamp("2023-06-21 11:00", tz="UTC")]))
    assert pv[0] == era[0] == pd.Timestamp("2023-06-21 10:00", tz="UTC")


def test_align_fixture_is_continuous(fixture_hourly):
    # 48 PVGIS + 48 ERA5 rows, shifted by one hour → 47 common keys.
    assert len(fixture_hourly) == 47
    assert fixture_hourly.index.is_monotonic_increasing
    assert (fixture_hourly.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()


def test_wind_converted_to_m_per_s():
    idx = pd.date_range("2023-01-01", periods=1, freq="h", tz="UTC")
    era5 = pd.DataFrame({c: [0.0] for c in clean.ERA5_COLUMNS}, index=idx)
    era5["wind_speed_10m"] = 36.0
    pv = pd.DataFrame({"G_poa": 0.0, "Gb": 0.0, "Gd": 0.0, "P_W": 0.0, "sun_height": 0.0, "T2m": 0.0, "WS10m": 0.0}, index=idx)
    _, _, era = clean.rename_and_convert(pv, pv, era5)
    assert era["wind_era5"].iloc[0] == pytest.approx(10.0)


def test_lag_correlation_peak_at_true_shift():
    idx = pd.date_range("2023-06-21", periods=200, freq="h", tz="UTC")
    a = pd.Series(np.random.default_rng(0).random(200), index=idx)
    b = a.shift(1, freq="h")  # b is a, labelled one hour late
    corr = clean.lag_correlation(a, b)
    assert max(corr, key=corr.__getitem__) == -1


def test_detect_gaps_sun_up_vs_night():
    df = pd.DataFrame({"elevation": [30.0, 3.0, -10.0, 30.0], "sun_height_pvgis": [0.0, 0.0, 0.0, 29.9]})
    assert clean.detect_gaps(df).tolist() == [True, False, False, False]


def test_fill_one_and_two_hour_gaps_three_stays_nan():
    # k = ghi / g0 with g0 = 500: valid hours 100, 200 | gap | 300, 400 | 2 gaps | 100 | 3 gaps | 500
    ghi = [100, np.nan, 300, 400, np.nan, np.nan, 100, np.nan, np.nan, np.nan, 500]
    out = clean.fill_short_gaps(synthetic(ghi), cols=["ghi"])
    assert out["ghi"].iloc[1] == pytest.approx(200)  # midway between 100 and 300
    assert out["ghi"].iloc[4] == pytest.approx(400 - 100)  # 400 → 100 over 3 steps
    assert out["ghi"].iloc[5] == pytest.approx(400 - 200)
    assert out["ghi"].iloc[7:10].isna().all()
    assert out["filled"].tolist() == [False, True, False, False, True, True, False, False, False, False, False]


def test_fill_is_k_times_g0_not_raw_interpolation():
    df = synthetic([100.0, np.nan, 300.0])
    df["g0"] = [200.0, 800.0, 600.0]  # k = 0.5, ?, 0.5
    out = clean.fill_short_gaps(df, cols=["ghi"])
    assert out["ghi"].iloc[1] == pytest.approx(0.5 * 800)


def test_fill_first_daylight_hour_holds_k_of_next_hour():
    df = synthetic([0.0, np.nan, 300.0])
    df["g0"] = [0.0, 200.0, 600.0]  # night, gap, k = 0.5
    out = clean.fill_short_gaps(df, cols=["ghi"])
    assert out["ghi"].iloc[1] == pytest.approx(0.5 * 200)


def test_value_checks_flag_injected_errors(fixture_hourly):
    counts, flag = clean.value_checks(fixture_hourly)
    assert sum(counts.values()) == 0 and not flag.any()

    bad = fixture_hourly.copy()
    noon = bad["elevation"].idxmax()
    night = bad["elevation"].idxmin()
    bad.loc[noon, "ghi"] = -5.0
    bad.loc[night, "dhi"] = 20.0
    counts, flag = clean.value_checks(bad)
    assert counts["negative_radiation"] == 1
    assert counts["radiation_at_night"] == 1
    assert flag[noon] and flag[night]


def test_gap_day_is_incomplete(fixture_hourly):
    df = fixture_hourly.copy()
    # 3-hour gap around noon on 22 June (local) → too long to fill.
    gap_hours = pd.date_range("2023-06-22 09:00", periods=3, freq="h", tz="UTC")
    df.loc[gap_hours, ["ghi", "bhi", "dhi", "p_w", "poa35", "p35_w", "sun_height_pvgis"]] = 0.0
    gap = clean.detect_gaps(df)
    assert gap.sum() == 3
    df = clean.fill_short_gaps(clean.apply_gaps(df, gap))
    daily = clean.build_daily(df)
    day = daily.loc["2023-06-22"]
    assert not day["complete"]
    assert day["n_gap_hours"] == 3 and day["n_filled_hours"] == 0


def test_short_gap_day_is_complete_after_fill(fixture_hourly):
    df = fixture_hourly.copy()
    df.loc[pd.Timestamp("2023-06-22 10:00", tz="UTC"), ["ghi", "p_w", "sun_height_pvgis"]] = 0.0
    df = clean.fill_short_gaps(clean.apply_gaps(df, clean.detect_gaps(df)))
    day = clean.build_daily(df).loc["2023-06-22"]
    assert day["complete"] and day["n_filled_hours"] == 1
    assert day["e_kwh"] > 20  # sunny June day, flat 5 kWp
