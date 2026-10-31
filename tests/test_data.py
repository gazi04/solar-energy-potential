"""Offline tests for parsing and basic aggregation. Fixtures are trimmed real API rows."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solar_gjilan import data, download

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def pvgis() -> pd.DataFrame:
    return data.load_pvgis_hourly(path=FIX / "pvgis_hourly_sample.csv")


def test_parse_pvgis_time_is_utc_at_minute_10():
    idx = data.parse_pvgis_time(pd.Series(["20230621:1010"]))
    assert idx[0] == pd.Timestamp("2023-06-21 10:10", tz="UTC")


def test_load_pvgis_columns_and_types(pvgis):
    assert len(pvgis) == 48
    assert str(pvgis.index.tz) == "UTC"
    for col in ["P_W", "Gb", "Gd", "Gr", "sun_height", "T2m", "WS10m", "reconstructed", "G_poa"]:
        assert col in pvgis
    assert pvgis["reconstructed"].dtype == bool
    np.testing.assert_allclose(pvgis["G_poa"], pvgis["Gb"] + pvgis["Gd"] + pvgis["Gr"])


def test_sun_height_near_solstice_noon(pvgis):
    # Expected noon elevation on 21 June in Gjilan ≈ 71°; hourly samples at HH:10 peak slightly lower.
    assert 69 < pvgis["sun_height"].max() < 71.5


def test_daily_energy_drops_partial_local_days(pvgis):
    # 48 UTC hours (21–22 June) → local days 21 (22 h), 22 (24 h), 23 (2 h); only 22 June is complete.
    daily = data.daily_energy(pvgis)
    assert list(pd.to_datetime(daily.index)) == [pd.Timestamp("2023-06-22")]
    local_22 = pvgis.loc["2023-06-21 22:00":"2023-06-22 21:59", "P_W"]
    assert len(local_22) == 24
    assert daily.iloc[0] == pytest.approx(local_22.sum() / 1000)


def test_sunniest_day_picks_highest_daily_irradiation():
    idx = pd.date_range("2023-06-01", "2023-06-30 23:00", freq="h", tz="UTC")
    df = pd.DataFrame({"G_poa": 100.0}, index=idx)
    df.loc["2023-06-15 08:00":"2023-06-15 14:00", "G_poa"] = 900.0
    assert data.sunniest_day(df, 2023, 6) == pd.Timestamp("2023-06-15")
    with pytest.raises(ValueError):
        data.sunniest_day(df, 2023, 7)


def test_quick_summary_detects_missing_hour(pvgis):
    assert data.quick_summary(pvgis)["missing_hours"] == 0
    gappy = pvgis.drop(pvgis.index[10])
    summary = data.quick_summary(gappy)
    assert summary["missing_hours"] == 1
    assert summary["rows"] == 47


def test_load_openmeteo():
    df = data.load_openmeteo_hourly(path=FIX / "openmeteo_hourly_sample.csv")
    assert len(df) == 48
    assert str(df.index.tz) == "UTC"
    assert df.index[0] == pd.Timestamp("2023-06-21 00:00", tz="UTC")
    assert {"temperature_2m", "cloud_cover", "shortwave_radiation"} <= set(df.columns)


def test_load_pvcalc():
    est = data.load_pvcalc(path=FIX / "pvcalc_sample.json")
    assert est["tilt"] == 35
    assert est["E_y"] == pytest.approx(6554.03)


def test_cached_skips_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(download.cfg, "ROOT", tmp_path)
    target = tmp_path / "exists.csv"
    target.write_text("x")
    calls = []
    download._cached(target, lambda: calls.append(1), force=False)
    assert calls == []
