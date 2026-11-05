import numpy as np
import pandas as pd
import pytest
from pvlib import solarposition

from solar_gjilan import config as cfg
from solar_gjilan.sun import sun_position


def _day(date: str) -> pd.DataFrame:
    idx = pd.date_range(date, periods=1440, freq="min", tz="UTC")
    return sun_position(idx)


def test_noon_elevation_summer_solstice():
    # 90 - latitude + declination = 70.99 deg
    assert _day("2023-06-21")["elevation"].max() == pytest.approx(70.99, abs=0.1)


def test_noon_elevation_winter_solstice():
    assert _day("2023-12-21")["elevation"].max() == pytest.approx(24.09, abs=0.1)


def test_solar_noon_clock_time():
    # ~12:36 CEST on 21 Jun (10:36 UTC), ~11:33 CET on 21 Dec (10:32 UTC), within 3 min
    for date, expected in [("2023-06-21", "2023-06-21 10:36"), ("2023-12-21", "2023-12-21 10:32")]:
        noon = _day(date)["elevation"].idxmax()
        assert abs((noon - pd.Timestamp(expected, tz="UTC")).total_seconds()) <= 180


def test_hour_angle_zero_at_solar_noon_and_solar_time_is_12():
    d = _day("2023-06-21")
    noon = d["elevation"].idxmax()
    assert d.loc[noon, "hour_angle"] == pytest.approx(0.0, abs=0.5)
    assert d.loc[noon, "solar_time"] == pytest.approx(12.0, abs=0.05)


def test_declination_solstices():
    d = sun_position(pd.DatetimeIndex(["2023-06-21 12:00", "2023-12-21 12:00"], tz="UTC"))
    assert d["declination"].iloc[0] == pytest.approx(23.45, abs=0.1)
    assert d["declination"].iloc[1] == pytest.approx(-23.45, abs=0.1)


def test_cooper_declination_vs_spencer_error_is_small():
    # Course formula (Cooper 1969) against the Spencer series used in sun_position
    n = np.arange(1, 366)
    cooper = np.rad2deg(solarposition.declination_cooper69(n))
    spencer = np.rad2deg(solarposition.declination_spencer71(n))
    err = np.abs(cooper - spencer)
    assert err.mean() < 0.6
    assert err.max() < 1.6


def test_requires_timezone_aware_index():
    with pytest.raises(ValueError):
        sun_position(pd.DatetimeIndex(["2023-06-21 12:00"]))
