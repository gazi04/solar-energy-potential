import numpy as np
import pandas as pd
import pytest
from pvlib import solarposition
from scipy.optimize import newton

from solar_gjilan import config as cfg
from solar_gjilan.sun import (
    horizon_f,
    horizon_fprime,
    newton_table,
    sun_position,
    sunrise_exact,
    sunrise_sunset,
    to_clock,
)


def _day(date: str) -> pd.DataFrame:
    idx = pd.date_range(date, periods=1440, freq="min", tz="UTC")
    return sun_position(idx)


def test_noon_elevation_summer_solstice():
    # 90 - latitude + declination = 70.99 deg
    assert _day("2023-06-21")["elevation"].to_numpy().max() == pytest.approx(70.99, abs=0.1)


def test_noon_elevation_winter_solstice():
    assert _day("2023-12-21")["elevation"].to_numpy().max() == pytest.approx(24.09, abs=0.1)


def test_solar_noon_clock_time():
    # ~12:36 CEST on 21 Jun (10:36 UTC), ~11:33 CET on 21 Dec (10:32 UTC), within 3 min
    for date, expected in [("2023-06-21", "2023-06-21 10:36"), ("2023-12-21", "2023-12-21 10:32")]:
        d = _day(date)
        noon = d.index[d["elevation"].to_numpy().argmax()]
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


# --- Step 3: sunrise / sunset with Newton's method -------------------------


def test_sunrise_june_solstice():
    # cos h0 = -tan(phi) tan(delta) -> h0 = 113.38 deg -> 12 - h0/15 = 4.441 h
    assert sunrise_exact(172) == pytest.approx(4.441, abs=1e-3)


def test_newton_matches_exact_all_days():
    n = np.arange(1, 366)
    assert np.abs(sunrise_sunset(n)["sunrise"].to_numpy() - sunrise_exact(n)).max() < 1e-9


def test_newton_converges_in_few_iterations():
    root, table = newton_table(172, t0=6.0)
    assert len(table) - 1 <= 4  # last row is SciPy's final check at the root
    assert root == pytest.approx(sunrise_exact(172), abs=1e-12)


def test_newton_fails_at_noon():
    # f'(12) = 0: tangent is flat, Newton cannot step
    delta = solarposition.declination_spencer71(172)
    with pytest.raises(RuntimeError):
        newton(horizon_f, 12.0, fprime=horizon_fprime, args=(delta, 0.0))


def test_day_length_solstices():
    d = sunrise_sunset([172, 355])["day_length"].to_numpy()
    assert d == pytest.approx([15.12, 8.88], abs=0.01)


def test_apparent_sunrise_matches_pvlib_spa():
    date = pd.Timestamp("2023-06-21")
    ours = to_clock(sunrise_sunset(172, alpha0=-0.833)["sunrise"], [date])[0]
    spa = solarposition.sun_rise_set_transit_spa(
        pd.DatetimeIndex([date]).tz_localize(cfg.TZ), cfg.LAT, cfg.LON
    )["sunrise"].iloc[0]
    assert abs((ours - spa).total_seconds()) < 60
