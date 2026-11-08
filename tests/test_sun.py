from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pvlib import solarposition
from scipy.optimize import newton

from solar_gjilan import config as cfg
from solar_gjilan import data
from solar_gjilan.sun import (
    elevation_course,
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
        noon = pd.Timestamp(d.index[int(d["elevation"].to_numpy().argmax())])
        assert abs((noon - pd.Timestamp(expected, tz="UTC")).total_seconds()) <= 180


def test_hour_angle_zero_at_solar_noon_and_solar_time_is_12():
    d = _day("2023-06-21")
    k = int(d["elevation"].to_numpy().argmax())
    assert d["hour_angle"].to_numpy()[k] == pytest.approx(0.0, abs=0.5)
    assert d["solar_time"].to_numpy()[k] == pytest.approx(12.0, abs=0.05)


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


# --- Validation against PVGIS, textbook values and pvlib SPA ----------------

FIX = Path(__file__).parent / "fixtures"


def test_equation_of_time_extremes():
    # Textbook: about +16.4 min in early November, about -14.2 min in mid February
    n = np.arange(1, 366)
    eot = solarposition.equation_of_time_spencer71(n)
    assert eot.max() == pytest.approx(16.4, abs=0.3)
    assert 300 <= n[eot.argmax()] <= 310
    assert eot.min() == pytest.approx(-14.2, abs=0.3)
    assert 40 <= n[eot.argmin()] <= 50


def test_milne_eot_vs_spencer_error_is_small():
    # Course formula (Milne 1921, pvlib "pvcdrom") against Spencer 1971
    n = np.arange(1, 366)
    err = np.abs(solarposition.equation_of_time_pvcdrom(n) - solarposition.equation_of_time_spencer71(n))
    assert err.mean() < 0.5
    assert err.max() < 1.0


def _pvgis_daylight() -> pd.DataFrame:
    pvgis = data.load_pvgis_hourly(path=FIX / "pvgis_hourly_sample.csv")
    return pvgis[pvgis["sun_height"] > 0]


def test_elevation_matches_pvgis_sun_height():
    pv = _pvgis_daylight()
    ours = sun_position(pd.DatetimeIndex(pv.index))["elevation"].to_numpy()
    err = np.abs(ours - pv["sun_height"].to_numpy())
    assert err.mean() < 0.5
    assert err.max() < 1.0


def test_pvgis_values_are_stamped_at_minute_10():
    # Our elevation agrees with PVGIS sun_height best at HH:10, not at the full or half hour
    pv = _pvgis_daylight()
    idx = pd.DatetimeIndex(pv.index)
    assert (idx.minute == 10).all()

    def mae(shift_min: int) -> float:
        ours = sun_position(idx + pd.Timedelta(minutes=shift_min))["elevation"].to_numpy()
        return float(np.abs(ours - pv["sun_height"].to_numpy()).mean())

    assert mae(0) < mae(-10) and mae(0) < mae(20)


def test_solar_noon_clock_time_all_days():
    # Solar noon in CET (UTC+1, no DST) stays between 11:15 and 11:50 all year (EoT ±16 min)
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    noon = to_clock(np.full(len(dates), 12.0), dates).tz_convert("Etc/GMT-1")
    hours = noon.hour + noon.minute / 60
    assert hours.min() > 11.25 and hours.max() < 11 + 50 / 60


def test_apparent_sunrise_sunset_match_pvlib_spa_whole_year():
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    rs = sunrise_sunset(dates.day_of_year.to_numpy(), alpha0=-0.833)
    spa = solarposition.sun_rise_set_transit_spa(dates.tz_localize(cfg.TZ), cfg.LAT, cfg.LON)
    for ours, ref in [(rs["sunrise"], spa["sunrise"]), (rs["sunset"], spa["sunset"])]:
        diff = (to_clock(ours.to_numpy(), dates) - pd.DatetimeIndex(ref)).total_seconds()
        assert np.abs(diff).max() < 2 * 60  # δ held constant over the day → ≤ 2 min


@pytest.mark.parametrize(("declination", "eot", "mae_max"), [("cooper", "milne", 0.5), ("spencer", "spencer", 0.25)])
def test_course_formula_elevation_vs_pvlib_spa_2023(declination, eot, mae_max):
    # M1 plan 2.4 item 2: course formulas vs NREL SPA, every hour of 2023 at HH:10, daylight only
    idx = pd.date_range("2023-01-01", "2023-12-31 23:00", freq="h", tz="UTC") + pd.Timedelta(minutes=10)
    spa = np.asarray(solarposition.get_solarposition(idx, cfg.LAT, cfg.LON)["elevation"])
    err = np.abs(elevation_course(idx, declination, eot) - spa)[spa > 0]
    assert err.mean() < mae_max
    assert err.max() < 1.2
