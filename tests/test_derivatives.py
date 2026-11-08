"""Offline tests for derivatives (M1 step 5): analytic formulas, synthetic frames, fixtures."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solar_gjilan import clean, data, derivatives, sun

FIX = Path(__file__).parent / "fixtures"


# --- difference formulas -----------------------------------------------------


def test_central_difference_of_sin_is_cos_second_order():
    t = np.linspace(0, 2 * np.pi, 50)
    err = [np.abs(derivatives.central_diff(np.sin, t, dt) - np.cos(t)).max() for dt in (0.1, 0.05)]
    # error ≈ dt²/6 → halving dt divides the error by 4
    assert err[0] == pytest.approx(0.1**2 / 6, rel=0.01)
    assert err[0] / err[1] == pytest.approx(4, rel=0.01)


def test_forward_difference_of_sin_is_cos_first_order():
    t = np.linspace(0, 2 * np.pi, 50)
    err = [np.abs(derivatives.forward_diff(np.sin, t, dt) - np.cos(t)).max() for dt in (0.01, 0.005)]
    # error ≈ dt/2 → halving dt halves the error
    assert err[0] == pytest.approx(0.01 / 2, rel=0.02)
    assert err[0] / err[1] == pytest.approx(2, rel=0.02)


def test_central_diff_matches_np_gradient_interior():
    dt = 0.1
    t = np.arange(0, 5, dt)
    own = derivatives.central_diff(np.exp, t[1:-1], dt)
    assert np.allclose(own, np.gradient(np.exp(t), dt)[1:-1], rtol=1e-12)


# --- sun elevation -----------------------------------------------------------


def test_elevation_formula_noon_summer_solstice():
    assert sun.elevation_formula(172, 12.0) == pytest.approx(70.99, abs=0.05)


def test_elevation_rate_zero_at_noon_and_sign():
    assert sun.elevation_rate(172, 12.0) == pytest.approx(0.0, abs=1e-12)
    assert sun.elevation_rate(172, 9.0) > 0
    assert sun.elevation_rate(172, 15.0) < 0
    # symmetric around noon
    assert sun.elevation_rate(172, 9.0) == pytest.approx(-sun.elevation_rate(172, 15.0))


@pytest.mark.parametrize("n", [21, 80, 172, 355])
def test_analytic_rate_matches_numerical(n):
    t = np.linspace(5, 19, 57) if n == 172 else np.linspace(9, 15, 25)
    dt = 1 / 60  # 1 minute
    numeric = derivatives.central_diff(lambda x: sun.elevation_formula(n, x), t, dt)
    assert np.abs(numeric - sun.elevation_rate(n, t)).max() < 1e-3


def test_elevation_curve_gradient_close_to_analytic():
    c = derivatives.elevation_curve(172)
    assert (c["rate_gradient"] - c["rate_analytic"]).abs().iloc[1:-1].max() < 1e-3


def test_error_study_orders():
    err = derivatives.step_error_study(172)
    order = derivatives.convergence_order(err)
    assert order["forward"] == pytest.approx(1.0, abs=0.1)
    assert order["central"] == pytest.approx(2.0, abs=0.1)
    assert (err["central"] < err["forward"]).all()


# --- dP/dt on data -----------------------------------------------------------


def synthetic_hourly(p_w: Sequence[float], elevation: Sequence[float] | None = None) -> pd.DataFrame:
    idx = pd.date_range("2023-06-21 03:00", periods=len(p_w), freq="h", tz="UTC")
    elev = elevation if elevation is not None else [30.0] * len(p_w)
    return pd.DataFrame({"p_w": p_w, "elevation": elev}, index=idx)


def test_power_rate_units_and_steps():
    r = derivatives.power_rate(synthetic_hourly([0, 1000, 3000, 2000]))
    assert str(data.time_index(r).tz) == "Europe/Belgrade"
    assert r["p_kw"].tolist() == [0, 1, 3, 2]
    assert r["step"].tolist()[1:] == [1, 2, -1]
    # central difference at hour 2: (3 - 0) / 2
    assert r["dpdt"].iloc[1] == pytest.approx(1.5)


def test_ramp_stats_threshold_share():
    # steps: +1, +2, -2, -0.5 → two of four above 1 kW/h (night row excluded)
    p = [0, 1000, 3000, 1000, 500]
    elev = [-5, 10, 20, 30, 40]
    stats = derivatives.ramp_stats(derivatives.power_rate(synthetic_hourly(p, elev)))
    assert stats["n_hours"] == 4
    assert stats["share_above_step"] == pytest.approx(0.5)
    assert stats["max_abs_step"] == pytest.approx(2.0)
    assert stats["top10"]["step"].abs().iloc[0] == pytest.approx(2.0)


def test_pick_days_clear_vs_variable():
    idx = pd.date_range("2023-06-01", periods=48, freq="h", tz="Europe/Belgrade")
    t = np.arange(24)
    smooth = np.clip(np.sin((t - 6) / 12 * np.pi), 0, None) * 3000
    bumpy = smooth.copy()
    bumpy[[10, 12, 14]] = 200  # passing clouds
    hourly = pd.DataFrame({"p_w": np.r_[smooth, bumpy], "elevation": 10.0}, index=idx.tz_convert("UTC"))
    daily = pd.DataFrame(
        {"kt_daily": [0.75, 0.5], "complete": [True, True]},
        index=pd.DatetimeIndex(["2023-06-01", "2023-06-02"], name="date"),
    )
    clear, variable = derivatives.pick_days(daily, derivatives.power_rate(hourly), 2023, 6)
    assert clear == pd.Timestamp("2023-06-01")
    assert variable == pd.Timestamp("2023-06-02")


def test_power_rate_on_fixture_clear_day():
    pvgis = data.load_pvgis_hourly(path=FIX / "pvgis_hourly_sample.csv")
    era5 = data.load_openmeteo_hourly(path=FIX / "openmeteo_hourly_sample.csv")
    pv, pv35, era = clean.rename_and_convert(pvgis, pvgis, era5)
    r = derivatives.power_rate(clean.add_geometry(clean.align(pv, pv35, era)))
    day = derivatives.one_day(r, "2023-06-21")
    up = day[day["elevation"] > 0]
    # power rises before noon and falls after: dP/dt changes sign once around midday
    assert up["dpdt"].iloc[:3].gt(0).all() and up["dpdt"].iloc[-3:].lt(0).all()
