"""Derivatives (milestone M1, step 5; syllabus weeks 4-5).

Two uses in the project:

1. **Sun elevation** dα/dt: analytic formula (``sun.elevation_rate``) vs numerical
   differences. The error study shows how the error shrinks with the step Δt:
   forward difference O(Δt), central difference O(Δt²) (Taylor series argument).
   dα/dt = 0 at solar noon is the first "derivative = 0 → maximum" example; M3
   uses the same idea for the optimal tilt (dE/dβ = 0).
2. **PV power** dP/dt on the real hourly data: smooth on clear days, jumps on
   cloudy days. Ramp statistics say how fast production swings.

Central differences on sampled data come from ``numpy.gradient``. NumPy has no
forward-difference function, so ``forward_diff`` is one line of own code; the
central difference of a callable is also written out so both appear side by side
in the error study (it matches ``numpy.gradient`` exactly, see tests).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from solar_gjilan import data, sun

Func = Callable[[np.ndarray], Any]

RAMP_THRESHOLD_KW_H = 1.0  # kW per hour, for a 5 kWp system = 20 % of peak per hour
SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}


# --- 1. Difference formulas --------------------------------------------------


def forward_diff(f: Func, t: np.ndarray, dt: float) -> np.ndarray:
    """f'(t) ≈ (f(t + Δt) − f(t)) / Δt. Taylor: error = f''(t)·Δt/2 + … → O(Δt)."""
    return (np.asarray(f(t + dt)) - np.asarray(f(t))) / dt


def central_diff(f: Func, t: np.ndarray, dt: float) -> np.ndarray:
    """f'(t) ≈ (f(t + Δt) − f(t − Δt)) / (2Δt). The f'' terms cancel: error = f'''(t)·Δt²/6 + … → O(Δt²)."""
    return (np.asarray(f(t + dt)) - np.asarray(f(t - dt))) / (2 * dt)


# --- 2. Error study on the sun elevation -------------------------------------


def elevation_curve(n: int, step_min: float = 1.0) -> pd.DataFrame:
    """α(t), analytic dα/dt and ``np.gradient`` dα/dt over one day (solar time grid)."""
    t = np.arange(0, 24 + 1e-9, step_min / 60)
    alpha = sun.elevation_formula(n, t)
    return pd.DataFrame(
        {
            "elevation": alpha,
            "rate_analytic": sun.elevation_rate(n, t),
            "rate_gradient": np.gradient(alpha, t),
        },
        index=pd.Index(t, name="solar_time"),
    )


def step_error_study(n: int = 172, steps_min: Sequence[float] = (60, 30, 15, 5, 1)) -> pd.DataFrame:
    """Max |numerical − analytic| dα/dt [°/h] over daylight hours for each step Δt.

    Evaluated at fixed points (every 10 min between sunrise + 30 min and sunset − 30 min),
    so only Δt changes between rows.
    """
    rs = sun.sunrise_sunset(n).iloc[0]
    t = np.arange(rs["sunrise"] + 0.5, rs["sunset"] - 0.5, 1 / 6)
    exact = sun.elevation_rate(n, t)

    def f(x: np.ndarray) -> Any:
        return sun.elevation_formula(n, x)

    rows = []
    for m in steps_min:
        dt = m / 60
        rows.append(
            {
                "step_min": m,
                "forward": np.abs(forward_diff(f, t, dt) - exact).max(),
                "central": np.abs(central_diff(f, t, dt) - exact).max(),
            }
        )
    return pd.DataFrame(rows).set_index("step_min")


def convergence_order(errors: pd.DataFrame) -> pd.Series:
    """Slope of log(error) vs log(Δt) per column (least squares): ≈ 1 forward, ≈ 2 central."""
    x = np.log(errors.index.to_numpy(dtype=float))
    return pd.Series({col: np.polyfit(x, np.log(errors[col].to_numpy()), 1)[0] for col in errors.columns})


# --- 3. dP/dt on the real hourly data ----------------------------------------


def power_rate(hourly: pd.DataFrame, col: str = "p_w") -> pd.DataFrame:
    """PV power and its rate of change, local time index.

    Columns:
        p_kw        power [kW] (5 kWp, flat; PVGIS)
        dpdt        central difference ``np.gradient`` [kW/h], P'(t) ≈ (P(t+1) − P(t−1)) / 2
        step        hour-to-hour change P(t) − P(t−1) [kW/h], the ramp the grid sees
        elevation   our sun elevation [°]
    NaN gap hours give NaN rates next to them (not filled).
    """
    local = data.to_local(hourly)
    p = local[col] / 1000
    return pd.DataFrame(
        {
            "p_kw": p,
            "dpdt": np.gradient(p.to_numpy()),
            "step": p.diff().to_numpy(),
            "elevation": local["elevation"],
        },
        index=local.index,
    )


def pick_days(daily: pd.DataFrame, rate: pd.DataFrame, year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Clearest day (max kt_daily) and most variable day of a month (complete days only).

    Variability = total variation Σ|ΔP| − 2·max P. A smooth one-hump curve rises once
    to its peak and falls once, so Σ|ΔP| = 2·max P and the excess is 0; every extra
    up/down swing (passing cloud) adds to it.
    """
    didx = data.time_index(daily)
    days = daily[(didx.year == year) & (didx.month == month) & daily["complete"]]
    clear = data.argmax_date(days["kt_daily"])
    date = pd.DatetimeIndex(data.time_index(rate).date)
    tv = rate["step"].abs().groupby(date).sum()
    peak = rate["p_kw"].groupby(date).max()
    excess = (tv - 2 * peak).reindex(days.index)
    return clear, data.argmax_date(excess)


def one_day(rate: pd.DataFrame, date: pd.Timestamp | str) -> pd.DataFrame:
    """Rows of ``rate`` for one local calendar day."""
    d = pd.Timestamp(date).date()
    return rate[data.time_index(rate).date == d]


def ramp_stats(rate: pd.DataFrame, threshold: float = RAMP_THRESHOLD_KW_H) -> dict[str, Any]:
    """Ramp statistics over daylight hours (elevation > 0, NaN dropped).

    Returns share of hours above ``threshold`` (central and hour-to-hour), |rate|
    quantiles per season, and the 10 largest hour-to-hour ramps.
    """
    day = rate[rate["elevation"] > 0].dropna(subset=["dpdt", "step"])
    season = pd.Series(data.time_index(day).month, index=day.index).map(SEASONS)
    per_season = (
        day["step"].abs().groupby(season).quantile(np.array([0.5, 0.9, 0.99])).unstack()
        .reindex(["Winter", "Spring", "Summer", "Autumn"])
    )
    per_season.columns = ["p50", "p90", "p99"]
    share_season = (day["step"].abs() > threshold).groupby(season).mean()
    per_season["share_above"] = share_season
    top = day.reindex(day["step"].abs().nlargest(10).index)[["p_kw", "step", "elevation"]]
    return {
        "n_hours": int(len(day)),
        "threshold_kw_h": threshold,
        "share_above_central": float((day["dpdt"].abs() > threshold).mean()),
        "share_above_step": float((day["step"].abs() > threshold).mean()),
        "max_abs_central": float(day["dpdt"].abs().max()),
        "max_abs_step": float(day["step"].abs().max()),
        "per_season": per_season,
        "top10": top,
    }
