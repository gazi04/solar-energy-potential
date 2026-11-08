"""Sun position for Gjilan, a thin layer over pvlib.

pvlib supplies the astronomy (NREL SPA for elevation/azimuth, Spencer 1971 for
declination and equation of time). The formulas behind them are explained in
the M1 notebook. Interface: degrees, timestamps are UTC.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pvlib import solarposition
from scipy.optimize import newton

from solar_gjilan import config as cfg

PHI = np.deg2rad(cfg.LAT)


def sun_position(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Sun position at each UTC timestamp for Gjilan.

    Columns:
        declination        degrees, Spencer 1971
        equation_of_time   minutes, Spencer 1971
        hour_angle         degrees, 0 at solar noon, negative in the morning
        solar_time         hours, 12 at solar noon
        elevation          degrees above the horizon, no refraction (geometric)
        apparent_elevation degrees, with atmospheric refraction
        azimuth            degrees clockwise from north
    """
    if index.tz is None:
        raise ValueError("index must be timezone-aware (UTC)")
    index = index.tz_convert("UTC")

    day = index.day_of_year
    declination = solarposition.declination_spencer71(day)
    eot = solarposition.equation_of_time_spencer71(day)
    hour_angle = solarposition.hour_angle(index, cfg.LON, eot)
    spa = solarposition.get_solarposition(index, cfg.LAT, cfg.LON)

    return pd.DataFrame(
        {
            "declination": np.rad2deg(declination),
            "equation_of_time": eot,
            "hour_angle": hour_angle,
            "solar_time": 12 + hour_angle / 15,
            "elevation": spa["elevation"],
            "apparent_elevation": spa["apparent_elevation"],
            "azimuth": spa["azimuth"],
        },
        index=index,
    )


def elevation_course(index: pd.DatetimeIndex, declination: str = "cooper", eot: str = "milne") -> np.ndarray:
    """Sun elevation [deg] from the course formulas, to measure their error against SPA.

    sin(alpha) = sin(phi) sin(delta) + cos(phi) cos(delta) cos(h), with
    delta from Cooper 1969 (or Spencer 1971), EoT from Milne 1921 / pvlib "pvcdrom"
    (or Spencer 1971), h from UTC + longitude + EoT (pvlib ``hour_angle``).
    delta and EoT are evaluated once per day (day of year), as in the course.
    """
    if index.tz is None:
        raise ValueError("index must be timezone-aware (UTC)")
    n = index.day_of_year.to_numpy()
    delta = {"cooper": solarposition.declination_cooper69, "spencer": solarposition.declination_spencer71}[declination](n)
    eq = {"milne": solarposition.equation_of_time_pvcdrom, "spencer": solarposition.equation_of_time_spencer71}[eot](n)
    h = np.deg2rad(np.asarray(solarposition.hour_angle(index, cfg.LON, eq)))
    zenith = np.asarray(solarposition.solar_zenith_analytical(PHI, h, delta))
    return 90.0 - np.rad2deg(zenith)


def horizon_f(t: ArrayLike, delta: ArrayLike, alpha0: float = 0.0) -> Any:
    """
    f(t) = sin(alpha(t)) - sin(alpha0); root = sun crosses alpha0.
    t: solar time [h]
    delta: declination [rad]
    alpha0: horizon angle [deg]
    (0 = geometric, -0.8333 = apparent sunrise).
    """
    h = np.deg2rad(15.0 * (np.asarray(t) - 12.0))
    return (
        np.sin(PHI) * np.sin(delta)
        + np.cos(PHI) * np.cos(delta) * np.cos(h)
        - np.sin(np.deg2rad(alpha0))
    )


def horizon_fprime(t: ArrayLike, delta: ArrayLike, _alpha0: float = 0.0) -> Any:
    """
    df/dt = -cos(phi) cos(delta) sin(h) * pi/12

    _alpha0 is unused (d/dt of the constant sin(alpha0) is 0) but must be
    accepted because scipy.optimize.newton passes the same args to f and fprime.
    """
    h = np.deg2rad(15.0 * (np.asarray(t) - 12.0))
    return -np.cos(PHI) * np.cos(delta) * np.sin(h) * np.pi / 12.0


def elevation_formula(n: ArrayLike, t: ArrayLike) -> Any:
    """Sun elevation alpha [deg] from the course formula (geometric, no refraction).

    sin(alpha) = sin(phi) sin(delta) + cos(phi) cos(delta) cos(h),  h = 15 deg * (t - 12)

    n: day of year, t: solar time [h]. Declination from Spencer 1971 (pvlib).
    Used for the derivative study, where analytic and numerical dalpha/dt must
    differentiate the same function.
    """
    delta = solarposition.declination_spencer71(n)
    return np.rad2deg(np.arcsin(horizon_f(t, delta)))


def elevation_rate(n: ArrayLike, t: ArrayLike) -> Any:
    """Analytic d(alpha)/dt [deg/h] of ``elevation_formula`` (chain rule).

    Differentiate both sides of sin(alpha) = sin(phi) sin(delta) + cos(phi) cos(delta) cos(h):

        cos(alpha) * dalpha/dt = -cos(phi) cos(delta) sin(h) * dh/dt,   dh/dt = pi/12 rad/h
        dalpha/dt = -cos(phi) cos(delta) sin(h) * (pi/12) / cos(alpha)   [rad/h]

    The right-hand numerator is ``horizon_fprime``. Zero at solar noon (h = 0),
    positive in the morning, negative in the afternoon. Undefined only at alpha = 90 deg,
    which never happens at 42.5 deg N.
    """
    delta = solarposition.declination_spencer71(n)
    cos_alpha = np.cos(np.deg2rad(elevation_formula(n, t)))
    return np.rad2deg(horizon_fprime(t, delta) / cos_alpha)


def newton_table(n: int, t0: float = 6.0, alpha0: float = 0.0) -> tuple[Any, pd.DataFrame]:
    delta = solarposition.declination_spencer71(n)
    path: list[float] = []

    def f_log(t: float, d: float, a0: float) -> Any:
        path.append(t)
        return horizon_f(t, d, a0)

    root = newton(f_log, t0, fprime=horizon_fprime, args=(delta, alpha0), tol=1e-12)
    table = pd.DataFrame({"k": range(len(path)), "t_k": path})
    table["error"] = (table["t_k"] - root).abs()
    return root, table


def sunrise_exact(n: ArrayLike, alpha0: float = 0.0) -> Any:
    delta = solarposition.declination_spencer71(n)
    cos_h0 = (np.sin(np.deg2rad(alpha0)) - np.sin(PHI) * np.sin(delta)) / (
        np.cos(PHI) * np.cos(delta)
    )

    return 12.0 - np.rad2deg(np.arccos(cos_h0)) / 15.0


def sunrise_sunset(n: ArrayLike, alpha0: float = 0.0) -> pd.DataFrame:
    """
    Sunrise sunset [solar h] and day length [h] for day(s) of year n, via Newton.
    """
    n = np.atleast_1d(np.asarray(n))
    delta = solarposition.declination_spencer71(n)
    t0 = np.full(n.shape, 6.0)
    sunrise = newton(
        horizon_f, t0, fprime=horizon_fprime, args=(delta, alpha0), tol=1e-12
    )
    sunset = 24.0 - sunrise
    return pd.DataFrame(
        {"sunrise": sunrise, "sunset": sunset, "day_length": sunset - sunrise},
        index=pd.Index(n, name="n"),
    )


def to_clock(t_solar: ArrayLike, date: Any) -> pd.DatetimeIndex:
    """Solar time [h] on given date(s) -> Europe/Belgrade clock time.

    UTC = t_solar - LON/15 - EoT/60.
    """
    date = pd.DatetimeIndex(pd.to_datetime(date))
    eot = solarposition.equation_of_time_spencer71(date.day_of_year)
    utc_hours = np.asarray(t_solar) - cfg.LON / 15.0 - eot / 60.0
    utc = date.tz_localize("UTC") + pd.to_timedelta(utc_hours, unit="h")
    return utc.tz_convert(cfg.TZ)
