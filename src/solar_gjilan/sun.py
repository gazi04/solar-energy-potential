"""Sun position for Gjilan, a thin layer over pvlib.

pvlib supplies the astronomy (NREL SPA for elevation/azimuth, Spencer 1971 for
declination and equation of time). The formulas behind them are explained in
the M1 notebook. Interface: degrees, timestamps are UTC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pvlib import solarposition

from solar_gjilan import config as cfg


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
