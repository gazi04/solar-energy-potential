"""Regenerate all M0/M1 figures into reports/figures/ from cached data.

    uv run python scripts/make_figures.py
"""

import matplotlib

matplotlib.use("Agg")

from solar_gjilan import config as cfg  # noqa: E402
from solar_gjilan import clean, data, plots  # noqa: E402

FIGURE_YEAR = 2023


def main() -> None:
    flat = data.load_pvgis_hourly(0)
    tilted = data.load_pvgis_hourly(cfg.REF_TILT)
    hourly = clean.load_hourly()

    summer = data.sunniest_day(flat, FIGURE_YEAR, 6)
    winter = data.sunniest_day(flat, FIGURE_YEAR, 12)
    print(f"sunniest June day: {summer:%Y-%m-%d}, sunniest December day: {winter:%Y-%m-%d}")

    figures = {
        "fig01_summer_vs_winter_day": plots.summer_vs_winter_day(flat, summer, winter),
        "fig02_monthly_energy": plots.monthly_energy(flat, tilted, cfg.REF_TILT),
        "fig03_irradiation_heatmap": plots.irradiation_heatmap(flat),
        "fig04_sun_elevation_days": plots.sun_elevation_days(
            hourly, [f"{FIGURE_YEAR}-06-21", f"{FIGURE_YEAR}-03-21", f"{FIGURE_YEAR}-12-21"]
        ),
        "fig07_day_length": plots.day_length_year(FIGURE_YEAR),
    }
    for name, fig in figures.items():
        print(f"saved {plots.save(fig, name).relative_to(cfg.ROOT)}")


if __name__ == "__main__":
    main()
