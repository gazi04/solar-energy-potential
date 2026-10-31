"""Figure functions. Each returns a matplotlib Figure; ``save`` writes it to ``reports/figures``.

Colors follow one fixed palette (categorical slots in fixed order, one-hue
sequential ramp), so every figure in the project reads as one system.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from solar_gjilan import config as cfg
from solar_gjilan import data

# --- palette -----------------------------------------------------------------
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "font.size": 9,
            "lines.linewidth": 2,
            "figure.dpi": 110,
        }
    )


def save(fig: plt.Figure, name: str, fig_dir: Path | None = None) -> Path:
    fig_dir = fig_dir or cfg.FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


def _clock_hours(index: pd.DatetimeIndex) -> np.ndarray:
    return index.hour + index.minute / 60


# --- figure 1 ------------------------------------------------------------------


def summer_vs_winter_day(hourly: pd.DataFrame, summer: str, winter: str) -> plt.Figure:
    """Irradiance (beam + diffuse) and 5 kWp PV power for one summer and one winter day.

    Small multiples: columns = days, rows = measure; each row shares its y-scale so
    the two seasons are directly comparable.
    """
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey="row",
                             gridspec_kw={"height_ratios": [1.3, 1]})
    for col, date in enumerate([summer, winter]):
        day = data.one_day(hourly, date)
        t = _clock_hours(day.index)
        e_kwh = day["P_W"].sum() / 1000
        h_kwh = day["G_poa"].sum() / 1000
        noon_sun = day["sun_height"].max()

        ax = axes[0, col]
        ax.stackplot(t, day["Gb"], day["Gd"], colors=SERIES[:2], edgecolor=SURFACE,
                     linewidth=1.5, labels=["Beam (direct)", "Diffuse"])
        ax.set_title(f"{pd.Timestamp(date):%d %B %Y}", pad=18)
        ax.text(0, 1.02, f"Daily irradiation {h_kwh:.2f} kWh/m²  ·  max sun height {noon_sun:.1f}°",
                transform=ax.transAxes, color=TEXT_2, fontsize=8.5)
        if col == 0:
            ax.set_ylabel("Horizontal irradiance (W/m²)")

        ax = axes[1, col]
        ax.fill_between(t, day["P_W"] / 1000, color=SERIES[0], alpha=0.18, linewidth=0)
        ax.plot(t, day["P_W"] / 1000, color=SERIES[0])
        ax.text(0.02, 0.88, f"{e_kwh:.1f} kWh produced", transform=ax.transAxes,
                color=TEXT, fontsize=9, fontweight="bold")
        ax.set_xlabel("Local clock time (h)")
        if col == 0:
            ax.set_ylabel(f"PV power, {cfg.PEAK_KWP:g} kWp flat (kW)")

    for ax in axes.flat:
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.set_ylim(bottom=0)
    axes[0, 1].legend(loc="upper right")
    fig.suptitle("A sunny summer day vs a sunny winter day in Gjilan (PVGIS-SARAH3, horizontal plane)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# --- figure 2 ------------------------------------------------------------------


def monthly_energy(flat: pd.DataFrame, tilted: pd.DataFrame, tilt: float) -> plt.Figure:
    """Mean daily PV energy per month, flat vs tilted, with year-to-year range."""
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    months = np.arange(1, 13)
    for i, (df, label) in enumerate([(flat, "Flat (0°)"), (tilted, f"Tilted {tilt:g}° south")]):
        daily = data.daily_energy(df)
        idx = pd.to_datetime(daily.index)
        per_year_month = daily.groupby([idx.year, idx.month]).mean().unstack()
        mean = per_year_month.mean()
        ax.fill_between(months, per_year_month.min(), per_year_month.max(),
                        color=SERIES[i], alpha=0.14, linewidth=0)
        ax.plot(months, mean, color=SERIES[i], marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax.annotate(f"{label}: {daily.sum() / len(set(idx.year)):,.0f} kWh/yr",
                    (12, mean.iloc[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=TEXT, fontsize=8.5)
    ax.set_xticks(months, ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_xlim(0.6, 12.4)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(f"Mean daily energy, {cfg.PEAK_KWP:g} kWp (kWh/day)")
    ax.set_title(f"Monthly production in Gjilan, {cfg.START_YEAR}–{cfg.END_YEAR} "
                 "(line = mean of all years, band = lowest-to-highest year)", pad=10)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


# --- figure 3 ------------------------------------------------------------------


def irradiation_heatmap(flat: pd.DataFrame) -> plt.Figure:
    """Daily horizontal irradiation, year × day of year: completeness and cloudiness at a glance."""
    apply_style()
    daily = data.daily_energy(flat, "G_poa")
    idx = pd.to_datetime(daily.index)
    table = pd.Series(daily.values, index=[idx.year, idx.dayofyear]).unstack()
    fig, ax = plt.subplots(figsize=(10, 4.4))
    im = ax.imshow(table.values, aspect="auto", cmap=SEQ_CMAP, interpolation="nearest",
                   extent=(0.5, table.shape[1] + 0.5, table.index.max() + 0.5, table.index.min() - 0.5))
    ax.grid(False)
    month_starts = pd.date_range("2023-01-01", periods=12, freq="MS").dayofyear
    ax.set_xticks(month_starts, ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], ha="left")
    ax.set_yticks(table.index[::2])
    ax.set_xlabel("Day of year")
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Daily irradiation (kWh/m²)", color=TEXT_2)
    cbar.outline.set_visible(False)
    ax.set_title("Every day of 19 years: daily solar irradiation on a horizontal surface in Gjilan "
                 "(dark = sunny, light = cloudy/winter)", pad=10)
    fig.tight_layout()
    return fig
