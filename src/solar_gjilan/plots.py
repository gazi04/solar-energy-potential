"""Figure functions. Each returns a matplotlib Figure; ``save`` writes it to ``reports/figures``.

Colors follow one fixed palette (categorical slots in fixed order, one-hue
sequential ramp), so every figure in the project reads as one system.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from solar_gjilan import config as cfg
from solar_gjilan import data, sun

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


def save(fig: Figure, name: str, fig_dir: Path | None = None) -> Path:
    fig_dir = fig_dir or cfg.FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


def _clock_hours(index: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(index.hour + index.minute / 60)


# --- figure 1 ------------------------------------------------------------------


def summer_vs_winter_day(hourly: pd.DataFrame, summer: pd.Timestamp | str, winter: pd.Timestamp | str) -> Figure:
    """Irradiance (beam + diffuse) and 5 kWp PV power for one summer and one winter day.

    Small multiples: columns = days, rows = measure; each row shares its y-scale so
    the two seasons are directly comparable.
    """
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey="row",
                             gridspec_kw={"height_ratios": [1.3, 1]})
    for col, date in enumerate([summer, winter]):
        day = data.one_day(hourly, date)
        t = _clock_hours(data.time_index(day))
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


def monthly_energy(flat: pd.DataFrame, tilted: pd.DataFrame, tilt: float) -> Figure:
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
        ax.annotate(f"{label}: {daily.sum() / (cfg.END_YEAR - cfg.START_YEAR + 1):,.0f} kWh/yr",
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


def irradiation_heatmap(flat: pd.DataFrame) -> Figure:
    """Daily horizontal irradiation, year × day of year: completeness and cloudiness at a glance."""
    apply_style()
    daily = data.daily_energy(flat, "G_poa")
    idx = pd.to_datetime(daily.index)
    table = pd.Series(daily.values, index=[idx.year, idx.day_of_year]).unstack()
    fig, ax = plt.subplots(figsize=(10, 4.4))
    im = ax.imshow(table.values, aspect="auto", cmap=SEQ_CMAP, interpolation="nearest",
                   extent=(0.5, table.shape[1] + 0.5, table.index.max() + 0.5, table.index.min() - 0.5))
    ax.grid(False)
    month_starts = pd.date_range("2023-01-01", periods=12, freq="MS").day_of_year
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


# --- figure 4 ------------------------------------------------------------------


def sun_elevation_days(hourly: pd.DataFrame, dates: Sequence[str]) -> Figure:
    """Sun elevation over one day for several dates: our pvlib curve vs PVGIS sun_height.

    Line: elevation on a 1-minute grid over the local day (sun.sun_position).
    Dots: PVGIS ``sun_height`` at its real stamp HH:10 UTC (night and gap hours dropped).
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for color, date in zip(SERIES, dates):
        local = pd.date_range(date, periods=24 * 60, freq="min", tz=cfg.TZ)
        elev = sun.sun_position(local.tz_convert("UTC"))["elevation"].to_numpy()
        t = np.asarray(_clock_hours(local))
        ax.plot(t, elev, color=color, label=f"{pd.Timestamp(date):%d %B}")

        day = hourly.loc[local[0].tz_convert("UTC"):local[-1].tz_convert("UTC")]
        day = day[day["sun_height_pvgis"] > 0]
        stamp = (data.time_index(day) + pd.Timedelta(minutes=10)).tz_convert(cfg.TZ)
        ax.scatter(_clock_hours(stamp), day["sun_height_pvgis"], color=color, s=22,
                   edgecolor=SURFACE, linewidth=1, zorder=3)

        k = elev.argmax()
        ax.annotate(f"max {elev[k]:.1f}° at {local[k]:%H:%M}", (t[k], elev[k]),
                    xytext=(0, 6), textcoords="offset points", ha="center",
                    color=TEXT, fontsize=8.5)

    ax.axhline(0, color=TEXT_2, linewidth=1)  # horizon
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_ylim(-5, 80)
    ax.set_xlabel("Local clock time (h)")
    ax.set_ylabel("Sun elevation α (°)")
    ax.set_title("Sun elevation in Gjilan (line = our pvlib curve, dots = PVGIS sun_height)", pad=10)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


# --- figure 7 ------------------------------------------------------------------


def day_length_year(year: int) -> Figure:
    """Sunrise/sunset local clock time and day length for every day of ``year``.

    Geometric sunrise (alpha = 0), solved with Newton (``sun.sunrise_sunset``) in solar
    time and converted to Europe/Belgrade clock time (``sun.to_clock``), so the
    CET/CEST jumps in March and October are visible.
    """
    apply_style()
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    rs = sun.sunrise_sunset(dates.day_of_year.to_numpy())
    rise = sun.to_clock(rs["sunrise"].to_numpy(), dates)
    sset = sun.to_clock(rs["sunset"].to_numpy(), dates)
    rise_h = np.asarray(_clock_hours(rise)) + rise.second / 3600
    set_h = np.asarray(_clock_hours(sset)) + sset.second / 3600
    length = rs["day_length"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [1.3, 1]})

    ax1.fill_between(dates, rise_h, set_h, color=SERIES[3], alpha=0.18, linewidth=0)
    ax1.plot(dates, rise_h, color=SERIES[1], label="Sunrise")
    ax1.plot(dates, set_h, color=SERIES[0], label="Sunset")
    dst = dates[rise.map(lambda ts: ts.dst() != pd.Timedelta(0))]
    for switch, text in [(dst[0], "CEST starts\n(+1 h)"), (dst[-1] + pd.Timedelta(days=1), "CET returns\n(−1 h)")]:
        ax1.axvline(switch, color=TEXT_2, linewidth=1, linestyle=":")
        ax1.text(switch, 12, f" {text}", color=TEXT_2, fontsize=8, va="center")
    ax1.set_ylim(3, 22)
    ax1.set_yticks(range(3, 23, 3))
    ax1.set_ylabel("Local clock time (h)")
    ax1.set_title(f"Sunrise and sunset in Gjilan, {year} (geometric, α = 0°, Newton's method)", pad=10)
    ax1.legend(loc="upper left", ncols=2)

    ax2.fill_between(dates, length, color=SERIES[0], alpha=0.14, linewidth=0)
    ax2.plot(dates, length, color=SERIES[0])
    ax2.axhline(12, color=TEXT_2, linewidth=1, linestyle=":")
    for k, va, ha, off in [(length.argmax(), "bottom", "center", 6), (length.argmin(), "top", "right", -6)]:
        hh, mm = divmod(round(length[k] * 60), 60)
        ax2.annotate(f"{length[k]:.2f} h ({hh} h {mm:02d} min), {dates[k]:%d %b}", (dates[k], length[k]),
                     xytext=(0, off), textcoords="offset points", ha=ha, va=va,
                     color=TEXT, fontsize=8.5)
        ax2.plot(dates[k], length[k], "o", color=SERIES[0], markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax2.set_ylim(6, 18)
    ax2.set_ylabel("Day length (h)")
    ax2.set_title("Day length = sunset − sunrise (dotted: 12 h at the equinoxes)", pad=8)
    ax2.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    ax2.set_xlim(dates[0], dates[-1])
    fig.tight_layout()
    return fig


# --- figure 8 ------------------------------------------------------------------


def derivatives_overview(hourly: pd.DataFrame, daily: pd.DataFrame, year: int, month: int = 6,
                         n: int = 172) -> Figure:
    """Derivatives (M1 step 5), six panels.

    Row 1: sun elevation α(t) on day ``n`` and its derivative dα/dt (analytic line,
           ``np.gradient`` dots); dα/dt = 0 at solar noon = maximum of α.
    Row 2: error of forward vs central differences vs step Δt (log-log, slopes 1 and 2);
           share of daylight hours with an hour-to-hour ramp |ΔP| > 1 kW/h per season.
    Row 3: PV power P(t) and dP/dt (``np.gradient``) on the clearest and the most
           variable day of ``month``/``year``.
    """
    from solar_gjilan import derivatives

    apply_style()
    fig, axes = plt.subplots(3, 2, figsize=(11, 10.5))

    # Row 1: alpha and dalpha/dt
    curve = derivatives.elevation_curve(n)
    t = curve.index.to_numpy()
    noon = 12.0
    ax = axes[0, 0]
    ax.plot(t, curve["elevation"], color=SERIES[0])
    ax.axhline(0, color=TEXT_2, linewidth=1)
    ax.axvline(noon, color=TEXT_2, linewidth=1, linestyle=":")
    peak = curve["elevation"].max()
    ax.annotate(f"max α = {peak:.2f}° at solar noon", (noon, peak), xytext=(6, -14),
                textcoords="offset points", color=TEXT, fontsize=8.5)
    ax.set_ylim(-75, 80)
    ax.set_ylabel("Sun elevation α (°)")
    ax.set_title(f"α(t), day {n} (21 June)")

    ax = axes[0, 1]
    ax.plot(t, curve["rate_analytic"], color=SERIES[0], label="Analytic (chain rule)")
    sample = curve.iloc[::60]
    ax.scatter(sample.index, sample["rate_gradient"], color=SERIES[1], s=22, zorder=3,
               edgecolor=SURFACE, linewidth=1, label="np.gradient (1-min grid)")
    ax.axhline(0, color=TEXT_2, linewidth=1)
    ax.axvline(noon, color=TEXT_2, linewidth=1, linestyle=":")
    ax.annotate("dα/dt = 0 at noon\n→ maximum of α", (noon, 0), xytext=(8, 14),
                textcoords="offset points", color=TEXT, fontsize=8.5)
    ax.set_ylabel("dα/dt (°/h)")
    ax.set_title("dα/dt: rising fastest in the morning, 0 at noon, negative after")
    ax.legend(loc="lower left")
    for a in axes[0]:
        a.set_xlim(0, 24)
        a.set_xticks(range(0, 25, 3))
        a.set_xlabel("Solar time (h)")

    # Row 2 left: error study
    err = derivatives.step_error_study(n)
    order = derivatives.convergence_order(err)
    ax = axes[1, 0]
    for i, col in enumerate(["forward", "central"]):
        ax.loglog(err.index, err[col], color=SERIES[i], marker="o", markersize=5,
                  markeredgecolor=SURFACE, markeredgewidth=1.5,
                  label=f"{col.capitalize()} difference, slope {order[col]:.2f}")
    ax.set_xticks(err.index, [f"{m:g}" for m in err.index])
    ax.minorticks_off()
    ax.set_xlabel("Step Δt (min)")
    ax.set_ylabel("Max error vs analytic (°/h)")
    ax.set_title("Error shrinks as O(Δt) forward, O(Δt²) central")
    ax.legend(loc="lower right")

    # Row 2 right: ramp share per season
    rate = derivatives.power_rate(hourly)
    stats = derivatives.ramp_stats(rate)
    seasons = stats["per_season"]
    ax = axes[1, 1]
    bars = ax.bar(seasons.index, seasons["share_above"] * 100, color=SERIES[0], width=0.6)
    ax.bar_label(bars, fmt="%.1f %%", padding=3, color=TEXT, fontsize=8.5)
    ax.set_ylabel("Daylight hours with |ΔP| > 1 kW/h (%)")
    ax.set_ylim(0, seasons["share_above"].max() * 100 * 1.25)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Fast power swings, {cfg.START_YEAR}–{cfg.END_YEAR} "
                 f"(all: {stats['share_above_step'] * 100:.1f} %, max {stats['max_abs_step']:.1f} kW/h)")

    # Row 3: P and dP/dt on a clear vs a variable day
    clear, variable = derivatives.pick_days(daily, rate, year, month)
    for i, (date, label) in enumerate([(clear, "clearest"), (variable, "most variable")]):
        day = derivatives.one_day(rate, date)
        th = np.asarray(_clock_hours(pd.DatetimeIndex(day.index)))
        e = daily.loc[date, "e_kwh"]
        axes[2, 0].plot(th, day["p_kw"], color=SERIES[i], label=f"{date:%d %b %Y} ({label}, {e:.1f} kWh)")
        axes[2, 1].plot(th, day["dpdt"], color=SERIES[i], marker="o", markersize=3.5)
    axes[2, 0].set_ylabel(f"PV power P, {cfg.PEAK_KWP:g} kWp flat (kW)")
    axes[2, 0].set_ylim(0, rate["p_kw"].max() * 1.35)
    axes[2, 0].set_title("P(t): smooth on a clear day, jumps when clouds pass")
    axes[2, 0].legend(loc="upper left")
    axes[2, 1].axhline(0, color=TEXT_2, linewidth=1)
    axes[2, 1].set_ylabel("dP/dt (kW/h), np.gradient")
    axes[2, 1].set_title("dP/dt: one smooth wave vs repeated sign changes")
    for a in axes[2]:
        a.set_xlim(3, 22)
        a.set_xticks(range(3, 23, 3))
        a.set_xlabel("Local clock time (h)")

    fig.suptitle("Derivatives: rate of change of sun height and PV power in Gjilan",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# --- figure 5 ------------------------------------------------------------------


def declination_eot(year: int = 2023) -> Figure:
    """Declination δ(n) and equation of time EoT(n): course formulas vs Spencer 1971.

    Course formulas: Cooper 1969 for δ, Milne 1921 (pvlib ``equation_of_time_pvcdrom``)
    for EoT. Reference: Spencer 1971 Fourier series (used by ``sun.sun_position``).
    Two panels, no dual axis.
    """
    from pvlib import solarposition

    apply_style()
    n = np.arange(1, 366)
    dates = pd.Timestamp(f"{year}-01-01") + pd.to_timedelta(n - 1, unit="D")
    cooper = np.rad2deg(solarposition.declination_cooper69(n))
    spencer_d = np.rad2deg(solarposition.declination_spencer71(n))
    milne = solarposition.equation_of_time_pvcdrom(n)
    spencer_e = solarposition.equation_of_time_spencer71(n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    ax1.plot(dates, spencer_d, color=SERIES[0], label="Spencer 1971 (reference)")
    ax1.plot(dates, cooper, color=SERIES[1], linestyle="--", label="Cooper 1969 (course formula)")
    ax1.axhline(0, color=TEXT_2, linewidth=1)
    ax1.set_ylabel("Declination δ (°)")
    err_d = np.abs(cooper - spencer_d)
    ax1.set_title(f"Declination: Cooper vs Spencer, mean |Δ| {err_d.mean():.2f}°, max {err_d.max():.2f}°")
    ax1.legend(loc="lower center")

    ax2.plot(dates, spencer_e, color=SERIES[0], label="Spencer 1971 (reference)")
    ax2.plot(dates, milne, color=SERIES[1], linestyle="--", label="Milne 1921 (course formula)")
    ax2.axhline(0, color=TEXT_2, linewidth=1)
    for k, va, off in [(spencer_e.argmax(), "bottom", 6), (spencer_e.argmin(), "top", -6)]:
        ax2.annotate(f"{spencer_e[k]:+.1f} min, {dates[k]:%d %b}", (dates[k], spencer_e[k]),
                     xytext=(0, off), textcoords="offset points", ha="center", va=va,
                     color=TEXT, fontsize=8.5)
    ax2.set_ylim(-19, 21)
    ax2.set_ylabel("Equation of time (min)")
    err_e = np.abs(milne - spencer_e)
    ax2.set_title(f"Equation of time: Milne vs Spencer, mean |Δ| {err_e.mean():.2f} min")
    ax2.legend(loc="upper left")
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=range(1, 13, 2)))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
        ax.set_xlim(dates[0], dates[-1])
    fig.tight_layout()
    return fig


# --- figure 6 ------------------------------------------------------------------


def newton_convergence(days: Sequence[tuple[int, str]] = ((172, "21 June"), (355, "21 December"))) -> Figure:
    """Newton's method for sunrise: error per iteration (log scale) vs bisection.

    Left: |t_k − t*| for Newton from t₀ = 6 h on each day (quadratic: the number of
    correct digits roughly doubles each step). Right: Newton vs bisection on [0, 12] h
    for the first day (bisection halves the interval each step: linear convergence).
    """
    from pvlib import solarposition
    from scipy.optimize import bisect

    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for color, style, (n, label) in zip(SERIES, ["-", "--"], days):
        root, table = sun.newton_table(n, t0=6.0)
        tab = table[table["error"] > 0]
        ax1.semilogy(tab["k"], tab["error"], color=color, linestyle=style, marker="o", markersize=6,
                     markeredgecolor=SURFACE, markeredgewidth=1.5,
                     label=f"{label}: sunrise {root:.4f} h solar time")
    ax1.text(0.1, 1e-7, "both days lie on top of each other", color=TEXT_2, fontsize=8)
    ax1.axhline(1e-9, color=TEXT_2, linewidth=1, linestyle=":")
    ax1.text(2.05, 1.6e-9, "1e-9 h ≈ 4 µs", color=TEXT_2, fontsize=8)
    ax1.set_xticks(range(0, 4))
    ax1.set_xlabel("Iteration k")
    ax1.set_ylabel("Error |t_k − t*| (h)")
    ax1.set_title("Newton from t₀ = 6 h: < 1e-9 h after 3 steps")
    ax1.legend(loc="upper right")

    n0, label0 = days[0]
    delta = solarposition.declination_spencer71(n0)
    root, newton_tab = sun.newton_table(n0, t0=6.0)
    mids: list[float] = []

    def f_log(t: float) -> float:
        mids.append(t)
        return float(sun.horizon_f(t, delta))

    bisect(f_log, 0.0, 12.0, xtol=1e-12)
    bis_err = np.abs(np.array(mids[2:]) - root)  # first two calls are the end points
    bis_err = bis_err[bis_err > 0]
    ax2.semilogy(np.arange(1, len(bis_err) + 1), bis_err, color=SERIES[2], marker="o",
                 markersize=3.5, label=f"Bisection on [0, 12] h ({len(bis_err)} steps)")
    newt = newton_tab[newton_tab["error"] > 0]
    ax2.semilogy(newt["k"], newt["error"], color=SERIES[0], marker="o", markersize=6,
                 markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"Newton from t₀ = 6 h ({len(newt) - 1} steps)")
    ax2.set_xlabel("Iteration k")
    ax2.set_ylabel("Error |t_k − t*| (h)")
    ax2.set_title(f"Newton (quadratic) vs bisection (linear), {label0}")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    return fig


# --- figure 9 ------------------------------------------------------------------


def data_quality(hourly: pd.DataFrame, daily: pd.DataFrame, gap_day: str = "2008-05-18") -> Figure:
    """PVGIS zero-filled gaps: how many, what one looks like, what cleaning does.

    A: gap hours per year, split into filled (≤ 2 h runs) and still NaN.
    B: a full-gap day: PVGIS reports 0 while ERA5 shows daylight and the sun is up.
    C: a short-gap day (clearest complete day with filled hours): raw zeros vs filled values.
    """
    apply_style()
    fig = plt.figure(figsize=(11, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1], sharey=ax_b)

    gap = hourly["pvgis_gap"]
    year = data.time_index(hourly).year
    filled = (gap & hourly["filled"]).groupby(year).sum()
    nan = (gap & ~hourly["filled"]).groupby(year).sum()
    years = filled.index.to_numpy()
    ax_a.bar(years, filled, color=SERIES[0], width=0.7, label=f"Filled (≤ 2 h runs): {filled.sum()} h")
    ax_a.bar(years, nan, bottom=filled, color=SERIES[1], width=0.7, label=f"Left as NaN (longer runs): {nan.sum()} h")
    ax_a.set_xticks(years, [str(y) for y in years])
    ax_a.grid(axis="x", visible=False)
    ax_a.set_ylabel("Gap hours")
    ax_a.set_title(f"PVGIS daylight hours stored as 0 (sun > 5° but sun_height = 0): {int(gap.sum())} h in "
                   f"{cfg.START_YEAR}–{cfg.END_YEAR}")
    ax_a.legend(loc="upper right")

    def raw_ghi(day: pd.DataFrame) -> pd.Series:
        """PVGIS GHI as delivered: gap hours were 0."""
        return day["ghi"].where(~day["pvgis_gap"], 0.0)

    def draw(ax: Axes, date: str, title: str) -> None:
        day = data.one_day(hourly, date)
        t = np.asarray(_clock_hours(pd.DatetimeIndex(day.index))) + 0.5  # centre of the hour
        up = day["elevation"] > 5
        ax.fill_between(t, 0, 1, where=up.to_numpy(), transform=ax.get_xaxis_transform(),
                        color=SERIES[3], alpha=0.12, linewidth=0, step="mid", label="Sun > 5°")
        ax.plot(t, day["ghi_era5"], color=SERIES[2], label="ERA5 GHI (Open-Meteo)")
        ax.plot(t, raw_ghi(day), color=SERIES[1], marker="o", markersize=3.5, label="PVGIS GHI, raw")
        g = day["pvgis_gap"].to_numpy()
        if day.loc[day["pvgis_gap"], "ghi"].notna().any():
            ax.plot(t, day["ghi"], color=SERIES[0], label="PVGIS GHI, cleaned")
            ax.scatter(t[g], day["ghi"].to_numpy()[g], color=SERIES[0], s=40, zorder=4,
                       edgecolor=SURFACE, linewidth=1.2, label="Filled hours")
        ax.set_xlim(3, 22)
        ax.set_xticks(range(3, 23, 3))
        ax.set_xlabel("Local clock time (h)")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)

    draw(ax_b, gap_day, f"Full-gap day {pd.Timestamp(gap_day):%d %b %Y}: stays NaN")
    short = data.argmax_date(daily[(daily["n_filled_hours"] > 0) & daily["complete"]]["kt_daily"])
    n_fill = int(daily.loc[[short], "n_filled_hours"].to_numpy()[0])
    draw(ax_c, f"{short:%Y-%m-%d}", f"Short gap {short:%d %b %Y}: {n_fill} h filled with k·g0")
    ax_b.set_ylabel("Horizontal irradiance (W/m²)")
    ax_b.set_ylim(0, 1150)
    plt.setp(ax_c.get_yticklabels(), visible=False)
    fig.tight_layout()
    return fig
