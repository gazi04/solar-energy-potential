# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Master's course project (*Mathematics for Machine Learning*, Kadri Zeka University): estimate rooftop PV production in Gjilan, Kosovo (42.46° N, 21.47° E, Europe/Belgrade) and find the best panel orientation, using real PVGIS and Open-Meteo data. The professor grades **course math applied in Python to real data**, so the math must be implemented explicitly.

- `docs/SOLAR_PROJECT_CONTEXT.md`: full background, formulas, data sources, expected sanity values. Read before major work.
- `docs/ROADMAP.md`: milestone checklist (M0–M6 ↔ HW1–HW4, final). Tick boxes and append to its decision log when tasks or decisions land.

Status: M0 done (setup, cached data, first figures, notebook, HW1 slides). Next: M1.

## Commands

Environment is managed with `uv` (Python 3.13 pinned; 3.14 avoided for package compatibility).

```bash
uv sync                                          # install deps
uv run python -m solar_gjilan.download           # fetch + cache raw data (skips existing; --force, --only pvgis|openmeteo)
uv run python scripts/make_figures.py            # regenerate reports/figures/*.png
uv run pytest                                    # all tests (offline, use tests/fixtures/)
uv run pytest tests/test_data.py::test_load_pvcalc   # single test
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/00_first_look.ipynb   # re-run a notebook
npx @marp-team/marp-cli reports/slides/hw1_idea.md --html -o reports/slides/hw1_idea.html  # slides
```

## Architecture

Package `src/solar_gjilan/` (src layout, installed editable by `uv sync`); notebooks in `notebooks/` (one per milestone) only call into it. Figures go through `plots.py` (fixed palette, `apply_style`, `save`) so script, notebook and slides share identical PNGs.

Existing: `config` (all constants/paths/endpoints) → `download` (cache API responses to `data/raw/` as JSON + flat CSV) → `data` (loaders return UTC-indexed DataFrames; `daily_energy`, `sunniest_day`, `quick_summary`) → `plots`.

Planned, one module per stage: `sun` (declination, equation of time, hour angle, elevation, sun vector, Newton sunrise) → `power` (dot-product incidence, isotropic transposition to tilted plane, P = P_peak·POA/1000·(1−losses)) → `energy` (hand-written trapezoid/Simpson, daily/yearly kWh) → `optimize` (tilt, then tilt+azimuth) → `model` (normal-equation regression vs sklearn, probability of bad days).

Defaults: 5 kWp system, 14 % losses, south-facing, albedo ρ for ground reflection.

Cached files: `data/raw/pvgis/input/hourly_horizontal_2005_2023.csv` (input), `data/raw/pvgis/reference/{hourly_tilt35_2005_2023.csv,pvcalc_optimal.json}` (reference answers, only for validation), `data/raw/openmeteo/era5_hourly_2005_2023.csv`; each CSV has a `.meta.json` (request + API units). Roles (input vs reference) and column units: `data/README.md`. 166,536 hourly rows each, no gaps. Paths come only from `download.pvgis_hourly_path/pvcalc_path/openmeteo_path`.

## Rules specific to this project

- **Implement math by hand in NumPy; use SciPy/sklearn/pvlib only to verify** (or pvlib only for downloading). Tests should compare own implementation against the library.
- **Never call APIs in loops.** Download once, cache in `data/raw/`; everything after must run offline.
- Vectorize with NumPy broadcasting (tilt sweep = angles × timesteps in one array op).
- Each notebook section labels the syllabus week it maps to and shows formula → code → result → interpretation; explanations double as study material for the written calculus test.
- English for code, notebook, report, slides.
- Do not run git commands; the user manages the repo.
- Notebooks must set `metadata.kernelspec.name = "solar_energy_potential_in_gjilan"` (user kernel pointing to this `.venv`). The user runs notebooks in Neovim (jupytext + molten), which starts the kernel named in the metadata; `python3` there is the Neovim venv without project deps.

## Data gotchas

- PVGIS (SARAH3, 2005–2023) timestamps are **UTC** and stamped at `HH:10`. Open-Meteo ERA5 (`timezone=GMT`) is on the full hour, radiation = mean of the preceding hour, and snaps to grid point 42.5 N / 21.5 E. Align on a UTC index before joining.
- PVGIS columns are renamed in `data.PVGIS_COLUMNS`; `sun_height` (H_sun) is a free reference for our own sun-elevation code; `reconstructed` (Int) is all 0.
- UTC→local shifts 31 Dec 23:10 into the next year; `daily_energy` drops local days with < 23 hours. Divide yearly sums by 19, not by the number of distinct years.
- PVGIS `seriescalc` with `components=1` returns irradiance **on the requested plane**. Download horizontal (`angle=0`) and do the transposition ourselves; download a 35° series and `PVcalc` only for validation.
- Formulas use **solar time**; converting to clock time needs longitude correction (−25.9 min vs 15° E meridian) and the equation of time. Solar noon ≈ 11:34 CET.
- `sin α = 0` gives geometric sunrise; published times use −0.833°. State which is used.
- Regression features must **exclude radiation** (target leakage). Split by time: train 2005–2022, test 2023.
- Bad day = daily kWh < 30 % of clear-sky kWh for that date.
- PVGIS output is satellite-modelled, not metered; mention in limitations.

## Sanity checks (use as test expectations)

- Noon elevation ≈ 71.0° (21 Jun), ≈ 24.1° (21 Dec)
- Sunrise 21 Jun ≈ 4.44 h solar time; Newton from t₀ = 6 converges in 3–4 iterations
- Clear June day, 5 kWp at 35° ≈ 33 kWh
- Optimal tilt low-to-mid 30s°; ~+19 % vs flat (clear-sky)
