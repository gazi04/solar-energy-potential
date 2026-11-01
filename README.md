# Solar Energy Potential in Gjilan

**Applying Calculus and Machine Learning to Real Radiation Data**

Course project for *Mathematics for Machine Learning* (MSc Control Systems and AI, Public University "Kadri Zeka", Gjilan).

Main question: **how much electricity would solar panels produce on a house in Gjilan (42.46° N, 21.47° E), and how should they be installed to produce the most?**

The course math (trigonometry, derivatives, Newton's method, numerical integration, optimization, least squares, probability) is implemented by hand in NumPy on 19 years of real hourly data; SciPy and scikit-learn are used only to verify the results.

## Data

| Source | Content | Period |
|---|---|---|
| [PVGIS-SARAH3](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis_en) (EU JRC) | hourly beam/diffuse irradiance, sun height, modelled PV power (5 kWp, 14 % loss), flat and 35° tilt; official PVcalc estimate | 2005–2023 |
| [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api) ERA5 | hourly temperature, humidity, cloud cover, wind, radiation | 2005–2023 |

166,536 hourly rows per source; no missing rows, no NaNs, but 437 daylight hours where PVGIS stores missing satellite data as 0 (handled in cleaning; see [data/README.md](data/README.md#known-issues)). Raw data is about 25 MB in `data/raw/`. File roles, columns and units: [data/README.md](data/README.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # Python 3.13 venv + dependencies
uv run python -m solar_gjilan.download    # download data once (~1 min); later runs use the cache
```

Register the project venv as a Jupyter kernel (once per machine). Notebooks name this kernel in their metadata, so editors (JupyterLab, VS Code, Neovim/molten) pick it automatically:

```bash
uv run python -m ipykernel install --user --name solar_energy_potential_in_gjilan
```

`--force` re-downloads; `--only pvgis` or `--only openmeteo` downloads a single source.

## Usage

```bash
uv run python scripts/make_figures.py     # regenerate reports/figures/*.png
uv run jupyter lab                        # open notebooks/
uv run pytest                             # tests (offline)
```

Slides (Marp markdown):

```bash
npx @marp-team/marp-cli --no-stdin reports/slides/hw1_idea.md --html -o reports/slides/hw1_idea.html
npx @marp-team/marp-cli --no-stdin reports/slides/hw1_idea.md --pdf --allow-local-files   # needs Chromium/Chrome
```

## Layout

```
src/solar_gjilan/   config, download, data loaders, plots (math modules follow in later milestones)
scripts/            figure generation
notebooks/          narrative notebooks, one per milestone
tests/              pytest, small fixtures from real data
data/raw/           cached downloads, one folder per source (CSV + .meta.json)
reports/            figures, slides, report
docs/               project context, roadmap, milestone plans
```

## Status

See [docs/ROADMAP.md](docs/ROADMAP.md). M0 (setup, data, first plots, idea slides) is done.
