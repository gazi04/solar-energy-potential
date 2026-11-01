# Data

All files in `raw/` are downloaded by `uv run python -m solar_gjilan.download` and never edited by hand.
Cleaned and combined tables produced by the project go to `processed/`.

Location: Gjilan, 42.46° N, 21.47° E · Period: 2005–2023 · All timestamps UTC.

## Files and their role

| File | Source | Role | Content |
|---|---|---|---|
| `raw/pvgis/input/hourly_horizontal_2005_2023.csv` | PVGIS `seriescalc` | **input** | hourly irradiance on a flat surface, sun height, PV power of a flat 5 kWp system |
| `raw/pvgis/reference/hourly_tilt35_2005_2023.csv` | PVGIS `seriescalc` | **reference** | same, on a 35° south-facing panel; checks our own tilted-panel calculation hour by hour |
| `raw/pvgis/reference/pvcalc_optimal.json` | PVGIS `PVcalc` | **reference** | official long-term estimate: optimal tilt (35°) and azimuth, yearly kWh (6,554) |
| `raw/openmeteo/era5_hourly_2005_2023.csv` | Open-Meteo archive (ERA5) | **input** | hourly weather: temperature, humidity, cloud cover, wind, radiation |

- **Input**: the project's math runs on it.
- **Reference**: PVGIS's own answer, used only to check our results.

PVGIS files are split into `input/` and `reference/` folders by role. All Open-Meteo data is input, so it has no subfolders.

Each hourly CSV has a `.meta.json` next to it: request URL and parameters, download time, and the API's description of the data (settings used, column units). The hourly rows exist only once, in the CSV.

Each CSV has 166,536 rows (19 years × 8,760 h + 4 leap days × 24 h), with no missing rows and no empty values. **But PVGIS stores some missing data as zeros; see Known issues.**

## Columns

### PVGIS hourly (`raw/pvgis/{input,reference}/hourly_*.csv`)

| Column | Renamed in code to | Unit | Meaning |
|---|---|---|---|
| `time` | index | UTC, `YYYYMMDD:HHMM` | values are stamped at minute 10 of each hour |
| `P` | `P_W` | W | PV power, 5 kWp system, 14 % losses (PVGIS model) |
| `Gb(i)` | `Gb` | W/m² | beam (direct) irradiance on the plane |
| `Gd(i)` | `Gd` | W/m² | diffuse irradiance on the plane |
| `Gr(i)` | `Gr` | W/m² | ground-reflected irradiance on the plane (0 for a flat surface) |
| `H_sun` | `sun_height` | ° | sun elevation above the horizon |
| `T2m` | `T2m` | °C | air temperature at 2 m |
| `WS10m` | `WS10m` | m/s | wind speed at 10 m |
| `Int` | `reconstructed` | 0/1 | 1 = radiation value reconstructed by PVGIS (all 0 here) |

"Plane" means the panel surface: horizontal in the `horizontal` file, 35° south in the `tilt35` file.

### Open-Meteo hourly (`raw/openmeteo/era5_hourly_2005_2023.csv`)

| Column | Unit | Meaning |
|---|---|---|
| `time` | UTC, ISO 8601 | full hour; radiation values are the mean of the **preceding** hour |
| `temperature_2m` | °C | air temperature at 2 m |
| `relative_humidity_2m` | % | relative humidity at 2 m |
| `cloud_cover` | % | total cloud cover |
| `wind_speed_10m` | **km/h** | wind speed at 10 m (PVGIS uses m/s) |
| `shortwave_radiation` | W/m² | global horizontal irradiance |
| `direct_radiation` | W/m² | direct radiation on a horizontal surface |
| `diffuse_radiation` | W/m² | diffuse radiation |

ERA5 is a 0.25° grid; the nearest grid point is 42.5° N, 21.5° E (513 m).

## Known issues

**PVGIS zero-filled gaps.** In 437 daylight hours (0.57 % of hours with the sun above 5°), PVGIS reports `Gb = Gd = Gr = P = 0` **and** `H_sun = 0` although the sun is up. These are missing satellite data, not clouds:

- the sun height is 0 too, which is geometry and cannot be caused by weather;
- Open-Meteo reports a mean of 335 W/m² in the same hours;
- even heavy overcast gives some diffuse light, never exactly 0 at midday;
- `Int` (reconstructed) is 0, so PVGIS does not flag them.

They occur as 14 full days (2005-04-18, 2008-01-23, 2008-03-01, 2008-03-06, 2008-05-17/18/19, 2009-04-18/19, 2009-08-15/16, 2014-12-02, 2016-10-16/17) plus scattered single hours (226 days have at least one). 2008 (85 h) and 2009 (111 h) are worst. The same hours are zero in the `tilt35` file. Impact: yearly totals about 0.5 % too low; the 14 days would look like 0 kWh days.

Detection: our computed sun elevation > ~1° while `H_sun == 0`. Handling is part of the cleaning step (milestone M1).

## Plausibility checks passed

- No negative values, no radiation at night, GHI never above extraterrestrial irradiance (max ratio 0.94).
- Sun height matches the textbook formula within 0.35° on average.
- No stuck (repeated) values; weather within plausible ranges (−21 to 41 °C).
- Yearly GHI 1,466 kWh/m² (±3.5 % between years).
- PVGIS vs Open-Meteo daily GHI: r = 0.94, Open-Meteo +1.5 % per year but +10–15 % in winter (PVGIS preferred for radiation).
- Daily clearness vs cloud cover: r = −0.78.

## Caveats

- PVGIS values come from satellite images and a PV model, not from a metered panel.
- ERA5 is a reanalysis (a weather model constrained by observations), not station measurements.
- PVGIS (`HH:10`) and Open-Meteo (`HH:00`, mean of the preceding hour) use different time stamps; they are aligned before joining (milestone M1).
