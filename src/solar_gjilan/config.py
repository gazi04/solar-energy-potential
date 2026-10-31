"""Project-wide constants: location, period, system defaults, paths, API endpoints."""

from pathlib import Path

# --- Location: Gjilan, Kosovo ---------------------------------------------
LAT = 42.46  # degrees north
LON = 21.47  # degrees east
TZ = "Europe/Belgrade"  # CET (UTC+1) / CEST (UTC+2)

# --- Study period (PVGIS-SARAH3 coverage) ----------------------------------
START_YEAR = 2005
END_YEAR = 2023

# --- Default residential PV system ----------------------------------------
PEAK_KWP = 5.0  # installed peak power, kWp
LOSS_PCT = 14.0  # total system losses, % (PVGIS default)
REF_TILT = 35  # reference tilt for validation series, degrees
AZIMUTH = 0  # PVGIS convention: 0 = south, -90 = east, 90 = west

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIG_DIR = ROOT / "reports" / "figures"

# --- APIs ------------------------------------------------------------------
PVGIS_BASE = "https://re.jrc.ec.europa.eu/api/v5_3"
PVGIS_DB = "PVGIS-SARAH3"
OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
]
