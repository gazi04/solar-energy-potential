"""Download raw data once and cache it in ``data/raw/``.

Layout (one folder per source, PVGIS split by role; see ``data/README.md``)::

    data/raw/pvgis/input/hourly_horizontal_2005_2023.csv       hourly irradiance, flat plane
    data/raw/pvgis/reference/hourly_tilt35_2005_2023.csv       same on a 35° plane (checks our math)
    data/raw/pvgis/reference/pvcalc_optimal.json               official yearly kWh + optimal tilt
    data/raw/openmeteo/era5_hourly_2005_2023.csv               hourly weather (input)

* **input**: data the project's math runs on.
* **reference**: PVGIS's own answers, used only to check our results.

Every hourly CSV has a ``.meta.json`` next to it with the provenance: the
request URL and parameters, the download time, and the API's own ``inputs`` /
``meta`` description (settings used, column units). The hourly rows are stored
only once, in the CSV.

Files that already exist are skipped, so after the first run the project works
offline. Use ``--force`` to re-download.

Usage::

    uv run python -m solar_gjilan.download            # everything
    uv run python -m solar_gjilan.download --only pvgis
    uv run python -m solar_gjilan.download --force
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from solar_gjilan import config as cfg

RETRIES = 3
TIMEOUT_S = 300

PVGIS_DIR = cfg.RAW_DIR / "pvgis"
PVGIS_INPUT_DIR = PVGIS_DIR / "input"
PVGIS_REFERENCE_DIR = PVGIS_DIR / "reference"
OPENMETEO_DIR = cfg.RAW_DIR / "openmeteo"
PERIOD = f"{cfg.START_YEAR}_{cfg.END_YEAR}"


# --- paths (used by data.py) ---------------------------------------------------


def pvgis_hourly_path(angle: float) -> Path:
    """Horizontal series is our input; any tilted series is a PVGIS reference answer."""
    if angle == 0:
        return PVGIS_INPUT_DIR / f"hourly_horizontal_{PERIOD}.csv"
    return PVGIS_REFERENCE_DIR / f"hourly_tilt{angle:g}_{PERIOD}.csv"


def pvcalc_path(name: str = "optimal") -> Path:
    return PVGIS_REFERENCE_DIR / f"pvcalc_{name}.json"


def openmeteo_path() -> Path:
    return OPENMETEO_DIR / f"era5_hourly_{PERIOD}.csv"


def meta_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".meta.json")


# --- generic helpers --------------------------------------------------------


def _fetch_json(url: str, params: dict) -> dict:
    """GET ``url`` and return parsed JSON, retrying on network errors, 429 and 5xx."""
    for attempt in range(1, RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT_S)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            return resp.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            if attempt == RETRIES or (
                isinstance(exc, requests.HTTPError)
                and exc.response is not None
                and 400 <= exc.response.status_code < 500
                and exc.response.status_code != 429
            ):
                raise
            wait = 5 * attempt
            print(f"  attempt {attempt} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _write_atomic(path: Path, text: str) -> None:
    """Write via a temp file + rename so an interrupted run never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _write_meta(path: Path, url: str, params: dict | list[dict], api_info: dict) -> None:
    meta = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "url": url,
        "params": params,
        **api_info,
    }
    _write_atomic(path, json.dumps(meta, indent=1, ensure_ascii=False))


def _cached(path: Path, produce: Callable[[], None], force: bool) -> None:
    if path.exists() and not force:
        print(f"cached, skipped: {path.relative_to(cfg.ROOT)}")
        return
    print(f"downloading: {path.relative_to(cfg.ROOT)}")
    produce()
    size_mb = path.stat().st_size / 1e6
    print(f"  saved {path.relative_to(cfg.ROOT)} ({size_mb:.1f} MB)")


# --- PVGIS ------------------------------------------------------------------


def _pvgis_hourly_params(angle: float, start: int, end: int) -> dict:
    return {
        "lat": cfg.LAT,
        "lon": cfg.LON,
        "raddatabase": cfg.PVGIS_DB,
        "startyear": start,
        "endyear": end,
        "angle": angle,
        "aspect": cfg.AZIMUTH,
        "components": 1,
        "pvcalculation": 1,
        "peakpower": cfg.PEAK_KWP,
        "loss": cfg.LOSS_PCT,
        "usehorizon": 1,
        "outputformat": "json",
    }


def fetch_pvgis_hourly(angle: float) -> tuple[dict, dict | list[dict]]:
    """Hourly series for the whole period; falls back to 5-year chunks if one call fails.

    Returns the (merged) payload and the request parameters used.
    """
    url = f"{cfg.PVGIS_BASE}/seriescalc"
    params = _pvgis_hourly_params(angle, cfg.START_YEAR, cfg.END_YEAR)
    try:
        return _fetch_json(url, params), params
    except requests.RequestException as exc:
        print(f"  full-range request failed ({exc}); falling back to 5-year chunks")
    merged: dict | None = None
    chunk_params = []
    for start in range(cfg.START_YEAR, cfg.END_YEAR + 1, 5):
        end = min(start + 4, cfg.END_YEAR)
        chunk_params.append(_pvgis_hourly_params(angle, start, end))
        part = _fetch_json(url, chunk_params[-1])
        if merged is None:
            merged = part
        else:
            merged["outputs"]["hourly"].extend(part["outputs"]["hourly"])
        time.sleep(1)
    assert merged is not None
    merged["inputs"]["meteo_data"]["year_min"] = cfg.START_YEAR
    merged["inputs"]["meteo_data"]["year_max"] = cfg.END_YEAR
    return merged, chunk_params


def download_pvgis_hourly(angle: float, force: bool) -> None:
    csv_path = pvgis_hourly_path(angle)

    def produce() -> None:
        payload, params = fetch_pvgis_hourly(angle)
        rows = pd.DataFrame(payload["outputs"]["hourly"])
        _write_atomic(csv_path, rows.to_csv(index=False))
        _write_meta(meta_path(csv_path), f"{cfg.PVGIS_BASE}/seriescalc", params,
                    {"inputs": payload["inputs"], "meta": payload["meta"]})

    _cached(csv_path, produce, force)


def download_pvcalc(force: bool) -> None:
    """Official PVGIS long-term estimate with PVGIS's own optimal tilt and azimuth.

    Small summary (no hourly rows), so the full response is kept as one JSON.
    """
    path = pvcalc_path()
    params = {
        "lat": cfg.LAT,
        "lon": cfg.LON,
        "raddatabase": cfg.PVGIS_DB,
        "peakpower": cfg.PEAK_KWP,
        "loss": cfg.LOSS_PCT,
        "optimalangles": 1,
        "usehorizon": 1,
        "outputformat": "json",
    }

    def produce() -> None:
        payload = _fetch_json(f"{cfg.PVGIS_BASE}/PVcalc", params)
        payload = {
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "url": f"{cfg.PVGIS_BASE}/PVcalc",
            "params": params,
            **payload,
        }
        _write_atomic(path, json.dumps(payload, indent=1, ensure_ascii=False))

    _cached(path, produce, force)


# --- Open-Meteo -------------------------------------------------------------


def _openmeteo_params(year: int) -> dict:
    return {
        "latitude": cfg.LAT,
        "longitude": cfg.LON,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": ",".join(cfg.OPENMETEO_HOURLY_VARS),
        "models": "era5",
        "timezone": "GMT",
    }


def download_openmeteo(force: bool) -> None:
    """One request per year (keeps each call light), combined into one CSV."""
    csv_path = openmeteo_path()

    def produce() -> None:
        frames, params, first = [], [], None
        for year in range(cfg.START_YEAR, cfg.END_YEAR + 1):
            params.append(_openmeteo_params(year))
            payload = _fetch_json(cfg.OPENMETEO_ARCHIVE, params[-1])
            first = first or payload
            frames.append(pd.DataFrame(payload["hourly"]))
            print(f"  {year}: {len(frames[-1])} rows")
            time.sleep(1)
        _write_atomic(csv_path, pd.concat(frames, ignore_index=True).to_csv(index=False))
        api_info = {k: v for k, v in first.items() if k not in ("hourly", "generationtime_ms")}
        _write_meta(meta_path(csv_path), cfg.OPENMETEO_ARCHIVE, params, {"response": api_info})

    _cached(csv_path, produce, force)


# --- entry point --------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument("--only", choices=["pvgis", "openmeteo"], help="download one source only")
    args = parser.parse_args(argv)

    if args.only in (None, "pvgis"):
        download_pvgis_hourly(0, args.force)
        download_pvgis_hourly(cfg.REF_TILT, args.force)
        download_pvcalc(args.force)
    if args.only in (None, "openmeteo"):
        download_openmeteo(args.force)


if __name__ == "__main__":
    main()
