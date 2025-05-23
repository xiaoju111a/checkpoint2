#!/usr/bin/env python3
# ------------------------------------------------------------
#  Checkpoint-2  (single-file refactor, PEP-8 compliant)
#  • gpt-4o-mini as LLM backend
#  • dataclass + modular loader pattern
#  • ready for GEE / OpenTopography / NASA / ESA / AWS datasets
# ------------------------------------------------------------
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.wkt import dumps as wkt_dumps
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

# ---------- optional external libs (gracefully degraded) ----------
try:
    import geemap, ee  # Google Earth Engine
    _GEE_ENABLED = True
except Exception:
    _GEE_ENABLED = False

try:
    import boto3  # AWS
except Exception:  # pragma: no cover
    boto3 = None  # type: ignore

try:
    from openai import OpenAI, RateLimitError
except ImportError as exc:  # pragma: no cover
    print("openai sdk missing:", exc, file=sys.stderr)
    OpenAI, RateLimitError = None, RuntimeError  # type: ignore

# --------------------------- config ----------------------------
RNG_SEED = 42
np.random.seed(RNG_SEED)
random.seed(RNG_SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
LOG = logging.getLogger("checkpoint2")

MODEL_NAME = "gpt-4o-mini"
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
SKIP_OPENAI = os.getenv("SKIP_OPENAI", "0") == "1"
openai_client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and not SKIP_OPENAI) else None

DATASET_IDS: dict[str, str] = {
    "gee_sentinel_2": "COPERNICUS/S2_SR",
    "gee_gedi": "LARSE/GEDI/GEDI02_A_002_MONTHLY",
    "nasa_cmr": "https://cmr.earthdata.nasa.gov/",
    "esa_hub": "https://scihub.copernicus.eu/",
    "opentopo": "https://portal.opentopography.org/",
    "aws_sentinel_2": "s3://sentinel-s2-l2a/",
    "archaeo_points": "data/archaeo_sites_acre.csv",
}

_DEG_PER_METER = 1.0 / 111_000.0  # lat/long conversion factor

# --------------------------- dataclasses ----------------------------
@dataclass(slots=True, frozen=True)
class Anomaly:
    id: str
    type: str
    lat: float
    lon: float
    radius_m: int
    confidence: float
    data_source: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update(self.extra)
        return d


@dataclass(slots=True, frozen=True)
class Footprint:
    id: str
    wkt: str
    center: str
    radius_m: int
    confidence: float
    type: str


@dataclass(slots=True)
class PromptLog:
    timestamp: str
    prompt_type: str
    anomaly_id: str | None
    model: str
    prompt: str


# --------------------------- data loader ----------------------------
class DataLoader:
    """All external data sources in one place."""

    # ---- Google Earth Engine ----
    @staticmethod
    def load_gee(
        collection_id: str,
        region: "ee.Geometry",  # type: ignore
        bands: list[str],
        start: str,
        end: str,
    ):
        if not _GEE_ENABLED:
            LOG.warning("GEE not available; returning None for %s.", collection_id)
            return None
        LOG.info("Fetching GEE %s from %s to %s …", collection_id, start, end)
        img = (
            ee.ImageCollection(collection_id)
            .filterBounds(region)
            .filterDate(start, end)
            .select(bands)
            .median()
        )
        return img.clip(region)

    # ---- OpenTopography ----
    @staticmethod
    def load_opentopo(bbox: list[float]) -> pd.DataFrame:
        LOG.info("Mock OpenTopography query for %s", bbox)
        return pd.DataFrame({"elev_mean": [155.4], "elev_std": [3.7], "pts_m2": [6.1]})

    # ---- NASA / ESA / AWS placeholders ----
    @staticmethod
    def load_nasa_placeholder() -> None:
        LOG.info("NASA loader placeholder invoked.")

    # ---- Archaeological points ----
    @staticmethod
    def load_archaeo_points() -> pd.DataFrame:
        path = Path(DATASET_IDS["archaeo_points"])
        if not path.exists():
            LOG.warning("Archaeological point file missing: %s", path)
            return pd.DataFrame(columns=["lat", "lon", "site_type"])
        return pd.read_csv(path)

    # ---- Mock GEDI point data ----
    @staticmethod
    def mock_gedi_points(n_samples: int = 800) -> pd.DataFrame:
        lat = np.random.uniform(-11.0, -8.5, n_samples)
        lon = np.random.uniform(-73.0, -66.5, n_samples)
        canopy = np.random.normal(30, 10, n_samples)
        idx = np.random.choice(n_samples, 12, replace=False)
        canopy[idx] = np.random.uniform(5, 15, idx.size)
        return pd.DataFrame({"lat": lat, "lon": lon, "canopy_height": canopy})


# --------------------------- processing ----------------------------
class AnomalyDetector:
    """Convert multi-source data to Anomaly objects."""

    def __init__(self) -> None:
        self.anomalies: list[Anomaly] = []

    def detect(
        self,
        gedi_df: pd.DataFrame,
        arch_df: pd.DataFrame,
        max_return: int = 10,
    ) -> list[Anomaly]:
        # -------- GEDI low canopy --------
        mean, std = gedi_df.canopy_height.mean(), gedi_df.canopy_height.std()
        low_thr = mean - 1.5 * std
        low_df = gedi_df[gedi_df.canopy_height < low_thr]

        for row in low_df.itertuples():
            self.anomalies.append(
                Anomaly(
                    id=f"gedi_{row.Index}",
                    type="low_canopy",
                    lat=row.lat,
                    lon=row.lon,
                    radius_m=120,
                    confidence=0.7,
                    data_source="GEDI",
                    extra={"canopy_height": row.canopy_height},
                )
            )

        # -------- catalogued archaeological points --------
        for idx, row in arch_df.iterrows():
            self.anomalies.append(
                Anomaly(
                    id=f"known_{idx}",
                    type=row.get("site_type", "archaeological_site"),
                    lat=row.lat,
                    lon=row.lon,
                    radius_m=150,
                    confidence=0.95,
                    data_source="Catalogued",
                    extra={},
                )
            )

        self.anomalies.sort(key=lambda a: (-a.confidence, a.radius_m))
        return self.anomalies[:max_return]


def build_footprints(anoms: list[Anomaly]) -> list[Footprint]:
    fps: list[Footprint] = []
    for a in anoms:
        delta = a.radius_m * _DEG_PER_METER
        bbox = box(a.lon - delta, a.lat - delta, a.lon + delta, a.lat + delta)
        fps.append(
            Footprint(
                id=a.id,
                wkt=wkt_dumps(bbox),
                center=f"POINT({a.lon} {a.lat})",
                radius_m=a.radius_m,
                confidence=a.confidence,
                type=a.type,
            )
        )
    return fps


# --------------------------- LLM utils ----------------------------
def _prompt_single(a: Anomaly) -> str:
    return (
        "You are a South-American landscape archaeologist.\n"
        f"Candidate at {a.lat:.4f}, {a.lon:.4f}\n"
        f"Signal: {a.type} | Source: {a.data_source}\n"
        f"Meta: {json.dumps(a.extra, ensure_ascii=False)}\n\n"
        "1. Probability of anthropic origin (0-1):\n"
        "2. Likely archaeological structures:\n"
        "3. Next imagery / in-situ checks (≤3 lines):"
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(2, 8), reraise=True)
def _chat(prompt: str) -> str:
    if SKIP_OPENAI or openai_client is None:
        return "OpenAI disabled."
    rsp = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an expert archaeologist."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=256,
    )
    return rsp.choices[0].message.content.strip()


def analyse_single(a: Anomaly, logs: list[PromptLog]) -> str:
    p = _prompt_single(a)
    logs.append(
        PromptLog(
            timestamp=datetime.utcnow().isoformat(),
            prompt_type="single",
            anomaly_id=a.id,
            model=MODEL_NAME,
            prompt=p,
        )
    )
    try:
        return _chat(p)
    except RateLimitError:  # type: ignore
        LOG.warning("OpenAI rate-limited.")
        return "Rate-limited."
    except Exception as exc:  # pragma: no cover
        LOG.error("LLM error: %s", exc)
        return "LLM error."


def analyse_leverage(anoms: list[Anomaly], logs: list[PromptLog]) -> str:
    summary = json.dumps(
        [
            {
                "id": a.id,
                "type": a.type,
                "conf": a.confidence,
                "loc": f"{a.lat:.4f},{a.lon:.4f}",
            }
            for a in anoms
        ],
        indent=2,
    )

    prompt = (
        f"{summary}\n\n"
        "Provide:\n"
        "1. Additional EO datasets to verify each anomaly;\n"
        "2. A 5-site ranked field survey plan;\n"
        "3. Recommended cross-disciplinary partners."
    )
    logs.append(
        PromptLog(
            timestamp=datetime.utcnow().isoformat(),
            prompt_type="leverage",
            anomaly_id=None,
            model=MODEL_NAME,
            prompt=prompt,
        )
    )
    try:
        return _chat(prompt)
    except Exception as exc:
        LOG.error("Leverage LLM error: %s", exc)
        return "LLM leverage error."


# --------------------------- reproducibility ----------------------------
def reproducibility_hash(anoms: list[Anomaly]) -> str:
    s = "".join(f"{a.lat:.4f},{a.lon:.4f},{a.radius_m};" for a in sorted(anoms, key=lambda x: x.id))
    return hashlib.md5(s.encode()).hexdigest()[:8]


# --------------------------- pipeline & CLI ----------------------------
def run_checkpoint2() -> Dict[str, Any]:
    LOG.info("Running checkpoint-2 (single-file)")

    # Earth Engine init if available
    if _GEE_ENABLED:
        try:
            if not ee.data._initialized:
                ee.Initialize()
        except Exception as exc:  # pragma: no cover
            LOG.warning("Earth Engine init failed: %s", exc)

    gedi_df = DataLoader.mock_gedi_points()
    arch_df = DataLoader.load_archaeo_points()

    detector = AnomalyDetector()
    anomalies = detector.detect(gedi_df, arch_df)
    footprints = build_footprints(anomalies)

    logs: list[PromptLog] = []
    analyses = {a.id: analyse_single(a, logs) for a in anomalies}
    leverage_analysis = analyse_leverage(anomalies, logs)

    rhash = reproducibility_hash(anomalies)
    LOG.info("Reproducibility hash: %s", rhash)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "dataset_ids": DATASET_IDS,
        "anomalies": [a.to_dict() for a in anomalies],
        "footprints": [asdict(fp) for fp in footprints],
        "analyses": analyses,
        "leverage_analysis": leverage_analysis,
        "prompts_log": [asdict(p) for p in logs],
        "reproducibility_hash": rhash,
    }


def main() -> None:
    res = run_checkpoint2()

    out = Path("checkpoint2_results.json")
    out.write_text(json.dumps(res, indent=2), "utf-8")

    print("\nCHECKPOINT-2 SUMMARY")
    print(f" Anomalies found:    {len(res['anomalies'])}")
    print(f" Reproducibility ID: {res['reproducibility_hash']}\n")
    for i, a in enumerate(res["anomalies"][:3], 1):
        print(f" {i}. {a['id']} @ {a['lat']:.4f},{a['lon']:.4f}")
    print("\nLeverage analysis preview:")
    print(res["leverage_analysis"][:400] + " …")
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.warning("Interrupted by user.")  # pragma: no cover