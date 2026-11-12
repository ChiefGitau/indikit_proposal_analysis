"""Paths shared by the app pages. ``LATEST_DIR`` points at the pipeline's ``outputs/latest/`` folder of the
``proposal_investigated`` repository (three levels up from the app); override with ``ZOA_PIPELINE_LATEST``."""

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = APP_DIR.parents[2]          # .../proposal_investigated
LATEST_DIR = Path(os.environ.get("ZOA_PIPELINE_LATEST") or PIPELINE_ROOT / "outputs" / "latest")
METADATA_JSONL = Path(os.environ.get("ZOA_PIPELINE_METADATA") or PIPELINE_ROOT / "pipeline" / "metadata" / "indikit_metadata.jsonl")
PROPOSALS_PREFIX = "proposals/"
INDIKIT_URL = "https://zoa.indikit.net/indicator/"
