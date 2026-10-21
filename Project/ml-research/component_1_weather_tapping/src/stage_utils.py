"""Shared helpers for the numbered data stages.

Re-exports the canonical paths from :mod:`common` so that every stage honours the
train-only redirect (``RUBBERSENSE_TRAIN_ONLY_DIR``) without duplicating logic.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (  # noqa: F401  (re-exported for ``from stage_utils import *``)
    ROOT, RAW, SPLITS, PROCESSED, REPORTS, FIGURES, ARTIFACTS,
    REG_MODELS, CLS_MODELS, PIPELINES, PREPROCESSING,
    REG_TASKS, CLS_TASK, CLS_TARGET, CLS_FEATURES, CLS_CLASSES,
    DATASET_PROVENANCE, TARGET_WARNING,
    ensure_dirs, load_raw, sha256_file, sha256_frame,
)

REPORTS.mkdir(parents=True, exist_ok=True)


def load_data():
    return load_raw()


def save_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
