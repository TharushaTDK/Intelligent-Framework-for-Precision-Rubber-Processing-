import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib


MODEL_DIR = Path(__file__).resolve().parent

WATER_ACID_MODEL_PATH = MODEL_DIR / "water_acid_pipeline.joblib"
COAGULATION_TIME_MODEL_PATH = MODEL_DIR / "coagulation_time_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "component2_model_metadata.json"
CONTRACT_PATH = MODEL_DIR / "component2_inference_contract.json"

# trusted hashes of the runtime models re-exported by src/33_export_backend_runtime.py
EXPECTED_WATER_ACID_SHA256 = (
    "b7071fd6d2002c640a5fd2af8366aa54facc7160a2683d24851e36291454daa2"
)
EXPECTED_COAGULATION_TIME_SHA256 = (
    "f74d244663b0938c90b78a46c2d008c7b66eb9cf5ea4d3d6b39f48d831a0f6c1"
)


class ModelLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class Component2Metadata:
    water_acid_version: str
    coagulation_time_version: str
    water_acid_features: tuple[str, ...]
    coagulation_time_features: tuple[str, ...]
    training_ranges: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class Component2Models:
    water_acid: Any
    coagulation_time: Any
    metadata: Component2Metadata


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise ModelLoadError(f"Missing Component 2 file: {path.name}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelLoadError(f"Could not read {path.name}.") from exc
    if not isinstance(data, dict):
        raise ModelLoadError(f"{path.name} must contain a JSON object.")
    return data


def _verify_file(path: Path, expected: str) -> None:
    if not path.is_file():
        raise ModelLoadError(f"Missing model artifact: {path.name}")
    try:
        actual = _sha256(path)
    except OSError as exc:
        raise ModelLoadError(f"Could not hash {path.name}.") from exc
    if actual != expected:
        raise ModelLoadError(
            f"{path.name} does not match the finalized research model."
        )


@lru_cache(maxsize=1)
def load_metadata() -> Component2Metadata:
    meta = _read_json(METADATA_PATH)
    contract = _read_json(CONTRACT_PATH)

    try:
        water_acid = meta["water_acid"]
        coag_time = meta["coagulation_time"]

        # metadata has to agree with the trusted hashes above
        if water_acid["sha256"] != EXPECTED_WATER_ACID_SHA256:
            raise ModelLoadError("Water/acid hash in metadata is not trusted.")
        if coag_time["sha256"] != EXPECTED_COAGULATION_TIME_SHA256:
            raise ModelLoadError("Coagulation time hash in metadata is not trusted.")

        ranges = {
            name: (float(r["min"]), float(r["max"]))
            for name, r in contract["training_ranges"].items()
        }

        return Component2Metadata(
            water_acid_version=str(water_acid["model_version"]),
            coagulation_time_version=str(coag_time["model_version"]),
            water_acid_features=tuple(contract["water_acid"]["features"]),
            coagulation_time_features=tuple(contract["coagulation_time"]["features"]),
            training_ranges=ranges,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelLoadError("Component 2 metadata or contract is incomplete.") from exc


@lru_cache(maxsize=1)
def load_models() -> Component2Models:
    metadata = load_metadata()

    _verify_file(WATER_ACID_MODEL_PATH, EXPECTED_WATER_ACID_SHA256)
    _verify_file(COAGULATION_TIME_MODEL_PATH, EXPECTED_COAGULATION_TIME_SHA256)

    try:
        water_acid = joblib.load(WATER_ACID_MODEL_PATH)
        coag_time = joblib.load(COAGULATION_TIME_MODEL_PATH)
    except Exception as exc:
        raise ModelLoadError("Component 2 models could not be loaded.") from exc

    for model, name in ((water_acid, "water/acid"), (coag_time, "coagulation time")):
        if not hasattr(model, "predict"):
            raise ModelLoadError(f"The {name} artifact has no predict().")

    # the artifacts must expect the same feature order as the contract
    for model, features in (
        (water_acid, metadata.water_acid_features),
        (coag_time, metadata.coagulation_time_features),
    ):
        trained_on = getattr(model, "feature_names_in_", None)
        if trained_on is not None and tuple(trained_on) != features:
            raise ModelLoadError("Model feature order does not match the contract.")

    return Component2Models(
        water_acid=water_acid,
        coagulation_time=coag_time,
        metadata=metadata,
    )


def get_models() -> Component2Models:
    return load_models()


def get_metadata() -> Component2Metadata:
    return load_metadata()
