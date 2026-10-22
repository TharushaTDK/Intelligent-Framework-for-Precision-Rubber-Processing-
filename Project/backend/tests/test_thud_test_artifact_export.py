"""Research artifact export checks; controlled temporary fitted pipeline only."""
import hashlib
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.app.ml.component_4_thud_test import model_loader
from backend.app.ml.component_4_thud_test.feature_contract import FEATURE_NAMES


def exporter():
    project = Path(__file__).resolve().parents[2]
    path = project / 'ml-research/component_4_thud_test/src/export_models.py'
    spec = importlib.util.spec_from_file_location('component4_export_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fitted(tmp_path, monkeypatch):
    runtime = tmp_path / 'runtime'
    (runtime / 'artifacts').mkdir(parents=True)
    x = np.random.default_rng(42).normal(size=(16, len(FEATURE_NAMES)))
    labels = np.array(['VALID', 'INVALID'] * 8)
    pipeline = Pipeline([('scale', StandardScaler()), ('classifier', LogisticRegression())])
    pipeline.fit(x, labels)
    path = runtime / 'artifacts/quality_pipeline.joblib'
    joblib.dump(pipeline, path)
    metadata = {
        'artifact': path.name, 'artifact_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'feature_names': list(FEATURE_NAMES), 'fit_rows': len(x),
        'model_version': 'test-model', 'sklearn_version': sklearn.__version__,
    }
    monkeypatch.setattr(model_loader, 'COMPONENT', runtime)
    monkeypatch.setattr(model_loader, 'get_quality_model', lambda: (pipeline, metadata))
    return runtime, tmp_path / 'research', pipeline, metadata, x


def test_export_preserves_pipeline_and_composes_identical_components_without_refit(fitted, monkeypatch):
    runtime, research, pipeline, metadata, x = fitted
    def forbidden_fit(*args, **kwargs):
        pytest.fail('Export must not refit either component.')
    monkeypatch.setattr(StandardScaler, 'fit', forbidden_fit)
    monkeypatch.setattr(LogisticRegression, 'fit', forbidden_fit)
    report = exporter().export_models(research, runtime)
    copy = research / 'artifacts/pipelines/quality_pipeline.joblib'
    assert copy.read_bytes() == (runtime / 'artifacts/quality_pipeline.joblib').read_bytes()
    scaler = joblib.load(research / 'artifacts/preprocessing/quality_scaler.joblib')
    classifier = joblib.load(research / 'artifacts/models/classification/quality_classifier.joblib')
    np.testing.assert_allclose(classifier.predict_proba(scaler.transform(x)), pipeline.predict_proba(x), rtol=0, atol=1e-12)
    assert report['refitted_during_export'] is False
    saved = json.loads((research / 'artifacts/preprocessing/quality_scaler_metadata.json').read_text())
    assert saved['source_pipeline_sha256'] == metadata['artifact_sha256']
    assert saved['feature_names'] == FEATURE_NAMES


def test_export_refuses_changed_source_artifact(fitted):
    runtime, research, pipeline, metadata, x = fitted
    (runtime / 'artifacts/quality_pipeline.joblib').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Source artifact changed'):
        exporter().export_models(research, runtime)
    assert not research.exists()


def test_export_refuses_inconsistent_feature_order(fitted):
    runtime, research, pipeline, metadata, x = fitted
    metadata['feature_names'] = list(reversed(FEATURE_NAMES))
    with pytest.raises(ValueError, match='feature order differs'):
        exporter().export_models(research, runtime)
    assert not research.exists()
