"""Signal/rule behavior and model integration contracts, not accuracy claims.

Signals are generated solely inside temporary test directories. Classifier
probabilities and joblib loading are controlled mocks, never training data.
"""
import hashlib
import json
import wave

import numpy as np
import pytest

from backend.app.ml.component_4_thud_test import model_loader as loader
from backend.app.ml.component_4_thud_test import inference
from backend.app.ml.component_4_thud_test.preprocessing import extract_features
from backend.app.ml.component_4_thud_test.preprocessing import check_recording
from backend.app.ml.component_4_thud_test.feature_contract import FEATURE_NAMES


RATE = 16000


def write_signal(tmp_path, events=((1.0, 0.4, 0.1),)):
    signal = np.zeros(3 * RATE)
    for onset, amplitude, duration in events:
        start, count = round(onset * RATE), round(duration * RATE)
        signal[start:start + count] = amplitude * np.sin(2 * np.pi * 500 * np.arange(count) / RATE)
    values = np.rint(signal * (2 ** 23)).astype(np.int64)
    raw = np.column_stack((values & 255, (values >> 8) & 255,
                           (values >> 16) & 255)).astype(np.uint8).tobytes()
    path = tmp_path / 'signal.wav'
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(3)
        wav.setframerate(RATE)
        wav.writeframes(raw)
    return path


class ControlledEstimator:
    # Deliberately opposite to alphabetical label order: indexing must use classes_.
    classes_ = np.array(['VALID', 'INVALID'])

    def __init__(self, invalid_score):
        self.invalid_score = invalid_score
        self.inputs = []

    def predict_proba(self, values):
        self.inputs.append(values.copy())
        return np.array([[1 - self.invalid_score, self.invalid_score]])


def attach_estimator(monkeypatch, score):
    estimator = ControlledEstimator(score)
    metadata = {
        'feature_names': list(reversed(FEATURE_NAMES)),
        'decision_threshold_invalid': 0.5,
        'model_version': 'controlled-test-model',
        'status': 'PRELIMINARY_DEMO',
    }
    monkeypatch.setattr(loader, 'get_quality_model', lambda: (estimator, metadata))
    return estimator, metadata


def test_known_tone_has_finite_features_and_correct_frequency(tmp_path):
    path = write_signal(tmp_path)
    original = path.read_bytes()
    features, checks = extract_features(path)
    assert checks['technical_status'] == 'TECHNICAL_PASSED'
    assert features is not None
    assert len(features) == 16
    assert set(features) == set(FEATURE_NAMES)
    assert all(np.isfinite(value) for value in features.values())
    assert features['dominant_frequency_hz'] == pytest.approx(500, abs=10)
    assert features['segment_rms'] > 0
    assert path.read_bytes() == original


def test_no_event_has_no_feature_vector(tmp_path):
    features, checks = extract_features(write_signal(tmp_path, events=()))
    assert features is None
    assert checks['technical_status'] == 'TECHNICAL_REJECTED'
    assert 'NO_EVENT' in checks['rejection_reasons']


@pytest.mark.parametrize('events,reason', [
    ((), 'SILENT_RECORDING'),
    (((0.8, 0.4, 0.02), (1.3, 0.4, 0.02)), 'MULTIPLE_EVENT_CANDIDATES'),
    (((2.8, 0.4, 0.02),), 'INCOMPLETE_RESPONSE'),
])
def test_technical_rejection_never_loads_classifier(tmp_path, monkeypatch, events, reason):
    def forbidden_model_load():
        pytest.fail('Technical rejection must bypass model loading.')
    monkeypatch.setattr(loader, 'get_quality_model', forbidden_model_load)
    result = inference.assess_recording(write_signal(tmp_path, events))
    assert result['quality_status'] == 'INVALID'
    assert result['technical_status'] == 'TECHNICAL_REJECTED'
    assert result['classifier_status'] == 'SKIPPED_TECHNICAL_REJECTION'
    assert result['decision_source'] == 'TECHNICAL_RULES'
    assert reason in result['rejection_reasons']
    assert result['invalid_probability_uncalibrated'] is None
    assert result['model_version'] is None
    assert result['moisture_percent'] is None


@pytest.mark.parametrize('score,expected', [(0.49, 'VALID'), (0.5, 'INVALID'), (0.9, 'INVALID')])
def test_score_uses_class_labels_threshold_and_metadata_feature_order(tmp_path, monkeypatch, score, expected):
    path = write_signal(tmp_path)
    estimator, metadata = attach_estimator(monkeypatch, score)
    result = inference.assess_recording(path)
    assert result['quality_status'] == expected
    assert result['classifier_status'] == 'MODEL_EVALUATED'
    assert result['decision_source'] == 'ML_CLASSIFIER'
    assert result['invalid_probability_uncalibrated'] == score
    assert result['moisture_percent'] is None
    assert len(estimator.inputs) == 1
    assert estimator.inputs[0].shape == (1, 16)
    features, _ = extract_features(path)
    np.testing.assert_allclose(estimator.inputs[0][0], [features[name] for name in metadata['feature_names']])
    expected_reasons = ['ML_RECORDING_QUALITY_REJECTION'] if expected == 'INVALID' else []
    assert result['rejection_reasons'] == expected_reasons


def test_missing_model_is_unavailable_not_valid_or_invalid(tmp_path, monkeypatch):
    def missing():
        raise loader.ModelLoadError('Test model is unavailable.')
    monkeypatch.setattr(loader, 'get_quality_model', missing)
    result = inference.assess_recording(write_signal(tmp_path))
    assert result['technical_status'] == 'TECHNICAL_PASSED'
    assert result['quality_status'] == 'UNAVAILABLE'
    assert result['classifier_status'] == 'QUALITY_MODEL_UNAVAILABLE'
    assert result['invalid_probability_uncalibrated'] is None


def test_high_peak_warning_does_not_itself_reject_recording(tmp_path, monkeypatch):
    path = write_signal(tmp_path, ((1.0, 0.995, 0.1),))
    checks = check_recording(path)
    assert checks['technical_status'] == 'TECHNICAL_PASSED'
    assert checks['near_full_scale_fraction'] == 0.0
    assert 'HIGH_PEAK_REVIEW' in checks['warnings']
    attach_estimator(monkeypatch, 0.2)
    result = inference.assess_recording(path)
    assert result['quality_status'] == 'VALID'
    assert 'HIGH_PEAK_REVIEW' in result['warnings']


@pytest.fixture
def model_files(tmp_path, monkeypatch):
    component = tmp_path / 'component'
    artifacts = component / 'artifacts'
    artifacts.mkdir(parents=True)
    artifact = artifacts / 'quality_pipeline.joblib'
    artifact.write_bytes(b'controlled fixture: never unpickled')
    source = component / 'audio_io.py'
    source.write_text('test processing bytes', encoding='utf-8')
    metadata = {
        'artifact': artifact.name,
        'artifact_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
        'code_hashes': {'audio_io.py': hashlib.sha256(source.read_bytes()).hexdigest()},
        'sklearn_version': loader.sklearn.__version__,
        'feature_version': loader.FEATURE_VERSION,
        'rule_version': loader.RULE_VERSION,
        'segmentation_config': dict(loader.CONFIG),
        'near_full_scale_limit': loader.NEAR_FULL_SCALE_LIMIT,
        'feature_names': list(FEATURE_NAMES),
    }
    metadata_path = artifacts / 'quality_model_metadata.json'
    metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
    loader.get_quality_model.cache_clear()
    monkeypatch.setattr(loader, 'COMPONENT', component)
    yield artifact, source, metadata_path, metadata
    loader.get_quality_model.cache_clear()


@pytest.mark.parametrize('change', ['artifact', 'processing_code', 'sklearn_version', 'segmentation_config', 'feature_order'])
def test_integrity_or_environment_mismatch_fails_before_deserialization(model_files, monkeypatch, change):
    artifact, source, metadata_path, metadata = model_files
    if change == 'artifact':
        artifact.write_bytes(b'changed artifact')
    elif change == 'processing_code':
        source.write_text('changed processing bytes', encoding='utf-8')
    elif change == 'sklearn_version':
        metadata['sklearn_version'] = 'deliberately-incompatible-test-version'
    elif change == 'segmentation_config':
        metadata['segmentation_config']['post_seconds'] += 0.1
    else:
        metadata['feature_names'] = list(reversed(FEATURE_NAMES))
    metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
    def forbidden_load(*args, **kwargs):
        pytest.fail('Mismatched model must not be deserialized.')
    monkeypatch.setattr(loader.joblib, 'load', forbidden_load)
    with pytest.raises(loader.ModelLoadError):
        loader.get_quality_model()


def test_verified_model_load_is_cached(model_files, monkeypatch):
    calls = []
    estimator = ControlledEstimator(0.2)
    def load(path):
        calls.append(path)
        return estimator
    monkeypatch.setattr(loader.joblib, 'load', load)
    first = loader.get_quality_model()
    second = loader.get_quality_model()
    assert first is second
    assert first[0] is estimator
    assert len(calls) == 1


def test_unexpected_model_classes_are_rejected(model_files, monkeypatch):
    estimator = ControlledEstimator(0.2)
    estimator.classes_ = np.array(['VALID', 'UNRELATED_LABEL'])
    monkeypatch.setattr(loader.joblib, 'load', lambda path: estimator)
    with pytest.raises(loader.ModelLoadError, match='Unexpected model classes'):
        loader.get_quality_model()


def dataset_fixture(tmp_path):
    """Small provenance-linked corpus for the research export checks only."""
    import csv
    from pathlib import Path
    research = tmp_path / 'research'
    raw = research / 'data/raw'
    metadata = research / 'data/metadata'
    raw.mkdir(parents=True)
    metadata.mkdir(parents=True)
    originals = []
    manifest = []
    for number, amplitude in enumerate([0.3, 0.4, 0.5], 1):
        directory = tmp_path / f'fixture_{number}'
        directory.mkdir()
        source = write_signal(directory, ((1.0, amplitude, 0.1),))
        filename = f'S{number}_T1.wav'
        (raw / filename).write_bytes(source.read_bytes())
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        row = {'recording_id': f'S{number}_T1', 'sheet_id': f'S{number}',
               'condition_id': 'C01', 'filename': filename, 'quality_label': 'VALID',
               'source_type': 'COLLECTED', 'parent_recording_id': ''}
        originals.append(row)
        manifest.append({**row, 'sha256': digest, 'parent_sha256': '',
                         'transformation': '', 'shift_seconds': '', 'noise_sigma': ''})
    parent = manifest[0]
    derived_name = 'S1_T1_A01.wav'
    # Same bytes are adequate for this split-isolation fixture; it is not a new sheet.
    (raw / derived_name).write_bytes((raw / parent['filename']).read_bytes())
    manifest.append({**parent, 'recording_id': 'S1_T1_A01', 'filename': derived_name,
                     'source_type': 'DERIVED', 'parent_recording_id': parent['recording_id'],
                     'parent_sha256': parent['sha256'],
                     'transformation': 'controlled fixture copy', 'shift_seconds': '0', 'noise_sigma': '0'})
    for name, rows in [('recordings.csv', originals), ('dataset_manifest.csv', manifest)]:
        with (metadata / name).open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return research


def load_dataset_preparer():
    import importlib.util
    from pathlib import Path
    project = Path(__file__).resolve().parents[2]
    path = project / 'ml-research/component_4_thud_test/src/preprocessing.py'
    spec = importlib.util.spec_from_file_location('component4_dataset_prepare_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dataset_exports_are_repeatable_and_keep_parent_sheet_out_of_test_training(tmp_path):
    import csv
    research = dataset_fixture(tmp_path)
    original_bytes = {p.name: p.read_bytes() for p in (research / 'data/raw').glob('*.wav')}
    prepare = load_dataset_preparer().prepare_dataset
    first = prepare(research)
    second = prepare(research)
    assert first == second
    assert first['collected_recordings'] == 3
    assert first['derived_recordings'] == 1
    assert first['features_are_scaled'] is False
    for path in (research / 'data/raw').glob('*.wav'):
        assert path.read_bytes() == original_bytes[path.name]
    split = research / 'data/splits/fold_01'
    with (split / 'train.csv').open() as file:
        training = list(csv.DictReader(file))
    with (split / 'test.csv').open() as file:
        testing = list(csv.DictReader(file))
    with (split / 'excluded.csv').open() as file:
        excluded = list(csv.DictReader(file))
    assert {row['sheet_id'] for row in training} == {'S2', 'S3'}
    assert {row['sheet_id'] for row in testing} == {'S1'}
    assert all(row['source_type'] == 'COLLECTED' for row in testing)
    assert 'S1_T1_A01' in {row['recording_id'] for row in excluded}


def test_dataset_preparation_refuses_changed_derived_input(tmp_path):
    research = dataset_fixture(tmp_path)
    (research / 'data/raw/S1_T1_A01.wav').write_bytes(b'changed')
    with pytest.raises(ValueError, match='refusing overwrite'):
        load_dataset_preparer().prepare_dataset(research)
