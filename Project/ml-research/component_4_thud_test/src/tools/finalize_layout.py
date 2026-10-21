"""Verify runtime equivalence, rebind code hashes and archive retired C4 files.

Requires a unique backup directory outside the repository. Nothing is deleted,
no model is refitted, and no peer component is modified.
"""
import argparse
import ast
import csv
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[4]
REPO = PROJECT.parent
RESEARCH = PROJECT / 'ml-research/component_4_thud_test'
COMPONENT = PROJECT / 'backend/app/ml/component_4_thud_test'
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from backend.app.ml.component_4_thud_test import inference, model_loader, preprocessing
from backend.app.ml.component_4_thud_test.feature_contract import FEATURE_NAMES


LAYOUT = 'preprocessing-inference-v2'
RUNTIME_FILES = ('preprocessing.py', 'feature_contract.py', 'model_loader.py', 'inference.py')
LEGACY_FUNCTIONS = {
    'audio_io.py': ('load_wav', 'inspect_wav'),
    'segmentation.py': ('segment_wav',),
    'technical_checks.py': ('check_recording',),
    'features.py': ('rms', 'extract_features'),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def functions(path):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    return {n.name: ast.dump(n, include_attributes=False)
            for n in tree.body if isinstance(n, ast.FunctionDef)}


def checked_legacy_results():
    from backend.app.database.models import AcousticMeasurement
    from backend.app.database.session import SessionLocal
    from backend.app.schemas.thud_test import AcousticMeasurementResponse
    from backend.app.services.thud_test_service import STORAGE_DIR
    from uuid import UUID
    paths = []
    with SessionLocal() as db:
        for path in sorted(STORAGE_DIR.glob('*.json')):
            identifier = UUID(path.stem)
            response = AcousticMeasurementResponse.model_validate_json(path.read_text(encoding='utf-8'))
            row = db.get(AcousticMeasurement, identifier)
            if row is None:
                raise ValueError(f'Legacy result not yet imported: {identifier}')
            stored = AcousticMeasurementResponse.model_validate(row, from_attributes=True)
            if response.measurement_id != identifier or stored.model_dump(mode='json') != response.model_dump(mode='json'):
                raise ValueError(f'Legacy JSON/database result differs: {identifier}')
            if row.raw_file_name != f'{identifier}.wav' or sha256(STORAGE_DIR / row.raw_file_name) != row.source_sha256:
                raise ValueError(f'Database recording reference/hash differs: {identifier}')
            paths.append(path)
    return paths


def finalize(backup, include_legacy_json=False):
    backup = Path(backup).resolve()
    if backup.is_relative_to(REPO.resolve()):
        raise ValueError('Backup must be outside the repository.')
    if backup.exists():
        raise ValueError('Use a new backup directory so previous backups cannot be overwritten.')
    metadata_path = COMPONENT / 'artifacts/quality_model_metadata.json'
    previous_bytes = metadata_path.read_bytes()
    metadata = json.loads(previous_bytes)
    already_new = metadata.get('runtime_layout_version') == LAYOUT
    if metadata['feature_names'] != FEATURE_NAMES:
        raise ValueError('Feature contract differs; stop rather than change model meaning.')
    if not already_new:
        if set(metadata['code_hashes']) != set(LEGACY_FUNCTIONS):
            raise ValueError('Unexpected previous runtime layout; review it before migration.')
        combined = functions(COMPONENT / 'preprocessing.py')
        for name, names in LEGACY_FUNCTIONS.items():
            old_path = COMPONENT / name
            if sha256(old_path) != metadata['code_hashes'][name]:
                raise ValueError(f'Previous processing source changed: {name}')
            old_functions = functions(old_path)
            for function in names:
                if old_functions[function] != combined[function]:
                    raise ValueError(f'Processing function changed: {function}')
        old_features = importlib.import_module('backend.app.ml.component_4_thud_test.features')
    else:
        old_features = preprocessing

    model_loader.get_quality_model.cache_clear()
    pipeline, verified_metadata = model_loader.get_quality_model()
    with (RESEARCH / 'data/metadata/dataset_manifest.csv').open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError('Prepared dataset manifest is empty.')
    for row in rows:
        filename = row['filename']
        if Path(filename).name != filename or '/' in filename or '\\' in filename:
            raise ValueError('Unexpected dataset filename.')
        path = RESEARCH / 'data/raw' / filename
        if sha256(path) != row['sha256']:
            raise ValueError(f'Dataset input hash changed: {filename}')
        previous_features, previous_checks = old_features.extract_features(path)
        current_features, current_checks = preprocessing.extract_features(path)
        if previous_checks != current_checks or previous_features != current_features:
            raise ValueError(f'Preprocessing output changed: {filename}')
        result = inference.assess_recording(path)
        if previous_checks['technical_status'] == 'TECHNICAL_REJECTED':
            if result['quality_status'] != 'INVALID' or result['decision_source'] != 'TECHNICAL_RULES':
                raise ValueError('Technical rejection behavior changed.')
            if result['rejection_reasons'] != previous_checks['rejection_reasons']:
                raise ValueError('Technical rejection reasons changed.')
        else:
            vector = np.array([[previous_features[name] for name in FEATURE_NAMES]])
            score = float(pipeline.predict_proba(vector)[0][list(pipeline.classes_).index('INVALID')])
            label = 'INVALID' if score >= metadata['decision_threshold_invalid'] else 'VALID'
            if result['quality_status'] != label or not np.isclose(result['invalid_probability_uncalibrated'], score, rtol=0, atol=1e-12):
                raise ValueError(f'Inference output changed: {filename}')

    retired = [COMPONENT / name for name in (*LEGACY_FUNCTIONS, 'preview_api.py')]
    retired += [RESEARCH / 'src' / name for name in (
        'prepare_dataset.py', 'export_features.py', 'review_audio.py',
        'import_legacy_measurements.py', 'archive_legacy_layout.py')]
    old_reports = RESEARCH / 'reports'
    report_copies = []
    if old_reports.exists():
        for path in old_reports.iterdir():
            if not path.is_file():
                raise ValueError('Review unexpected nested legacy report directories before migration.')
            target = RESEARCH / 'outputs/reports' / path.name
            if target.exists() and target.read_bytes() != path.read_bytes():
                raise ValueError(f'Report destination differs; refusing overwrite: {path.name}')
            report_copies.append((path, target))
        retired.append(old_reports)
    retired.append(RESEARCH / 'data/review/waveforms.html')
    if include_legacy_json:
        retired += checked_legacy_results()
    retired = [path for path in retired if path.exists()]
    destinations = [(path, backup / 'Project' / path.relative_to(PROJECT)) for path in retired]
    backup.mkdir(parents=True)
    saved_metadata = backup / 'Project/backend/app/ml/component_4_thud_test/artifacts/quality_model_metadata.json'
    saved_metadata.parent.mkdir(parents=True, exist_ok=True)
    saved_metadata.write_bytes(previous_bytes)
    updated = dict(metadata, runtime_layout_version=LAYOUT,
                   code_hashes={name: sha256(COMPONENT / name) for name in RUNTIME_FILES})
    pending = metadata_path.with_suffix('.json.pending')
    pending.write_text(json.dumps(updated, indent=2, allow_nan=False), encoding='utf-8')
    pending.replace(metadata_path)
    model_loader.get_quality_model.cache_clear()
    _, reloaded = model_loader.get_quality_model()
    if reloaded['artifact_sha256'] != verified_metadata['artifact_sha256']:
        raise RuntimeError('Model artifact changed unexpectedly.')
    for source, destination in report_copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    for source, destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        print('ARCHIVED', source.relative_to(PROJECT))
    report = {
        'runtime_layout_version': LAYOUT, 'checked_recordings': len(rows),
        'preprocessing_and_inference_equivalent': True, 'model_refitted': False,
        'model_artifact_sha256': updated['artifact_sha256'],
        'code_hashes': updated['code_hashes'],
        'archived_files': [str(source.relative_to(PROJECT)) for source, _ in destinations],
        'note': 'Structural equivalence check only, not independent accuracy validation.',
    }
    output = RESEARCH / 'outputs/reports/runtime_layout_verification.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Verified {len(rows)} recordings; model bytes unchanged; runtime layout finalized.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup-dir', type=Path, required=True)
    parser.add_argument('--include-legacy-json', action='store_true')
    args = parser.parse_args()
    finalize(args.backup_dir, args.include_legacy_json)
