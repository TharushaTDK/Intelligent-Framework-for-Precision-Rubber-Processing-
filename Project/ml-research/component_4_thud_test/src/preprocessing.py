"""Organize existing audio inputs and export traceable sheet-grouped splits.

Copies existing derived recordings into raw/ without changing their bytes.
Never generates recordings, relabels their origin, deletes files or fits a scaler.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from backend.app.ml.component_4_thud_test.feature_contract import FEATURE_NAMES
from backend.app.ml.component_4_thud_test.preprocessing import extract_features


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, fieldnames=None):
    if not rows and fieldnames is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def basename(name):
    if not name or Path(name).name != name or '/' in name or '\\' in name:
        raise ValueError('Dataset filenames must be basenames.')
    return name


def prepare_dataset(research):
    research = Path(research).resolve()
    data = research / 'data'
    raw = data / 'raw'
    annotations = read_csv(data / 'metadata' / 'recordings.csv')
    if not annotations:
        raise ValueError('No collected-recording annotations.')
    parents = {}
    rows = []
    for annotation in annotations:
        identifier = annotation['recording_id']
        if identifier in parents or annotation['source_type'] != 'COLLECTED':
            raise ValueError('Collected metadata contains duplicate IDs or non-collected rows.')
        if annotation['quality_label'] not in ('VALID', 'INVALID'):
            raise ValueError(f'Quality annotation is incomplete: {identifier}')
        path = raw / basename(annotation['filename'])
        digest = sha256(path)
        row = {
            'recording_id': identifier, 'sheet_id': annotation['sheet_id'],
            'condition_id': annotation['condition_id'], 'filename': path.name,
            'quality_label': annotation['quality_label'], 'source_type': 'COLLECTED',
            'parent_recording_id': '', 'sha256': digest, 'parent_sha256': '',
            'transformation': '', 'shift_seconds': '', 'noise_sigma': '',
        }
        if not row['sheet_id'] or not row['condition_id']:
            raise ValueError('Collected recordings require sheet and condition IDs.')
        rows.append(row)
        parents[identifier] = row

    manifest_path = data / 'metadata' / 'dataset_manifest.csv'
    legacy_path = data / 'metadata' / 'augmented_recordings.csv'
    if manifest_path.exists():
        derived = [r for r in read_csv(manifest_path) if r['source_type'] == 'DERIVED']
        legacy = False
    elif legacy_path.exists():
        derived = read_csv(legacy_path)
        legacy = True
    else:
        derived, legacy = [], False

    known_hashes = {}
    legacy_features = data / 'features' / 'quality_features_with_augmentation.csv'
    if legacy_features.exists():
        known_hashes = {r['recording_id']: r['sha256'] for r in read_csv(legacy_features)}

    identifiers = set(parents)
    filenames = {r['filename'] for r in rows}
    for variant in derived:
        identifier = variant['recording_id']
        parent = parents[variant['parent_recording_id']]
        if identifier in identifiers:
            raise ValueError('Duplicate dataset recording ID.')
        for key in ('sheet_id', 'condition_id', 'quality_label'):
            if variant[key] != parent[key]:
                raise ValueError(f'Derived recording has inconsistent parent metadata: {identifier}')
        expected_parent_hash = variant['source_sha256'] if legacy else variant['parent_sha256']
        if expected_parent_hash != parent['sha256']:
            raise ValueError(f'Parent WAV hash differs: {identifier}')
        filename = basename(variant['filename'])
        if filename in filenames:
            raise ValueError('Duplicate dataset filename.')
        destination = raw / filename
        old = data / 'augmented' / filename
        expected_hash = known_hashes.get(identifier) if legacy else variant['sha256']
        if legacy and not expected_hash:
            raise ValueError(f'No recorded output hash for legacy variant: {identifier}')
        if destination.exists():
            if sha256(destination) != expected_hash:
                raise ValueError(f'Existing destination differs; refusing overwrite: {filename}')
        else:
            if not old.is_file() or sha256(old) != expected_hash:
                raise ValueError(f'Existing variant missing or changed: {filename}')
            shutil.copyfile(old, destination)
        rows.append({
            'recording_id': identifier, 'sheet_id': parent['sheet_id'],
            'condition_id': parent['condition_id'], 'filename': filename,
            'quality_label': parent['quality_label'], 'source_type': 'DERIVED',
            'parent_recording_id': parent['recording_id'], 'sha256': expected_hash,
            'parent_sha256': parent['sha256'],
            'transformation': variant.get('augmentation', '') if legacy else variant['transformation'],
            'shift_seconds': variant['shift_seconds'], 'noise_sigma': variant['noise_sigma'],
        })
        identifiers.add(identifier)
        filenames.add(filename)

    unknown = {p.name for p in raw.glob('*.wav')} - filenames
    if unknown:
        raise ValueError(f'WAV files need metadata before use: {sorted(unknown)}')
    original_hashes = [r['sha256'] for r in rows if r['source_type'] == 'COLLECTED']
    if len(original_hashes) != len(set(original_hashes)):
        raise ValueError('Duplicate original WAV contents; review provenance.')

    feature_rows = []
    for row in rows:
        features, checks = extract_features(raw / row['filename'])
        feature_row = {
            'recording_id': row['recording_id'], 'sheet_id': row['sheet_id'],
            'condition_id': row['condition_id'], 'quality_label': row['quality_label'],
            'source_type': row['source_type'], 'parent_recording_id': row['parent_recording_id'],
            'technical_status': checks['technical_status'], 'has_features': features is not None,
            'sha256': row['sha256'],
        }
        feature_row.update(features or {name: None for name in FEATURE_NAMES})
        feature_rows.append(feature_row)

    write_csv(manifest_path, rows)
    write_csv(data / 'processed' / 'quality_features.csv', feature_rows)
    ready = [r for r in feature_rows if r['technical_status'] == 'TECHNICAL_PASSED' and r['has_features']]
    write_csv(data / 'model_ready' / 'quality_features.csv', ready, fieldnames=list(feature_rows[0]))

    sheets = sorted({r['sheet_id'] for r in rows if r['source_type'] == 'COLLECTED'})
    if len(sheets) < 3:
        raise ValueError('At least three independent sheets are needed for this pilot split procedure.')
    fold_report = []
    for fold, held_sheet in enumerate(sheets, 1):
        training, testing, excluded = [], [], []
        for row in feature_rows:
            if row['sheet_id'] == held_sheet:
                (testing if row['source_type'] == 'COLLECTED' else excluded).append(row)
            elif row['technical_status'] == 'TECHNICAL_PASSED' and row['has_features']:
                training.append(row)
            else:
                excluded.append(row)
        if set(r['sheet_id'] for r in training) & set(r['sheet_id'] for r in testing):
            raise AssertionError('Sheet leakage in split export.')
        directory = data / 'splits' / f'fold_{fold:02d}'
        write_csv(directory / 'train.csv', training, fieldnames=list(feature_rows[0]))
        write_csv(directory / 'test.csv', testing, fieldnames=list(feature_rows[0]))
        write_csv(directory / 'excluded.csv', excluded, fieldnames=list(feature_rows[0]))
        fold_report.append({
            'fold': fold, 'training_sheets': sorted({r['sheet_id'] for r in training}),
            'held_out_sheets': [held_sheet], 'training_rows': len(training),
            'test_original_rows': len(testing), 'excluded_rows': len(excluded),
        })
    report = {
        'evaluation': 'leave_one_sheet_out; score held-out originals only',
        'dataset_manifest_sha256': sha256(manifest_path),
        'collected_recordings': len(parents), 'derived_recordings': len(derived),
        'total_input_recordings': len(rows), 'model_ready_rows': len(ready),
        'independent_sheets': len(sheets), 'features_are_scaled': False, 'folds': fold_report,
        'validation_note': 'No extra validation set is used; hyperparameters are fixed. Scaler fits only inside each training fold.',
    }
    split_report = data / 'splits' / 'folds.json'
    split_report.parent.mkdir(parents=True, exist_ok=True)
    split_report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f"Input recordings: {len(rows)} ({len(parents)} collected, {len(derived)} derived)")
    print(f'Numeric features: {len(FEATURE_NAMES)}; model-ready rows: {len(ready)}')
    print(f'Sheet-grouped folds saved: {len(sheets)}')
    print('Original WAVs and existing model artifacts were not changed.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-dir', type=Path,
                        default=PROJECT / 'ml-research' / 'component_4_thud_test')
    args = parser.parse_args()
    prepare_dataset(args.research_dir)
