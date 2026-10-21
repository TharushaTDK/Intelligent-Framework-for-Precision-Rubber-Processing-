"""Export the existing verified pilot pipeline and its fitted components.

No fitting or tuning occurs. The backend pipeline is the source artifact;
research exports are reproducible snapshots, not new independently trained models.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from backend.app.ml.component_4_thud_test import model_loader
from backend.app.ml.component_4_thud_test.feature_contract import FEATURE_NAMES


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def export_models(research=None, runtime=None):
    research = Path(research) if research is not None else PROJECT / 'ml-research/component_4_thud_test'
    runtime = Path(runtime) if runtime is not None else model_loader.COMPONENT
    if runtime.resolve() != model_loader.COMPONENT.resolve():
        raise ValueError('Export source must match the verified runtime model location.')
    pipeline, metadata = model_loader.get_quality_model()
    source = runtime / 'artifacts' / metadata['artifact']
    if sha256(source) != metadata['artifact_sha256']:
        raise ValueError('Source artifact changed; refusing export.')
    if metadata['feature_names'] != FEATURE_NAMES:
        raise ValueError('Export feature order differs from the runtime contract.')
    if set(pipeline.named_steps) != {'scale', 'classifier'}:
        raise ValueError('Expected the existing scale/classifier pilot pipeline.')
    scaler = pipeline.named_steps['scale']
    classifier = pipeline.named_steps['classifier']
    artifacts = research / 'artifacts'
    paths = {
        'pipeline': artifacts / 'pipelines/quality_pipeline.joblib',
        'preprocessor': artifacts / 'preprocessing/quality_scaler.joblib',
        'classifier': artifacts / 'models/classification/quality_classifier.joblib',
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, paths['pipeline'])
    joblib.dump(scaler, paths['preprocessor'], compress=3)
    joblib.dump(classifier, paths['classifier'], compress=3)
    if sha256(paths['pipeline']) != metadata['artifact_sha256']:
        raise RuntimeError('Pipeline snapshot hash differs from the source.')
    # Checks only serialization/composition. These inputs are not accuracy tests.
    probes = np.vstack((scaler.mean_, scaler.mean_ + 0.01 * scaler.scale_))
    saved_pipeline = joblib.load(paths['pipeline'])
    saved_scaler = joblib.load(paths['preprocessor'])
    saved_classifier = joblib.load(paths['classifier'])
    expected = pipeline.predict_proba(probes)
    np.testing.assert_allclose(saved_pipeline.predict_proba(probes), expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(saved_classifier.predict_proba(saved_scaler.transform(probes)), expected,
                               rtol=0, atol=1e-12)
    common = {
        'model_version': metadata['model_version'], 'status': 'PRELIMINARY_DEMO',
        'source_pipeline_sha256': metadata['artifact_sha256'],
        'feature_names': FEATURE_NAMES, 'fit_rows': metadata['fit_rows'],
        'fit_scope': 'ALL_RULE_PASSED_PILOT_ROWS_FINAL_REFIT',
        'refitted_during_export': False,
        'evaluation_note': 'Use saved sheet-grouped OOF metrics; component reload checks are not accuracy validation.',
        'sklearn_version': metadata['sklearn_version'],
    }
    write_json(paths['pipeline'].with_name('quality_pipeline_metadata.json'), metadata)
    write_json(paths['preprocessor'].with_name('quality_scaler_metadata.json'), {
        **common, 'artifact_sha256': sha256(paths['preprocessor']),
        'class': type(scaler).__name__, 'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(), 'variance': scaler.var_.tolist(),
    })
    write_json(paths['classifier'].with_name('quality_classifier_metadata.json'), {
        **common, 'artifact_sha256': sha256(paths['classifier']),
        'class': type(classifier).__name__, 'classes': list(classifier.classes_),
        'input_requirement': 'Use the matching fitted scaler first; prefer the complete pipeline for inference.',
    })
    report = {
        **common,
        'artifacts': {key: {'path': str(path.relative_to(research)), 'sha256': sha256(path)}
                      for key, path in paths.items()},
        'pipeline_copy_matches_source': True, 'component_reload_equivalence': True,
    }
    write_json(research / 'outputs/reports/model_export_report.json', report)
    print('Research pipeline, fitted preprocessor and classifier snapshots exported; no refit performed.')
    return report


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    export_models()
