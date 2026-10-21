# Component 4 — acoustic recording quality pilot

The mobile app calls the shared FastAPI backend. The backend captures or accepts a WAV, checks its technical quality, extracts the strike, evaluates the preliminary quality classifier and stores the result in PostgreSQL. Moisture estimation is not implemented; `moisture_percent` remains null.

## Runtime responsibilities

| File under backend/app/ml/component_4_thud_test | Responsibility |
| --- | --- |
| preprocessing.py | PCM reading, inspection, strike segmentation, technical rules and 16 acoustic features |
| feature_contract.py | Ordered model input names |
| model_loader.py | Cached pipeline loading; artifact, processing-source, version and feature-order checks |
| inference.py | Technical rules followed by classifier decision; file/folder inspection CLI |
| artifacts/quality_pipeline.joblib | Existing fitted StandardScaler + logistic classifier |
| artifacts/quality_model_metadata.json | Model identity, hashes, thresholds and pilot status |

The separate service, route, schema and database model retain their existing responsibilities. There is no change to the mobile API or database schema in this update.

## Research responsibilities

| Path | Contents |
| --- | --- |
| data/raw/ | Existing 60 input WAVs, unchanged |
| data/metadata/recordings.csv | Annotations for 12 collected recordings |
| data/metadata/dataset_manifest.csv | All inputs, collected/derived origin, parent identity, hashes and transforms |
| data/processed/quality_features.csv | 60 feature rows including technical status |
| data/model_ready/quality_features.csv | 53 rule-passed rows; numeric values remain unscaled |
| data/splits/folds.json and fold_01..04/ | Sheet-grouped training, held-out original and excluded rows |
| src/preprocessing.py | Verify inputs and regenerate features and grouped splits using runtime preprocessing |
| src/train_quality.py | Evaluate grouped folds and deliberately refit the final runtime pipeline |
| src/export_models.py | Export existing fitted pipeline, scaler and classifier without fitting |
| src/tools/ | Audio review, legacy database import and verified layout migration utilities |
| artifacts/preprocessing/ | Exported fitted scaler and its metadata |
| artifacts/models/classification/ | Exported fitted classifier and its metadata |
| artifacts/pipelines/ | Byte-identical snapshot of the complete runtime pipeline |
| outputs/reports/ | Pilot metrics, out-of-fold predictions, export and layout verification reports |

The scaler lives inside the runtime pipeline so inference cannot accidentally skip scaling. Research exports expose its fitted parameters for inspection. These are snapshots of the same model, not additional models or validation results.

## Commands

Run from `Project`, using `..\.venv\Scripts\python.exe` on Windows:

```powershell
..\.venv\Scripts\python.exe .\ml-research\component_4_thud_test\src\preprocessing.py
..\.venv\Scripts\python.exe .\ml-research\component_4_thud_test\src\export_models.py
..\.venv\Scripts\python.exe -m backend.app.ml.component_4_thud_test.inference .\ml-research\component_4_thud_test\data\raw\S001_C01_T01.wav
```

Training is a separate deliberate operation; it replaces the runtime model and regenerates pilot evaluation reports:

```powershell
..\.venv\Scripts\python.exe .\ml-research\component_4_thud_test\src\train_quality.py
```

Do not train just to apply the layout update. Run the provided finalizer first while the previous processing files still exist. It checks function equivalence, all manifest hashes, features and decisions before updating source hashes and archiving retired files outside the repository. It never refits the model. See `APPLY_FINAL_STRUCTURE.md` in the delivery ZIP.

## Pilot scope

There are 12 collected recordings from four independent sheets and 48 derived variants. Variants retain their parent labels and do not constitute new independent sheets. Keep these origin fields when presenting dataset counts.

Evaluation holds out an entire sheet and scores its original recordings only. Variants of the held-out sheet are excluded. A fresh scaler is fitted on training rows within each fold. The deployed final pipeline is then refitted on all 53 rule-passed pilot rows; its exported scaler is consequently the final-fit scaler, not one fold's scaler.

Saved pilot complete-workflow accuracy is 0.5833 and invalid F1 is 0.4444. Rules-only accuracy is 0.75 and invalid F1 is 0.4. These small-pilot results do not establish reliable generalization. Tests and serialization checks verify software behavior, not model accuracy. No separate independent validation set or moisture model is claimed.

The organization follows Component 3's separation of preprocessing, model loading, fitted artifacts and research outputs. It retains Component 4's own acoustic logic and classifier.
