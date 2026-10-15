from pathlib import Path
from collections import Counter
from itertools import permutations

import pandas as pd


# ============================================================
# COMPONENT 3 - STAGE 30 INTEGRITY AUDIT
# ============================================================
#
# Purpose:
#   Verify whether the model-ready classification Train data
#   was resampled/duplicated or otherwise changed before
#   GroupKFold cross-validation.
#
# IMPORTANT:
#   - Reads Train + Validation only.
#   - DOES NOT load Component3_Test.csv.
#   - Does not modify any data.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

MODEL_READY_DIR = ROOT / "data" / "model_ready"
SPLIT_DIR = ROOT / "data" / "splits"
REPORT_DIR = ROOT / "outputs" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

X_TRAIN_PATH = MODEL_READY_DIR / "Component3_X_Train.csv"
X_VAL_PATH = MODEL_READY_DIR / "Component3_X_Validation.csv"

Y_TRAIN_PATH = MODEL_READY_DIR / "Component3_y_classification_Train.csv"
Y_VAL_PATH = MODEL_READY_DIR / "Component3_y_classification_Validation.csv"

META_TRAIN_PATH = MODEL_READY_DIR / "Component3_metadata_Train.csv"
META_VAL_PATH = MODEL_READY_DIR / "Component3_metadata_Validation.csv"

SPLIT_TRAIN_PATH = SPLIT_DIR / "Component3_Train.csv"
SPLIT_VAL_PATH = SPLIT_DIR / "Component3_Validation.csv"
TEST_PATH = SPLIT_DIR / "Component3_Test.csv"

REPORT_PATH = REPORT_DIR / "30_cv_data_integrity_audit.txt"


def read_single_target(path):
    df = pd.read_csv(path)

    if "Fire_Risk_Label" in df.columns:
        return df["Fire_Risk_Label"]

    if df.shape[1] == 1:
        return df.iloc[:, 0]

    raise ValueError(
        f"Could not identify classification target in {path.name}. "
        f"Columns: {list(df.columns)}"
    )


def counts(series):
    return dict(
        pd.Series(series)
        .value_counts(dropna=False)
        .sort_index()
    )


def infer_numeric_mapping(model_ready_series, raw_label_series):
    """
    Infer numeric target mapping by matching Validation class counts.
    This avoids assuming 0/1/2 meaning.
    """
    mr = pd.to_numeric(model_ready_series, errors="coerce")

    if mr.isna().any():
        return None

    numeric_values = sorted(mr.astype(int).unique().tolist())
    labels = ["Normal", "Warning", "Critical"]

    if len(numeric_values) != 3:
        return None

    raw_counts = raw_label_series.astype(
        str).str.strip().value_counts().to_dict()
    numeric_counts = mr.astype(int).value_counts().to_dict()

    for label_perm in permutations(labels):
        mapping = dict(zip(numeric_values, label_perm))

        mapped_counts = {
            mapping[k]: int(v)
            for k, v in numeric_counts.items()
        }

        if all(
            mapped_counts.get(label, 0) == raw_counts.get(label, 0)
            for label in labels
        ):
            return mapping

    return None


def detect_string_label_mapping(model_ready_series, label_order):
    """
    Detect whether the model-ready target is ALREADY stored as the raw
    string labels (current Stage-10 format: Normal/Warning/Critical)
    rather than a numeric 0/1/2 encoding. If so, the "mapping" is simply
    the identity on those labels, and infer_numeric_mapping() must not be
    used (it assumes a numeric series and would incorrectly return None).
    """
    values = set(
        pd.Series(model_ready_series)
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    if values and values.issubset(set(label_order)):
        return {label: label for label in label_order}

    return None


# Documented runtime numeric encoding used by Stage 20/26/28/29 (hard-coded
# LABEL_TO_INT) and exported in
# backend/app/ml/component_3_smokehouse/component3_inference_contract.json
# ("class_id_to_label"). Reported here for cross-check only -- Stage-10
# model-ready target files are string-labeled, not numerically encoded, so
# this mapping is informational and is not applied to them.
RUNTIME_LABEL_TO_INT = {
    "Normal": 0,
    "Warning": 1,
    "Critical": 2,
}


print("=" * 78)
print("COMPONENT 3 - STAGE 30: CV DATA INTEGRITY AUDIT")
print("=" * 78)

required = [
    X_TRAIN_PATH,
    X_VAL_PATH,
    Y_TRAIN_PATH,
    Y_VAL_PATH,
    META_TRAIN_PATH,
    META_VAL_PATH,
    TEST_PATH,
]

for p in required:
    if not p.exists():
        raise FileNotFoundError(f"Required file not found:\n{p}")

# Final Test existence only.
test_content_read = False

print("\n--- TEST PROTECTION ---")
print("Final Test exists:", TEST_PATH.exists())
print("Component3_Test.csv content read:", test_content_read)

X_train = pd.read_csv(X_TRAIN_PATH)
X_val = pd.read_csv(X_VAL_PATH)
y_train = read_single_target(Y_TRAIN_PATH)
y_val = read_single_target(Y_VAL_PATH)
meta_train = pd.read_csv(META_TRAIN_PATH)
meta_val = pd.read_csv(META_VAL_PATH)

print("\n--- MODEL-READY FILES ---")
print(f"X Train rows      : {len(X_train):,}")
print(f"X Validation rows : {len(X_val):,}")
print(f"y Train rows      : {len(y_train):,}")
print(f"y Validation rows : {len(y_val):,}")
print(f"Meta Train rows   : {len(meta_train):,}")
print(f"Meta Val rows     : {len(meta_val):,}")

print("\nModel-ready y Train counts:")
print(y_train.value_counts(dropna=False).sort_index())

print("\nModel-ready y Validation counts:")
print(y_val.value_counts(dropna=False).sort_index())

# Metadata duplicate-key checks.
possible_key_columns = [
    c for c in [
        "Smokehouse_ID",
        "Batch_ID",
        "Layer_ID",
        "Timestamp",
        "Elapsed_Drying_Hours",
    ]
    if c in meta_train.columns
]

print("\n--- TRAIN METADATA DUPLICATION ---")

if possible_key_columns:
    print("Available metadata key columns:")
    print(possible_key_columns)

    full_key = [
        c for c in [
            "Smokehouse_ID",
            "Batch_ID",
            "Layer_ID",
            "Timestamp",
        ]
        if c in meta_train.columns
    ]

    if len(full_key) >= 3:
        dup_count = int(
            meta_train.duplicated(
                subset=full_key,
                keep=False
            ).sum()
        )

        unique_count = int(
            meta_train.drop_duplicates(
                subset=full_key
            ).shape[0]
        )

        print(
            f"Train rows involved in duplicate observation keys: {dup_count:,}")
        print(
            f"Unique Train observation keys                : {unique_count:,}")
    else:
        print("Not enough metadata columns for observation-key duplicate check.")
else:
    print("No suitable metadata key columns found.")

split_files_available = (
    SPLIT_TRAIN_PATH.exists()
    and SPLIT_VAL_PATH.exists()
)

print("\n--- ORIGINAL SPLIT FILES ---")
print("Component3_Train.csv exists      :", SPLIT_TRAIN_PATH.exists())
print("Component3_Validation.csv exists :", SPLIT_VAL_PATH.exists())

status = "INCONCLUSIVE"

report_lines = []

if split_files_available:
    split_train = pd.read_csv(SPLIT_TRAIN_PATH)
    split_val = pd.read_csv(SPLIT_VAL_PATH)

    if "Fire_Risk_Label" not in split_train.columns:
        raise ValueError(
            "Fire_Risk_Label missing from Component3_Train.csv"
        )

    if "Fire_Risk_Label" not in split_val.columns:
        raise ValueError(
            "Fire_Risk_Label missing from Component3_Validation.csv"
        )

    raw_train_labels = (
        split_train["Fire_Risk_Label"]
        .astype(str)
        .str.strip()
    )

    raw_val_labels = (
        split_val["Fire_Risk_Label"]
        .astype(str)
        .str.strip()
    )

    print(f"\nOriginal split Train rows      : {len(split_train):,}")
    print(f"Original split Validation rows : {len(split_val):,}")

    print("\nOriginal Train class counts:")
    print(
        raw_train_labels.value_counts()
        .reindex(["Normal", "Warning", "Critical"], fill_value=0)
    )

    print("\nOriginal Validation class counts:")
    print(
        raw_val_labels.value_counts()
        .reindex(["Normal", "Warning", "Critical"], fill_value=0)
    )

    # Detect target encoding from untouched Validation. Stage-10 currently
    # saves the RAW STRING labels directly (Normal/Warning/Critical), not a
    # numeric 0/1/2 encoding -- check for that format first, and only fall
    # back to numeric-mapping inference if the target is not string-labeled.
    labels_order = ["Normal", "Warning", "Critical"]

    string_mapping = detect_string_label_mapping(
        y_val,
        labels_order,
    )

    if string_mapping is not None:
        mapping = string_mapping

        print(
            "\nModel-ready target format: raw string labels "
            "(no numeric encoding present)."
        )
        print("Identity mapping used for comparison:")
        for label in labels_order:
            print(f"  '{label}' -> '{label}'")

        mapped_train = (
            pd.Series(y_train)
            .astype(str)
            .str.strip()
        )

        mapped_val = (
            pd.Series(y_val)
            .astype(str)
            .str.strip()
        )

    else:
        mapping = infer_numeric_mapping(
            y_val,
            raw_val_labels,
        )

        if mapping is not None:
            print("\nInferred model-ready numeric class mapping:")
            for numeric, label in mapping.items():
                print(f"  {numeric} -> {label}")

            y_train_num = pd.to_numeric(
                y_train,
                errors="raise"
            ).astype(int)

            y_val_num = pd.to_numeric(
                y_val,
                errors="raise"
            ).astype(int)

            mapped_train = y_train_num.map(mapping)
            mapped_val = y_val_num.map(mapping)

    if mapping is not None:
        print(
            "\nCross-check vs documented runtime numeric encoding "
            "(Stage-26 LABEL_TO_INT / component3_inference_contract.json):"
        )
        for label in labels_order:
            print(f"  {label} -> {RUNTIME_LABEL_TO_INT[label]}")

        mapped_train_counts = (
            mapped_train.value_counts()
            .reindex(
                ["Normal", "Warning", "Critical"],
                fill_value=0,
            )
        )

        mapped_val_counts = (
            mapped_val.value_counts()
            .reindex(
                ["Normal", "Warning", "Critical"],
                fill_value=0,
            )
        )

        raw_train_counts = (
            raw_train_labels.value_counts()
            .reindex(
                ["Normal", "Warning", "Critical"],
                fill_value=0,
            )
        )

        raw_val_counts = (
            raw_val_labels.value_counts()
            .reindex(
                ["Normal", "Warning", "Critical"],
                fill_value=0,
            )
        )

        train_same = (
            len(mapped_train) == len(raw_train_labels)
            and mapped_train_counts.equals(raw_train_counts)
        )

        val_same = (
            len(mapped_val) == len(raw_val_labels)
            and mapped_val_counts.equals(raw_val_counts)
        )

        print("\n--- MODEL-READY VS ORIGINAL SPLIT ---")
        print("Validation target unchanged :", val_same)
        print("Training target unchanged   :", train_same)

        if val_same and train_same:
            status = "NO RESAMPLING DETECTED"
            explanation = (
                "Model-ready Train and Validation class counts match "
                "the original batch-aware split."
            )

        elif val_same and not train_same:
            status = "TRAINING RESAMPLING / DUPLICATION DETECTED"
            explanation = (
                "Validation target matches the original split, but "
                "Training target does not. Group CV must therefore be "
                "rebuilt from the original unresampled Train + Validation "
                "data and any weighting/resampling must occur inside each "
                "training fold."
            )

        else:
            status = "TARGET / SPLIT INCONSISTENCY DETECTED"
            explanation = (
                "Validation model-ready target does not match the original "
                "Validation split. Target encoding or preprocessing must be "
                "reviewed before further model evaluation."
            )

        print("\nAUDIT STATUS:", status)
        print(explanation)

        report_lines.extend([
            f"Validation target unchanged: {val_same}",
            f"Training target unchanged: {train_same}",
            f"Audit status: {status}",
            explanation,
        ])

    else:
        status = "CLASS MAPPING COULD NOT BE DETERMINED"
        explanation = (
            "Validation target is neither the recognized string-label "
            "format (Normal/Warning/Critical) nor a numeric encoding that "
            "matches the original split's class counts. Review Stage-10/"
            "Stage-20 target encoding before continuing."
        )

        print("\nAUDIT STATUS:", status)
        print(explanation)

        report_lines.extend([
            f"Audit status: {status}",
            explanation,
        ])

else:
    status = "ORIGINAL TRAIN/VALIDATION SPLIT FILES NOT FOUND"
    explanation = (
        "The audit could not compare model-ready targets with the "
        "original split files. Do not proceed to Stage 31 until the "
        "classification data lineage is confirmed."
    )

    print("\nAUDIT STATUS:", status)
    print(explanation)

    report_lines.extend([
        f"Audit status: {status}",
        explanation,
    ])

with REPORT_PATH.open("w", encoding="utf-8") as report:
    report.write(
        "COMPONENT 3 - STAGE 30 CV DATA INTEGRITY AUDIT\n"
    )
    report.write("=" * 78 + "\n\n")
    report.write(
        f"Model-ready Train rows: {len(X_train):,}\n"
    )
    report.write(
        f"Model-ready Validation rows: {len(X_val):,}\n"
    )
    report.write(
        "Final Test contents loaded: False\n\n"
    )

    for line in report_lines:
        report.write(str(line) + "\n")

print("\n--- FINAL TEST PROTECTION ---")
print(
    "Component3_Test.csv content read:",
    test_content_read,
)

print("\nReport saved to:")
print(REPORT_PATH)
