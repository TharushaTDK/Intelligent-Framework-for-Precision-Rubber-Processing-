"""
COMPONENT 3 - CLASSIFICATION V2 EXPERIMENT AREA - SHARED UTILITIES
====================================================================

This module is shared by every script under src/classification_v2/.

It intentionally duplicates (rather than imports from) the relevant
logic in src/30_group_cross_validation.py so that:

  - src/30_group_cross_validation.py (frozen V1 evidence) is never
    imported, modified, or re-executed as a side effect of V2 work.
  - The V2 harness can be inspected/audited independently of V1.

SCOPE / SAFETY RULES ENFORCED HERE:
  - Only Train + Validation (Development) are ever loaded.
  - data/splits/Component3_Test.csv is NEVER read by anything in this
    package. A module-level guard raises if anyone tries.
  - The regression side (Stage 19 GB_02 / drying_time_pipeline.pkl) is
    never touched anywhere in this package.
  - All preprocessing (imputation/scaling/one-hot) and all class-weight
    computation happens strictly inside each fold's TRAINING portion.
"""

from pathlib import Path
import ast
import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


# ============================================================
# 1. PATHS (ROOT = ml-research/component_3_smokehouse)
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = ROOT / "data" / "splits"
REPORT_DIR = ROOT / "outputs" / "reports"

TRAIN_PATH = SPLIT_DIR / "Component3_Train.csv"
VALIDATION_PATH = SPLIT_DIR / "Component3_Validation.csv"
TEST_PATH = SPLIT_DIR / "Component3_Test.csv"  # existence-check ONLY, never read

CLS_FEATURE_LIST_PATH = REPORT_DIR / "08_classification_selected_features.csv"
CLS_SELECTION_CONFIG_PATH = REPORT_DIR / "29_selected_classification_configuration.csv"

OUT_DIR = ROOT / "outputs" / "classification_v2"
FIG_DIR = OUT_DIR / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CONSTANTS (identical to Stage 30 / Stage 20-29)
# ============================================================

N_SPLITS = 5

CLASS_ORDER_TEXT = ["Normal", "Warning", "Critical"]

CLASS_TO_INT = {"Normal": 0, "Warning": 1, "Critical": 2}
INT_TO_CLASS = {value: key for key, value in CLASS_TO_INT.items()}

CATEGORICAL_FEATURES = ["Smokehouse_ID", "Layer_ID"]

# Predefined BEFORE any outer-fold result was inspected (Experiment A).
COST_SENSITIVE_CRITICAL_MULTIPLIERS = [1.0, 1.5, 2.0, 3.0]

# Predefined BEFORE any outer-fold result was inspected (Experiment B).
CRITICAL_THRESHOLD_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def forbid_test_access():
    """Call this nowhere except inside a test, as a guard/documentation
    anchor. The Test file is never opened by this package."""
    raise RuntimeError(
        "classification_v2 must never read data/splits/Component3_Test.csv"
    )


# ============================================================
# 3. DEVELOPMENT DATA LOADING (Train + Validation ONLY)
# ============================================================

def load_development():
    """
    Reproduces Stage 30 Sections 5-8 exactly for the classification task:
    loads Train + Validation only, recovers the frozen Stage-08 selected
    feature list, and returns everything needed to run GroupKFold.

    Component3_Test.csv existence is checked but its content is never read.
    """
    required = [TRAIN_PATH, VALIDATION_PATH, TEST_PATH, CLS_FEATURE_LIST_PATH]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found:\n{path}")

    test_exists = TEST_PATH.exists()  # existence check only

    cls_feature_df = pd.read_csv(CLS_FEATURE_LIST_PATH)
    selected_features = cls_feature_df["Feature"].astype(str).tolist()

    if len(selected_features) != 17:
        raise RuntimeError(
            f"Expected 17 Stage-08 selected features, found {len(selected_features)}."
        )

    numeric_features = [f for f in selected_features if f not in CATEGORICAL_FEATURES]

    if len(numeric_features) != 15:
        raise RuntimeError(f"Expected 15 numeric features, found {len(numeric_features)}.")

    for forbidden in [
        "Batch_ID", "Timestamp", "Fire_Risk_Assessment_Timestamp",
        "Actual_Total_Drying_Time", "Remaining_Drying_Time", "Fire_Risk_Label",
    ]:
        if forbidden in selected_features:
            raise RuntimeError(f"Leakage field unexpectedly selected: {forbidden}")

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    development = pd.concat(
        [
            train_df.assign(_Original_Split="Train"),
            validation_df.assign(_Original_Split="Validation"),
        ],
        ignore_index=True,
    )

    required_columns = selected_features + ["Batch_ID", "Fire_Risk_Label"]
    missing_columns = [c for c in required_columns if c not in development.columns]
    if missing_columns:
        raise RuntimeError(f"Development data missing required columns: {missing_columns}")

    for feature in numeric_features:
        development[feature] = pd.to_numeric(development[feature], errors="coerce")

    development["Fire_Risk_Label"] = development["Fire_Risk_Label"].astype(str).str.strip()

    unexpected_labels = sorted(
        set(development["Fire_Risk_Label"].unique()) - set(CLASS_ORDER_TEXT)
    )
    if unexpected_labels:
        raise RuntimeError(f"Unexpected Fire_Risk_Label values: {unexpected_labels}")

    development["Fire_Risk_Int"] = development["Fire_Risk_Label"].map(CLASS_TO_INT)

    groups = development["Batch_ID"].astype(str)
    unique_batches = sorted(groups.unique().tolist())

    if len(unique_batches) < N_SPLITS:
        raise RuntimeError(f"Not enough development batches for {N_SPLITS}-fold GroupKFold.")

    return {
        "development": development,
        "selected_features": selected_features,
        "numeric_features": numeric_features,
        "categorical_features": CATEGORICAL_FEATURES,
        "groups": groups,
        "unique_batches": unique_batches,
        "test_exists": test_exists,
        "test_content_read": False,
    }


# ============================================================
# 4. RECOVER THE EXACT STAGE-29 SELECTED XGBOOST CONFIG (XGB_03)
# ============================================================

def _parse_dict_like(value):
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def recover_selected_xgb_configuration():
    """Identical logic to Stage 30's recover_selected_xgb_configuration()."""
    frame = pd.read_csv(CLS_SELECTION_CONFIG_PATH)
    if frame.empty:
        raise RuntimeError("Stage-29 selected classification configuration is empty.")

    selected = frame.iloc[0]
    family = str(selected.get("Model_Family", "Unknown"))
    trial = str(selected.get("Trial", "Unknown"))

    params = {}
    for column in ["Parameters_JSON", "Best_Parameters", "Parameters", "Params"]:
        if column in frame.columns:
            params = _parse_dict_like(selected[column])
            if params:
                break

    required = ["n_estimators", "learning_rate", "max_depth"]
    missing = [p for p in required if p not in params]
    if missing:
        raise RuntimeError(f"Could not recover exact XGBoost parameters. Missing: {missing}")

    return family, trial, params


# ============================================================
# 5. PREPROCESSOR (identical structure to Stage 10 / Stage 30)
# ============================================================

def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[("onehot", make_one_hot_encoder())])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fold_sample_weight(y_fold_train, critical_multiplier=1.0):
    """
    Balanced sample weights computed from THIS fold's training labels only
    (never from validation/outer-test). critical_multiplier additionally
    scales the Critical-class component (Experiment A); 1.0 reproduces the
    exact Stage-30/Stage-28 balanced-weighting recipe unchanged.
    """
    weights = compute_sample_weight(class_weight="balanced", y=y_fold_train)
    if critical_multiplier != 1.0:
        critical_mask = np.asarray(y_fold_train) == CLASS_TO_INT["Critical"]
        weights = weights.copy()
        weights[critical_mask] = weights[critical_mask] * critical_multiplier
    return weights


# ============================================================
# 6. METRICS (NaN convention: recall/F1/F2 are NaN when TRUE support = 0;
#    precision is NaN when PREDICTED count = 0 for that class)
# ============================================================

def safe_recall(y_true, y_pred, class_id):
    support = int(np.sum(np.asarray(y_true) == class_id))
    if support == 0:
        return np.nan
    return float(recall_score(y_true, y_pred, labels=[class_id], average="macro", zero_division=0))


def safe_precision(y_true, y_pred, class_id):
    predicted_count = int(np.sum(np.asarray(y_pred) == class_id))
    if predicted_count == 0:
        return np.nan
    return float(precision_score(y_true, y_pred, labels=[class_id], average="macro", zero_division=0))


def safe_f1(y_true, y_pred, class_id):
    support = int(np.sum(np.asarray(y_true) == class_id))
    if support == 0:
        return np.nan
    return float(f1_score(y_true, y_pred, labels=[class_id], average="macro", zero_division=0))


def safe_fbeta(y_true, y_pred, class_id, beta=2.0):
    support = int(np.sum(np.asarray(y_true) == class_id))
    if support == 0:
        return np.nan
    return float(
        fbeta_score(y_true, y_pred, labels=[class_id], beta=beta, average="macro", zero_division=0)
    )


def supported_macro_f1(y_true, y_pred):
    supported = sorted(np.unique(np.asarray(y_true)).tolist())
    return float(f1_score(y_true, y_pred, labels=supported, average="macro", zero_division=0))


def supported_balanced_accuracy(y_true, y_pred):
    supported = sorted(np.unique(np.asarray(y_true)).tolist())
    return float(recall_score(y_true, y_pred, labels=supported, average="macro", zero_division=0))


def full_classification_metrics(y_true, y_pred, prefix=""):
    """Every metric required by the experiment spec, for one (sub)set of rows."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    out = {
        f"{prefix}Accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}Balanced_Accuracy_AllThree": float(
            balanced_accuracy_score(y_true, y_pred)
        ) if len(np.unique(y_true)) == 3 else supported_balanced_accuracy(y_true, y_pred),
        f"{prefix}Supported_Balanced_Accuracy": supported_balanced_accuracy(y_true, y_pred),
        f"{prefix}Macro_F1_AllThree": float(
            f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        f"{prefix}Supported_Macro_F1": supported_macro_f1(y_true, y_pred),
        f"{prefix}Weighted_F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        f"{prefix}Normal_Recall": safe_recall(y_true, y_pred, 0),
        f"{prefix}Normal_Precision": safe_precision(y_true, y_pred, 0),
        f"{prefix}Warning_Recall": safe_recall(y_true, y_pred, 1),
        f"{prefix}Warning_Precision": safe_precision(y_true, y_pred, 1),
        f"{prefix}Warning_F1": safe_f1(y_true, y_pred, 1),
        f"{prefix}Critical_Recall": safe_recall(y_true, y_pred, 2),
        f"{prefix}Critical_Precision": safe_precision(y_true, y_pred, 2),
        f"{prefix}Critical_F1": safe_f1(y_true, y_pred, 2),
        f"{prefix}Critical_F2": safe_fbeta(y_true, y_pred, 2, beta=2.0),
        f"{prefix}Actual_Normal_Count": int(np.sum(y_true == 0)),
        f"{prefix}Actual_Warning_Count": int(np.sum(y_true == 1)),
        f"{prefix}Actual_Critical_Count": int(np.sum(y_true == 2)),
        f"{prefix}Predicted_Normal_Count": int(np.sum(y_pred == 0)),
        f"{prefix}Predicted_Warning_Count": int(np.sum(y_pred == 1)),
        f"{prefix}Predicted_Critical_Count": int(np.sum(y_pred == 2)),
    }
    return out


def confusion_df(y_true, y_pred, labels=(0, 1, 2), label_names=("Normal", "Warning", "Critical")):
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    return pd.DataFrame(
        cm,
        index=[f"Actual_{n}" for n in label_names],
        columns=[f"Predicted_{n}" for n in label_names],
    )


def inner_group_kfold_splits(outer_train_idx, group_values, n_splits=4):
    """
    Shared nested-CV mechanic for the final round (Experiments D and E).

    Yields (inner_train_idx, inner_val_idx) as GLOBAL row-index arrays
    (positions into the full X_raw/y_cls arrays passed by the caller),
    computed via GroupKFold restricted to the batches present inside
    outer_train_idx. Implemented once here so every nested threshold-
    selection routine in this round uses identical inner-CV mechanics.

    If fewer than 2 distinct batches are available inside outer_train_idx,
    nothing is yielded; the caller must handle that (documented) fallback.
    """
    from sklearn.model_selection import GroupKFold

    outer_train_idx = np.asarray(outer_train_idx)
    inner_groups_all = group_values[outer_train_idx]
    unique_inner_groups = sorted(set(inner_groups_all))
    n_inner = min(n_splits, len(unique_inner_groups))

    if n_inner < 2:
        return

    inner_kfold = GroupKFold(n_splits=n_inner)
    local_positions = np.arange(len(outer_train_idx))

    for inner_train_local, inner_val_local in inner_kfold.split(
        local_positions, groups=inner_groups_all
    ):
        yield outer_train_idx[inner_train_local], outer_train_idx[inner_val_local]


def scale_pos_weight_from_labels(y_binary):
    """
    XGBoost-style scale_pos_weight = n_negative / n_positive, computed from
    the given (training-fold-only) binary labels. Returns 1.0 if there are
    no positive examples in this fold (documented edge case for the caller).
    """
    y_binary = np.asarray(y_binary)
    n_pos = int(np.sum(y_binary == 1))
    n_neg = int(np.sum(y_binary == 0))
    if n_pos == 0:
        return 1.0
    return float(n_neg) / float(n_pos)


def batch_classification_metrics(frame, y_true_col="y_true_int", y_pred_col="y_pred_int"):
    rows = []
    for batch_id, batch in frame.groupby("Batch_ID", sort=True):
        y_true = batch[y_true_col].to_numpy()
        y_pred = batch[y_pred_col].to_numpy()
        row = {"Batch_ID": batch_id, "Rows": len(batch)}
        row.update(full_classification_metrics(y_true, y_pred))
        rows.append(row)
    return pd.DataFrame(rows)
