"""
Component 3 - Ordinal XGBoost Classifier with Argmax Decision Rule
====================================================================

Stable, importable module for the promoted Fire_Risk_Label classifier.

This module intentionally has NO dependency on any `app.*` package path
(backend FastAPI package structure does not exist yet at the time this was
written - no __init__.py files anywhere under backend/). To load a
serialized OrdinalFireRiskPipeline instance in ANY Python process (research
or backend), the caller must:

    import sys
    sys.path.insert(0, "<directory containing this file>")
    import ordinal_fire_risk_pipeline  # registers the class for unpickling
    import joblib
    model = joblib.load("ordinal_fire_risk_pipeline.joblib")

Architecture (research-frozen, do not modify without a new research round):

    shared frozen preprocessing (median impute + StandardScaler for numeric,
    OneHotEncoder for categorical - identical to Stage-10/Stage-30 recipe)
        -> O1: XGBoost binary classifier, P(Y >= Warning)
               target: Normal=0, Warning=1, Critical=1
        -> O2: XGBoost binary classifier, P(Y >= Critical)
               target: Normal=0, Warning=0, Critical=1
        -> monotonicity correction: p_ge_critical_corrected =
               min(p_ge_critical_raw, p_ge_warning)
        -> probability reconstruction:
               P(Normal)   = 1 - p_ge_warning
               P(Warning)  = p_ge_warning - p_ge_critical_corrected
               P(Critical) = p_ge_critical_corrected
        -> decision rule: argmax(P(Normal), P(Warning), P(Critical))

IMPORTANT - deployment note (do not silently violate):
This pipeline does NOT perform Stage-07 temporal feature engineering
(lags, rolling windows, rate-of-10-min features, etc.). The caller MUST
construct the 17 required post-feature-engineering inputs (see
`SELECTED_FEATURES` below) using the same definitions as
src/07_feature_engineering.py before calling predict()/predict_proba().

Research/prototype safety status: this is a research/prototype decision-
support classifier. It is NOT a certified fire alarm, guaranteed fire
detector, or production safety device. Critical-class generalization
remains limited and batch-sensitive (see
outputs/final_classification/final_classifier_selection.md and
component3_inference_contract.json for full documented limitations).
"""

from __future__ import annotations

import numpy as np

CLASS_ORDER = ("Normal", "Warning", "Critical")
CLASS_TO_INT = {"Normal": 0, "Warning": 1, "Critical": 2}

SELECTED_FEATURES = [
    "Smokehouse_ID",
    "Layer_ID",
    "Temperature_C",
    "Humidity_%",
    "Elapsed_Drying_Hours",
    "Number_of_Sheets",
    "Layer_Area_m2",
    "Temperature_Rate_per_10min",
    "Humidity_Rate_per_10min",
    "Temperature_Change_30m",
    "Humidity_Change_30m",
    "Temperature_Change_60m",
    "Humidity_Change_60m",
    "Temperature_RollStd_30m",
    "Humidity_RollStd_30m",
    "Temperature_RollStd_60m",
    "Humidity_RollStd_60m",
]


class OrdinalFireRiskPipeline:
    """
    Ordinal XGBoost Classifier with Argmax Decision Rule for Fire_Risk_Label.

    Parameters
    ----------
    preprocessor : fitted sklearn ColumnTransformer
        Median imputation + StandardScaler (numeric) and OneHotEncoder
        (categorical), fitted on Development (Train+Validation) only.
    model_o1 : fitted xgboost.XGBClassifier
        Binary model for P(Y >= Warning).
    model_o2 : fitted xgboost.XGBClassifier
        Binary model for P(Y >= Critical).
    selected_features : list[str]
        The exact 17 required input columns, in order.
    """

    def __init__(self, preprocessor, model_o1, model_o2, selected_features=None):
        self.preprocessor = preprocessor
        self.model_o1 = model_o1
        self.model_o2 = model_o2
        self.selected_features = list(selected_features or SELECTED_FEATURES)
        self.class_order_ = CLASS_ORDER

    def _validate_input(self, X):
        missing = [c for c in self.selected_features if c not in X.columns]
        if missing:
            raise ValueError(
                f"OrdinalFireRiskPipeline input is missing required columns: {missing}. "
                f"Required (in order): {self.selected_features}"
            )
        return X[self.selected_features]

    def _reconstruct(self, X_transformed):
        p_ge_warning = self.model_o1.predict_proba(X_transformed)[:, 1]
        p_ge_critical_raw = self.model_o2.predict_proba(X_transformed)[:, 1]

        p_ge_critical_corrected = np.minimum(p_ge_critical_raw, p_ge_warning)

        p_normal = np.clip(1.0 - p_ge_warning, 0.0, 1.0)
        p_critical = np.clip(p_ge_critical_corrected, 0.0, 1.0)
        p_warning = np.clip(p_ge_warning - p_ge_critical_corrected, 0.0, 1.0)

        total = p_normal + p_warning + p_critical
        total_safe = np.where(total == 0, 1.0, total)
        p_normal = p_normal / total_safe
        p_warning = p_warning / total_safe
        p_critical = p_critical / total_safe

        return np.stack([p_normal, p_warning, p_critical], axis=1)

    def predict_proba(self, X):
        """
        Returns an (n_samples, 3) array. Column order is fixed:
        0 = Normal, 1 = Warning, 2 = Critical. Each row sums to 1 within
        floating-point tolerance.
        """
        X_selected = self._validate_input(X)
        X_transformed = self.preprocessor.transform(X_selected)
        return self._reconstruct(X_transformed)

    def predict(self, X):
        """
        Returns an array of string labels ("Normal"/"Warning"/"Critical"),
        derived strictly from argmax(predict_proba(X)). No hidden
        thresholding is applied.
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return np.array([self.class_order_[i] for i in indices])

    def predict_int(self, X):
        """Same decision as predict(), returned as the 0/1/2 integer
        encoding (0=Normal, 1=Warning, 2=Critical) used throughout the
        research pipeline (Stage 20-33, classification_v2)."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
