from pathlib import Path
import hashlib
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# COMPONENT 3 - STAGE 10: TRAIN-ONLY PREPROCESSING
# ============================================================
#
# Purpose:
#   Fit all LEARNED preprocessing operations on TRAIN ONLY,
#   then apply the already-fitted transformer to VALIDATION.
#
# Final TEST is deliberately NOT loaded or transformed here.
# It remains frozen until Stage 31 - Final Test Evaluation.
#
# Learned preprocessing:
#   Numeric:
#     1) Median imputation fitted on TRAIN only
#     2) StandardScaler fitted on TRAIN only
#
#   Categorical:
#     1) OneHotEncoder fitted on TRAIN only
#     2) handle_unknown="ignore" for future unseen categories
#
# Why scale even though many later models are tree-based?
#   - Regression/classification baselines and linear models
#     need appropriately scaled numeric inputs.
#   - Standard scaling is a monotonic linear transform and is
#     also acceptable for the planned tree ensembles.
#   - A single reproducible preprocessor simplifies later
#     comparison and final pipeline export.
#
# Important:
#   - NO model is trained here.
#   - NO class weights are calculated here.
#   - NO SMOTE/resampling is performed here.
#   - NO hyperparameters are tuned here.
#   - TEST is NOT opened, fitted, transformed, or inspected.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = (
    ROOT
    / "data"
    / "splits"
)

TRAIN_PATH = (
    SPLIT_DIR
    / "Component3_Train.csv"
)

VALIDATION_PATH = (
    SPLIT_DIR
    / "Component3_Validation.csv"
)

TEST_PATH = (
    SPLIT_DIR
    / "Component3_Test.csv"
)

MODEL_READY_DIR = (
    ROOT
    / "data"
    / "model_ready"
)

ARTIFACT_DIR = (
    ROOT
    / "artifacts"
    / "preprocessing"
)

REPORT_DIR = (
    ROOT
    / "outputs"
    / "reports"
)

MODEL_READY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. INPUT FEATURE LISTS FROM STAGE 08
# ============================================================

REG_FEATURE_LIST_PATH = (
    REPORT_DIR
    / "08_regression_selected_features.csv"
)

CLS_FEATURE_LIST_PATH = (
    REPORT_DIR
    / "08_classification_selected_features.csv"
)


# ============================================================
# 3. OUTPUT FILES
# ============================================================

PREPROCESSOR_PATH = (
    ARTIFACT_DIR
    / "component3_feature_preprocessor.joblib"
)

PREPROCESSOR_METADATA_PATH = (
    ARTIFACT_DIR
    / "component3_feature_preprocessor_metadata.json"
)

X_TRAIN_PATH = (
    MODEL_READY_DIR
    / "Component3_X_Train.csv"
)

X_VALIDATION_PATH = (
    MODEL_READY_DIR
    / "Component3_X_Validation.csv"
)

Y_REG_TRAIN_PATH = (
    MODEL_READY_DIR
    / "Component3_y_regression_Train.csv"
)

Y_REG_VALIDATION_PATH = (
    MODEL_READY_DIR
    / "Component3_y_regression_Validation.csv"
)

Y_CLS_TRAIN_PATH = (
    MODEL_READY_DIR
    / "Component3_y_classification_Train.csv"
)

Y_CLS_VALIDATION_PATH = (
    MODEL_READY_DIR
    / "Component3_y_classification_Validation.csv"
)

META_TRAIN_PATH = (
    MODEL_READY_DIR
    / "Component3_metadata_Train.csv"
)

META_VALIDATION_PATH = (
    MODEL_READY_DIR
    / "Component3_metadata_Validation.csv"
)

REPORT_PATH = (
    REPORT_DIR
    / "10_train_only_preprocessing_report.txt"
)

CHECKS_PATH = (
    REPORT_DIR
    / "10_train_only_preprocessing_checks.csv"
)

NUMERIC_STATS_PATH = (
    REPORT_DIR
    / "10_numeric_preprocessing_statistics.csv"
)

CATEGORY_STATS_PATH = (
    REPORT_DIR
    / "10_categorical_encoding_statistics.csv"
)

TRANSFORMED_FEATURES_PATH = (
    REPORT_DIR
    / "10_transformed_feature_names.csv"
)

MISSINGNESS_PATH = (
    REPORT_DIR
    / "10_missingness_before_after.csv"
)


# ============================================================
# 4. EXPECTED SPLIT SIZES / IDENTIFIERS
# ============================================================

EXPECTED_TRAIN_ROWS = 16246
EXPECTED_VALIDATION_ROWS = 4159

EXPECTED_TRAIN_BATCHES = {
    "SH01_B01",
    "SH01_B04",
    "SH01_B05",
    "SH01_B06",
    "SH02_B01",
    "SH02_B03",
    "SH02_B05",
    "SH02_B06",
}

EXPECTED_VALIDATION_BATCHES = {
    "SH01_B02",
    "SH02_B02",
}


# ============================================================
# 5. LOAD STAGE-08 FEATURE LISTS
# ============================================================

print("=" * 78)
print("COMPONENT 3 - STAGE 10: TRAIN-ONLY PREPROCESSING")
print("=" * 78)

for required_path in [
    TRAIN_PATH,
    VALIDATION_PATH,
    TEST_PATH,
    REG_FEATURE_LIST_PATH,
    CLS_FEATURE_LIST_PATH,
]:
    if not required_path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{required_path}"
        )

reg_feature_df = pd.read_csv(
    REG_FEATURE_LIST_PATH
)

cls_feature_df = pd.read_csv(
    CLS_FEATURE_LIST_PATH
)

REGRESSION_FEATURES = (
    reg_feature_df[
        "Feature"
    ]
    .tolist()
)

CLASSIFICATION_FEATURES = (
    cls_feature_df[
        "Feature"
    ]
    .tolist()
)

if REGRESSION_FEATURES != CLASSIFICATION_FEATURES:
    raise RuntimeError(
        "Stage 10 currently expects the frozen Stage-08 "
        "regression and classification feature lists to match."
    )

SELECTED_FEATURES = (
    REGRESSION_FEATURES
)

if len(
    SELECTED_FEATURES
) != 17:
    raise RuntimeError(
        f"Expected 17 Stage-08 selected features, "
        f"found {len(SELECTED_FEATURES)}."
    )


# ============================================================
# 6. DEFINE NUMERIC / CATEGORICAL FEATURES
# ============================================================

CATEGORICAL_FEATURES = [
    "Smokehouse_ID",
    "Layer_ID",
]

NUMERIC_FEATURES = [
    feature
    for feature in (
        SELECTED_FEATURES
    )
    if feature
    not in CATEGORICAL_FEATURES
]

if len(
    CATEGORICAL_FEATURES
) != 2:
    raise RuntimeError(
        "Expected exactly 2 categorical selected features."
    )

if len(
    NUMERIC_FEATURES
) != 15:
    raise RuntimeError(
        "Expected exactly 15 numeric selected features."
    )


# ============================================================
# 7. LOAD TRAIN + VALIDATION ONLY
# ============================================================
#
# IMPORTANT:
# TEST_PATH existence was checked above, but its CONTENT IS
# NOT READ in this stage.

with open(
    TRAIN_PATH,
    "rb",
) as file:
    train_sha256 = hashlib.sha256(
        file.read()
    ).hexdigest()

with open(
    VALIDATION_PATH,
    "rb",
) as file:
    validation_sha256 = hashlib.sha256(
        file.read()
    ).hexdigest()

train_df = pd.read_csv(
    TRAIN_PATH
)

validation_df = pd.read_csv(
    VALIDATION_PATH
)

print("\nTrain and Validation datasets loaded.")
print("Final Test dataset was NOT loaded.")


# ============================================================
# 8. TYPE CONVERSION
# ============================================================

for frame in [
    train_df,
    validation_df,
]:
    frame["Timestamp"] = pd.to_datetime(
        frame[
            "Timestamp"
        ],
        errors="raise",
    )

    frame[
        "Fire_Risk_Assessment_Timestamp"
    ] = pd.to_datetime(
        frame[
            "Fire_Risk_Assessment_Timestamp"
        ],
        errors="raise",
    )

    frame[
        "Remaining_Drying_Time"
    ] = pd.to_numeric(
        frame[
            "Remaining_Drying_Time"
        ],
        errors="raise",
    )

    for feature in (
        NUMERIC_FEATURES
    ):
        frame[
            feature
        ] = pd.to_numeric(
            frame[
                feature
            ],
            errors="coerce",
        )


# ============================================================
# 9. VERIFY SPLIT IDENTITY
# ============================================================

train_batches = set(
    train_df[
        "Batch_ID"
    ]
    .unique()
)

validation_batches = set(
    validation_df[
        "Batch_ID"
    ]
    .unique()
)

if train_batches != EXPECTED_TRAIN_BATCHES:
    raise RuntimeError(
        "Train batch assignment does not match frozen Stage 09."
    )

if validation_batches != EXPECTED_VALIDATION_BATCHES:
    raise RuntimeError(
        "Validation batch assignment does not match frozen Stage 09."
    )

if len(
    train_df
) != EXPECTED_TRAIN_ROWS:
    raise RuntimeError(
        f"Unexpected Train row count: {len(train_df):,}"
    )

if len(
    validation_df
) != EXPECTED_VALIDATION_ROWS:
    raise RuntimeError(
        f"Unexpected Validation row count: {len(validation_df):,}"
    )


# ============================================================
# 10. CREATE X / TARGET / METADATA
# ============================================================

X_train = train_df[
    SELECTED_FEATURES
].copy()

X_validation = validation_df[
    SELECTED_FEATURES
].copy()

y_reg_train = train_df[
    "Remaining_Drying_Time"
].copy()

y_reg_validation = validation_df[
    "Remaining_Drying_Time"
].copy()

y_cls_train = train_df[
    "Fire_Risk_Label"
].copy()

y_cls_validation = validation_df[
    "Fire_Risk_Label"
].copy()

metadata_columns = [
    "Timestamp",
    "Smokehouse_ID",
    "Batch_ID",
    "Layer_ID",
    "Fire_Risk_Assessment_Timestamp",
]

meta_train = train_df[
    metadata_columns
].copy()

meta_validation = validation_df[
    metadata_columns
].copy()


# ============================================================
# 11. MISSINGNESS BEFORE TRANSFORMATION
# ============================================================

missingness_rows = []

for split_name, X in [
    (
        "Train",
        X_train,
    ),
    (
        "Validation",
        X_validation,
    ),
]:
    for feature in (
        SELECTED_FEATURES
    ):
        missingness_rows.append({
            "Split": split_name,
            "Feature": feature,
            "Missing_Before": int(
                X[
                    feature
                ]
                .isna()
                .sum()
            ),
        })

missingness_before = pd.DataFrame(
    missingness_rows
)


# ============================================================
# 12. CREATE PREPROCESSOR
# ============================================================
#
# Compatibility with older/newer scikit-learn:
#   sparse_output=False -> newer sklearn
#   sparse=False        -> older sklearn

try:
    one_hot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )
except TypeError:
    one_hot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False,
    )

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "onehot",
            one_hot_encoder,
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES,
        ),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)


# ============================================================
# 13. FIT ON TRAIN ONLY
# ============================================================

preprocessor.fit(
    X_train
)

print("\nPreprocessor fitted on TRAIN ONLY.")


# ============================================================
# 14. TRANSFORM TRAIN + VALIDATION
# ============================================================

X_train_transformed = (
    preprocessor.transform(
        X_train
    )
)

X_validation_transformed = (
    preprocessor.transform(
        X_validation
    )
)

transformed_feature_names = (
    preprocessor
    .get_feature_names_out()
    .tolist()
)


# ============================================================
# 15. BUILD MODEL-READY DATAFRAMES
# ============================================================

X_train_ready = pd.DataFrame(
    X_train_transformed,
    columns=transformed_feature_names,
    index=train_df.index,
)

X_validation_ready = pd.DataFrame(
    X_validation_transformed,
    columns=transformed_feature_names,
    index=validation_df.index,
)

X_train_ready = X_train_ready.reset_index(
    drop=True
)

X_validation_ready = (
    X_validation_ready.reset_index(
        drop=True
    )
)


# ============================================================
# 16. VALIDATE NO MISSING / NON-FINITE VALUES
# ============================================================

train_missing_after = int(
    X_train_ready
    .isna()
    .sum()
    .sum()
)

validation_missing_after = int(
    X_validation_ready
    .isna()
    .sum()
    .sum()
)

train_nonfinite = int(
    (
        ~np.isfinite(
            X_train_ready
            .to_numpy(
                dtype=float
            )
        )
    ).sum()
)

validation_nonfinite = int(
    (
        ~np.isfinite(
            X_validation_ready
            .to_numpy(
                dtype=float
            )
        )
    ).sum()
)


# ============================================================
# 17. EXTRACT TRAIN-FITTED NUMERIC STATISTICS
# ============================================================

fitted_numeric_pipeline = (
    preprocessor
    .named_transformers_[
        "numeric"
    ]
)

fitted_imputer = (
    fitted_numeric_pipeline
    .named_steps[
        "imputer"
    ]
)

fitted_scaler = (
    fitted_numeric_pipeline
    .named_steps[
        "scaler"
    ]
)

numeric_stats = pd.DataFrame({
    "Feature": (
        NUMERIC_FEATURES
    ),
    "Train_Median_Imputation_Value": (
        fitted_imputer
        .statistics_
    ),
    "Scaler_Train_Mean_After_Imputation": (
        fitted_scaler
        .mean_
    ),
    "Scaler_Train_Scale": (
        fitted_scaler
        .scale_
    ),
})

# Independent validation:
# Compare fitted imputer stats against medians calculated
# directly from TRAIN raw selected numeric columns.

direct_train_medians = (
    X_train[
        NUMERIC_FEATURES
    ]
    .median()
    .to_numpy()
)

median_stat_max_error = float(
    np.max(
        np.abs(
            fitted_imputer
            .statistics_
            -
            direct_train_medians
        )
    )
)

numeric_stats.to_csv(
    NUMERIC_STATS_PATH,
    index=False,
)


# ============================================================
# 18. EXTRACT TRAIN-FITTED CATEGORICAL STATISTICS
# ============================================================

fitted_ohe = (
    preprocessor
    .named_transformers_[
        "categorical"
    ]
    .named_steps[
        "onehot"
    ]
)

category_rows = []

for feature, categories in zip(
    CATEGORICAL_FEATURES,
    fitted_ohe.categories_,
):
    validation_values = set(
        X_validation[
            feature
        ]
        .dropna()
        .unique()
    )

    fitted_values = set(
        categories
    )

    unknown_validation = sorted(
        str(
            value
        )
        for value in (
            validation_values
            -
            fitted_values
        )
    )

    category_rows.append({
        "Feature": feature,
        "Train_Fitted_Categories": (
            " | ".join(
                str(
                    value
                )
                for value in categories
            )
        ),
        "Category_Count": len(
            categories
        ),
        "Unknown_Validation_Categories": (
            " | ".join(
                unknown_validation
            )
        ),
        "Unknown_Validation_Category_Count": (
            len(
                unknown_validation
            )
        ),
    })

category_stats = pd.DataFrame(
    category_rows
)

category_stats.to_csv(
    CATEGORY_STATS_PATH,
    index=False,
)


# ============================================================
# 19. TRANSFORMED TRAIN SCALE CHECK
# ============================================================
#
# First 15 transformed columns correspond to the scaled
# numeric features in NUMERIC_FEATURES order.

numeric_train_ready = (
    X_train_ready[
        NUMERIC_FEATURES
    ]
)

max_abs_train_scaled_mean = float(
    numeric_train_ready
    .mean()
    .abs()
    .max()
)

# population SD after StandardScaler should be ~1 except
# constant columns, if any.
train_scaled_population_sd = (
    numeric_train_ready
    .std(
        ddof=0
    )
)

nonconstant_scaled_sd = (
    train_scaled_population_sd[
        train_scaled_population_sd
        >
        0
    ]
)

max_scaled_sd_error = float(
    (
        nonconstant_scaled_sd
        -
        1.0
    )
    .abs()
    .max()
) if len(
    nonconstant_scaled_sd
) > 0 else 0.0


# ============================================================
# 20. VALIDATION WAS NOT REFIT CHECK
# ============================================================
#
# Validation numeric transformed means should generally not
# be forced to zero. This is supporting evidence that the
# Train-fitted scaler was applied rather than refitted.

validation_numeric_means = (
    X_validation_ready[
        NUMERIC_FEATURES
    ]
    .mean()
    .abs()
)

validation_mean_not_artificially_zero = bool(
    validation_numeric_means
    .max()
    >
    1e-4
)


# ============================================================
# 21. COMPLETE MISSINGNESS REPORT
# ============================================================

missingness_after_lookup = {
    "Train": train_missing_after,
    "Validation": validation_missing_after,
}

missingness_before[
    "Missing_After_Whole_Transformed_Matrix"
] = (
    missingness_before[
        "Split"
    ]
    .map(
        missingness_after_lookup
    )
)

missingness_before.to_csv(
    MISSINGNESS_PATH,
    index=False,
)


# ============================================================
# 22. SAVE TRANSFORMED FEATURE NAMES
# ============================================================

transformed_features_df = pd.DataFrame({
    "Order": range(
        1,
        len(
            transformed_feature_names
        )
        + 1,
    ),
    "Transformed_Feature": (
        transformed_feature_names
    ),
})

transformed_features_df.to_csv(
    TRANSFORMED_FEATURES_PATH,
    index=False,
)


# ============================================================
# 23. SAVE MODEL-READY TRAIN / VALIDATION
# ============================================================

X_train_ready.to_csv(
    X_TRAIN_PATH,
    index=False,
)

X_validation_ready.to_csv(
    X_VALIDATION_PATH,
    index=False,
)

pd.DataFrame({
    "Remaining_Drying_Time": (
        y_reg_train
        .reset_index(
            drop=True
        )
    ),
}).to_csv(
    Y_REG_TRAIN_PATH,
    index=False,
)

pd.DataFrame({
    "Remaining_Drying_Time": (
        y_reg_validation
        .reset_index(
            drop=True
        )
    ),
}).to_csv(
    Y_REG_VALIDATION_PATH,
    index=False,
)

pd.DataFrame({
    "Fire_Risk_Label": (
        y_cls_train
        .reset_index(
            drop=True
        )
    ),
}).to_csv(
    Y_CLS_TRAIN_PATH,
    index=False,
)

pd.DataFrame({
    "Fire_Risk_Label": (
        y_cls_validation
        .reset_index(
            drop=True
        )
    ),
}).to_csv(
    Y_CLS_VALIDATION_PATH,
    index=False,
)

meta_train.reset_index(
    drop=True
).to_csv(
    META_TRAIN_PATH,
    index=False,
    date_format="%Y-%m-%d %H:%M:%S",
)

meta_validation.reset_index(
    drop=True
).to_csv(
    META_VALIDATION_PATH,
    index=False,
    date_format="%Y-%m-%d %H:%M:%S",
)


# ============================================================
# 24. SAVE FITTED PREPROCESSOR
# ============================================================

joblib.dump(
    preprocessor,
    PREPROCESSOR_PATH,
)

metadata = {
    "stage": (
        "10_train_only_preprocessing"
    ),
    "fit_scope": (
        "TRAIN_ONLY"
    ),
    "test_loaded_or_transformed": (
        False
    ),
    "selected_input_feature_count": (
        len(
            SELECTED_FEATURES
        )
    ),
    "selected_input_features": (
        SELECTED_FEATURES
    ),
    "numeric_features": (
        NUMERIC_FEATURES
    ),
    "categorical_features": (
        CATEGORICAL_FEATURES
    ),
    "transformed_feature_count": (
        len(
            transformed_feature_names
        )
    ),
    "transformed_feature_names": (
        transformed_feature_names
    ),
    "train_rows": len(
        train_df
    ),
    "validation_rows": len(
        validation_df
    ),
    "train_sha256": (
        train_sha256
    ),
    "validation_sha256": (
        validation_sha256
    ),
}

with open(
    PREPROCESSOR_METADATA_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metadata,
        file,
        indent=2,
    )


# ============================================================
# 25. OUTPUT HASHES
# ============================================================

def sha256_file(
    path,
):
    with open(
        path,
        "rb",
    ) as file:
        return hashlib.sha256(
            file.read()
        ).hexdigest()


x_train_sha256 = sha256_file(
    X_TRAIN_PATH
)

x_validation_sha256 = sha256_file(
    X_VALIDATION_PATH
)

preprocessor_sha256 = sha256_file(
    PREPROCESSOR_PATH
)


# ============================================================
# 26. TARGET / ROW PRESERVATION
# ============================================================

targets_complete = (
    y_reg_train
    .notna()
    .all()
    and
    y_reg_validation
    .notna()
    .all()
    and
    y_cls_train
    .notna()
    .all()
    and
    y_cls_validation
    .notna()
    .all()
)

row_counts_preserved = (
    len(
        X_train_ready
    )
    ==
    len(
        train_df
    )
    ==
    len(
        y_reg_train
    )
    ==
    len(
        y_cls_train
    )
    and
    len(
        X_validation_ready
    )
    ==
    len(
        validation_df
    )
    ==
    len(
        y_reg_validation
    )
    ==
    len(
        y_cls_validation
    )
)


# ============================================================
# 27. PASS / FAIL CHECKS
# ============================================================

unknown_validation_category_total = int(
    category_stats[
        "Unknown_Validation_Category_Count"
    ]
    .sum()
)

checks = pd.DataFrame(
    [
        {
            "Check": (
                "Frozen Stage-09 Train row count"
            ),
            "Status": (
                "PASS"
                if len(
                    train_df
                )
                ==
                EXPECTED_TRAIN_ROWS
                else "FAIL"
            ),
            "Details": (
                f"Rows="
                f"{len(train_df):,}"
            ),
        },
        {
            "Check": (
                "Frozen Stage-09 Validation row count"
            ),
            "Status": (
                "PASS"
                if len(
                    validation_df
                )
                ==
                EXPECTED_VALIDATION_ROWS
                else "FAIL"
            ),
            "Details": (
                f"Rows="
                f"{len(validation_df):,}"
            ),
        },
        {
            "Check": (
                "Stage-08 selected feature count"
            ),
            "Status": (
                "PASS"
                if len(
                    SELECTED_FEATURES
                )
                ==
                17
                else "FAIL"
            ),
            "Details": (
                f"Features="
                f"{len(SELECTED_FEATURES)}"
            ),
        },
        {
            "Check": (
                "Train-only numeric medians verified"
            ),
            "Status": (
                "PASS"
                if median_stat_max_error
                <
                1e-12
                else "FAIL"
            ),
            "Details": (
                f"Max median error="
                f"{median_stat_max_error:.12g}"
            ),
        },
        {
            "Check": (
                "No missing values after Train transformation"
            ),
            "Status": (
                "PASS"
                if train_missing_after
                ==
                0
                else "FAIL"
            ),
            "Details": (
                f"Missing="
                f"{train_missing_after}"
            ),
        },
        {
            "Check": (
                "No missing values after Validation transformation"
            ),
            "Status": (
                "PASS"
                if validation_missing_after
                ==
                0
                else "FAIL"
            ),
            "Details": (
                f"Missing="
                f"{validation_missing_after}"
            ),
        },
        {
            "Check": (
                "No non-finite Train values"
            ),
            "Status": (
                "PASS"
                if train_nonfinite
                ==
                0
                else "FAIL"
            ),
            "Details": (
                f"Non-finite="
                f"{train_nonfinite}"
            ),
        },
        {
            "Check": (
                "No non-finite Validation values"
            ),
            "Status": (
                "PASS"
                if validation_nonfinite
                ==
                0
                else "FAIL"
            ),
            "Details": (
                f"Non-finite="
                f"{validation_nonfinite}"
            ),
        },
        {
            "Check": (
                "Train scaled numeric means approximately zero"
            ),
            "Status": (
                "PASS"
                if max_abs_train_scaled_mean
                <
                1e-10
                else "FAIL"
            ),
            "Details": (
                f"Max |mean|="
                f"{max_abs_train_scaled_mean:.12g}"
            ),
        },
        {
            "Check": (
                "Train scaled numeric SD approximately one"
            ),
            "Status": (
                "PASS"
                if max_scaled_sd_error
                <
                1e-10
                else "FAIL"
            ),
            "Details": (
                f"Max SD error="
                f"{max_scaled_sd_error:.12g}"
            ),
        },
        {
            "Check": (
                "Validation not independently standardized"
            ),
            "Status": (
                "PASS"
                if validation_mean_not_artificially_zero
                else "FAIL"
            ),
            "Details": (
                "Validation transformed with Train-fitted scaler."
            ),
        },
        {
            "Check": (
                "No unknown Validation categories"
            ),
            "Status": (
                "PASS"
                if unknown_validation_category_total
                ==
                0
                else "FAIL"
            ),
            "Details": (
                f"Unknown categories="
                f"{unknown_validation_category_total}"
            ),
        },
        {
            "Check": (
                "Targets and row counts preserved"
            ),
            "Status": (
                "PASS"
                if (
                    targets_complete
                    and
                    row_counts_preserved
                )
                else "FAIL"
            ),
            "Details": (
                "Train/Validation X-y alignment preserved."
            ),
        },
        {
            "Check": (
                "Final Test not loaded/transformed"
            ),
            "Status": "PASS",
            "Details": (
                "Stage 10 only checked that Component3_Test.csv exists; "
                "its contents were not read."
            ),
        },
    ]
)

checks.to_csv(
    CHECKS_PATH,
    index=False,
)

overall_status = (
    "PASS"
    if not (
        checks[
            "Status"
        ]
        ==
        "FAIL"
    ).any()
    else "FAIL"
)


# ============================================================
# 28. CONSOLE OUTPUT
# ============================================================

train_missing_before_total = int(
    X_train
    .isna()
    .sum()
    .sum()
)

validation_missing_before_total = int(
    X_validation
    .isna()
    .sum()
    .sum()
)

print(
    "\n--- INPUT SPLITS ---"
)

print(
    f"Train rows                       : "
    f"{len(train_df):,}"
)

print(
    f"Validation rows                  : "
    f"{len(validation_df):,}"
)

print(
    f"Selected input features          : "
    f"{len(SELECTED_FEATURES)}"
)

print(
    f"Numeric / categorical            : "
    f"{len(NUMERIC_FEATURES)} / "
    f"{len(CATEGORICAL_FEATURES)}"
)

print(
    f"Train SHA-256                    : "
    f"{train_sha256}"
)

print(
    f"Validation SHA-256               : "
    f"{validation_sha256}"
)


print(
    "\n--- TRAIN-ONLY FIT ---"
)

print(
    "Numeric imputer                 : "
    "Median fitted on TRAIN only"
)

print(
    "Numeric scaler                  : "
    "StandardScaler fitted on TRAIN only"
)

print(
    "Categorical encoder             : "
    "OneHotEncoder fitted on TRAIN only"
)

print(
    f"Max train-median verification err: "
    f"{median_stat_max_error:.12g}"
)


print(
    "\n--- MISSINGNESS ---"
)

print(
    f"Train missing cells before       : "
    f"{train_missing_before_total}"
)

print(
    f"Train missing cells after        : "
    f"{train_missing_after}"
)

print(
    f"Validation missing cells before  : "
    f"{validation_missing_before_total}"
)

print(
    f"Validation missing cells after   : "
    f"{validation_missing_after}"
)


print(
    "\n--- TRANSFORMED MATRICES ---"
)

print(
    f"Transformed feature count        : "
    f"{len(transformed_feature_names)}"
)

print(
    f"X_train shape                    : "
    f"{X_train_ready.shape}"
)

print(
    f"X_validation shape               : "
    f"{X_validation_ready.shape}"
)

print(
    f"Train non-finite values          : "
    f"{train_nonfinite}"
)

print(
    f"Validation non-finite values     : "
    f"{validation_nonfinite}"
)

print(
    f"Max |train numeric mean|         : "
    f"{max_abs_train_scaled_mean:.3e}"
)

print(
    f"Max train numeric SD error       : "
    f"{max_scaled_sd_error:.3e}"
)

print(
    f"Unknown Validation categories    : "
    f"{unknown_validation_category_total}"
)


print(
    "\n--- TEST PROTECTION ---"
)

print(
    "Component3_Test.csv content read : False"
)

print(
    "Component3_Test.csv transformed  : False"
)

print(
    "Final Test remains frozen until Stage 31."
)


print(
    "\n--- OUTPUT ARTIFACTS ---"
)

print(
    f"Preprocessor:\n"
    f"{PREPROCESSOR_PATH}"
)

print(
    f"\nModel-ready Train X:\n"
    f"{X_TRAIN_PATH}"
)

print(
    f"\nModel-ready Validation X:\n"
    f"{X_VALIDATION_PATH}"
)

print(
    "\nSHA-256 fingerprints:"
)

print(
    f"  Preprocessor : "
    f"{preprocessor_sha256}"
)

print(
    f"  X Train      : "
    f"{x_train_sha256}"
)

print(
    f"  X Validation : "
    f"{x_validation_sha256}"
)


# ============================================================
# 29. HUMAN-READABLE REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "COMPONENT 3 - STAGE 10: TRAIN-ONLY PREPROCESSING\n"
    )

    report.write(
        "=" * 78
        + "\n\n"
    )

    report.write(
        f"OVERALL STATUS: "
        f"{overall_status}\n\n"
    )

    report.write(
        "METHODOLOGY\n"
        "-----------\n"
    )

    report.write(
        "All learned preprocessing was fitted using the "
        "Stage-09 TRAIN split only. Validation was transformed "
        "with the already-fitted Train transformer. Final Test "
        "was not opened or transformed and remains frozen until "
        "Stage 31.\n\n"
    )

    report.write(
        "Numeric preprocessing: Train-only median imputation "
        "followed by Train-only StandardScaler.\n"
    )

    report.write(
        "Categorical preprocessing: Train-only OneHotEncoder "
        "with handle_unknown='ignore'.\n\n"
    )

    report.write(
        "INPUT SPLITS\n"
        "------------\n"
    )

    report.write(
        f"Train rows: "
        f"{len(train_df):,}\n"
    )

    report.write(
        f"Validation rows: "
        f"{len(validation_df):,}\n"
    )

    report.write(
        f"Selected raw input features: "
        f"{len(SELECTED_FEATURES)}\n"
    )

    report.write(
        f"Numeric features: "
        f"{len(NUMERIC_FEATURES)}\n"
    )

    report.write(
        f"Categorical features: "
        f"{len(CATEGORICAL_FEATURES)}\n"
    )

    report.write(
        f"Train SHA-256: "
        f"{train_sha256}\n"
    )

    report.write(
        f"Validation SHA-256: "
        f"{validation_sha256}\n\n"
    )

    report.write(
        "SELECTED INPUT FEATURES\n"
        "-----------------------\n"
    )

    for feature in (
        SELECTED_FEATURES
    ):
        report.write(
            f"- {feature}\n"
        )

    report.write(
        "\n"
    )

    report.write(
        "TRAIN-FITTED NUMERIC STATISTICS\n"
        "-------------------------------\n"
    )

    report.write(
        numeric_stats
        .round(8)
        .to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )

    report.write(
        "TRAIN-FITTED CATEGORICAL LEVELS\n"
        "-------------------------------\n"
    )

    report.write(
        category_stats
        .to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )

    report.write(
        "TRANSFORMED FEATURE SPACE\n"
        "-------------------------\n"
    )

    report.write(
        f"Transformed feature count: "
        f"{len(transformed_feature_names)}\n"
    )

    for feature in (
        transformed_feature_names
    ):
        report.write(
            f"- {feature}\n"
        )

    report.write(
        "\n"
    )

    report.write(
        "MISSINGNESS\n"
        "-----------\n"
    )

    report.write(
        f"Train selected-feature missing cells before: "
        f"{train_missing_before_total}\n"
    )

    report.write(
        f"Train transformed missing cells after: "
        f"{train_missing_after}\n"
    )

    report.write(
        f"Validation selected-feature missing cells before: "
        f"{validation_missing_before_total}\n"
    )

    report.write(
        f"Validation transformed missing cells after: "
        f"{validation_missing_after}\n\n"
    )

    report.write(
        "LEAKAGE PROTECTION\n"
        "------------------\n"
    )

    report.write(
        "- Median imputation statistics learned from Train only.\n"
    )

    report.write(
        "- StandardScaler statistics learned from Train only.\n"
    )

    report.write(
        "- OneHotEncoder categories learned from Train only.\n"
    )

    report.write(
        "- Validation was transformed, never refitted.\n"
    )

    report.write(
        "- Test file contents were not loaded or transformed.\n"
    )

    report.write(
        "- No target values were used to fit preprocessing.\n"
    )

    report.write(
        "- No class balancing or SMOTE was performed.\n"
    )

    report.write(
        "- No regression/classification model was trained.\n\n"
    )

    report.write(
        "ARTIFACTS\n"
        "---------\n"
    )

    report.write(
        f"Fitted preprocessor: "
        f"{PREPROCESSOR_PATH}\n"
    )

    report.write(
        f"Preprocessor metadata: "
        f"{PREPROCESSOR_METADATA_PATH}\n"
    )

    report.write(
        f"Model-ready Train X: "
        f"{X_TRAIN_PATH}\n"
    )

    report.write(
        f"Model-ready Validation X: "
        f"{X_VALIDATION_PATH}\n\n"
    )

    report.write(
        "IMPORTANT NEXT-STAGE RULE\n"
        "-------------------------\n"
    )

    report.write(
        "Stages 11-29 may use TRAIN and VALIDATION only. "
        "Component3_Test.csv remains untouched. At Stage 31, "
        "the already-fitted Stage-10 preprocessor will be "
        "loaded and applied to the frozen Test split exactly "
        "once for final evaluation.\n\n"
    )

    report.write(
        "PASS / FAIL CHECKS\n"
        "------------------\n"
    )

    report.write(
        checks
        .to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )

    report.write(
        f"STAGE 10 STATUS: "
        f"{overall_status}\n"
    )


# ============================================================
# 30. COMPLETE
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    f"STAGE 10 TRAIN-ONLY PREPROCESSING: "
    f"{overall_status}"
)

print(
    "=" * 78
)

print(
    f"\nReports saved in:\n"
    f"{REPORT_DIR}"
)

print(
    "\nNo prediction model was trained."
)

if overall_status != "PASS":
    raise RuntimeError(
        "Stage 10 failed. Review "
        "10_train_only_preprocessing_report.txt."
    )
