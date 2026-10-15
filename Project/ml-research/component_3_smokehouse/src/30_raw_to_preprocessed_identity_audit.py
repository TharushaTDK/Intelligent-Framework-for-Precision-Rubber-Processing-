from pathlib import Path
import pandas as pd

# ============================================================
# COMPONENT 3 - RAW -> PREPROCESSED IDENTITY AUDIT
# ============================================================
#
# Purpose:
#   Verify that Stage 4 preprocessing did not create rows
#   or alter the research targets.
#
# IMPORTANT:
#   - Final Test is NOT loaded.
#   - No files are modified.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = (
    ROOT
    / "data"
    / "raw"
    / "Component3_Dataset.csv"
)

PREPROCESSED_PATH = (
    ROOT
    / "data"
    / "processed"
    / "Component3_Dataset_Preprocessed.csv"
)

TEST_PATH = (
    ROOT
    / "data"
    / "splits"
    / "Component3_Test.csv"
)

REPORT_DIR = ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = (
    REPORT_DIR
    / "30_raw_to_preprocessed_identity_audit.txt"
)

for path in [
    RAW_PATH,
    PREPROCESSED_PATH,
    TEST_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

print("=" * 78)
print(
    "COMPONENT 3 - RAW -> PREPROCESSED "
    "IDENTITY AUDIT"
)
print("=" * 78)

# Final Test existence only.
final_test_content_read = False

print("\n--- TEST PROTECTION ---")
print("Final Test exists:", TEST_PATH.exists())
print(
    "Component3_Test.csv content read:",
    final_test_content_read,
)

raw = pd.read_csv(RAW_PATH)
pre = pd.read_csv(PREPROCESSED_PATH)

required_columns = [
    "Smokehouse_ID",
    "Batch_ID",
    "Layer_ID",
    "Elapsed_Drying_Hours",
    "Remaining_Drying_Time",
    "Fire_Risk_Label",
]

for name, frame in [
    ("Raw", raw),
    ("Preprocessed", pre),
]:
    missing = [
        c for c in required_columns
        if c not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{name} missing columns: {missing}"
        )

print("\n--- ROW COUNTS ---")
print(f"Raw rows                  : {len(raw):,}")
print(f"Preprocessed rows         : {len(pre):,}")

raw_exact_duplicates = int(
    raw.duplicated().sum()
)

raw_after_exact_dedup = (
    raw.drop_duplicates()
)

print(
    f"Raw exact duplicate rows  : "
    f"{raw_exact_duplicates:,}"
)
print(
    f"Raw rows after exact dedup : "
    f"{len(raw_after_exact_dedup):,}"
)

row_count_match = (
    len(raw_after_exact_dedup)
    == len(pre)
)

print(
    "Preprocessed row count equals "
    "raw-after-dedup:",
    row_count_match,
)


# ============================================================
# TARGET DISTRIBUTION CHECK
# ============================================================

RISK_ORDER = [
    "Normal",
    "Warning",
    "Critical",
]

def risk_counts(frame):
    return (
        frame["Fire_Risk_Label"]
        .astype(str)
        .str.strip()
        .value_counts()
        .reindex(
            RISK_ORDER,
            fill_value=0,
        )
    )

raw_risk = risk_counts(
    raw_after_exact_dedup
)

pre_risk = risk_counts(
    pre
)

print("\n--- FIRE-RISK TARGET COUNTS ---")
print("\nRaw after exact dedup:")
print(raw_risk.to_string())

print("\nPreprocessed:")
print(pre_risk.to_string())

risk_counts_match = (
    raw_risk.equals(pre_risk)
)

print(
    "\nFire_Risk_Label counts preserved:",
    risk_counts_match,
)


# ============================================================
# REGRESSION TARGET CHECK
# ============================================================

raw_remaining = (
    pd.to_numeric(
        raw_after_exact_dedup[
            "Remaining_Drying_Time"
        ],
        errors="raise",
    )
)

pre_remaining = (
    pd.to_numeric(
        pre[
            "Remaining_Drying_Time"
        ],
        errors="raise",
    )
)

print("\n--- REGRESSION TARGET SUMMARY ---")
print(
    "Raw Remaining_Drying_Time:"
)
print(
    raw_remaining.describe()[
        ["count", "min", "max", "mean"]
    ].to_string()
)

print(
    "\nPreprocessed Remaining_Drying_Time:"
)
print(
    pre_remaining.describe()[
        ["count", "min", "max", "mean"]
    ].to_string()
)


# ============================================================
# STABLE OBSERVATION KEY CHECK
# ============================================================

stable_key = [
    "Smokehouse_ID",
    "Batch_ID",
    "Layer_ID",
    "Elapsed_Drying_Hours",
]

raw_keyed = (
    raw_after_exact_dedup[
        stable_key
        + [
            "Remaining_Drying_Time",
            "Fire_Risk_Label",
        ]
    ]
    .copy()
)

pre_keyed = (
    pre[
        stable_key
        + [
            "Remaining_Drying_Time",
            "Fire_Risk_Label",
        ]
    ]
    .copy()
)

for frame in [
    raw_keyed,
    pre_keyed,
]:
    for column in [
        "Smokehouse_ID",
        "Batch_ID",
        "Layer_ID",
        "Fire_Risk_Label",
    ]:
        frame[column] = (
            frame[column]
            .astype(str)
            .str.strip()
        )

    frame[
        "Elapsed_Drying_Hours"
    ] = pd.to_numeric(
        frame[
            "Elapsed_Drying_Hours"
        ],
        errors="raise",
    ).round(6)

    frame[
        "Remaining_Drying_Time"
    ] = pd.to_numeric(
        frame[
            "Remaining_Drying_Time"
        ],
        errors="raise",
    ).round(6)

raw_duplicate_stable_keys = int(
    raw_keyed.duplicated(
        subset=stable_key,
        keep=False,
    ).sum()
)

pre_duplicate_stable_keys = int(
    pre_keyed.duplicated(
        subset=stable_key,
        keep=False,
    ).sum()
)

print("\n--- STABLE OBSERVATION KEY ---")
print(
    "Raw duplicate stable-key rows:",
    raw_duplicate_stable_keys,
)
print(
    "Preprocessed duplicate stable-key rows:",
    pre_duplicate_stable_keys,
)

# If the stable key is unique, compare row-by-row targets.
target_changes = None
missing_rows = None

if (
    raw_duplicate_stable_keys == 0
    and pre_duplicate_stable_keys == 0
):
    raw_ref = raw_keyed.rename(
        columns={
            "Remaining_Drying_Time":
                "Raw_Remaining_Drying_Time",
            "Fire_Risk_Label":
                "Raw_Fire_Risk_Label",
        }
    )

    compare = pre_keyed.merge(
        raw_ref,
        on=stable_key,
        how="left",
        validate="one_to_one",
    )

    missing_rows = int(
        compare[
            "Raw_Fire_Risk_Label"
        ].isna().sum()
    )

    class_changes = int(
        (
            compare[
                "Fire_Risk_Label"
            ]
            != compare[
                "Raw_Fire_Risk_Label"
            ]
        )
        .fillna(False)
        .sum()
    )

    remaining_changes = int(
        (
            (
                compare[
                    "Remaining_Drying_Time"
                ]
                - compare[
                    "Raw_Remaining_Drying_Time"
                ]
            ).abs()
            > 1e-9
        )
        .fillna(False)
        .sum()
    )

    target_changes = (
        class_changes
        + remaining_changes
    )

    print(
        "Preprocessed rows not found in raw:",
        missing_rows,
    )
    print(
        "Fire_Risk_Label changes:",
        class_changes,
    )
    print(
        "Remaining_Drying_Time changes:",
        remaining_changes,
    )

else:
    print(
        "Stable key is not unique, so exact "
        "row-level target comparison was skipped."
    )


# ============================================================
# FINAL STATUS
# ============================================================

if (
    row_count_match
    and risk_counts_match
    and (
        target_changes is None
        or (
            missing_rows == 0
            and target_changes == 0
        )
    )
):
    status = "PASS"
    interpretation = (
        "Stage-4 preprocessing is consistent with the current "
        "raw dataset: no extra rows were created after exact "
        "duplicate removal and the research targets were preserved."
    )

else:
    status = "REVIEW REQUIRED"
    interpretation = (
        "The current raw and preprocessed datasets are not fully "
        "consistent with Stage-4 expectations. Review the active "
        "dataset version before proceeding to the locked Final Test."
    )

print("\n--- FINAL AUDIT RESULT ---")
print("Status:", status)
print(interpretation)

with REPORT_PATH.open(
    "w",
    encoding="utf-8",
) as report:
    report.write(
        "COMPONENT 3 - RAW -> PREPROCESSED "
        "IDENTITY AUDIT\n"
    )
    report.write("=" * 78 + "\n\n")

    report.write(
        f"Raw rows: {len(raw):,}\n"
    )
    report.write(
        f"Raw exact duplicates: "
        f"{raw_exact_duplicates:,}\n"
    )
    report.write(
        f"Raw after exact dedup: "
        f"{len(raw_after_exact_dedup):,}\n"
    )
    report.write(
        f"Preprocessed rows: "
        f"{len(pre):,}\n\n"
    )

    report.write(
        "Raw-after-dedup fire-risk counts:\n"
        + raw_risk.to_string()
        + "\n\n"
    )

    report.write(
        "Preprocessed fire-risk counts:\n"
        + pre_risk.to_string()
        + "\n\n"
    )

    report.write(
        f"Row count match: {row_count_match}\n"
    )
    report.write(
        f"Fire-risk counts preserved: "
        f"{risk_counts_match}\n"
    )

    if target_changes is not None:
        report.write(
            f"Rows missing from raw by stable key: "
            f"{missing_rows}\n"
        )
        report.write(
            f"Total target changes: "
            f"{target_changes}\n"
        )

    report.write("\n")
    report.write(
        f"FINAL STATUS: {status}\n"
    )
    report.write(
        interpretation + "\n\n"
    )

    report.write(
        "Component3_Test.csv contents loaded: False\n"
    )

print("\n--- FINAL TEST PROTECTION ---")
print(
    "Component3_Test.csv content read:",
    final_test_content_read,
)

print("\nReport saved to:")
print(REPORT_PATH)
