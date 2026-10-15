from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS / FINAL MODEL CONTRACT
# ============================================================

MODEL_DIR = Path(__file__).resolve().parent

INFERENCE_CONTRACT_PATH = (
    MODEL_DIR / "component3_inference_contract.json"
)


# Exact finalized 17-feature contract.
#
# IMPORTANT:
# The order must remain identical to the frozen inference contract
# because both final ML pipelines were trained with this feature order.
EXPECTED_FEATURES = (
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
)


NUMERIC_FEATURES = (
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
)


# ============================================================
# EXCEPTIONS
# ============================================================


class FeatureBuildError(ValueError):
    """
    Raised when runtime sensor/history data cannot safely be
    converted into the finalized Component 3 feature contract.
    """


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass(frozen=True)
class HistoricalReading:
    """
    One previous sensor observation from the SAME:

        Batch
        +
        Layer

    sequence.

    recorded_at must be earlier than the current reading.
    """

    recorded_at: datetime
    temperature_c: float
    humidity_pct: float


@dataclass(frozen=True)
class FeatureBuildResult:
    """
    Result returned by the runtime feature builder.

    features:
        One-row pandas DataFrame containing the exact 17 inputs
        expected by the final ML pipelines.

    missing_numeric_features:
        History-dependent features that cannot yet be calculated.
        These may legitimately occur near the start of a batch or
        around packet-loss gaps.

        The finalized ML pipelines contain trained median imputers
        for numeric missing values.

    historical_reading_count:
        Number of previous observations supplied to this builder.
    """

    features: pd.DataFrame

    missing_numeric_features: tuple[str, ...]

    historical_reading_count: int


# ============================================================
# VERIFY FINAL INFERENCE CONTRACT
# ============================================================


def _load_contract_features() -> tuple[str, ...]:
    """
    Read the frozen inference contract and make sure this runtime
    implementation still matches the final ML feature contract.
    """

    if not INFERENCE_CONTRACT_PATH.is_file():
        raise RuntimeError(
            "Component 3 inference contract was not found."
        )

    with INFERENCE_CONTRACT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        contract = json.load(file)

    ordered_features = tuple(
        contract["input_contract"]["ordered_features"]
    )

    if ordered_features != EXPECTED_FEATURES:
        raise RuntimeError(
            "Runtime feature builder does not match the finalized "
            "Component 3 inference contract."
        )

    return ordered_features


ORDERED_FEATURES = _load_contract_features()


# ============================================================
# DATETIME NORMALIZATION
# ============================================================


def _to_utc(
    value: datetime,
    field_name: str,
) -> datetime:
    """
    Require timezone-aware timestamps.

    All backend runtime calculations use UTC.
    """

    if value.tzinfo is None or value.utcoffset() is None:
        raise FeatureBuildError(
            f"{field_name} must be timezone-aware."
        )

    return value.astimezone(timezone.utc)


# ============================================================
# MAIN FEATURE BUILDER
# ============================================================


def build_feature_row(
    *,
    smokehouse_id: str,
    layer_id: str,
    temperature_c: float,
    humidity_pct: float,
    recorded_at: datetime,
    batch_started_at: datetime,
    number_of_sheets: int,
    layer_area_m2: float,
    history: Sequence[HistoricalReading],
) -> FeatureBuildResult:
    """
    Construct the exact 17-feature runtime input required by both
    finalized Component 3 ML pipelines.

    Inputs
    ------
    smokehouse_id:
        Example: SH01

    layer_id:
        L1 / L2 / L3

    temperature_c:
        Current DHT22 temperature reading.

    humidity_pct:
        Current DHT22 relative humidity reading.

    recorded_at:
        Timestamp of the current sensor reading.

    batch_started_at:
        Timestamp when the drying batch began.

    number_of_sheets:
        Fixed configuration for this batch/layer.

    layer_area_m2:
        Fixed configuration for this batch/layer.

    history:
        Previous readings from the SAME batch + layer only.

    IMPORTANT
    ---------
    Historical readings must contain ONLY past observations.

    No future information is permitted.

    This reproduces the required Stage-07 temporal feature logic:
    - previous-available-reading normalized 10-minute rates
    - exact timestamp 30-minute changes
    - exact timestamp 60-minute changes
    - timestamp-aware 30-minute rolling population std
    - timestamp-aware 60-minute rolling population std
    """

    # ========================================================
    # 1. VALIDATE DIRECT / CONFIGURATION INPUTS
    # ========================================================

    smokehouse_id = str(
        smokehouse_id
    ).strip()

    layer_id = str(
        layer_id
    ).strip()

    if not smokehouse_id:
        raise FeatureBuildError(
            "smokehouse_id must not be empty."
        )

    if not layer_id:
        raise FeatureBuildError(
            "layer_id must not be empty."
        )

    if not 0 <= float(humidity_pct) <= 100:
        raise FeatureBuildError(
            "humidity_pct must be between 0 and 100."
        )

    if int(number_of_sheets) <= 0:
        raise FeatureBuildError(
            "number_of_sheets must be greater than zero."
        )

    if float(layer_area_m2) <= 0:
        raise FeatureBuildError(
            "layer_area_m2 must be greater than zero."
        )

    current_time = _to_utc(
        recorded_at,
        "recorded_at",
    )

    batch_start = _to_utc(
        batch_started_at,
        "batch_started_at",
    )

    elapsed_seconds = (
        current_time
        - batch_start
    ).total_seconds()

    if elapsed_seconds < 0:
        raise FeatureBuildError(
            "recorded_at cannot be earlier than "
            "batch_started_at."
        )

    elapsed_drying_hours = (
        elapsed_seconds / 3600.0
    )

    # ========================================================
    # 2. PREPARE HISTORICAL OBSERVATIONS
    # ========================================================

    historical_rows: list[
        dict[str, object]
    ] = []

    seen_timestamps: set[datetime] = set()

    for reading in history:

        timestamp = _to_utc(
            reading.recorded_at,
            "history.recorded_at",
        )

        # ----------------------------------------------------
        # Critical leakage protection:
        # only observations STRICTLY before the current
        # timestamp are allowed.
        # ----------------------------------------------------

        if timestamp >= current_time:
            raise FeatureBuildError(
                "Historical readings must be strictly earlier "
                "than the current recorded_at timestamp."
            )

        if timestamp in seen_timestamps:
            raise FeatureBuildError(
                "Duplicate historical timestamp detected."
            )

        if not (
            0
            <= float(reading.humidity_pct)
            <= 100
        ):
            raise FeatureBuildError(
                "Historical humidity must be between "
                "0 and 100."
            )

        seen_timestamps.add(
            timestamp
        )

        historical_rows.append(
            {
                "Timestamp": timestamp,
                "Temperature_C": float(
                    reading.temperature_c
                ),
                "Humidity_%": float(
                    reading.humidity_pct
                ),
            }
        )

    # Always ensure chronological sequence.
    historical_rows.sort(
        key=lambda row: row["Timestamp"]
    )

    # ========================================================
    # 3. RATE PER 10 MINUTES
    # ========================================================
    #
    # Stage-07 definition:
    #
    #        current - previous available reading
    # rate = ------------------------------------ × 10
    #              actual minutes elapsed
    #
    # This is NOT simply current - previous row.
    #
    # If packet loss causes the previous available reading to
    # be 20 minutes earlier, the change is normalized back to a
    # 10-minute equivalent.
    # ========================================================

    temperature_rate_per_10min = np.nan
    humidity_rate_per_10min = np.nan

    if historical_rows:

        previous = historical_rows[-1]

        previous_timestamp = previous[
            "Timestamp"
        ]

        minutes_since_previous = (
            current_time
            - previous_timestamp
        ).total_seconds() / 60.0

        if minutes_since_previous <= 0:
            raise FeatureBuildError(
                "Previous-reading interval must be positive."
            )

        temperature_rate_per_10min = (
            (
                float(temperature_c)
                - float(
                    previous[
                        "Temperature_C"
                    ]
                )
            )
            / minutes_since_previous
            * 10.0
        )

        humidity_rate_per_10min = (
            (
                float(humidity_pct)
                - float(
                    previous[
                        "Humidity_%"
                    ]
                )
            )
            / minutes_since_previous
            * 10.0
        )

    # ========================================================
    # 4. EXACT-TIMESTAMP LAG LOOKUP
    # ========================================================
    #
    # Stage 07 intentionally used exact timestamp matching.
    #
    # Example:
    #
    # current = 01:00
    #
    # 30-minute lag requires EXACTLY:
    # 00:30
    #
    # If 00:30 does not exist because of packet loss:
    # result = NaN
    #
    # We DO NOT substitute 00:20 or 00:40.
    #
    # This prevents fake lag features.
    # ========================================================

    historical_lookup = {
        row["Timestamp"]: row
        for row in historical_rows
    }

    def exact_change(
        minutes: int,
        value_name: str,
        current_value: float,
    ) -> float:
        """
        Current value minus the value at the exact historical
        timestamp.

        Missing exact timestamp -> NaN.
        """

        target_timestamp = (
            current_time
            - timedelta(
                minutes=minutes
            )
        )

        historical = (
            historical_lookup.get(
                target_timestamp
            )
        )

        if historical is None:
            return np.nan

        return (
            float(current_value)
            - float(
                historical[
                    value_name
                ]
            )
        )

    temperature_change_30m = exact_change(
        30,
        "Temperature_C",
        float(temperature_c),
    )

    humidity_change_30m = exact_change(
        30,
        "Humidity_%",
        float(humidity_pct),
    )

    temperature_change_60m = exact_change(
        60,
        "Temperature_C",
        float(temperature_c),
    )

    humidity_change_60m = exact_change(
        60,
        "Humidity_%",
        float(humidity_pct),
    )

    # ========================================================
    # 5. TIMESTAMP-AWARE ROLLING WINDOWS
    # ========================================================
    #
    # Exact Stage-07 behaviour:
    #
    # rolling(
    #     window="30min" / "60min",
    #     min_periods=1,
    #     closed="both",
    # )
    #
    # Population standard deviation:
    #
    # std(ddof=0)
    #
    # CURRENT reading is included.
    #
    # With a perfect 10-minute sequence, the 60-minute window
    # contains:
    #
    # current reading
    # +
    # six previous readings
    #
    # = 7 total observations.
    # ========================================================

    all_rows = (
        historical_rows
        + [
            {
                "Timestamp": current_time,
                "Temperature_C": float(
                    temperature_c
                ),
                "Humidity_%": float(
                    humidity_pct
                ),
            }
        ]
    )

    sequence_df = pd.DataFrame(
        all_rows
    )

    sequence_df = (
        sequence_df
        .sort_values(
            "Timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    sequence_df[
        "Timestamp"
    ] = pd.to_datetime(
        sequence_df[
            "Timestamp"
        ],
        utc=True,
    )

    sequence_df = (
        sequence_df
        .set_index(
            "Timestamp"
        )
    )

    def rolling_population_std(
        minutes: int,
        column: str,
    ) -> float:
        """
        Reproduce Stage-07 timestamp-aware rolling population
        standard deviation for the CURRENT reading.
        """

        rolling_result = (
            sequence_df[
                [column]
            ]
            .rolling(
                window=f"{minutes}min",
                min_periods=1,
                closed="both",
            )[
                column
            ]
            .std(
                ddof=0
            )
        )

        return float(
            rolling_result.iloc[-1]
        )

    temperature_rollstd_30m = (
        rolling_population_std(
            30,
            "Temperature_C",
        )
    )

    humidity_rollstd_30m = (
        rolling_population_std(
            30,
            "Humidity_%",
        )
    )

    temperature_rollstd_60m = (
        rolling_population_std(
            60,
            "Temperature_C",
        )
    )

    humidity_rollstd_60m = (
        rolling_population_std(
            60,
            "Humidity_%",
        )
    )

    # ========================================================
    # 6. CREATE EXACT FINAL 17-FEATURE ROW
    # ========================================================

    feature_values = {
        "Smokehouse_ID": (
            smokehouse_id
        ),

        "Layer_ID": (
            layer_id
        ),

        "Temperature_C": float(
            temperature_c
        ),

        "Humidity_%": float(
            humidity_pct
        ),

        "Elapsed_Drying_Hours": float(
            elapsed_drying_hours
        ),

        "Number_of_Sheets": int(
            number_of_sheets
        ),

        "Layer_Area_m2": float(
            layer_area_m2
        ),

        "Temperature_Rate_per_10min": (
            temperature_rate_per_10min
        ),

        "Humidity_Rate_per_10min": (
            humidity_rate_per_10min
        ),

        "Temperature_Change_30m": (
            temperature_change_30m
        ),

        "Humidity_Change_30m": (
            humidity_change_30m
        ),

        "Temperature_Change_60m": (
            temperature_change_60m
        ),

        "Humidity_Change_60m": (
            humidity_change_60m
        ),

        "Temperature_RollStd_30m": (
            temperature_rollstd_30m
        ),

        "Humidity_RollStd_30m": (
            humidity_rollstd_30m
        ),

        "Temperature_RollStd_60m": (
            temperature_rollstd_60m
        ),

        "Humidity_RollStd_60m": (
            humidity_rollstd_60m
        ),
    }

    features = pd.DataFrame(
        [
            feature_values
        ],
        columns=ORDERED_FEATURES,
    )

    # ========================================================
    # 7. VERIFY FINAL FEATURE ORDER
    # ========================================================

    if tuple(
        features.columns
    ) != ORDERED_FEATURES:
        raise RuntimeError(
            "Generated feature order does not match "
            "the finalized inference contract."
        )

    if features.shape != (
        1,
        len(ORDERED_FEATURES),
    ):
        raise RuntimeError(
            "Runtime feature builder did not generate "
            "exactly one 17-feature row."
        )

    # ========================================================
    # 8. IDENTIFY LEGITIMATE MISSING NUMERIC FEATURES
    # ========================================================
    #
    # Examples:
    #
    # First sensor reading:
    #
    # Temperature_Rate_per_10min = NaN
    # Temperature_Change_30m     = NaN
    # Temperature_Change_60m     = NaN
    #
    # These are legitimate.
    #
    # Both final pipelines contain their own trained numeric
    # median imputers.
    # ========================================================

    missing_numeric_features = tuple(
        feature
        for feature in NUMERIC_FEATURES
        if pd.isna(
            features.iloc[0][
                feature
            ]
        )
    )

    # ========================================================
    # 9. RETURN RESULT
    # ========================================================

    return FeatureBuildResult(
        features=features,
        missing_numeric_features=(
            missing_numeric_features
        ),
        historical_reading_count=len(
            historical_rows
        ),
    )
