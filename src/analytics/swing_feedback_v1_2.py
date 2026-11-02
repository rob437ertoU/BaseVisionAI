"""
BaseVisionAI - Swing Feedback V1.2
==================================

Robust session-relative swing feedback.

V1.2 improvements
-----------------
- Uses Swing Validity V1.1.
- Uses baseline_eligible=True to define the session baseline.
- Uses leave-one-out comparison when the target itself is baseline eligible.
- Uses Median + MAD + Robust Z-score as the primary comparison.
- Preserves Mean + SD + conventional Z-score for traceability.
- Reports metric coverage.
- Does NOT generate biomechanical GOOD/BAD judgments.

IMPORTANT
---------
HIGH / LOW / SLIGHTLY_HIGH / SLIGHTLY_LOW describe statistical
position relative to the player's own session baseline.

They do NOT indicate swing quality or baseball performance.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


VERSION = "Swing Feedback V1.2"


# =============================================================================
# METRICS
# =============================================================================

METRICS = [
    {
        "group": "WRIST MOTION",
        "name": "Right wrist motion",
        "column": "right_wrist_peak_torsoscale_frame",
        "unit": "torso scales/frame",
    },
    {
        "group": "WRIST MOTION",
        "name": "Left wrist motion",
        "column": "left_wrist_peak_torsoscale_frame",
        "unit": "torso scales/frame",
    },

    {
        "group": "BODY CENTER MOTION",
        "name": "Hip center motion",
        "column": "hip_center_peak_px_frame",
        "unit": "px/frame",
    },
    {
        "group": "BODY CENTER MOTION",
        "name": "Shoulder center motion",
        "column": "shoulder_center_peak_px_frame",
        "unit": "px/frame",
    },

    {
        "group": "PROJECTED ROTATION",
        "name": "Shoulder rotation",
        "column": "shoulder_rotation_projected_deg",
        "unit": "deg projected",
    },
    {
        "group": "PROJECTED ROTATION",
        "name": "Hip rotation",
        "column": "hip_rotation_projected_deg",
        "unit": "deg projected",
    },
    {
        "group": "PROJECTED ROTATION",
        "name": "Hip-shoulder separation",
        "column": "hip_shoulder_separation_projected_deg",
        "unit": "deg projected",
    },

    {
        "group": "PRE-PEAK STABILITY",
        "name": "Head pre-peak movement",
        "column": "head_prepeak_torso",
        "unit": "torso lengths",
    },
    {
        "group": "PRE-PEAK STABILITY",
        "name": "Left ankle pre-peak movement",
        "column": "left_ankle_prepeak_torso",
        "unit": "torso lengths",
    },
    {
        "group": "PRE-PEAK STABILITY",
        "name": "Right ankle pre-peak movement",
        "column": "right_ankle_prepeak_torso",
        "unit": "torso lengths",
    },

    {
        "group": "TIMING",
        "name": "Load duration",
        "column": "load_duration_sec",
        "unit": "s",
    },
    {
        "group": "TIMING",
        "name": "Swing duration",
        "column": "swing_duration_sec",
        "unit": "s",
    },
    {
        "group": "TIMING",
        "name": "Load to peak timing",
        "column": "load_to_peak_sec",
        "unit": "s",
    },
    {
        "group": "TIMING",
        "name": "Swing to peak timing",
        "column": "swing_to_peak_sec",
        "unit": "s",
    },
]


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Swing Feedback V1.2"
    )

    parser.add_argument(
        "--database",
        required=True,
        help="Player Swing Database V1 CSV",
    )

    parser.add_argument(
        "--validity",
        required=True,
        help="Swing Validity V1.1 CSV",
    )

    parser.add_argument(
        "--video-id",
        required=True,
        help="Target video ID",
    )

    parser.add_argument(
        "--player-id",
        required=True,
        help="Player ID",
    )

    parser.add_argument(
        "--session-id",
        required=True,
        help="Session ID",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory",
    )

    return parser.parse_args()


# =============================================================================
# HELPERS
# =============================================================================

def separator(char="-", width=96):
    print(char * width)


def safe_float(value):

    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except (TypeError, ValueError):
        pass

    return np.nan


def bool_from_value(value):

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def format_value(value, decimals=3):

    value = safe_float(value)

    if not np.isfinite(value):
        return "N/A"

    return f"{value:.{decimals}f}"


def median_absolute_deviation(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    median = np.median(values)

    return float(
        np.median(
            np.abs(values - median)
        )
    )


def conventional_z(
    target,
    mean,
    sd,
):

    if not all(
        np.isfinite(x)
        for x in [target, mean, sd]
    ):
        return np.nan

    if abs(sd) < 1e-12:
        return np.nan

    return (
        target - mean
    ) / sd


def robust_z(
    target,
    baseline_values,
):

    values = np.asarray(
        baseline_values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if (
        not np.isfinite(target)
        or len(values) == 0
    ):
        return np.nan, "UNAVAILABLE"

    median = np.median(values)

    mad = median_absolute_deviation(
        values
    )

    # Primary robust method
    if (
        np.isfinite(mad)
        and mad > 1e-12
    ):

        z = (
            0.67448975
            * (target - median)
            / mad
        )

        return float(z), "MAD"

    # -------------------------------------------------------------------------
    # Fallback if MAD collapses to zero
    # -------------------------------------------------------------------------

    if len(values) >= 2:

        mean = np.mean(values)

        sd = np.std(
            values,
            ddof=1,
        )

        if (
            np.isfinite(sd)
            and sd > 1e-12
        ):

            z = (
                target - mean
            ) / sd

            return float(z), "SD_FALLBACK"

    # Baseline values are effectively identical.
    if np.isclose(
        target,
        median,
        atol=1e-12,
    ):

        return 0.0, "ZERO_VARIANCE"

    return np.nan, "ZERO_VARIANCE"


def percent_difference(
    target,
    baseline_center,
):

    if not all(
        np.isfinite(x)
        for x in [
            target,
            baseline_center,
        ]
    ):
        return np.nan

    if abs(
        baseline_center
    ) < 1e-12:
        return np.nan

    return (
        (target - baseline_center)
        / abs(baseline_center)
        * 100.0
    )


# =============================================================================
# CLASSIFICATION
# =============================================================================

def classify_robust_z(z):

    if not np.isfinite(z):
        return "INSUFFICIENT_DATA"

    az = abs(z)

    if az < 0.75:
        return "CONSISTENT"

    if az < 1.50:

        if z > 0:
            return "SLIGHTLY_HIGH"

        return "SLIGHTLY_LOW"

    if z > 0:
        return "HIGH"

    return "LOW"


def feedback_text(
    metric_name,
    classification,
):

    mapping = {

        "CONSISTENT":
            (
                f"{metric_name}: consistent with the "
                "player's robust session baseline."
            ),

        "SLIGHTLY_HIGH":
            (
                f"{metric_name}: slightly above the "
                "player's robust session baseline."
            ),

        "SLIGHTLY_LOW":
            (
                f"{metric_name}: slightly below the "
                "player's robust session baseline."
            ),

        "HIGH":
            (
                f"{metric_name}: above the player's "
                "typical robust session range."
            ),

        "LOW":
            (
                f"{metric_name}: below the player's "
                "typical robust session range."
            ),

        "INSUFFICIENT_DATA":
            (
                f"{metric_name}: insufficient data for "
                "a robust session comparison."
            ),
    }

    return mapping[
        classification
    ]


# =============================================================================
# LOAD DATA
# =============================================================================

def load_database(path):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"Player Swing Database not found: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "video_id",
        "player_id",
        "session_id",
    }

    missing = required - set(
        df.columns
    )

    if missing:

        raise ValueError(
            "Player Swing Database is missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    return df


def load_validity(path):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"Swing Validity V1.1 not found: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "video_id",
        "player_id",
        "session_id",
        "validity_status_v1_1",
        "baseline_eligible",
    }

    missing = required - set(
        df.columns
    )

    if missing:

        raise ValueError(
            "Swing Validity V1.1 is missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    return df


# =============================================================================
# SESSION PREPARATION
# =============================================================================

def prepare_session(
    database,
    validity,
    player_id,
    session_id,
):

    profiles = database[
        (
            database["player_id"].astype(str)
            == str(player_id)
        )
        &
        (
            database["session_id"].astype(str)
            == str(session_id)
        )
    ].copy()

    if profiles.empty:

        raise ValueError(
            f"No profiles found for "
            f"{player_id} / {session_id}"
        )

    validity_session = validity[
        (
            validity["player_id"].astype(str)
            == str(player_id)
        )
        &
        (
            validity["session_id"].astype(str)
            == str(session_id)
        )
    ].copy()

    if validity_session.empty:

        raise ValueError(
            f"No validity data found for "
            f"{player_id} / {session_id}"
        )

    columns = [
        "video_id",
        "validity_status_v1_1",
        "baseline_eligible",
    ]

    optional = [
        "evidence_quality_v1_1",
        "temporal_confidence",
        "validity_reason_v1_1",
    ]

    for column in optional:

        if column in validity_session.columns:
            columns.append(column)

    validity_subset = (
        validity_session[
            columns
        ]
        .drop_duplicates(
            subset=["video_id"],
            keep="last",
        )
    )

    merged = profiles.merge(
        validity_subset,
        on="video_id",
        how="left",
        validate="many_to_one",
    )

    merged[
        "baseline_eligible"
    ] = (
        merged[
            "baseline_eligible"
        ]
        .apply(bool_from_value)
    )

    merged[
        "validity_status_v1_1"
    ] = (
        merged[
            "validity_status_v1_1"
        ]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return merged


# =============================================================================
# TARGET + BASELINE
# =============================================================================

def select_target_and_baseline(
    session,
    video_id,
):

    target_rows = session[
        session["video_id"].astype(str)
        == str(video_id)
    ].copy()

    if target_rows.empty:

        raise ValueError(
            f"Target video not found in session: {video_id}"
        )

    if len(target_rows) > 1:

        raise ValueError(
            f"Multiple target profiles found for video: {video_id}"
        )

    target = target_rows.iloc[0]

    eligible = session[
        session["baseline_eligible"]
    ].copy()

    target_eligible = bool(
        target[
            "baseline_eligible"
        ]
    )

    if target_eligible:

        baseline = eligible[
            eligible["video_id"].astype(str)
            != str(video_id)
        ].copy()

        baseline_mode = "LEAVE_ONE_OUT"
        target_excluded = True

    else:

        baseline = eligible.copy()

        baseline_mode = "FULL_ELIGIBLE_BASELINE"
        target_excluded = False

    if len(baseline) < 2:

        raise ValueError(
            "Swing Feedback V1.2 requires at least "
            "2 baseline swings after target selection."
        )

    return (
        target,
        baseline,
        baseline_mode,
        target_excluded,
    )


# =============================================================================
# METRIC ANALYSIS
# =============================================================================

def analyze_metric(
    target,
    baseline,
    config,
):

    column = config[
        "column"
    ]

    if column not in target.index:

        return {
            "available": False,
            "reason": "metric_not_available",
        }

    target_value = safe_float(
        target[column]
    )

    if column not in baseline.columns:

        return {
            "available": False,
            "reason": "baseline_metric_not_available",
        }

    baseline_values = (
        pd.to_numeric(
            baseline[column],
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=float)
    )

    if (
        not np.isfinite(target_value)
        or len(baseline_values) < 2
    ):

        return {
            "available": False,
            "reason": "insufficient_metric_data",
            "target_value": target_value,
            "baseline_n": len(
                baseline_values
            ),
        }

    mean = float(
        np.mean(
            baseline_values
        )
    )

    median = float(
        np.median(
            baseline_values
        )
    )

    if len(
        baseline_values
    ) >= 2:

        sd = float(
            np.std(
                baseline_values,
                ddof=1,
            )
        )

    else:
        sd = np.nan

    mad = (
        median_absolute_deviation(
            baseline_values
        )
    )

    q1 = float(
        np.percentile(
            baseline_values,
            25,
        )
    )

    q3 = float(
        np.percentile(
            baseline_values,
            75,
        )
    )

    iqr = (
        q3 - q1
    )

    rz, rz_method = robust_z(
        target_value,
        baseline_values,
    )

    z = conventional_z(
        target_value,
        mean,
        sd,
    )

    difference = percent_difference(
        target_value,
        median,
    )

    classification = (
        classify_robust_z(
            rz
        )
    )

    feedback = (
        feedback_text(
            config["name"],
            classification,
        )
    )

    return {
        "available": True,

        "target_value":
            target_value,

        "baseline_n":
            len(baseline_values),

        "baseline_mean":
            mean,

        "baseline_sd":
            sd,

        "baseline_median":
            median,

        "baseline_mad":
            mad,

        "baseline_q1":
            q1,

        "baseline_q3":
            q3,

        "baseline_iqr":
            iqr,

        "difference_percent":
            difference,

        "z_score":
            z,

        "robust_z_score":
            rz,

        "robust_z_method":
            rz_method,

        "classification":
            classification,

        "feedback":
            feedback,
    }


# =============================================================================
# FEEDBACK DATABASE
# =============================================================================

def build_feedback(
    target,
    baseline,
    baseline_mode,
    target_excluded,
):

    rows = []

    for config in METRICS:

        result = analyze_metric(
            target=target,
            baseline=baseline,
            config=config,
        )

        row = {

            "feedback_version":
                VERSION,

            "video_id":
                target["video_id"],

            "player_id":
                target["player_id"],

            "session_id":
                target["session_id"],

            "validity_status":
                target[
                    "validity_status_v1_1"
                ],

            "target_baseline_eligible":
                bool(
                    target[
                        "baseline_eligible"
                    ]
                ),

            "baseline_mode":
                baseline_mode,

            "target_excluded_from_baseline":
                target_excluded,

            "baseline_profiles":
                len(baseline),

            "group":
                config["group"],

            "metric":
                config["name"],

            "source_column":
                config["column"],

            "unit":
                config["unit"],

            "available":
                result.get(
                    "available",
                    False,
                ),

            "reason":
                result.get(
                    "reason",
                    "",
                ),

            "target_value":
                result.get(
                    "target_value",
                    np.nan,
                ),

            "baseline_n":
                result.get(
                    "baseline_n",
                    0,
                ),

            "baseline_mean":
                result.get(
                    "baseline_mean",
                    np.nan,
                ),

            "baseline_sd":
                result.get(
                    "baseline_sd",
                    np.nan,
                ),

            "baseline_median":
                result.get(
                    "baseline_median",
                    np.nan,
                ),

            "baseline_mad":
                result.get(
                    "baseline_mad",
                    np.nan,
                ),

            "baseline_q1":
                result.get(
                    "baseline_q1",
                    np.nan,
                ),

            "baseline_q3":
                result.get(
                    "baseline_q3",
                    np.nan,
                ),

            "baseline_iqr":
                result.get(
                    "baseline_iqr",
                    np.nan,
                ),

            "difference_percent":
                result.get(
                    "difference_percent",
                    np.nan,
                ),

            "z_score":
                result.get(
                    "z_score",
                    np.nan,
                ),

            "robust_z_score":
                result.get(
                    "robust_z_score",
                    np.nan,
                ),

            "robust_z_method":
                result.get(
                    "robust_z_method",
                    "",
                ),

            "classification":
                result.get(
                    "classification",
                    "INSUFFICIENT_DATA",
                ),

            "feedback":
                result.get(
                    "feedback",
                    (
                        f"{config['name']}: insufficient "
                        "data for comparison."
                    ),
                ),
        }

        rows.append(row)

    return pd.DataFrame(
        rows
    )


# =============================================================================
# MEASUREMENT CONFIDENCE
# =============================================================================

def measurement_confidence(
    target,
):

    pose_quality = safe_float(
        target.get(
            "mean_pose_quality",
            np.nan,
        )
    )

    if not np.isfinite(
        pose_quality
    ):

        return (
            pose_quality,
            "UNKNOWN",
        )

    if pose_quality >= 0.90:

        confidence = "HIGH"

    elif pose_quality >= 0.75:

        confidence = "MODERATE"

    else:

        confidence = "LOW"

    return (
        pose_quality,
        confidence,
    )


# =============================================================================
# SUMMARY
# =============================================================================

def build_summary(
    target,
    baseline,
    feedback,
    baseline_mode,
    target_excluded,
):

    available = feedback[
        feedback["available"] == True
    ].copy()

    total_metrics = len(
        feedback
    )

    available_metrics = len(
        available
    )

    coverage = (
        available_metrics
        / total_metrics
        * 100.0
        if total_metrics > 0
        else np.nan
    )

    classifications = (
        available[
            "classification"
        ]
        .value_counts()
        .to_dict()
    )

    consistent = int(
        classifications.get(
            "CONSISTENT",
            0,
        )
    )

    slightly_high = int(
        classifications.get(
            "SLIGHTLY_HIGH",
            0,
        )
    )

    slightly_low = int(
        classifications.get(
            "SLIGHTLY_LOW",
            0,
        )
    )

    high = int(
        classifications.get(
            "HIGH",
            0,
        )
    )

    low = int(
        classifications.get(
            "LOW",
            0,
        )
    )

    slightly_different = (
        slightly_high
        + slightly_low
    )

    outside_range = (
        high
        + low
    )

    pose_quality, pose_confidence = (
        measurement_confidence(
            target
        )
    )

    robust_z_values = pd.to_numeric(
        available[
            "robust_z_score"
        ],
        errors="coerce",
    ).abs().dropna()

    mean_abs_rz = (
        float(
            robust_z_values.mean()
        )
        if len(robust_z_values)
        else np.nan
    )

    median_abs_rz = (
        float(
            robust_z_values.median()
        )
        if len(robust_z_values)
        else np.nan
    )

    row = {

        "feedback_version":
            VERSION,

        "video_id":
            target["video_id"],

        "player_id":
            target["player_id"],

        "session_id":
            target["session_id"],

        "validity_status":
            target[
                "validity_status_v1_1"
            ],

        "target_baseline_eligible":
            bool(
                target[
                    "baseline_eligible"
                ]
            ),

        "baseline_mode":
            baseline_mode,

        "target_excluded_from_baseline":
            target_excluded,

        "baseline_profiles":
            len(baseline),

        "metrics_configured":
            total_metrics,

        "metrics_available":
            available_metrics,

        "metric_coverage_percent":
            coverage,

        "consistent":
            consistent,

        "slightly_high":
            slightly_high,

        "slightly_low":
            slightly_low,

        "slightly_different":
            slightly_different,

        "high":
            high,

        "low":
            low,

        "outside_typical_range":
            outside_range,

        "mean_abs_robust_z":
            mean_abs_rz,

        "median_abs_robust_z":
            median_abs_rz,

        "mean_pose_quality":
            pose_quality,

        "measurement_confidence":
            pose_confidence,

        "baseline_filter":
            "baseline_eligible=True",

        "primary_statistics":
            "Median/MAD/Robust Z",

        "performance_score_generated":
            False,

        "calibrated_biomechanics":
            False,
    }

    return pd.DataFrame(
        [row]
    )


# =============================================================================
# REPORT
# =============================================================================

def print_report(
    session,
    target,
    baseline,
    feedback,
    summary,
    baseline_mode,
    target_excluded,
    output_detail,
    output_summary,
):

    print()
    separator("=")

    print(
        "BASEVISIONAI - SWING FEEDBACK V1.2 COMPLETE"
    )

    separator("=")

    print()
    print("TARGET")
    separator()

    print(
        f"Video ID                        : "
        f"{target['video_id']}"
    )

    print(
        f"Player ID                       : "
        f"{target['player_id']}"
    )

    print(
        f"Session ID                      : "
        f"{target['session_id']}"
    )

    print(
        f"Swing validity                  : "
        f"{target['validity_status_v1_1']}"
    )

    print(
        f"Target baseline eligible        : "
        f"{bool(target['baseline_eligible'])}"
    )

    print()
    print("SESSION BASELINE")
    separator()

    print(
        f"Session profiles                : "
        f"{len(session)}"
    )

    print(
        f"Baseline eligible profiles      : "
        f"{int(session['baseline_eligible'].sum())}"
    )

    print(
        f"Profiles used for this target   : "
        f"{len(baseline)}"
    )

    print(
        f"Baseline mode                   : "
        f"{baseline_mode}"
    )

    print(
        f"Target excluded                 : "
        f"{target_excluded}"
    )

    print()
    print("BASELINE VIDEOS")
    separator()

    for video_id in baseline[
        "video_id"
    ].tolist():

        print(
            f"  {video_id}"
        )

    print()
    print("ROBUST COMPARATIVE FEEDBACK")
    separator("=")

    current_group = None

    for _, row in feedback.iterrows():

        if row[
            "group"
        ] != current_group:

            print()
            print(
                row["group"]
            )

            separator()

            current_group = row[
                "group"
            ]

        print()
        print(
            row["metric"]
        )

        if not bool(
            row["available"]
        ):

            print(
                "  Status                         : "
                "INSUFFICIENT_DATA"
            )

            print(
                f"  Reason                         : "
                f"{row['reason']}"
            )

            continue

        print(
            f"  Target                         : "
            f"{format_value(row['target_value'])} "
            f"{row['unit']}"
        )

        print(
            f"  Baseline samples               : "
            f"{int(row['baseline_n'])}"
        )

        print(
            f"  Baseline median                : "
            f"{format_value(row['baseline_median'])} "
            f"{row['unit']}"
        )

        print(
            f"  Baseline MAD                   : "
            f"{format_value(row['baseline_mad'])} "
            f"{row['unit']}"
        )

        print(
            f"  Difference from median         : "
            f"{format_value(row['difference_percent'], 1)}%"
        )

        print(
            f"  Robust Z-score                 : "
            f"{format_value(row['robust_z_score'], 2)}"
        )

        print(
            f"  Robust Z method                : "
            f"{row['robust_z_method']}"
        )

        print(
            f"  Classification                 : "
            f"{row['classification']}"
        )

        print(
            f"  Feedback                       : "
            f"{row['feedback']}"
        )

        print(
            f"  Traditional mean / SD          : "
            f"{format_value(row['baseline_mean'])} / "
            f"{format_value(row['baseline_sd'])}"
        )

    s = summary.iloc[0]

    print()
    print("MEASUREMENT CONFIDENCE")
    separator()

    print(
        f"Mean pose quality               : "
        f"{format_value(s['mean_pose_quality'])}"
    )

    print(
        f"Measurement confidence          : "
        f"{s['measurement_confidence']}"
    )

    print()
    print(
        "Measurement confidence describes pose-estimation "
        "reliability, not swing quality."
    )

    print()
    print("SUMMARY")
    separator()

    print(
        f"Metrics configured              : "
        f"{int(s['metrics_configured'])}"
    )

    print(
        f"Metrics available               : "
        f"{int(s['metrics_available'])}"
    )

    print(
        f"Metric coverage                 : "
        f"{format_value(s['metric_coverage_percent'], 1)}%"
    )

    print(
        f"Consistent                      : "
        f"{int(s['consistent'])}"
    )

    print(
        f"Slightly different              : "
        f"{int(s['slightly_different'])}"
    )

    print(
        f"Outside typical range           : "
        f"{int(s['outside_typical_range'])}"
    )

    print(
        f"  HIGH                          : "
        f"{int(s['high'])}"
    )

    print(
        f"  LOW                           : "
        f"{int(s['low'])}"
    )

    print(
        f"Mean |robust Z|                 : "
        f"{format_value(s['mean_abs_robust_z'], 2)}"
    )

    print(
        f"Median |robust Z|               : "
        f"{format_value(s['median_abs_robust_z'], 2)}"
    )

    print()
    print("OUTPUT FILES")
    separator()

    print(
        f"Detailed feedback               : "
        f"{output_detail}"
    )

    print(
        f"Summary                         : "
        f"{output_summary}"
    )

    print()
    print("IMPORTANT")
    separator()

    print(
        "Feedback is relative to this player's "
        "baseline_eligible session swings."
    )

    print(
        "A baseline-eligible target is excluded from its "
        "own baseline using leave-one-out comparison."
    )

    print(
        "Median/MAD/Robust Z are the primary comparison "
        "statistics in V1.2."
    )

    print(
        "HIGH/LOW describe statistical position, "
        "not swing quality."
    )

    print(
        "No GOOD/BAD, biomechanical quality, or "
        "performance score is inferred."
    )

    print(
        "Projected angles remain comparative 2D measurements."
    )

    separator("=")


# =============================================================================
# MAIN
# =============================================================================

def main():

    args = parse_args()

    try:

        database = load_database(
            args.database
        )

        validity = load_validity(
            args.validity
        )

        session = prepare_session(
            database=database,
            validity=validity,
            player_id=args.player_id,
            session_id=args.session_id,
        )

        (
            target,
            baseline,
            baseline_mode,
            target_excluded,
        ) = select_target_and_baseline(
            session=session,
            video_id=args.video_id,
        )

        feedback = build_feedback(
            target=target,
            baseline=baseline,
            baseline_mode=baseline_mode,
            target_excluded=target_excluded,
        )

        summary = build_summary(
            target=target,
            baseline=baseline,
            feedback=feedback,
            baseline_mode=baseline_mode,
            target_excluded=target_excluded,
        )

        output_dir = Path(
            args.output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        detail_path = (
            output_dir
            / (
                f"{args.video_id}"
                "_swing_feedback_v1_2.csv"
            )
        )

        summary_path = (
            output_dir
            / (
                f"{args.video_id}"
                "_swing_feedback_v1_2_summary.csv"
            )
        )

        feedback.to_csv(
            detail_path,
            index=False,
        )

        summary.to_csv(
            summary_path,
            index=False,
        )

        print_report(
            session=session,
            target=target,
            baseline=baseline,
            feedback=feedback,
            summary=summary,
            baseline_mode=baseline_mode,
            target_excluded=target_excluded,
            output_detail=detail_path,
            output_summary=summary_path,
        )

    except Exception as exc:

        print()
        separator("=")

        print(
            "BASEVISIONAI - SWING FEEDBACK V1.2 FAILED"
        )

        separator("=")

        print()
        print(
            f"ERROR: {exc}"
        )
        print()

        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())