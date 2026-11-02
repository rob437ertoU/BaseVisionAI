"""
BaseVisionAI - Swing Feedback V1.1

Generates comparative feedback for one swing relative to the player's
VALID_SWING session baseline.

The target swing is excluded from its own baseline whenever possible
(leave-one-out comparison).

IMPORTANT:
- This is NOT a biomechanical quality score.
- This does NOT classify a swing as good or bad.
- Comparisons are relative to the player's own session.
- Projected 2D measurements must not be interpreted as calibrated
  3D biomechanics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

METRICS = {

    # =========================================================================
    # WRIST MOTION - BODY NORMALIZED
    # =========================================================================

    "right_wrist_motion": {
        "category": "WRIST MOTION",
        "label": "Right wrist motion",
        "unit": "torso scales/frame",
        "candidates": [
            "right_wrist_peak_torsoscale_frame",
        ],
    },

    "left_wrist_motion": {
        "category": "WRIST MOTION",
        "label": "Left wrist motion",
        "unit": "torso scales/frame",
        "candidates": [
            "left_wrist_peak_torsoscale_frame",
        ],
    },

    # =========================================================================
    # BODY CENTER MOTION
    # =========================================================================

    "hip_center_motion": {
        "category": "BODY CENTER MOTION",
        "label": "Hip center motion",
        "unit": "px/frame",
        "candidates": [
            "hip_center_peak_px_frame",
        ],
    },

    "shoulder_center_motion": {
        "category": "BODY CENTER MOTION",
        "label": "Shoulder center motion",
        "unit": "px/frame",
        "candidates": [
            "shoulder_center_peak_px_frame",
        ],
    },

    # =========================================================================
    # PROJECTED ROTATION
    # =========================================================================

    "shoulder_rotation": {
        "category": "PROJECTED ROTATION",
        "label": "Shoulder rotation",
        "unit": "deg projected",
        "candidates": [
            "shoulder_rotation_projected_deg",
        ],
    },

    "hip_rotation": {
        "category": "PROJECTED ROTATION",
        "label": "Hip rotation",
        "unit": "deg projected",
        "candidates": [
            "hip_rotation_projected_deg",
        ],
    },

    "hip_shoulder_separation": {
        "category": "PROJECTED ROTATION",
        "label": "Hip-shoulder separation",
        "unit": "deg projected",
        "candidates": [
            "hip_shoulder_separation_projected_deg",
        ],
    },

    # =========================================================================
    # PRE-PEAK STABILITY
    # =========================================================================

    "head_prepeak": {
        "category": "PRE-PEAK STABILITY",
        "label": "Head pre-peak movement",
        "unit": "torso lengths",
        "candidates": [
            "head_prepeak_torso",
        ],
    },

    "left_ankle_prepeak": {
        "category": "PRE-PEAK STABILITY",
        "label": "Left ankle pre-peak movement",
        "unit": "torso lengths",
        "candidates": [
            "left_ankle_prepeak_torso",
        ],
    },

    "right_ankle_prepeak": {
        "category": "PRE-PEAK STABILITY",
        "label": "Right ankle pre-peak movement",
        "unit": "torso lengths",
        "candidates": [
            "right_ankle_prepeak_torso",
        ],
    },

    # =========================================================================
    # TIMING
    # =========================================================================

    "load_duration": {
        "category": "TIMING",
        "label": "Load duration",
        "unit": "s",
        "candidates": [
            "load_duration_sec",
        ],
    },

    "swing_duration": {
        "category": "TIMING",
        "label": "Swing duration",
        "unit": "s",
        "candidates": [
            "swing_duration_sec",
        ],
    },

    "load_to_peak": {
        "category": "TIMING",
        "label": "Load to peak timing",
        "unit": "s",
        "candidates": [
            "load_to_peak_sec",
        ],
    },

    "swing_to_peak": {
        "category": "TIMING",
        "label": "Swing to peak timing",
        "unit": "s",
        "candidates": [
            "swing_to_peak_sec",
        ],
    },
}


# =============================================================================
# COLUMN HELPERS
# =============================================================================

def normalize_name(name: str) -> str:
    """
    Normalize column names for tolerant matching.
    """

    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(">", "to")
    )


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """
    Search for a column using normalized candidate names.
    """

    normalized = {
        normalize_name(column): column
        for column in df.columns
    }

    for candidate in candidates:
        key = normalize_name(candidate)

        if key in normalized:
            return normalized[key]

    return None


def require_column(
    df: pd.DataFrame,
    candidates: list[str],
    description: str,
) -> str:
    """
    Find a required column or raise a descriptive error.
    """

    column = find_column(df, candidates)

    if column is None:
        raise ValueError(
            f"Could not detect {description} column.\n"
            f"Candidates: {candidates}\n\n"
            f"Available columns:\n  "
            + "\n  ".join(map(str, df.columns))
        )

    return column


# =============================================================================
# NUMERIC HELPERS
# =============================================================================

def safe_float(value) -> float:
    """
    Convert a value to finite float.
    """

    try:
        number = float(value)

        if np.isfinite(number):
            return number

    except (TypeError, ValueError):
        pass

    return np.nan


def format_value(
    value: float,
    unit: str,
) -> str:
    """
    Human-readable metric formatting.
    """

    if not np.isfinite(value):
        return "N/A"

    if unit == "s":
        return f"{value:.3f} s"

    if unit == "px/frame":
        return f"{value:.2f} px/frame"

    if unit == "torso scales/frame":
        return f"{value:.3f} torso scales/frame"

    if unit == "torso lengths":
        return f"{value:.3f} torso lengths"

    if unit == "deg projected":
        return f"{value:.2f} deg projected"

    return f"{value:.3f}"


# =============================================================================
# COMPARISON LOGIC
# =============================================================================

def classify_difference(
    value: float,
    mean: float,
    std: float,
) -> tuple[str, float]:
    """
    Compare target metric against player's own session baseline.

    Classification based on standard deviation:

        |z| < 0.5
            CONSISTENT

        0.5 <= |z| < 1.0
            SLIGHTLY_HIGH / SLIGHTLY_LOW

        |z| >= 1.0
            HIGH / LOW

    HIGH and LOW are statistical descriptions only.
    They do NOT represent swing quality.

    If baseline SD is unavailable or effectively zero,
    percentage difference is used as fallback.
    """

    if (
        not np.isfinite(value)
        or not np.isfinite(mean)
    ):
        return "INSUFFICIENT_DATA", np.nan

    # -------------------------------------------------------------------------
    # Z-score method
    # -------------------------------------------------------------------------

    if (
        np.isfinite(std)
        and std > 1e-9
    ):

        z_score = (value - mean) / std

        if abs(z_score) < 0.5:
            return "CONSISTENT", z_score

        if abs(z_score) < 1.0:

            if z_score > 0:
                return "SLIGHTLY_HIGH", z_score

            return "SLIGHTLY_LOW", z_score

        if z_score > 0:
            return "HIGH", z_score

        return "LOW", z_score

    # -------------------------------------------------------------------------
    # Percentage fallback
    # -------------------------------------------------------------------------

    if abs(mean) > 1e-9:

        relative_difference = (
            value - mean
        ) / abs(mean)

        if abs(relative_difference) < 0.10:
            return "CONSISTENT", np.nan

        if abs(relative_difference) < 0.20:

            if relative_difference > 0:
                return "SLIGHTLY_HIGH", np.nan

            return "SLIGHTLY_LOW", np.nan

        if relative_difference > 0:
            return "HIGH", np.nan

        return "LOW", np.nan

    return "CONSISTENT", np.nan


def make_feedback(
    label: str,
    status: str,
) -> str:
    """
    Generate neutral comparative language.

    Deliberately avoids:
    - good
    - bad
    - better
    - worse
    - optimal
    """

    if status == "INSUFFICIENT_DATA":
        return (
            f"{label}: insufficient data "
            f"for session comparison."
        )

    if status == "CONSISTENT":
        return (
            f"{label}: consistent with the "
            f"player's session baseline."
        )

    if status == "SLIGHTLY_HIGH":
        return (
            f"{label}: slightly above the "
            f"player's session baseline."
        )

    if status == "SLIGHTLY_LOW":
        return (
            f"{label}: slightly below the "
            f"player's session baseline."
        )

    if status == "HIGH":
        return (
            f"{label}: above the player's "
            f"typical session range."
        )

    if status == "LOW":
        return (
            f"{label}: below the player's "
            f"typical session range."
        )

    return (
        f"{label}: comparative status unavailable."
    )


# =============================================================================
# MEASUREMENT CONFIDENCE
# =============================================================================

def classify_measurement_confidence(
    pose_quality: float,
) -> str:
    """
    Pose quality is handled separately from swing metrics.

    It describes confidence in the pose-derived measurement,
    NOT quality of the player's swing.
    """

    if not np.isfinite(pose_quality):
        return "UNKNOWN"

    if pose_quality >= 0.90:
        return "HIGH"

    if pose_quality >= 0.80:
        return "MODERATE"

    return "LOW"


# =============================================================================
# MAIN
# =============================================================================

def main():

    # =========================================================================
    # ARGUMENTS
    # =========================================================================

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Swing Feedback V1.1"
    )

    parser.add_argument(
        "--database",
        required=True,
        help="Player Swing Database V1 CSV",
    )

    parser.add_argument(
        "--validity",
        required=True,
        help="Swing Validity V1 CSV",
    )

    parser.add_argument(
        "--video-id",
        required=True,
        help="Swing/video to analyze",
    )

    parser.add_argument(
        "--player-id",
        required=True,
        help="Player identifier",
    )

    parser.add_argument(
        "--session-id",
        required=True,
        help="Session identifier",
    )

    parser.add_argument(
        "--output-dir",
        default="results/swing_feedback",
        help="Output directory",
    )

    args = parser.parse_args()

    # =========================================================================
    # PATHS
    # =========================================================================

    database_path = Path(args.database)
    validity_path = Path(args.validity)
    output_dir = Path(args.output_dir)

    if not database_path.exists():
        print(
            f"ERROR: Database not found:\n"
            f"{database_path}"
        )
        sys.exit(1)

    if not validity_path.exists():
        print(
            f"ERROR: Validity file not found:\n"
            f"{validity_path}"
        )
        sys.exit(1)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    database = pd.read_csv(database_path)
    validity = pd.read_csv(validity_path)

    # =========================================================================
    # REQUIRED DATABASE COLUMNS
    # =========================================================================

    video_col = require_column(
        database,
        [
            "video_id",
            "video",
        ],
        "database video ID",
    )

    player_col = require_column(
        database,
        [
            "player_id",
            "player",
        ],
        "database player ID",
    )

    session_col = require_column(
        database,
        [
            "session_id",
            "session",
        ],
        "database session ID",
    )

    # =========================================================================
    # REQUIRED VALIDITY COLUMNS
    # =========================================================================

    validity_video_col = require_column(
        validity,
        [
            "video_id",
            "video",
        ],
        "validity video ID",
    )

    validity_status_col = require_column(
        validity,
        [
            "swing_validity",
            "validity",
            "validity_status",
            "swing_status",
            "status",
        ],
        "swing validity status",
    )

    # =========================================================================
    # SELECT PLAYER + SESSION
    # =========================================================================

    session_db = database[
        (
            database[player_col].astype(str)
            == str(args.player_id)
        )
        &
        (
            database[session_col].astype(str)
            == str(args.session_id)
        )
    ].copy()

    if session_db.empty:

        print(
            "ERROR: No profiles found for:\n"
            f"Player  : {args.player_id}\n"
            f"Session : {args.session_id}"
        )

        sys.exit(1)

    # =========================================================================
    # SELECT TARGET
    # =========================================================================

    target_rows = session_db[
        session_db[video_col].astype(str)
        == str(args.video_id)
    ]

    if target_rows.empty:

        print(
            "ERROR: Target video not found "
            "in database:\n"
            f"{args.video_id}"
        )

        sys.exit(1)

    target = target_rows.iloc[0]

    # =========================================================================
    # MERGE VALIDITY
    # =========================================================================

    validity_small = validity[
        [
            validity_video_col,
            validity_status_col,
        ]
    ].copy()

    validity_small.columns = [
        "__validity_video",
        "__validity_status",
    ]

    session_db = session_db.merge(
        validity_small,
        left_on=video_col,
        right_on="__validity_video",
        how="left",
    )

    # =========================================================================
    # VALID SWINGS
    # =========================================================================

    valid_mask = (
        session_db["__validity_status"]
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("VALID_SWING")
    )

    valid_db = session_db[
        valid_mask
    ].copy()

    if valid_db.empty:

        print(
            "ERROR: No VALID_SWING profiles "
            "available for baseline."
        )

        sys.exit(1)

    # =========================================================================
    # LEAVE-ONE-OUT BASELINE
    # =========================================================================

    baseline = valid_db[
        valid_db[video_col].astype(str)
        != str(args.video_id)
    ].copy()

    baseline_mode = "LEAVE_ONE_OUT"

    # If there are no other valid swings,
    # use the complete valid subset.
    if baseline.empty:

        baseline = valid_db.copy()

        baseline_mode = "FULL_VALID_BASELINE"

    # =========================================================================
    # TARGET VALIDITY
    # =========================================================================

    target_validity_rows = validity[
        validity[validity_video_col]
        .astype(str)
        .eq(str(args.video_id))
    ]

    if not target_validity_rows.empty:

        target_validity = str(
            target_validity_rows
            .iloc[0][validity_status_col]
        )

    else:

        target_validity = "UNKNOWN"

    # =========================================================================
    # ANALYZE METRICS
    # =========================================================================

    rows = []

    missing_metrics = []

    for metric_key, config in METRICS.items():

        metric_column = find_column(
            session_db,
            config["candidates"],
        )

        # ---------------------------------------------------------------------
        # Metric unavailable
        # ---------------------------------------------------------------------

        if metric_column is None:

            missing_metrics.append(
                config["label"]
            )

            continue

        # ---------------------------------------------------------------------
        # Target value
        # ---------------------------------------------------------------------

        target_value = safe_float(
            target.get(
                metric_column,
                np.nan,
            )
        )

        # ---------------------------------------------------------------------
        # Baseline values
        # ---------------------------------------------------------------------

        baseline_values = pd.to_numeric(
            baseline[metric_column],
            errors="coerce",
        ).dropna()

        baseline_n = len(
            baseline_values
        )

        # ---------------------------------------------------------------------
        # Statistics
        # ---------------------------------------------------------------------

        if baseline_n == 0:

            baseline_mean = np.nan
            baseline_std = np.nan
            baseline_median = np.nan
            baseline_min = np.nan
            baseline_max = np.nan

        else:

            baseline_mean = float(
                baseline_values.mean()
            )

            baseline_median = float(
                baseline_values.median()
            )

            baseline_min = float(
                baseline_values.min()
            )

            baseline_max = float(
                baseline_values.max()
            )

            if baseline_n >= 2:

                baseline_std = float(
                    baseline_values.std(
                        ddof=1
                    )
                )

            else:

                baseline_std = np.nan

        # ---------------------------------------------------------------------
        # Classification
        # ---------------------------------------------------------------------

        comparison, z_score = (
            classify_difference(
                target_value,
                baseline_mean,
                baseline_std,
            )
        )

        # ---------------------------------------------------------------------
        # Delta
        # ---------------------------------------------------------------------

        if (
            np.isfinite(target_value)
            and np.isfinite(baseline_mean)
        ):

            delta = (
                target_value
                - baseline_mean
            )

        else:

            delta = np.nan

        # ---------------------------------------------------------------------
        # Percentage difference
        # ---------------------------------------------------------------------

        if (
            np.isfinite(delta)
            and np.isfinite(baseline_mean)
            and abs(baseline_mean) > 1e-9
        ):

            percent_delta = (
                100.0
                * delta
                / abs(baseline_mean)
            )

        else:

            percent_delta = np.nan

        # ---------------------------------------------------------------------
        # Feedback
        # ---------------------------------------------------------------------

        feedback = make_feedback(
            config["label"],
            comparison,
        )

        # ---------------------------------------------------------------------
        # Save row
        # ---------------------------------------------------------------------

        rows.append(
            {
                "feedback_version":
                    "Swing Feedback V1.1",

                "video_id":
                    args.video_id,

                "player_id":
                    args.player_id,

                "session_id":
                    args.session_id,

                "target_validity":
                    target_validity,

                "category":
                    config["category"],

                "metric_key":
                    metric_key,

                "metric":
                    config["label"],

                "source_column":
                    metric_column,

                "unit":
                    config["unit"],

                "target_value":
                    target_value,

                "baseline_n":
                    baseline_n,

                "baseline_mean":
                    baseline_mean,

                "baseline_std":
                    baseline_std,

                "baseline_median":
                    baseline_median,

                "baseline_min":
                    baseline_min,

                "baseline_max":
                    baseline_max,

                "delta_from_mean":
                    delta,

                "percent_delta":
                    percent_delta,

                "z_score":
                    z_score,

                "comparison":
                    comparison,

                "feedback":
                    feedback,

                "baseline_mode":
                    baseline_mode,
            }
        )

    # =========================================================================
    # CHECK METRICS
    # =========================================================================

    if not rows:

        print(
            "ERROR: No compatible metric "
            "columns were detected."
        )

        print()
        print("Available database columns:")

        for column in database.columns:
            print(f"  - {column}")

        sys.exit(1)

    feedback_df = pd.DataFrame(
        rows
    )

    # =========================================================================
    # MEASUREMENT CONFIDENCE
    # =========================================================================

    pose_quality_col = find_column(
        session_db,
        [
            "mean_pose_quality",
            "pose_quality",
        ],
    )

    if pose_quality_col is not None:

        pose_quality = safe_float(
            target.get(
                pose_quality_col,
                np.nan,
            )
        )

    else:

        pose_quality = np.nan

    measurement_confidence = (
        classify_measurement_confidence(
            pose_quality
        )
    )

    # =========================================================================
    # SUMMARY COUNTS
    # =========================================================================

    comparable = feedback_df[
        feedback_df["comparison"]
        != "INSUFFICIENT_DATA"
    ]

    consistent_count = int(
        comparable[
            "comparison"
        ]
        .eq("CONSISTENT")
        .sum()
    )

    slight_count = int(
        comparable[
            "comparison"
        ]
        .isin(
            [
                "SLIGHTLY_HIGH",
                "SLIGHTLY_LOW",
            ]
        )
        .sum()
    )

    outside_count = int(
        comparable[
            "comparison"
        ]
        .isin(
            [
                "HIGH",
                "LOW",
            ]
        )
        .sum()
    )

    high_count = int(
        comparable[
            "comparison"
        ]
        .eq("HIGH")
        .sum()
    )

    low_count = int(
        comparable[
            "comparison"
        ]
        .eq("LOW")
        .sum()
    )

    # =========================================================================
    # OUTPUT PATHS
    # =========================================================================

    detail_output = (
        output_dir
        / f"{args.video_id}_swing_feedback_v1_1.csv"
    )

    summary_output = (
        output_dir
        / f"{args.video_id}_swing_feedback_v1_1_summary.csv"
    )

    # =========================================================================
    # SAVE DETAILED FEEDBACK
    # =========================================================================

    feedback_df.to_csv(
        detail_output,
        index=False,
    )

    # =========================================================================
    # SAVE SUMMARY
    # =========================================================================

    summary_df = pd.DataFrame(
        [
            {
                "feedback_version":
                    "Swing Feedback V1.1",

                "video_id":
                    args.video_id,

                "player_id":
                    args.player_id,

                "session_id":
                    args.session_id,

                "target_validity":
                    target_validity,

                "session_profiles":
                    len(session_db),

                "valid_session_profiles":
                    len(valid_db),

                "baseline_profiles":
                    len(baseline),

                "baseline_mode":
                    baseline_mode,

                "metrics_available":
                    len(feedback_df),

                "metrics_compared":
                    len(comparable),

                "consistent_metrics":
                    consistent_count,

                "slightly_different_metrics":
                    slight_count,

                "outside_typical_range_metrics":
                    outside_count,

                "high_metrics":
                    high_count,

                "low_metrics":
                    low_count,

                "mean_pose_quality":
                    pose_quality,

                "measurement_confidence":
                    measurement_confidence,

                "missing_metrics":
                    "; ".join(
                        missing_metrics
                    ),
            }
        ]
    )

    summary_df.to_csv(
        summary_output,
        index=False,
    )

    # =========================================================================
    # TERMINAL REPORT
    # =========================================================================

    print()
    print("=" * 88)
    print(
        "BASEVISIONAI - "
        "SWING FEEDBACK V1.1 COMPLETE"
    )
    print("=" * 88)

    # -------------------------------------------------------------------------
    # TARGET
    # -------------------------------------------------------------------------

    print()
    print("TARGET")
    print("-" * 88)

    print(
        f"Video ID                  : "
        f"{args.video_id}"
    )

    print(
        f"Player ID                 : "
        f"{args.player_id}"
    )

    print(
        f"Session ID                : "
        f"{args.session_id}"
    )

    print(
        f"Swing validity            : "
        f"{target_validity}"
    )

    # -------------------------------------------------------------------------
    # BASELINE
    # -------------------------------------------------------------------------

    print()
    print("SESSION BASELINE")
    print("-" * 88)

    print(
        f"Session profiles          : "
        f"{len(session_db)}"
    )

    print(
        f"VALID_SWING profiles      : "
        f"{len(valid_db)}"
    )

    print(
        f"Baseline profiles         : "
        f"{len(baseline)}"
    )

    print(
        f"Baseline mode             : "
        f"{baseline_mode}"
    )

    if baseline_mode == "LEAVE_ONE_OUT":

        print(
            "Target excluded           : "
            "Yes (prevents self-comparison bias)"
        )

    else:

        print(
            "Target excluded           : "
            "No (insufficient other valid swings)"
        )

    # -------------------------------------------------------------------------
    # COMPARATIVE FEEDBACK BY CATEGORY
    # -------------------------------------------------------------------------

    print()
    print("COMPARATIVE FEEDBACK")
    print("=" * 88)

    categories = (
        feedback_df["category"]
        .drop_duplicates()
        .tolist()
    )

    for category in categories:

        category_df = feedback_df[
            feedback_df["category"]
            == category
        ]

        print()
        print(category)
        print("-" * 88)

        for _, row in category_df.iterrows():

            target_text = format_value(
                row["target_value"],
                row["unit"],
            )

            mean_text = format_value(
                row["baseline_mean"],
                row["unit"],
            )

            print()
            print(
                f"{row['metric']}"
            )

            print(
                f"  Target                  : "
                f"{target_text}"
            )

            print(
                f"  Session baseline        : "
                f"{mean_text}"
            )

            if np.isfinite(
                row["baseline_std"]
            ):

                std_text = format_value(
                    row["baseline_std"],
                    row["unit"],
                )

                print(
                    f"  Baseline SD             : "
                    f"{std_text}"
                )

            if np.isfinite(
                row["percent_delta"]
            ):

                print(
                    f"  Difference              : "
                    f"{row['percent_delta']:+.1f}%"
                )

            if np.isfinite(
                row["z_score"]
            ):

                print(
                    f"  Z-score                 : "
                    f"{row['z_score']:+.2f}"
                )

            print(
                f"  Classification          : "
                f"{row['comparison']}"
            )

            print(
                f"  Feedback                : "
                f"{row['feedback']}"
            )

    # -------------------------------------------------------------------------
    # MEASUREMENT CONFIDENCE
    # -------------------------------------------------------------------------

    print()
    print("MEASUREMENT CONFIDENCE")
    print("-" * 88)

    if np.isfinite(
        pose_quality
    ):

        print(
            f"Mean pose quality         : "
            f"{pose_quality:.3f}"
        )

    else:

        print(
            "Mean pose quality         : "
            "N/A"
        )

    print(
        f"Measurement confidence    : "
        f"{measurement_confidence}"
    )

    print()
    print(
        "Measurement confidence describes "
        "pose-estimation reliability, "
        "not swing quality."
    )

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------

    print()
    print("SUMMARY")
    print("-" * 88)

    print(
        f"Metrics available         : "
        f"{len(feedback_df)}"
    )

    print(
        f"Metrics compared          : "
        f"{len(comparable)}"
    )

    print(
        f"Consistent                : "
        f"{consistent_count}"
    )

    print(
        f"Slightly different        : "
        f"{slight_count}"
    )

    print(
        f"Outside typical range     : "
        f"{outside_count}"
    )

    print(
        f"  HIGH                    : "
        f"{high_count}"
    )

    print(
        f"  LOW                     : "
        f"{low_count}"
    )

    # -------------------------------------------------------------------------
    # MISSING METRICS
    # -------------------------------------------------------------------------

    if missing_metrics:

        print()
        print("MISSING METRICS")
        print("-" * 88)

        for metric in missing_metrics:
            print(f"  - {metric}")

    # -------------------------------------------------------------------------
    # OUTPUT
    # -------------------------------------------------------------------------

    print()
    print("OUTPUT FILES")
    print("-" * 88)

    print(
        f"Detailed feedback         : "
        f"{detail_output.resolve()}"
    )

    print(
        f"Summary                   : "
        f"{summary_output.resolve()}"
    )

    # -------------------------------------------------------------------------
    # IMPORTANT
    # -------------------------------------------------------------------------

    print()
    print("IMPORTANT")
    print("-" * 88)

    print(
        "Feedback is relative to this player's "
        "VALID_SWING session baseline."
    )

    print(
        "The target swing is excluded from its own "
        "baseline whenever other valid swings exist."
    )

    print(
        "HIGH/LOW describe statistical position, "
        "not swing quality."
    )

    print(
        "Pose quality is reported separately as "
        "measurement confidence."
    )

    print(
        "No GOOD/BAD, biomechanical quality, "
        "or performance score is inferred."
    )

    print(
        "Motion and projected-angle metrics remain "
        "comparative 2D measurements."
    )

    print("=" * 88)


if __name__ == "__main__":
    main()