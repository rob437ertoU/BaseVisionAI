"""
BaseVisionAI
Swing Insights V1

Creates a human-readable statistical interpretation of Swing Feedback V1.2.

IMPORTANT
---------
This module does NOT:
- generate a baseball performance score
- classify swings as good/bad
- provide biomechanical coaching
- infer calibrated 3D biomechanics

Insights describe only the selected swing's statistical position relative
to the player's observed session baseline.

Author: BaseVisionAI
Version: Swing Insights V1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


VERSION = "Swing Insights V1"

SLIGHT_CLASSES = {
    "SLIGHTLY_HIGH",
    "SLIGHTLY_LOW",
}

OUTSIDE_CLASSES = {
    "HIGH",
    "LOW",
}

CONSISTENT_CLASSES = {
    "CONSISTENT",
}


# =============================================================================
# HELPERS
# =============================================================================

def clean_string(value, default="N/A"):
    if pd.isna(value):
        return default

    value = str(value).strip()

    if not value:
        return default

    return value


def to_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default

        return float(value)

    except (TypeError, ValueError):
        return default


def to_int(value, default=0):
    value = to_float(value, np.nan)

    if np.isnan(value):
        return default

    return int(round(value))


def to_bool(value, default=False):
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return default

    text = str(value).strip().lower()

    if text in {"true", "1", "yes", "y"}:
        return True

    if text in {"false", "0", "no", "n"}:
        return False

    return default


def format_bool(value):
    return "Yes" if to_bool(value) else "No"


def safe_percent(value):
    value = to_float(value)

    if np.isnan(value):
        return "N/A"

    return f"{value:.1f}%"


def safe_number(value, decimals=2):
    value = to_float(value)

    if np.isnan(value):
        return "N/A"

    return f"{value:.{decimals}f}"


def robust_abs_z(row):
    value = to_float(row.get("robust_z_score"))

    if np.isnan(value):
        return np.nan

    return abs(value)


def normalized_classification(value):
    return clean_string(value, "").upper()


def metric_name(row):
    return clean_string(row.get("metric"), "Unknown metric")


def feedback_text(row):
    return clean_string(row.get("feedback"), "")


# =============================================================================
# VALIDATION
# =============================================================================

def validate_feedback_columns(df: pd.DataFrame):
    required = {
        "feedback_version",
        "video_id",
        "player_id",
        "session_id",
        "validity_status",
        "target_baseline_eligible",
        "baseline_mode",
        "baseline_profiles",
        "group",
        "metric",
        "available",
        "target_value",
        "baseline_median",
        "baseline_mad",
        "difference_percent",
        "robust_z_score",
        "classification",
        "feedback",
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            "Feedback database is missing required columns: "
            + ", ".join(missing)
        )


def validate_summary_columns(df: pd.DataFrame):
    required = {
        "feedback_version",
        "video_id",
        "player_id",
        "session_id",
        "validity_status",
        "target_baseline_eligible",
        "baseline_mode",
        "baseline_profiles",
        "metrics_configured",
        "metrics_available",
        "metric_coverage_percent",
        "consistent",
        "slightly_different",
        "outside_typical_range",
        "mean_abs_robust_z",
        "median_abs_robust_z",
        "mean_pose_quality",
        "measurement_confidence",
        "performance_score_generated",
        "calibrated_biomechanics",
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            "Feedback summary is missing required columns: "
            + ", ".join(missing)
        )


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_feedback(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["classification"] = (
        result["classification"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    result["available_bool"] = result["available"].apply(to_bool)

    result["robust_z_numeric"] = pd.to_numeric(
        result["robust_z_score"],
        errors="coerce",
    )

    result["abs_robust_z"] = result["robust_z_numeric"].abs()

    result["difference_percent_numeric"] = pd.to_numeric(
        result["difference_percent"],
        errors="coerce",
    )

    return result


# =============================================================================
# INSIGHT SELECTION
# =============================================================================

def get_available_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["available_bool"]].copy()


def get_outside_metrics(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[
        df["classification"].isin(OUTSIDE_CLASSES)
    ].copy()

    return subset.sort_values(
        by="abs_robust_z",
        ascending=False,
        na_position="last",
    )


def get_slight_metrics(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[
        df["classification"].isin(SLIGHT_CLASSES)
    ].copy()

    return subset.sort_values(
        by="abs_robust_z",
        ascending=False,
        na_position="last",
    )


def get_consistent_metrics(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[
        df["classification"].isin(CONSISTENT_CLASSES)
    ].copy()

    return subset.sort_values(
        by="abs_robust_z",
        ascending=True,
        na_position="last",
    )


def get_largest_differences(df: pd.DataFrame, n=3) -> pd.DataFrame:
    """
    Select metrics with the largest absolute robust Z-score.

    Priority:
        HIGH / LOW
        then SLIGHTLY_HIGH / SLIGHTLY_LOW

    CONSISTENT metrics are not reported as largest differences.
    """

    candidates = df[
        df["classification"].isin(
            OUTSIDE_CLASSES | SLIGHT_CLASSES
        )
    ].copy()

    candidates = candidates[
        candidates["abs_robust_z"].notna()
    ]

    candidates = candidates.sort_values(
        by="abs_robust_z",
        ascending=False,
    )

    return candidates.head(n)


def get_most_stable(df: pd.DataFrame, n=3) -> pd.DataFrame:
    """
    Stable features are selected only from metrics classified CONSISTENT.
    Within those metrics, the smallest absolute robust Z-score is preferred.
    """

    candidates = get_consistent_metrics(df)

    return candidates.head(n)


# =============================================================================
# SUMMARY LANGUAGE
# =============================================================================

def comparison_level(
    consistent: int,
    slight: int,
    outside: int,
    total: int,
):
    if total <= 0:
        return "INSUFFICIENT_DATA"

    outside_ratio = outside / total
    deviation_ratio = (outside + slight) / total

    if outside_ratio >= 0.50:
        return "NOTICEABLE_DIFFERENCE"

    if outside_ratio >= 0.25:
        return "MODERATE_DIFFERENCE"

    if deviation_ratio >= 0.50:
        return "SOME_DIFFERENCE"

    return "MOSTLY_SIMILAR"


def build_overall_summary(
    video_id: str,
    consistent: int,
    slight: int,
    outside: int,
    total: int,
):
    level = comparison_level(
        consistent,
        slight,
        outside,
        total,
    )

    if total <= 0:
        text = (
            "There are not enough available metrics to summarize this "
            "swing relative to the player's session baseline."
        )

    elif level == "NOTICEABLE_DIFFERENCE":
        text = (
            f"This swing differs noticeably from the player's observed "
            f"session pattern in {outside} of {total} compared metrics."
        )

    elif level == "MODERATE_DIFFERENCE":
        text = (
            f"This swing shows several differences from the player's "
            f"observed session pattern, with {outside} of {total} metrics "
            f"outside the typical robust session range."
        )

    elif level == "SOME_DIFFERENCE":
        text = (
            f"This swing shows some variation from the player's observed "
            f"session pattern, although most differences are limited or "
            f"moderate."
        )

    else:
        text = (
            f"This swing is broadly similar to the player's observed "
            f"session pattern across the available compared metrics."
        )

    return level, text


def build_reliability_summary(
    measurement_confidence: str,
    metric_coverage: float,
    validity_status: str,
):
    confidence = clean_string(
        measurement_confidence,
        "UNKNOWN",
    ).upper()

    validity = clean_string(
        validity_status,
        "UNKNOWN",
    ).upper()

    if np.isnan(metric_coverage):
        coverage_text = "Metric coverage is unavailable."
    else:
        coverage_text = (
            f"Metric coverage is {metric_coverage:.1f}%."
        )

    if confidence == "HIGH":
        confidence_text = (
            "Pose-estimation confidence is high."
        )

    elif confidence == "MODERATE":
        confidence_text = (
            "Pose-estimation confidence is moderate."
        )

    elif confidence == "LOW":
        confidence_text = (
            "Pose-estimation confidence is low, so the comparison should "
            "be interpreted with additional caution."
        )

    else:
        confidence_text = (
            "Pose-estimation confidence is not available."
        )

    if validity == "VALID_SWING":
        validity_text = (
            "The swing has sufficient structural and temporal support "
            "for the current analysis."
        )

    elif validity == "TEMPORAL_LOW_CONFIDENCE":
        validity_text = (
            "The swing has limited temporal support and is not eligible "
            "to define the session baseline."
        )

    elif validity == "LOW_CONFIDENCE_SWING":
        validity_text = (
            "The swing has limited analysis evidence and is not eligible "
            "to define the session baseline."
        )

    else:
        validity_text = (
            f"Swing validity status is {validity}."
        )

    return (
        f"{confidence_text} {coverage_text} {validity_text}"
    )


# =============================================================================
# INSIGHT TABLE
# =============================================================================

def make_insight_row(
    video_id,
    player_id,
    session_id,
    insight_type,
    rank,
    row,
):
    return {
        "insights_version": VERSION,
        "video_id": video_id,
        "player_id": player_id,
        "session_id": session_id,
        "insight_type": insight_type,
        "rank": rank,
        "group": clean_string(row.get("group"), ""),
        "metric": metric_name(row),
        "classification": normalized_classification(
            row.get("classification")
        ),
        "robust_z_score": to_float(
            row.get("robust_z_score")
        ),
        "abs_robust_z": robust_abs_z(row),
        "difference_percent": to_float(
            row.get("difference_percent")
        ),
        "feedback": feedback_text(row),
    }


def build_insights_table(
    feedback_df: pd.DataFrame,
    video_id: str,
    player_id: str,
    session_id: str,
):
    rows = []

    largest = get_largest_differences(
        feedback_df,
        n=3,
    )

    stable = get_most_stable(
        feedback_df,
        n=3,
    )

    slight = get_slight_metrics(
        feedback_df
    ).head(3)

    for rank, (_, row) in enumerate(
        largest.iterrows(),
        start=1,
    ):
        rows.append(
            make_insight_row(
                video_id,
                player_id,
                session_id,
                "LARGEST_DIFFERENCE",
                rank,
                row,
            )
        )

    for rank, (_, row) in enumerate(
        stable.iterrows(),
        start=1,
    ):
        rows.append(
            make_insight_row(
                video_id,
                player_id,
                session_id,
                "MOST_STABLE",
                rank,
                row,
            )
        )

    for rank, (_, row) in enumerate(
        slight.iterrows(),
        start=1,
    ):
        rows.append(
            make_insight_row(
                video_id,
                player_id,
                session_id,
                "SLIGHT_DEVIATION",
                rank,
                row,
            )
        )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_swing(
    feedback_path: Path,
    summary_path: Path,
    output_dir: Path,
):
    feedback_raw = pd.read_csv(feedback_path)
    summary_df = pd.read_csv(summary_path)

    validate_feedback_columns(feedback_raw)
    validate_summary_columns(summary_df)

    if len(summary_df) != 1:
        raise ValueError(
            "Swing Feedback V1.2 summary must contain exactly one row."
        )

    summary = summary_df.iloc[0]

    feedback_df = prepare_feedback(feedback_raw)
    available_df = get_available_metrics(feedback_df)

    video_id = clean_string(summary["video_id"])
    player_id = clean_string(summary["player_id"])
    session_id = clean_string(summary["session_id"])

    validity_status = clean_string(
        summary["validity_status"]
    )

    target_baseline_eligible = to_bool(
        summary["target_baseline_eligible"]
    )

    baseline_mode = clean_string(
        summary["baseline_mode"]
    )

    baseline_profiles = to_int(
        summary["baseline_profiles"]
    )

    metrics_configured = to_int(
        summary["metrics_configured"]
    )

    metrics_available = to_int(
        summary["metrics_available"]
    )

    metric_coverage = to_float(
        summary["metric_coverage_percent"]
    )

    consistent = to_int(
        summary["consistent"]
    )

    slightly_different = to_int(
        summary["slightly_different"]
    )

    outside_typical = to_int(
        summary["outside_typical_range"]
    )

    mean_abs_robust_z = to_float(
        summary["mean_abs_robust_z"]
    )

    median_abs_robust_z = to_float(
        summary["median_abs_robust_z"]
    )

    mean_pose_quality = to_float(
        summary["mean_pose_quality"]
    )

    measurement_confidence = clean_string(
        summary["measurement_confidence"]
    )

    performance_score_generated = to_bool(
        summary["performance_score_generated"]
    )

    calibrated_biomechanics = to_bool(
        summary["calibrated_biomechanics"]
    )

    level, overall_summary = build_overall_summary(
        video_id,
        consistent,
        slightly_different,
        outside_typical,
        metrics_available,
    )

    reliability_summary = build_reliability_summary(
        measurement_confidence,
        metric_coverage,
        validity_status,
    )

    largest = get_largest_differences(
        available_df,
        n=3,
    )

    stable = get_most_stable(
        available_df,
        n=3,
    )

    insight_df = build_insights_table(
        available_df,
        video_id,
        player_id,
        session_id,
    )

    # -------------------------------------------------------------------------
    # Human-readable metric lists
    # -------------------------------------------------------------------------

    largest_metrics = [
        metric_name(row)
        for _, row in largest.iterrows()
    ]

    stable_metrics = [
        metric_name(row)
        for _, row in stable.iterrows()
    ]

    largest_text = "; ".join(largest_metrics)
    stable_text = "; ".join(stable_metrics)

    if not largest_text:
        largest_text = "None identified"

    if not stable_text:
        stable_text = "None identified"

    # -------------------------------------------------------------------------
    # Summary row
    # -------------------------------------------------------------------------

    summary_output = pd.DataFrame(
        [
            {
                "insights_version": VERSION,
                "video_id": video_id,
                "player_id": player_id,
                "session_id": session_id,
                "validity_status": validity_status,
                "target_baseline_eligible": target_baseline_eligible,
                "baseline_mode": baseline_mode,
                "baseline_profiles": baseline_profiles,
                "metrics_configured": metrics_configured,
                "metrics_available": metrics_available,
                "metric_coverage_percent": metric_coverage,
                "consistent": consistent,
                "slightly_different": slightly_different,
                "outside_typical_range": outside_typical,
                "comparison_level": level,
                "mean_abs_robust_z": mean_abs_robust_z,
                "median_abs_robust_z": median_abs_robust_z,
                "largest_difference_1": (
                    largest_metrics[0]
                    if len(largest_metrics) > 0
                    else ""
                ),
                "largest_difference_2": (
                    largest_metrics[1]
                    if len(largest_metrics) > 1
                    else ""
                ),
                "largest_difference_3": (
                    largest_metrics[2]
                    if len(largest_metrics) > 2
                    else ""
                ),
                "most_stable_1": (
                    stable_metrics[0]
                    if len(stable_metrics) > 0
                    else ""
                ),
                "most_stable_2": (
                    stable_metrics[1]
                    if len(stable_metrics) > 1
                    else ""
                ),
                "most_stable_3": (
                    stable_metrics[2]
                    if len(stable_metrics) > 2
                    else ""
                ),
                "mean_pose_quality": mean_pose_quality,
                "measurement_confidence": measurement_confidence,
                "overall_summary": overall_summary,
                "reliability_summary": reliability_summary,
                "performance_score_generated": (
                    performance_score_generated
                ),
                "calibrated_biomechanics": (
                    calibrated_biomechanics
                ),
            }
        ]
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    detail_output = (
        output_dir
        / f"{video_id}_swing_insights_v1.csv"
    )

    summary_output_path = (
        output_dir
        / f"{video_id}_swing_insights_v1_summary.csv"
    )

    insight_df.to_csv(
        detail_output,
        index=False,
    )

    summary_output.to_csv(
        summary_output_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Console report
    # -------------------------------------------------------------------------

    print()
    print("=" * 92)
    print("BASEVISIONAI - SWING INSIGHTS V1 COMPLETE")
    print("=" * 92)

    print()
    print("TARGET")
    print("-" * 92)

    print(
        f"{'Video ID':26}: {video_id}"
    )
    print(
        f"{'Player ID':26}: {player_id}"
    )
    print(
        f"{'Session ID':26}: {session_id}"
    )
    print(
        f"{'Swing validity':26}: {validity_status}"
    )
    print(
        f"{'Baseline eligible':26}: "
        f"{format_bool(target_baseline_eligible)}"
    )
    print(
        f"{'Baseline mode':26}: {baseline_mode}"
    )
    print(
        f"{'Baseline profiles':26}: {baseline_profiles}"
    )

    print()
    print("OVERALL COMPARISON")
    print("-" * 92)

    print(
        f"{'Metrics configured':26}: {metrics_configured}"
    )
    print(
        f"{'Metrics available':26}: {metrics_available}"
    )
    print(
        f"{'Metric coverage':26}: "
        f"{safe_percent(metric_coverage)}"
    )
    print(
        f"{'Consistent':26}: {consistent}"
    )
    print(
        f"{'Slight deviations':26}: {slightly_different}"
    )
    print(
        f"{'Outside typical range':26}: {outside_typical}"
    )
    print(
        f"{'Comparison level':26}: {level}"
    )

    print()
    print("LARGEST OBSERVED DIFFERENCES")
    print("-" * 92)

    if largest.empty:
        print(
            "No non-consistent metrics with valid robust Z-scores "
            "were available."
        )

    else:
        for rank, (_, row) in enumerate(
            largest.iterrows(),
            start=1,
        ):
            rz = to_float(
                row["robust_z_score"]
            )

            classification = normalized_classification(
                row["classification"]
            )

            print()
            print(
                f"{rank}. {metric_name(row)}"
            )
            print(
                f"   Classification         : {classification}"
            )
            print(
                f"   Robust Z               : "
                f"{safe_number(rz, 2)}"
            )
            print(
                f"   Difference             : "
                f"{safe_percent(row['difference_percent'])}"
            )
            print(
                f"   Observation            : "
                f"{feedback_text(row)}"
            )

    print()
    print("MOST STABLE OBSERVED FEATURES")
    print("-" * 92)

    if stable.empty:
        print(
            "No metrics classified CONSISTENT were available."
        )

    else:
        for rank, (_, row) in enumerate(
            stable.iterrows(),
            start=1,
        ):
            rz = to_float(
                row["robust_z_score"]
            )

            print()
            print(
                f"{rank}. {metric_name(row)}"
            )
            print(
                f"   Robust Z               : "
                f"{safe_number(rz, 2)}"
            )
            print(
                f"   Observation            : "
                f"{feedback_text(row)}"
            )

    print()
    print("MEASUREMENT RELIABILITY")
    print("-" * 92)

    print(
        f"{'Pose confidence':26}: "
        f"{measurement_confidence}"
    )
    print(
        f"{'Mean pose quality':26}: "
        f"{safe_number(mean_pose_quality, 3)}"
    )
    print(
        f"{'Metric coverage':26}: "
        f"{safe_percent(metric_coverage)}"
    )

    print()
    print("SUMMARY")
    print("-" * 92)

    print(overall_summary)

    print()
    print(reliability_summary)

    print()
    print("OUTPUT FILES")
    print("-" * 92)

    print(
        f"{'Insights':26}: {detail_output}"
    )
    print(
        f"{'Summary':26}: {summary_output_path}"
    )

    print()
    print("IMPORTANT")
    print("-" * 92)

    print(
        "Insights describe statistical differences relative to the "
        "player's observed session baseline."
    )
    print(
        "Largest differences are ranked by absolute robust Z-score, "
        "not by baseball quality."
    )
    print(
        "Stable features indicate similarity to the observed session "
        "baseline, not biomechanical correctness."
    )
    print(
        "No GOOD/BAD swing label, performance score, or coaching "
        "recommendation is generated."
    )
    print(
        "Projected rotation and motion metrics remain comparative "
        "2D measurements."
    )
    print(
        "No calibrated 3D biomechanics are inferred."
    )

    print("=" * 92)

    return insight_df, summary_output


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate human-readable Swing Insights V1 from "
            "BaseVisionAI Swing Feedback V1.2."
        )
    )

    parser.add_argument(
        "--feedback",
        required=True,
        help=(
            "Path to the detailed "
            "*_swing_feedback_v1_2.csv file."
        ),
    )

    parser.add_argument(
        "--summary",
        required=True,
        help=(
            "Path to the "
            "*_swing_feedback_v1_2_summary.csv file."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for Swing Insights V1 outputs.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    feedback_path = Path(args.feedback)
    summary_path = Path(args.summary)
    output_dir = Path(args.output_dir)

    if not feedback_path.exists():
        raise FileNotFoundError(
            f"Feedback file not found: {feedback_path}"
        )

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Feedback summary not found: {summary_path}"
        )

    analyze_swing(
        feedback_path=feedback_path,
        summary_path=summary_path,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print()
        print("=" * 92)
        print("BASEVISIONAI - SWING INSIGHTS V1 FAILED")
        print("=" * 92)
        print()
        print(f"ERROR: {exc}")
        print()

        sys.exit(1)