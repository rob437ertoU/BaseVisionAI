"""
BaseVisionAI - Session Insights V1

Session-level descriptive analysis using:
- Swing Consistency V1.1
- Swing Validity V1.1
- Swing Insights V1

IMPORTANT
---------
This module summarizes observed session-relative measurements.

It does NOT generate:
- GOOD/BAD swing labels
- biomechanical quality scores
- baseball performance scores
- coaching recommendations
- calibrated 3D biomechanics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# VERSION / FILES
# =============================================================================

VERSION = "Session Insights V1"

CONSISTENCY_SUMMARY_FILE = "swing_consistency_v1_1_summary.csv"
CONSISTENCY_METRICS_FILE = "swing_consistency_v1_1_metrics.csv"
CONSISTENCY_SWINGS_FILE = "swing_consistency_v1_1_swings.csv"
CONSISTENCY_GROUPS_FILE = "swing_consistency_v1_1_groups.csv"


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def clean_string(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def to_float(value, default=np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
        "t",
    }


def format_float(value, decimals=3, default="N/A") -> str:
    value = to_float(value)

    if not np.isfinite(value):
        return default

    return f"{value:.{decimals}f}"


def format_percent(value, decimals=1, default="N/A") -> str:
    value = to_float(value)

    if not np.isfinite(value):
        return default

    return f"{value:.{decimals}f}%"


def safe_read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read {label}:\n"
            f"{path}\n\n"
            f"{exc}"
        ) from exc

    if df.empty:
        raise ValueError(
            f"{label} is empty:\n{path}"
        )

    return df


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{label} is missing required columns: "
            + ", ".join(missing)
        )


def first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> Optional[str]:

    for column in candidates:
        if column in df.columns:
            return column

    return None


# =============================================================================
# LOAD INPUT FILES
# =============================================================================

def load_inputs(
    consistency_dir: Path,
    validity_path: Path,
    swing_insights_path: Path,
):

    consistency_summary_path = (
        consistency_dir
        / CONSISTENCY_SUMMARY_FILE
    )

    consistency_metrics_path = (
        consistency_dir
        / CONSISTENCY_METRICS_FILE
    )

    consistency_swings_path = (
        consistency_dir
        / CONSISTENCY_SWINGS_FILE
    )

    consistency_groups_path = (
        consistency_dir
        / CONSISTENCY_GROUPS_FILE
    )

    consistency_summary = safe_read_csv(
        consistency_summary_path,
        "Consistency summary",
    )

    consistency_metrics = safe_read_csv(
        consistency_metrics_path,
        "Consistency metrics",
    )

    validity = safe_read_csv(
        validity_path,
        "Validity database",
    )

    swing_insights = safe_read_csv(
        swing_insights_path,
        "Batch Swing Insights summary",
    )

    consistency_swings = None
    consistency_groups = None

    if consistency_swings_path.exists():
        consistency_swings = pd.read_csv(
            consistency_swings_path
        )

    if consistency_groups_path.exists():
        consistency_groups = pd.read_csv(
            consistency_groups_path
        )

    return {
        "consistency_summary": consistency_summary,
        "consistency_metrics": consistency_metrics,
        "consistency_swings": consistency_swings,
        "consistency_groups": consistency_groups,
        "validity": validity,
        "swing_insights": swing_insights,
    }


# =============================================================================
# SESSION FILTER
# =============================================================================

def filter_session(
    df: pd.DataFrame,
    player_id: str,
    session_id: str,
    label: str,
) -> pd.DataFrame:

    require_columns(
        df,
        [
            "player_id",
            "session_id",
        ],
        label,
    )

    filtered = df[
        (
            df["player_id"].astype(str)
            == str(player_id)
        )
        &
        (
            df["session_id"].astype(str)
            == str(session_id)
        )
    ].copy()

    if filtered.empty:
        raise ValueError(
            f"No rows found in {label} for:\n"
            f"player_id={player_id}\n"
            f"session_id={session_id}"
        )

    return filtered


# =============================================================================
# VALIDITY V1.1
# =============================================================================

def prepare_validity(
    validity: pd.DataFrame,
    player_id: str,
    session_id: str,
) -> pd.DataFrame:

    validity = filter_session(
        validity,
        player_id,
        session_id,
        "Validity database",
    )

    require_columns(
        validity,
        [
            "video_id",
            "validity_status",
        ],
        "Validity database",
    )

    validity = validity.copy()

    # =========================================================================
    # AUTHORITATIVE FINAL CLASSIFICATION
    # =========================================================================
    #
    # Validity V1.1 contains:
    #
    # validity_status
    #     Original classification inherited from V1.
    #
    # validity_status_v1
    #     Explicit preserved V1 classification.
    #
    # validity_status_v1_1
    #     FINAL classification after temporal-support analysis.
    #
    # Example:
    #
    # baseball_eval6:
    # validity_status       = VALID_SWING
    # validity_status_v1    = VALID_SWING
    # validity_status_v1_1  = TEMPORAL_LOW_CONFIDENCE
    #
    # Session Insights MUST therefore use validity_status_v1_1.
    # =========================================================================

    if "validity_status_v1_1" in validity.columns:
        final_status_column = "validity_status_v1_1"
    else:
        # Compatibility fallback for older databases.
        final_status_column = "validity_status"

    validity["final_validity_status"] = (
        validity[final_status_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Keep the original V1 status for traceability.
    if "validity_status_v1" in validity.columns:
        validity["original_validity_status"] = (
            validity["validity_status_v1"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )
    else:
        validity["original_validity_status"] = (
            validity["validity_status"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

    # =========================================================================
    # BASELINE ELIGIBILITY
    # =========================================================================

    baseline_column = first_existing_column(
        validity,
        [
            "baseline_eligible",
            "target_baseline_eligible",
        ],
    )

    if baseline_column is not None:
        validity["baseline_eligible_final"] = (
            validity[baseline_column]
            .apply(to_bool)
        )
    else:
        # Compatibility fallback:
        # final VALID_SWING profiles are eligible.
        validity["baseline_eligible_final"] = (
            validity["final_validity_status"]
            == "VALID_SWING"
        )

    # =========================================================================
    # SAFETY CHECK
    # =========================================================================
    #
    # The final V1.1 status has priority.
    #
    # Even if baseline_eligible accidentally contains True,
    # TEMPORAL_LOW_CONFIDENCE and LOW_CONFIDENCE_SWING cannot define
    # the session baseline.
    # =========================================================================

    validity["baseline_eligible_final"] = (
        validity["baseline_eligible_final"]
        &
        (
            validity["final_validity_status"]
            == "VALID_SWING"
        )
    )

    if "temporal_confidence" not in validity.columns:
        validity["temporal_confidence"] = ""

    if "evidence_quality_v1_1" not in validity.columns:
        validity["evidence_quality_v1_1"] = ""

    return validity


# =============================================================================
# SESSION EVIDENCE
# =============================================================================

def calculate_session_evidence(
    validity: pd.DataFrame,
) -> dict:

    statuses = validity[
        "final_validity_status"
    ]

    profiles_analyzed = len(validity)

    valid_swing_profiles = int(
        (
            statuses
            == "VALID_SWING"
        ).sum()
    )

    temporal_low_confidence = int(
        (
            statuses
            == "TEMPORAL_LOW_CONFIDENCE"
        ).sum()
    )

    low_confidence_swings = int(
        (
            statuses
            == "LOW_CONFIDENCE_SWING"
        ).sum()
    )

    baseline_eligible = int(
        validity[
            "baseline_eligible_final"
        ].sum()
    )

    excluded_profiles = (
        profiles_analyzed
        - baseline_eligible
    )

    known_statuses = {
        "VALID_SWING",
        "TEMPORAL_LOW_CONFIDENCE",
        "LOW_CONFIDENCE_SWING",
    }

    unknown_status_profiles = int(
        (
            ~statuses.isin(
                known_statuses
            )
        ).sum()
    )

    return {
        "profiles_analyzed":
            profiles_analyzed,

        "valid_swing_profiles":
            valid_swing_profiles,

        "baseline_eligible":
            baseline_eligible,

        "excluded_profiles":
            excluded_profiles,

        "temporal_low_confidence":
            temporal_low_confidence,

        "low_confidence_swings":
            low_confidence_swings,

        "unknown_status_profiles":
            unknown_status_profiles,
    }


# =============================================================================
# CONSISTENCY SUMMARY
# =============================================================================

def prepare_consistency_summary(
    consistency_summary: pd.DataFrame,
    player_id: str,
    session_id: str,
) -> pd.Series:

    filtered = filter_session(
        consistency_summary,
        player_id,
        session_id,
        "Consistency summary",
    )

    return filtered.iloc[0]


# =============================================================================
# CONSISTENCY METRICS
# =============================================================================

def prepare_consistency_metrics(
    metrics: pd.DataFrame,
) -> pd.DataFrame:

    require_columns(
        metrics,
        [
            "group",
            "metric",
        ],
        "Consistency metrics",
    )

    robust_cv_column = first_existing_column(
        metrics,
        [
            "robust_cv_percent",
            "robust_cv",
            "median_cv_percent",
        ],
    )

    conventional_cv_column = first_existing_column(
        metrics,
        [
            "cv_percent",
            "cv",
            "conventional_cv_percent",
        ],
    )

    if robust_cv_column is None:
        raise ValueError(
            "Consistency metrics do not contain "
            "a robust CV column."
        )

    metrics = metrics.copy()

    metrics["_robust_cv"] = pd.to_numeric(
        metrics[robust_cv_column],
        errors="coerce",
    )

    if conventional_cv_column is not None:
        metrics["_cv"] = pd.to_numeric(
            metrics[conventional_cv_column],
            errors="coerce",
        )
    else:
        metrics["_cv"] = np.nan

    metrics = metrics[
        np.isfinite(
            metrics["_robust_cv"]
        )
    ].copy()

    if metrics.empty:
        raise ValueError(
            "No usable robust CV measurements "
            "were found in the consistency metrics."
        )

    return metrics


# =============================================================================
# SWING INSIGHTS
# =============================================================================

def prepare_swing_insights(
    insights: pd.DataFrame,
    player_id: str,
    session_id: str,
) -> pd.DataFrame:

    insights = filter_session(
        insights,
        player_id,
        session_id,
        "Batch Swing Insights summary",
    )

    require_columns(
        insights,
        [
            "video_id",
            "validity_status",
            "comparison_level",
            "metric_coverage_percent",
            "outside_typical_range",
            "measurement_confidence",
        ],
        "Batch Swing Insights summary",
    )

    insights = insights.copy()

    insights["validity_status"] = (
        insights["validity_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if "target_baseline_eligible" in insights.columns:
        insights["target_baseline_eligible"] = (
            insights[
                "target_baseline_eligible"
            ].apply(to_bool)
        )
    else:
        insights["target_baseline_eligible"] = False

    numeric_columns = [
        "metric_coverage_percent",
        "outside_typical_range",
        "mean_abs_robust_z",
        "median_abs_robust_z",
    ]

    for column in numeric_columns:
        if column in insights.columns:
            insights[column] = pd.to_numeric(
                insights[column],
                errors="coerce",
            )

    return insights


# =============================================================================
# BUILD SESSION INSIGHTS
# =============================================================================

def build_session_insights(
    player_id: str,
    session_id: str,
    consistency_summary: pd.Series,
    consistency_metrics: pd.DataFrame,
    validity: pd.DataFrame,
    swing_insights: pd.DataFrame,
):

    # =========================================================================
    # SESSION EVIDENCE
    # =========================================================================

    evidence = calculate_session_evidence(
        validity
    )

    profiles_analyzed = evidence[
        "profiles_analyzed"
    ]

    valid_swing_profiles = evidence[
        "valid_swing_profiles"
    ]

    baseline_eligible = evidence[
        "baseline_eligible"
    ]

    excluded_profiles = evidence[
        "excluded_profiles"
    ]

    temporal_low_confidence = evidence[
        "temporal_low_confidence"
    ]

    low_confidence_swings = evidence[
        "low_confidence_swings"
    ]

    unknown_status_profiles = evidence[
        "unknown_status_profiles"
    ]

    # =========================================================================
    # CONSISTENCY VALUES
    # =========================================================================

    mean_robust_cv = to_float(
        consistency_summary.get(
            "mean_robust_cv_percent",
            np.nan,
        )
    )

    median_robust_cv = to_float(
        consistency_summary.get(
            "median_robust_cv_percent",
            np.nan,
        )
    )

    mean_cv = to_float(
        consistency_summary.get(
            "mean_cv_percent",
            np.nan,
        )
    )

    median_cv = to_float(
        consistency_summary.get(
            "median_cv_percent",
            np.nan,
        )
    )

    # =========================================================================
    # REPRESENTATIVE SWING
    # =========================================================================

    representative_swing = clean_string(
        consistency_summary.get(
            "representative_swing",
            "",
        )
    )

    representative_distance = to_float(
        consistency_summary.get(
            "representative_distance",
            np.nan,
        )
    )

    most_different_swing = clean_string(
        consistency_summary.get(
            "most_different_swing",
            "",
        )
    )

    most_different_distance = to_float(
        consistency_summary.get(
            "most_different_distance",
            np.nan,
        )
    )

    distance_method = clean_string(
        consistency_summary.get(
            "distance_method",
            "",
        )
    )

    # =========================================================================
    # VARIABILITY
    # =========================================================================

    metrics_sorted = (
        consistency_metrics
        .sort_values(
            "_robust_cv",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    most_repeatable = (
        metrics_sorted.iloc[0]
    )

    most_variable = (
        metrics_sorted.iloc[-1]
    )

    most_repeatable_metric = clean_string(
        most_repeatable["metric"]
    )

    most_repeatable_group = clean_string(
        most_repeatable["group"]
    )

    most_repeatable_robust_cv = to_float(
        most_repeatable["_robust_cv"]
    )

    most_variable_metric = clean_string(
        most_variable["metric"]
    )

    most_variable_group = clean_string(
        most_variable["group"]
    )

    most_variable_robust_cv = to_float(
        most_variable["_robust_cv"]
    )

    lowest_variability = (
        metrics_sorted
        .head(3)
        .copy()
    )

    highest_variability = (
        metrics_sorted
        .sort_values(
            "_robust_cv",
            ascending=False,
        )
        .head(3)
        .copy()
    )

    # =========================================================================
    # AUTHORITATIVE BASELINE-ELIGIBLE VIDEO IDS
    # =========================================================================

    eligible_video_ids = set(
        validity.loc[
            validity[
                "baseline_eligible_final"
            ],
            "video_id",
        ]
        .astype(str)
    )

    eligible_insights = (
        swing_insights[
            swing_insights[
                "video_id"
            ]
            .astype(str)
            .isin(eligible_video_ids)
        ]
        .copy()
    )

    if not eligible_insights.empty:
        mean_metric_coverage = (
            eligible_insights[
                "metric_coverage_percent"
            ].mean()
        )

        mean_outside_range = (
            eligible_insights[
                "outside_typical_range"
            ].mean()
        )
    else:
        mean_metric_coverage = np.nan
        mean_outside_range = np.nan

    # =========================================================================
    # COMPARISON LEVEL DISTRIBUTION
    # =========================================================================

    comparison_counts = {}

    if not eligible_insights.empty:
        comparison_counts = (
            eligible_insights[
                "comparison_level"
            ]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
            .str.upper()
            .value_counts()
            .to_dict()
        )

    # =========================================================================
    # MEASUREMENT CONFIDENCE DISTRIBUTION
    # =========================================================================

    confidence_counts = {}

    if not swing_insights.empty:
        confidence_counts = (
            swing_insights[
                "measurement_confidence"
            ]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
            .str.upper()
            .value_counts()
            .to_dict()
        )

    # =========================================================================
    # SESSION OBSERVATIONS
    # =========================================================================

    observation_lowest = (
        f"{most_repeatable_metric} showed the lowest "
        f"observed session variability among the "
        f"configured metrics "
        f"(Robust CV "
        f"{most_repeatable_robust_cv:.1f}%)."
    )

    observation_highest = (
        f"{most_variable_metric} showed the highest "
        f"observed session variability among the "
        f"configured metrics "
        f"(Robust CV "
        f"{most_variable_robust_cv:.1f}%)."
    )

    if representative_swing:
        observation_representative = (
            f"{representative_swing} was the "
            f"baseline-eligible swing closest to the "
            f"robust session median profile."
        )
    else:
        observation_representative = (
            "A representative baseline-eligible swing "
            "could not be identified."
        )

    if most_different_swing:
        observation_different = (
            f"{most_different_swing} was the "
            f"baseline-eligible swing with the largest "
            f"observed robust distance from the session "
            f"median profile."
        )
    else:
        observation_different = (
            "A most-different baseline-eligible swing "
            "could not be identified."
        )

    observation_baseline = (
        f"{baseline_eligible} of "
        f"{profiles_analyzed} analyzed swings were "
        f"eligible to define the observed session "
        f"baseline; {excluded_profiles} were excluded "
        f"from baseline estimation."
    )

    # =========================================================================
    # DETAILED INSIGHTS TABLE
    # =========================================================================

    rows = []

    def add_row(
        category,
        rank,
        item,
        value,
        unit,
        observation,
    ):

        rows.append(
            {
                "session_insights_version":
                    VERSION,

                "player_id":
                    player_id,

                "session_id":
                    session_id,

                "category":
                    category,

                "rank":
                    rank,

                "item":
                    item,

                "value":
                    value,

                "unit":
                    unit,

                "observation":
                    observation,

                "performance_interpretation":
                    False,
            }
        )

    # -------------------------------------------------------------------------
    # LOWEST VARIABILITY
    # -------------------------------------------------------------------------

    for rank, (_, metric_row) in enumerate(
        lowest_variability.iterrows(),
        start=1,
    ):

        metric = clean_string(
            metric_row["metric"]
        )

        robust_cv = to_float(
            metric_row["_robust_cv"]
        )

        add_row(
            category=(
                "LOWEST_OBSERVED_VARIABILITY"
            ),
            rank=rank,
            item=metric,
            value=robust_cv,
            unit="robust CV percent",
            observation=(
                f"{metric} was among the lowest "
                f"observed session-variability metrics."
            ),
        )

    # -------------------------------------------------------------------------
    # HIGHEST VARIABILITY
    # -------------------------------------------------------------------------

    for rank, (_, metric_row) in enumerate(
        highest_variability.iterrows(),
        start=1,
    ):

        metric = clean_string(
            metric_row["metric"]
        )

        robust_cv = to_float(
            metric_row["_robust_cv"]
        )

        add_row(
            category=(
                "HIGHEST_OBSERVED_VARIABILITY"
            ),
            rank=rank,
            item=metric,
            value=robust_cv,
            unit="robust CV percent",
            observation=(
                f"{metric} was among the highest "
                f"observed session-variability metrics."
            ),
        )

    # -------------------------------------------------------------------------
    # REPRESENTATIVE SWING
    # -------------------------------------------------------------------------

    if representative_swing:
        add_row(
            category="REPRESENTATIVE_SWING",
            rank=1,
            item=representative_swing,
            value=representative_distance,
            unit=(
                "mean absolute robust z-distance"
            ),
            observation=(
                observation_representative
            ),
        )

    # -------------------------------------------------------------------------
    # MOST DIFFERENT SWING
    # -------------------------------------------------------------------------

    if most_different_swing:
        add_row(
            category=(
                "MOST_DIFFERENT_ELIGIBLE_SWING"
            ),
            rank=1,
            item=most_different_swing,
            value=most_different_distance,
            unit=(
                "mean absolute robust z-distance"
            ),
            observation=(
                observation_different
            ),
        )

    insights_df = pd.DataFrame(rows)

    # =========================================================================
    # SUMMARY
    # =========================================================================

    summary = {
        "session_insights_version":
            VERSION,

        "player_id":
            player_id,

        "session_id":
            session_id,

        # ---------------------------------------------------------------------
        # FINAL V1.1 SESSION EVIDENCE
        # ---------------------------------------------------------------------

        "profiles_analyzed":
            profiles_analyzed,

        "valid_swing_profiles":
            valid_swing_profiles,

        "baseline_eligible":
            baseline_eligible,

        "excluded_profiles":
            excluded_profiles,

        "temporal_low_confidence":
            temporal_low_confidence,

        "low_confidence_swings":
            low_confidence_swings,

        "unknown_status_profiles":
            unknown_status_profiles,

        # ---------------------------------------------------------------------
        # COMPATIBILITY COLUMN NAMES
        # ---------------------------------------------------------------------

        "profiles_total":
            profiles_analyzed,

        "baseline_eligible_swings":
            baseline_eligible,

        "temporal_low_confidence_swings":
            temporal_low_confidence,

        # ---------------------------------------------------------------------
        # SESSION VARIABILITY
        # ---------------------------------------------------------------------

        "metrics_available":
            len(consistency_metrics),

        "mean_cv_percent":
            mean_cv,

        "median_cv_percent":
            median_cv,

        "mean_robust_cv_percent":
            mean_robust_cv,

        "median_robust_cv_percent":
            median_robust_cv,

        # ---------------------------------------------------------------------
        # LOWEST / HIGHEST VARIABILITY
        # ---------------------------------------------------------------------

        "most_repeatable_group":
            most_repeatable_group,

        "most_repeatable_metric":
            most_repeatable_metric,

        "most_repeatable_robust_cv_percent":
            most_repeatable_robust_cv,

        "most_variable_group":
            most_variable_group,

        "most_variable_metric":
            most_variable_metric,

        "most_variable_robust_cv_percent":
            most_variable_robust_cv,

        # ---------------------------------------------------------------------
        # REPRESENTATIVE / MOST DIFFERENT
        # ---------------------------------------------------------------------

        "representative_swing":
            representative_swing,

        "representative_distance":
            representative_distance,

        "most_different_swing":
            most_different_swing,

        "most_different_distance":
            most_different_distance,

        "distance_method":
            distance_method,

        # ---------------------------------------------------------------------
        # ELIGIBLE SWING COMPARISONS
        # ---------------------------------------------------------------------

        "eligible_mean_metric_coverage_percent":
            mean_metric_coverage,

        "eligible_mean_outside_typical_range":
            mean_outside_range,

        "eligible_noticeable_difference_count":
            comparison_counts.get(
                "NOTICEABLE_DIFFERENCE",
                0,
            ),

        "eligible_moderate_difference_count":
            comparison_counts.get(
                "MODERATE_DIFFERENCE",
                0,
            ),

        "eligible_some_difference_count":
            comparison_counts.get(
                "SOME_DIFFERENCE",
                0,
            ),

        "eligible_mostly_similar_count":
            comparison_counts.get(
                "MOSTLY_SIMILAR",
                0,
            ),

        # ---------------------------------------------------------------------
        # MEASUREMENT CONFIDENCE
        # ---------------------------------------------------------------------

        "measurement_confidence_high_count":
            confidence_counts.get(
                "HIGH",
                0,
            ),

        "measurement_confidence_moderate_count":
            confidence_counts.get(
                "MODERATE",
                0,
            ),

        "measurement_confidence_low_count":
            confidence_counts.get(
                "LOW",
                0,
            ),

        # ---------------------------------------------------------------------
        # OBSERVATIONS
        # ---------------------------------------------------------------------

        "observation_lowest_variability":
            observation_lowest,

        "observation_highest_variability":
            observation_highest,

        "observation_representative_swing":
            observation_representative,

        "observation_most_different_swing":
            observation_different,

        "observation_baseline_support":
            observation_baseline,

        # ---------------------------------------------------------------------
        # TRACEABILITY
        # ---------------------------------------------------------------------

        "baseline_filter":
            (
                "Swing Validity V1.1 "
                "baseline_eligible=True"
            ),

        "validity_authority":
            (
                "validity_status_v1_1 from "
                "Swing Validity V1.1"
            ),

        "performance_score_generated":
            False,

        "coaching_recommendation_generated":
            False,

        "calibrated_biomechanics":
            False,
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    return insights_df, summary_df


# =============================================================================
# TERMINAL REPORT
# =============================================================================

def print_report(
    summary_df: pd.DataFrame,
    output_insights: Path,
    output_summary: Path,
):

    row = summary_df.iloc[0]

    width = 92

    print()
    print("=" * width)
    print(
        "BASEVISIONAI - SESSION INSIGHTS V1 COMPLETE"
    )
    print("=" * width)

    # =========================================================================
    # SESSION
    # =========================================================================

    print()
    print("SESSION")
    print("-" * width)

    print(
        f"{'Player ID':27}: "
        f"{row['player_id']}"
    )

    print(
        f"{'Session ID':27}: "
        f"{row['session_id']}"
    )

    # =========================================================================
    # SESSION EVIDENCE
    # =========================================================================

    print()
    print("SESSION EVIDENCE")
    print("-" * width)

    print(
        f"{'Profiles analyzed':27}: "
        f"{to_int(row['profiles_analyzed'])}"
    )

    print(
        f"{'VALID_SWING profiles':27}: "
        f"{to_int(row['valid_swing_profiles'])}"
    )

    print(
        f"{'Baseline eligible':27}: "
        f"{to_int(row['baseline_eligible'])}"
    )

    print(
        f"{'Excluded':27}: "
        f"{to_int(row['excluded_profiles'])}"
    )

    print(
        f"{'Temporal low confidence':27}: "
        f"{to_int(row['temporal_low_confidence'])}"
    )

    print(
        f"{'Low confidence swings':27}: "
        f"{to_int(row['low_confidence_swings'])}"
    )

    unknown = to_int(
        row["unknown_status_profiles"]
    )

    if unknown > 0:
        print(
            f"{'Unknown validity status':27}: "
            f"{unknown}"
        )

    # =========================================================================
    # VARIABILITY
    # =========================================================================

    print()
    print(
        "OBSERVED SESSION VARIABILITY"
    )
    print("-" * width)

    print(
        f"{'Mean Robust CV':27}: "
        f"{format_percent(row['mean_robust_cv_percent'], 1)}"
    )

    print(
        f"{'Median Robust CV':27}: "
        f"{format_percent(row['median_robust_cv_percent'], 1)}"
    )

    print(
        f"{'Mean conventional CV':27}: "
        f"{format_percent(row['mean_cv_percent'], 1)}"
    )

    print(
        f"{'Median conventional CV':27}: "
        f"{format_percent(row['median_cv_percent'], 1)}"
    )

    # =========================================================================
    # LOWEST VARIABILITY
    # =========================================================================

    print()
    print(
        "LOWEST OBSERVED VARIABILITY"
    )
    print("-" * width)

    print(
        f"{'Metric':27}: "
        f"{row['most_repeatable_metric']}"
    )

    print(
        f"{'Group':27}: "
        f"{row['most_repeatable_group']}"
    )

    print(
        f"{'Robust CV':27}: "
        f"{format_percent(row['most_repeatable_robust_cv_percent'], 1)}"
    )

    # =========================================================================
    # HIGHEST VARIABILITY
    # =========================================================================

    print()
    print(
        "HIGHEST OBSERVED VARIABILITY"
    )
    print("-" * width)

    print(
        f"{'Metric':27}: "
        f"{row['most_variable_metric']}"
    )

    print(
        f"{'Group':27}: "
        f"{row['most_variable_group']}"
    )

    print(
        f"{'Robust CV':27}: "
        f"{format_percent(row['most_variable_robust_cv_percent'], 1)}"
    )

    # =========================================================================
    # REPRESENTATIVE SWING
    # =========================================================================

    print()
    print(
        "REPRESENTATIVE SWING"
    )
    print("-" * width)

    print(
        f"{'Video ID':27}: "
        f"{row['representative_swing']}"
    )

    print(
        f"{'Robust distance':27}: "
        f"{format_float(row['representative_distance'], 3)}"
    )

    # =========================================================================
    # MOST DIFFERENT SWING
    # =========================================================================

    print()
    print(
        "MOST DIFFERENT ELIGIBLE SWING"
    )
    print("-" * width)

    print(
        f"{'Video ID':27}: "
        f"{row['most_different_swing']}"
    )

    print(
        f"{'Robust distance':27}: "
        f"{format_float(row['most_different_distance'], 3)}"
    )

    # =========================================================================
    # ELIGIBLE COMPARISONS
    # =========================================================================

    print()
    print(
        "ELIGIBLE SWING COMPARISONS"
    )
    print("-" * width)

    print(
        f"{'Mean metric coverage':27}: "
        f"{format_percent(row['eligible_mean_metric_coverage_percent'], 1)}"
    )

    print(
        f"{'Mean outside-range metrics':27}: "
        f"{format_float(row['eligible_mean_outside_typical_range'], 2)}"
    )

    # =========================================================================
    # OBSERVATIONS
    # =========================================================================

    print()
    print(
        "SESSION OBSERVATIONS"
    )
    print("-" * width)

    observations = [
        row[
            "observation_lowest_variability"
        ],
        row[
            "observation_highest_variability"
        ],
        row[
            "observation_representative_swing"
        ],
        row[
            "observation_most_different_swing"
        ],
        row[
            "observation_baseline_support"
        ],
    ]

    for index, observation in enumerate(
        observations,
        start=1,
    ):
        print()
        print(
            f"{index}. {observation}"
        )

    # =========================================================================
    # OUTPUT
    # =========================================================================

    print()
    print(
        "OUTPUT FILES"
    )
    print("-" * width)

    print(
        f"{'Insights':27}: "
        f"{output_insights}"
    )

    print(
        f"{'Summary':27}: "
        f"{output_summary}"
    )

    # =========================================================================
    # IMPORTANT
    # =========================================================================

    print()
    print(
        "IMPORTANT"
    )
    print("-" * width)

    print(
        "Session insights describe observed repeatability and "
        "statistical differences across baseline-eligible swings."
    )

    print(
        "Swing Validity V1.1 validity_status_v1_1 is used as the "
        "authoritative final swing classification."
    )

    print(
        "TEMPORAL_LOW_CONFIDENCE and LOW_CONFIDENCE_SWING "
        "profiles are excluded from baseline estimation."
    )

    print(
        "Lower variability means greater observed similarity "
        "across the session; it does not mean better baseball mechanics."
    )

    print(
        "The representative swing is the eligible swing closest "
        "to the observed robust session median profile, not the 'best' swing."
    )

    print(
        "The most different swing is the eligible swing farthest "
        "from the observed session median profile, not the 'worst' swing."
    )

    print(
        "No GOOD/BAD label, coaching recommendation, biomechanical "
        "quality score, or performance score is generated."
    )

    print(
        "Projected rotation and motion measurements remain "
        "comparative 2D camera-dependent measurements."
    )

    print("=" * width)


# =============================================================================
# CLI
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Generate BaseVisionAI Session Insights V1 "
            "from Consistency V1.1, Validity V1.1, "
            "and Swing Insights V1."
        )
    )

    parser.add_argument(
        "--consistency-dir",
        required=True,
        type=Path,
        help=(
            "Directory containing "
            "Swing Consistency V1.1 outputs."
        ),
    )

    parser.add_argument(
        "--validity",
        required=True,
        type=Path,
        help=(
            "Path to swing_validity_v1_1.csv"
        ),
    )

    parser.add_argument(
        "--swing-insights",
        required=True,
        type=Path,
        help=(
            "Path to "
            "batch_swing_insights_v1_summary.csv"
        ),
    )

    parser.add_argument(
        "--player-id",
        required=True,
        help="Player ID.",
    )

    parser.add_argument(
        "--session-id",
        required=True,
        help="Session ID.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory.",
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():

    args = parse_args()

    try:

        # =====================================================================
        # LOAD INPUTS
        # =====================================================================

        inputs = load_inputs(
            consistency_dir=(
                args.consistency_dir
            ),
            validity_path=(
                args.validity
            ),
            swing_insights_path=(
                args.swing_insights
            ),
        )

        # =====================================================================
        # PREPARE VALIDITY V1.1
        # =====================================================================

        validity = prepare_validity(
            inputs["validity"],
            args.player_id,
            args.session_id,
        )

        # =====================================================================
        # PREPARE CONSISTENCY
        # =====================================================================

        consistency_summary = (
            prepare_consistency_summary(
                inputs[
                    "consistency_summary"
                ],
                args.player_id,
                args.session_id,
            )
        )

        consistency_metrics = (
            prepare_consistency_metrics(
                inputs[
                    "consistency_metrics"
                ]
            )
        )

        # =====================================================================
        # PREPARE SWING INSIGHTS
        # =====================================================================

        swing_insights = (
            prepare_swing_insights(
                inputs[
                    "swing_insights"
                ],
                args.player_id,
                args.session_id,
            )
        )

        # =====================================================================
        # BUILD SESSION INSIGHTS
        # =====================================================================

        (
            insights_df,
            summary_df,
        ) = build_session_insights(
            player_id=(
                args.player_id
            ),
            session_id=(
                args.session_id
            ),
            consistency_summary=(
                consistency_summary
            ),
            consistency_metrics=(
                consistency_metrics
            ),
            validity=(
                validity
            ),
            swing_insights=(
                swing_insights
            ),
        )

        # =====================================================================
        # CREATE OUTPUT DIRECTORY
        # =====================================================================

        args.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_insights = (
            args.output_dir
            / "session_insights_v1.csv"
        )

        output_summary = (
            args.output_dir
            / "session_insights_v1_summary.csv"
        )

        # =====================================================================
        # SAVE
        # =====================================================================

        insights_df.to_csv(
            output_insights,
            index=False,
        )

        summary_df.to_csv(
            output_summary,
            index=False,
        )

        # =====================================================================
        # REPORT
        # =====================================================================

        print_report(
            summary_df=summary_df,
            output_insights=(
                output_insights
            ),
            output_summary=(
                output_summary
            ),
        )

    except Exception as exc:

        print()
        print("=" * 92)
        print(
            "BASEVISIONAI - SESSION INSIGHTS V1 FAILED"
        )
        print("=" * 92)
        print()
        print(
            f"ERROR: {exc}"
        )
        print()

        sys.exit(1)


if __name__ == "__main__":
    main()