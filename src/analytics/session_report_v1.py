"""
BaseVisionAI
Session Report V1

Builds a consolidated session-level report from already-generated
BaseVisionAI analytics outputs.

This module DOES NOT recompute biomechanics, swing validity,
session baselines, consistency metrics, or swing feedback.

Inputs
------
1. Session Insights V1 summary
2. Session Insights V1 detailed insights
3. Batch Swing Insights V1 summary

Outputs
-------
1. session_report_v1_summary.csv
2. session_report_v1_swings.csv
3. session_report_v1_observations.csv

Important
---------
This report describes observed statistical patterns relative to the
player's session.

It does NOT generate:
- GOOD/BAD swing labels
- baseball performance scores
- biomechanical quality scores
- coaching recommendations
- calibrated 3D biomechanics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONSTANTS
# =============================================================================

REPORT_VERSION = "Session Report V1"

SEPARATOR = "=" * 92
SUB_SEPARATOR = "-" * 92


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def fail(message: str) -> None:
    raise RuntimeError(message)


def clean_string(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def safe_float(value, default=np.nan) -> float:
    try:
        if pd.isna(value):
            return default

        text = str(value).strip()

        if text == "":
            return default

        return float(text)

    except (TypeError, ValueError):
        return default


def safe_int(value, default=0) -> int:
    number = safe_float(value, np.nan)

    if pd.isna(number):
        return default

    return int(round(number))


def safe_bool(value, default=False) -> bool:
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


def fmt_float(value, decimals=3, fallback="N/A") -> str:
    number = safe_float(value, np.nan)

    if pd.isna(number):
        return fallback

    return f"{number:.{decimals}f}"


def fmt_percent(value, decimals=1, fallback="N/A") -> str:
    number = safe_float(value, np.nan)

    if pd.isna(number):
        return fallback

    return f"{number:.{decimals}f}%"


def first_existing(row: pd.Series, candidates, default=""):
    for column in candidates:
        if column in row.index:
            value = row[column]

            if not pd.isna(value) and str(value).strip() != "":
                return value

    return default


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        fail(
            f"{description} not found:\n\n"
            f"{path.resolve()}"
        )


def require_columns(
    df: pd.DataFrame,
    required_columns,
    description: str,
) -> None:

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        fail(
            f"{description} is missing required columns: "
            + ", ".join(missing)
        )


def print_field(label: str, value) -> None:
    print(f"{label:<27}: {value}")


# =============================================================================
# LOAD INPUTS
# =============================================================================

def load_session_summary(path: Path) -> pd.Series:

    require_file(path, "Session Insights V1 summary")

    df = pd.read_csv(path)

    if df.empty:
        fail("Session Insights V1 summary is empty.")

    require_columns(
        df,
        [
            "player_id",
            "session_id",
            "profiles_analyzed",
            "valid_swing_profiles",
            "baseline_eligible",
            "excluded_profiles",
            "temporal_low_confidence",
            "low_confidence_swings",
            "mean_cv_percent",
            "median_cv_percent",
            "mean_robust_cv_percent",
            "median_robust_cv_percent",
            "most_repeatable_group",
            "most_repeatable_metric",
            "most_repeatable_robust_cv_percent",
            "most_variable_group",
            "most_variable_metric",
            "most_variable_robust_cv_percent",
            "representative_swing",
            "representative_distance",
            "most_different_swing",
            "most_different_distance",
        ],
        "Session Insights V1 summary",
    )

    return df.iloc[0]


def load_session_insights(path: Path) -> pd.DataFrame:

    require_file(path, "Session Insights V1 detail")

    df = pd.read_csv(path)

    if df.empty:
        fail("Session Insights V1 detail is empty.")

    require_columns(
        df,
        [
            "player_id",
            "session_id",
            "category",
            "rank",
            "item",
            "value",
            "unit",
            "observation",
            "performance_interpretation",
        ],
        "Session Insights V1 detail",
    )

    return df


def load_swing_insights(path: Path) -> pd.DataFrame:

    require_file(path, "Batch Swing Insights V1 summary")

    df = pd.read_csv(path)

    if df.empty:
        fail("Batch Swing Insights V1 summary is empty.")

    require_columns(
        df,
        [
            "video_id",
            "player_id",
            "session_id",
            "validity_status",
            "target_baseline_eligible",
            "baseline_mode",
            "baseline_profiles",
            "metric_coverage_percent",
            "comparison_level",
            "outside_typical_range",
            "mean_abs_robust_z",
            "measurement_confidence",
        ],
        "Batch Swing Insights V1 summary",
    )

    return df


# =============================================================================
# VALIDATION
# =============================================================================

def validate_identity(
    session_summary: pd.Series,
    session_insights: pd.DataFrame,
    swing_insights: pd.DataFrame,
    player_id: str,
    session_id: str,
) -> None:

    summary_player = clean_string(session_summary["player_id"])
    summary_session = clean_string(session_summary["session_id"])

    if summary_player != player_id:
        fail(
            "Player ID mismatch in Session Insights summary. "
            f"Expected '{player_id}', found '{summary_player}'."
        )

    if summary_session != session_id:
        fail(
            "Session ID mismatch in Session Insights summary. "
            f"Expected '{session_id}', found '{summary_session}'."
        )

    detail_subset = session_insights[
        (session_insights["player_id"].astype(str) == player_id)
        & (session_insights["session_id"].astype(str) == session_id)
    ]

    if detail_subset.empty:
        fail(
            "No Session Insights V1 detail rows found for "
            f"{player_id} / {session_id}."
        )

    swing_subset = swing_insights[
        (swing_insights["player_id"].astype(str) == player_id)
        & (swing_insights["session_id"].astype(str) == session_id)
    ]

    if swing_subset.empty:
        fail(
            "No Swing Insights V1 rows found for "
            f"{player_id} / {session_id}."
        )


# =============================================================================
# SESSION SUMMARY
# =============================================================================

def build_report_summary(
    session_summary: pd.Series,
    swing_rows: pd.DataFrame,
) -> pd.DataFrame:

    profiles_analyzed = safe_int(
        session_summary["profiles_analyzed"]
    )

    valid_swings = safe_int(
        session_summary["valid_swing_profiles"]
    )

    baseline_eligible = safe_int(
        session_summary["baseline_eligible"]
    )

    excluded_profiles = safe_int(
        session_summary["excluded_profiles"]
    )

    temporal_low_confidence = safe_int(
        session_summary["temporal_low_confidence"]
    )

    low_confidence_swings = safe_int(
        session_summary["low_confidence_swings"]
    )

    unknown_status_profiles = safe_int(
        session_summary.get(
            "unknown_status_profiles",
            0,
        )
    )

    eligible_mean_metric_coverage = safe_float(
        session_summary.get(
            "eligible_mean_metric_coverage_percent",
            np.nan,
        )
    )

    eligible_mean_outside_range = safe_float(
        session_summary.get(
            "eligible_mean_outside_typical_range",
            np.nan,
        )
    )

    report = {
        "report_version": REPORT_VERSION,

        "player_id": clean_string(
            session_summary["player_id"]
        ),

        "session_id": clean_string(
            session_summary["session_id"]
        ),

        # ---------------------------------------------------------
        # SESSION EVIDENCE
        # ---------------------------------------------------------

        "profiles_analyzed": profiles_analyzed,
        "valid_swing_profiles": valid_swings,
        "baseline_eligible_swings": baseline_eligible,
        "excluded_profiles": excluded_profiles,
        "temporal_low_confidence_swings":
            temporal_low_confidence,
        "low_confidence_swings":
            low_confidence_swings,
        "unknown_status_profiles":
            unknown_status_profiles,

        # ---------------------------------------------------------
        # METRIC COVERAGE
        # ---------------------------------------------------------

        "metrics_available": safe_int(
            session_summary.get(
                "metrics_available",
                0,
            )
        ),

        "eligible_mean_metric_coverage_percent":
            eligible_mean_metric_coverage,

        "eligible_mean_outside_typical_range":
            eligible_mean_outside_range,

        # ---------------------------------------------------------
        # SESSION VARIABILITY
        # ---------------------------------------------------------

        "mean_cv_percent": safe_float(
            session_summary["mean_cv_percent"]
        ),

        "median_cv_percent": safe_float(
            session_summary["median_cv_percent"]
        ),

        "mean_robust_cv_percent": safe_float(
            session_summary["mean_robust_cv_percent"]
        ),

        "median_robust_cv_percent": safe_float(
            session_summary[
                "median_robust_cv_percent"
            ]
        ),

        # ---------------------------------------------------------
        # LOWEST VARIABILITY
        # ---------------------------------------------------------

        "lowest_variability_group": clean_string(
            session_summary[
                "most_repeatable_group"
            ]
        ),

        "lowest_variability_metric": clean_string(
            session_summary[
                "most_repeatable_metric"
            ]
        ),

        "lowest_variability_robust_cv_percent":
            safe_float(
                session_summary[
                    "most_repeatable_robust_cv_percent"
                ]
            ),

        # ---------------------------------------------------------
        # HIGHEST VARIABILITY
        # ---------------------------------------------------------

        "highest_variability_group": clean_string(
            session_summary[
                "most_variable_group"
            ]
        ),

        "highest_variability_metric": clean_string(
            session_summary[
                "most_variable_metric"
            ]
        ),

        "highest_variability_robust_cv_percent":
            safe_float(
                session_summary[
                    "most_variable_robust_cv_percent"
                ]
            ),

        # ---------------------------------------------------------
        # SESSION PATTERN
        # ---------------------------------------------------------

        "representative_swing": clean_string(
            session_summary[
                "representative_swing"
            ]
        ),

        "representative_distance": safe_float(
            session_summary[
                "representative_distance"
            ]
        ),

        "most_different_eligible_swing":
            clean_string(
                session_summary[
                    "most_different_swing"
                ]
            ),

        "most_different_distance":
            safe_float(
                session_summary[
                    "most_different_distance"
                ]
            ),

        "distance_method": clean_string(
            session_summary.get(
                "distance_method",
                "",
            )
        ),

        # ---------------------------------------------------------
        # ELIGIBLE SWING COMPARISON DISTRIBUTION
        # ---------------------------------------------------------

        "eligible_noticeable_difference_count":
            safe_int(
                session_summary.get(
                    "eligible_noticeable_difference_count",
                    0,
                )
            ),

        "eligible_moderate_difference_count":
            safe_int(
                session_summary.get(
                    "eligible_moderate_difference_count",
                    0,
                )
            ),

        "eligible_some_difference_count":
            safe_int(
                session_summary.get(
                    "eligible_some_difference_count",
                    0,
                )
            ),

        "eligible_mostly_similar_count":
            safe_int(
                session_summary.get(
                    "eligible_mostly_similar_count",
                    0,
                )
            ),

        # ---------------------------------------------------------
        # MEASUREMENT CONFIDENCE
        # ---------------------------------------------------------

        "measurement_confidence_high_count":
            safe_int(
                session_summary.get(
                    "measurement_confidence_high_count",
                    0,
                )
            ),

        "measurement_confidence_moderate_count":
            safe_int(
                session_summary.get(
                    "measurement_confidence_moderate_count",
                    0,
                )
            ),

        "measurement_confidence_low_count":
            safe_int(
                session_summary.get(
                    "measurement_confidence_low_count",
                    0,
                )
            ),

        # ---------------------------------------------------------
        # TRACEABILITY
        # ---------------------------------------------------------

        "swing_rows_in_report": len(swing_rows),

        "baseline_filter": clean_string(
            session_summary.get(
                "baseline_filter",
                "",
            )
        ),

        "validity_authority": clean_string(
            session_summary.get(
                "validity_authority",
                "",
            )
        ),

        # ---------------------------------------------------------
        # INTERPRETATION FLAGS
        # ---------------------------------------------------------

        "performance_score_generated": False,
        "coaching_recommendation_generated": False,
        "calibrated_biomechanics": False,
    }

    return pd.DataFrame([report])


# =============================================================================
# PER-SWING REPORT
# =============================================================================

def build_swing_report(
    swing_insights: pd.DataFrame,
    player_id: str,
    session_id: str,
    representative_swing: str,
    most_different_swing: str,
) -> pd.DataFrame:

    df = swing_insights[
        (swing_insights["player_id"].astype(str) == player_id)
        & (swing_insights["session_id"].astype(str) == session_id)
    ].copy()

    if df.empty:
        fail(
            "No swing insight rows available for the requested "
            "player/session."
        )

    output_rows = []

    for _, row in df.iterrows():

        video_id = clean_string(row["video_id"])

        baseline_eligible = safe_bool(
            row["target_baseline_eligible"]
        )

        validity_status = clean_string(
            row["validity_status"]
        )

        comparison_level = clean_string(
            row["comparison_level"]
        )

        measurement_confidence = clean_string(
            row["measurement_confidence"]
        )

        report_row = {
            "report_version": REPORT_VERSION,

            "video_id": video_id,
            "player_id": player_id,
            "session_id": session_id,

            "validity_status": validity_status,

            "baseline_eligible":
                baseline_eligible,

            "baseline_mode": clean_string(
                row["baseline_mode"]
            ),

            "baseline_profiles": safe_int(
                row["baseline_profiles"]
            ),

            "metric_coverage_percent": safe_float(
                row["metric_coverage_percent"]
            ),

            "comparison_level":
                comparison_level,

            "outside_typical_range": safe_int(
                row["outside_typical_range"]
            ),

            "mean_abs_robust_z": safe_float(
                row["mean_abs_robust_z"]
            ),

            "median_abs_robust_z": safe_float(
                row.get(
                    "median_abs_robust_z",
                    np.nan,
                )
            ),

            "measurement_confidence":
                measurement_confidence,

            "is_representative_swing":
                video_id == representative_swing,

            "is_most_different_eligible_swing":
                video_id == most_different_swing,

            "performance_interpretation": False,
        }

        output_rows.append(report_row)

    output = pd.DataFrame(output_rows)

    # Preserve the original swing order when possible.
    return output.reset_index(drop=True)


# =============================================================================
# OBSERVATIONS REPORT
# =============================================================================

def build_observations_report(
    session_summary: pd.Series,
    session_insights: pd.DataFrame,
    player_id: str,
    session_id: str,
) -> pd.DataFrame:

    subset = session_insights[
        (session_insights["player_id"].astype(str) == player_id)
        & (session_insights["session_id"].astype(str) == session_id)
    ].copy()

    rows = []

    # -------------------------------------------------------------
    # Detailed ranked observations from Session Insights V1
    # -------------------------------------------------------------

    for _, row in subset.iterrows():

        rows.append(
            {
                "report_version": REPORT_VERSION,
                "player_id": player_id,
                "session_id": session_id,
                "source": "Session Insights V1",
                "category": clean_string(
                    row["category"]
                ),
                "rank": safe_int(
                    row["rank"]
                ),
                "item": clean_string(
                    row["item"]
                ),
                "value": safe_float(
                    row["value"]
                ),
                "unit": clean_string(
                    row["unit"]
                ),
                "observation": clean_string(
                    row["observation"]
                ),
                "performance_interpretation":
                    safe_bool(
                        row[
                            "performance_interpretation"
                        ]
                    ),
            }
        )

    # -------------------------------------------------------------
    # High-level session observations from summary
    # -------------------------------------------------------------

    summary_observation_columns = [
        (
            "SESSION_LOWEST_VARIABILITY",
            "observation_lowest_variability",
        ),
        (
            "SESSION_HIGHEST_VARIABILITY",
            "observation_highest_variability",
        ),
        (
            "SESSION_REPRESENTATIVE_SWING",
            "observation_representative_swing",
        ),
        (
            "SESSION_MOST_DIFFERENT_SWING",
            "observation_most_different_swing",
        ),
        (
            "SESSION_BASELINE_SUPPORT",
            "observation_baseline_support",
        ),
    ]

    for category, column in summary_observation_columns:

        if column not in session_summary.index:
            continue

        observation = clean_string(
            session_summary[column]
        )

        if observation == "":
            continue

        rows.append(
            {
                "report_version": REPORT_VERSION,
                "player_id": player_id,
                "session_id": session_id,
                "source":
                    "Session Insights V1 Summary",
                "category": category,
                "rank": 1,
                "item": "",
                "value": np.nan,
                "unit": "",
                "observation": observation,
                "performance_interpretation":
                    False,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# CONSOLE REPORT
# =============================================================================

def print_console_report(
    summary_df: pd.DataFrame,
    swings_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    output_dir: Path,
) -> None:

    row = summary_df.iloc[0]

    print()
    print(SEPARATOR)
    print("BASEVISIONAI - SESSION REPORT V1 COMPLETE")
    print(SEPARATOR)

    print()
    print("SESSION")
    print(SUB_SEPARATOR)

    print_field(
        "Player ID",
        row["player_id"],
    )

    print_field(
        "Session ID",
        row["session_id"],
    )

    # -----------------------------------------------------------------
    # EVIDENCE
    # -----------------------------------------------------------------

    print()
    print("SESSION EVIDENCE")
    print(SUB_SEPARATOR)

    print_field(
        "Profiles analyzed",
        row["profiles_analyzed"],
    )

    print_field(
        "VALID_SWING profiles",
        row["valid_swing_profiles"],
    )

    print_field(
        "Baseline eligible",
        row["baseline_eligible_swings"],
    )

    print_field(
        "Excluded",
        row["excluded_profiles"],
    )

    print_field(
        "Temporal low confidence",
        row["temporal_low_confidence_swings"],
    )

    print_field(
        "Low confidence swings",
        row["low_confidence_swings"],
    )

    # -----------------------------------------------------------------
    # VARIABILITY
    # -----------------------------------------------------------------

    print()
    print("OBSERVED SESSION VARIABILITY")
    print(SUB_SEPARATOR)

    print_field(
        "Mean Robust CV",
        fmt_percent(
            row["mean_robust_cv_percent"]
        ),
    )

    print_field(
        "Median Robust CV",
        fmt_percent(
            row["median_robust_cv_percent"]
        ),
    )

    print_field(
        "Mean conventional CV",
        fmt_percent(
            row["mean_cv_percent"]
        ),
    )

    print_field(
        "Median conventional CV",
        fmt_percent(
            row["median_cv_percent"]
        ),
    )

    # -----------------------------------------------------------------
    # SESSION PATTERN
    # -----------------------------------------------------------------

    print()
    print("SESSION PATTERN")
    print(SUB_SEPARATOR)

    print()
    print("Lowest observed variability")

    print_field(
        "Metric",
        row["lowest_variability_metric"],
    )

    print_field(
        "Group",
        row["lowest_variability_group"],
    )

    print_field(
        "Robust CV",
        fmt_percent(
            row[
                "lowest_variability_robust_cv_percent"
            ]
        ),
    )

    print()
    print("Highest observed variability")

    print_field(
        "Metric",
        row["highest_variability_metric"],
    )

    print_field(
        "Group",
        row["highest_variability_group"],
    )

    print_field(
        "Robust CV",
        fmt_percent(
            row[
                "highest_variability_robust_cv_percent"
            ]
        ),
    )

    print()
    print("Representative swing")

    print_field(
        "Video ID",
        row["representative_swing"],
    )

    print_field(
        "Robust distance",
        fmt_float(
            row["representative_distance"]
        ),
    )

    print()
    print("Most different eligible swing")

    print_field(
        "Video ID",
        row[
            "most_different_eligible_swing"
        ],
    )

    print_field(
        "Robust distance",
        fmt_float(
            row["most_different_distance"]
        ),
    )

    # -----------------------------------------------------------------
    # PER-SWING
    # -----------------------------------------------------------------

    print()
    print("SWING SUMMARY")
    print(SUB_SEPARATOR)

    for _, swing in swings_df.iterrows():

        video_id = clean_string(
            swing["video_id"]
        )

        flags = []

        if safe_bool(
            swing["is_representative_swing"]
        ):
            flags.append("REPRESENTATIVE")

        if safe_bool(
            swing[
                "is_most_different_eligible_swing"
            ]
        ):
            flags.append("MOST_DIFFERENT_ELIGIBLE")

        flag_text = ""

        if flags:
            flag_text = " [" + ", ".join(flags) + "]"

        print()
        print(f"{video_id}{flag_text}")

        print_field(
            "  Validity",
            swing["validity_status"],
        )

        print_field(
            "  Baseline eligible",
            (
                "Yes"
                if safe_bool(
                    swing["baseline_eligible"]
                )
                else "No"
            ),
        )

        print_field(
            "  Comparison",
            swing["comparison_level"],
        )

        print_field(
            "  Outside range",
            safe_int(
                swing["outside_typical_range"]
            ),
        )

        print_field(
            "  Metric coverage",
            fmt_percent(
                swing[
                    "metric_coverage_percent"
                ]
            ),
        )

        print_field(
            "  Measurement confidence",
            swing["measurement_confidence"],
        )

    # -----------------------------------------------------------------
    # SESSION OBSERVATIONS
    # -----------------------------------------------------------------

    print()
    print("SESSION OBSERVATIONS")
    print(SUB_SEPARATOR)

    summary_categories = {
        "SESSION_LOWEST_VARIABILITY",
        "SESSION_HIGHEST_VARIABILITY",
        "SESSION_REPRESENTATIVE_SWING",
        "SESSION_MOST_DIFFERENT_SWING",
        "SESSION_BASELINE_SUPPORT",
    }

    summary_observations = observations_df[
        observations_df["category"].isin(
            summary_categories
        )
    ]

    if summary_observations.empty:
        print("No high-level observations available.")

    else:
        for index, (_, obs) in enumerate(
            summary_observations.iterrows(),
            start=1,
        ):
            print()
            print(
                f"{index}. "
                f"{clean_string(obs['observation'])}"
            )

    # -----------------------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------------------

    print()
    print("OUTPUT FILES")
    print(SUB_SEPARATOR)

    print_field(
        "Summary",
        output_dir
        / "session_report_v1_summary.csv",
    )

    print_field(
        "Swings",
        output_dir
        / "session_report_v1_swings.csv",
    )

    print_field(
        "Observations",
        output_dir
        / "session_report_v1_observations.csv",
    )

    # -----------------------------------------------------------------
    # IMPORTANT
    # -----------------------------------------------------------------

    print()
    print("IMPORTANT")
    print(SUB_SEPARATOR)

    print(
        "Session Report V1 consolidates previously generated "
        "BaseVisionAI session and swing analytics."
    )

    print(
        "It does not recompute swing validity, session baselines, "
        "consistency metrics, or swing feedback."
    )

    print(
        "Observed variability and robust distance describe "
        "statistical similarity within the player's analyzed session."
    )

    print(
        "Lower variability does not mean better baseball mechanics."
    )

    print(
        "The representative swing is the baseline-eligible swing "
        "closest to the robust session median profile, not the "
        "'best' swing."
    )

    print(
        "The most different eligible swing is the eligible swing "
        "farthest from the robust session median profile, not the "
        "'worst' swing."
    )

    print(
        "TEMPORAL_LOW_CONFIDENCE and LOW_CONFIDENCE_SWING "
        "profiles may remain visible in the report but are not used "
        "to define the eligible session baseline."
    )

    print(
        "No GOOD/BAD label, coaching recommendation, biomechanical "
        "quality score, or performance score is generated."
    )

    print(
        "Projected rotation and motion measurements remain "
        "comparative 2D camera-dependent measurements."
    )

    print(SEPARATOR)


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build BaseVisionAI Session Report V1 from "
            "Session Insights V1 and Swing Insights V1."
        )
    )

    parser.add_argument(
        "--session-summary",
        required=True,
        help=(
            "Path to "
            "session_insights_v1_summary.csv"
        ),
    )

    parser.add_argument(
        "--session-insights",
        required=True,
        help=(
            "Path to session_insights_v1.csv"
        ),
    )

    parser.add_argument(
        "--swing-insights",
        required=True,
        help=(
            "Path to "
            "batch_swing_insights_v1_summary.csv"
        ),
    )

    parser.add_argument(
        "--player-id",
        required=True,
        help="Player identifier.",
    )

    parser.add_argument(
        "--session-id",
        required=True,
        help="Session identifier.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory.",
    )

    args = parser.parse_args()

    session_summary_path = Path(
        args.session_summary
    )

    session_insights_path = Path(
        args.session_insights
    )

    swing_insights_path = Path(
        args.swing_insights
    )

    output_dir = Path(
        args.output_dir
    )

    try:

        # =============================================================
        # LOAD
        # =============================================================

        session_summary = load_session_summary(
            session_summary_path
        )

        session_insights = load_session_insights(
            session_insights_path
        )

        swing_insights = load_swing_insights(
            swing_insights_path
        )

        # =============================================================
        # VALIDATE SESSION IDENTITY
        # =============================================================

        validate_identity(
            session_summary=session_summary,
            session_insights=session_insights,
            swing_insights=swing_insights,
            player_id=args.player_id,
            session_id=args.session_id,
        )

        # =============================================================
        # FILTER SWINGS
        # =============================================================

        swing_subset = swing_insights[
            (
                swing_insights[
                    "player_id"
                ].astype(str)
                == args.player_id
            )
            &
            (
                swing_insights[
                    "session_id"
                ].astype(str)
                == args.session_id
            )
        ].copy()

        # =============================================================
        # BUILD REPORT TABLES
        # =============================================================

        representative_swing = clean_string(
            session_summary[
                "representative_swing"
            ]
        )

        most_different_swing = clean_string(
            session_summary[
                "most_different_swing"
            ]
        )

        swings_df = build_swing_report(
            swing_insights=swing_subset,
            player_id=args.player_id,
            session_id=args.session_id,
            representative_swing=representative_swing,
            most_different_swing=most_different_swing,
        )

        summary_df = build_report_summary(
            session_summary=session_summary,
            swing_rows=swings_df,
        )

        observations_df = build_observations_report(
            session_summary=session_summary,
            session_insights=session_insights,
            player_id=args.player_id,
            session_id=args.session_id,
        )

        # =============================================================
        # CREATE OUTPUT DIRECTORY
        # =============================================================

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary_output = (
            output_dir
            / "session_report_v1_summary.csv"
        )

        swings_output = (
            output_dir
            / "session_report_v1_swings.csv"
        )

        observations_output = (
            output_dir
            / "session_report_v1_observations.csv"
        )

        # =============================================================
        # SAVE
        # =============================================================

        summary_df.to_csv(
            summary_output,
            index=False,
        )

        swings_df.to_csv(
            swings_output,
            index=False,
        )

        observations_df.to_csv(
            observations_output,
            index=False,
        )

        # =============================================================
        # CONSOLE
        # =============================================================

        print_console_report(
            summary_df=summary_df,
            swings_df=swings_df,
            observations_df=observations_df,
            output_dir=output_dir,
        )

    except Exception as exc:

        print()
        print(SEPARATOR)
        print(
            "BASEVISIONAI - SESSION REPORT V1 FAILED"
        )
        print(SEPARATOR)
        print()
        print(f"ERROR: {exc}")
        print()

        sys.exit(1)


if __name__ == "__main__":
    main()