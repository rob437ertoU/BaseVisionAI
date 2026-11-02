"""
BaseVisionAI - Swing Consistency V1

Computes session-level swing consistency metrics using the player's
VALID_SWING profiles.

The purpose of this module is to quantify repeatability across swings,
not to assign biomechanical quality or performance scores.

Input:
    - Player Swing Database V1
    - Swing Validity V1

Output:
    - swing_consistency_v1_metrics.csv
    - swing_consistency_v1_summary.csv
    - swing_consistency_v1_swings.csv

Important:
    - Only VALID_SWING profiles contribute to the main consistency baseline.
    - CV represents statistical variability, not swing quality.
    - Projected rotations are comparative 2D measurements.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

VERSION = "Swing Consistency V1"


METRICS = [
    {
        "group": "WRIST MOTION",
        "metric": "Left wrist peak normalized",
        "column": "left_wrist_peak_torsoscale_frame",
        "unit": "torso scales/frame",
    },
    {
        "group": "WRIST MOTION",
        "metric": "Right wrist peak normalized",
        "column": "right_wrist_peak_torsoscale_frame",
        "unit": "torso scales/frame",
    },
    {
        "group": "BODY CENTER MOTION",
        "metric": "Hip center peak",
        "column": "hip_center_peak_px_frame",
        "unit": "px/frame",
    },
    {
        "group": "BODY CENTER MOTION",
        "metric": "Shoulder center peak",
        "column": "shoulder_center_peak_px_frame",
        "unit": "px/frame",
    },
    {
        "group": "PROJECTED ROTATION",
        "metric": "Shoulder rotation",
        "column": "shoulder_rotation_projected_deg",
        "unit": "deg projected",
    },
    {
        "group": "PROJECTED ROTATION",
        "metric": "Hip rotation",
        "column": "hip_rotation_projected_deg",
        "unit": "deg projected",
    },
    {
        "group": "PROJECTED ROTATION",
        "metric": "Hip-shoulder separation",
        "column": "hip_shoulder_separation_projected_deg",
        "unit": "deg projected",
    },
    {
        "group": "PRE-PEAK STABILITY",
        "metric": "Head pre-peak movement",
        "column": "head_prepeak_torso",
        "unit": "torso lengths",
    },
    {
        "group": "PRE-PEAK STABILITY",
        "metric": "Left ankle pre-peak movement",
        "column": "left_ankle_prepeak_torso",
        "unit": "torso lengths",
    },
    {
        "group": "PRE-PEAK STABILITY",
        "metric": "Right ankle pre-peak movement",
        "column": "right_ankle_prepeak_torso",
        "unit": "torso lengths",
    },
    {
        "group": "TIMING",
        "metric": "Load duration",
        "column": "load_duration_sec",
        "unit": "s",
    },
    {
        "group": "TIMING",
        "metric": "Swing duration",
        "column": "swing_duration_sec",
        "unit": "s",
    },
    {
        "group": "TIMING",
        "metric": "Load to peak",
        "column": "load_to_peak_sec",
        "unit": "s",
    },
    {
        "group": "TIMING",
        "metric": "Swing to peak",
        "column": "swing_to_peak_sec",
        "unit": "s",
    },
]


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Swing Consistency V1"
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

def print_header(title: str, char: str = "=") -> None:
    print()
    print(char * 88)
    print(title)
    print(char * 88)


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * 88)


def safe_float(value) -> float:
    try:
        result = float(value)

        if np.isfinite(result):
            return result

    except (TypeError, ValueError):
        pass

    return np.nan


def safe_mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()

    if len(values) == 0:
        return np.nan

    return float(values.mean())


def safe_sd(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()

    if len(values) < 2:
        return np.nan

    return float(values.std(ddof=1))


def safe_median(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()

    if len(values) == 0:
        return np.nan

    return float(values.median())


def safe_min(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()

    if len(values) == 0:
        return np.nan

    return float(values.min())


def safe_max(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()

    if len(values) == 0:
        return np.nan

    return float(values.max())


def coefficient_of_variation(mean: float, sd: float) -> float:
    """
    Returns CV as percentage.

    CV = SD / |Mean| * 100
    """

    if not np.isfinite(mean):
        return np.nan

    if not np.isfinite(sd):
        return np.nan

    if abs(mean) < 1e-12:
        return np.nan

    return float((sd / abs(mean)) * 100.0)


def classify_cv(cv: float) -> str:
    """
    Descriptive repeatability classification.

    IMPORTANT:
    This is not a swing-quality classification.

    Lower CV = greater repeatability across the session.
    """

    if not np.isfinite(cv):
        return "INSUFFICIENT_DATA"

    if cv <= 10.0:
        return "VERY_CONSISTENT"

    if cv <= 20.0:
        return "CONSISTENT"

    if cv <= 35.0:
        return "MODERATE_VARIABILITY"

    if cv <= 50.0:
        return "HIGH_VARIABILITY"

    return "VERY_HIGH_VARIABILITY"


def variability_score(cv: float) -> float:
    """
    Converts CV into a bounded descriptive repeatability index.

    100 = low observed variability
    0   = very high observed variability

    This is NOT a performance score.
    """

    if not np.isfinite(cv):
        return np.nan

    return float(np.clip(100.0 - cv, 0.0, 100.0))


def format_number(value: float, decimals: int = 3) -> str:
    if not np.isfinite(value):
        return "N/A"

    return f"{value:.{decimals}f}"


def normalize_string(series: pd.Series) -> pd.Series:
    return (
        series
        .astype(str)
        .str.strip()
        .str.upper()
    )


# =============================================================================
# LOAD DATA
# =============================================================================

def load_database(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Player swing database not found: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "video_id",
        "player_id",
        "session_id",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Player swing database is missing required columns: "
            + ", ".join(sorted(missing))
        )

    return df


def load_validity(path: Path) -> tuple[pd.DataFrame, str]:
    """
    Supports the current BaseVisionAI schema:

        validity_status

    and remains backward-compatible with:

        validity
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Validity database not found: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "video_id",
        "player_id",
        "session_id",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Validity database is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if "validity_status" in df.columns:
        validity_column = "validity_status"

    elif "validity" in df.columns:
        validity_column = "validity"

    else:
        raise ValueError(
            "Validity database must contain either "
            "'validity_status' or 'validity'."
        )

    return df, validity_column


# =============================================================================
# SESSION SELECTION
# =============================================================================

def select_session(
    database: pd.DataFrame,
    validity: pd.DataFrame,
    validity_column: str,
    player_id: str,
    session_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    profiles = database[
        (database["player_id"].astype(str) == str(player_id))
        & (database["session_id"].astype(str) == str(session_id))
    ].copy()

    if profiles.empty:
        raise ValueError(
            f"No profiles found for player={player_id}, "
            f"session={session_id}"
        )

    validity_session = validity[
        (validity["player_id"].astype(str) == str(player_id))
        & (validity["session_id"].astype(str) == str(session_id))
    ].copy()

    if validity_session.empty:
        raise ValueError(
            f"No validity records found for player={player_id}, "
            f"session={session_id}"
        )

    validity_columns = [
        "video_id",
        validity_column,
    ]

    if "evidence_quality" in validity_session.columns:
        validity_columns.append("evidence_quality")

    if "reason" in validity_session.columns:
        validity_columns.append("reason")

    validity_subset = validity_session[
        validity_columns
    ].copy()

    # Avoid column collision if the profile database already contains
    # validity information.
    rename_map = {
        validity_column: "validity_status",
    }

    validity_subset = validity_subset.rename(
        columns=rename_map
    )

    # If validity_column was already validity_status, rename() simply keeps it.
    if "validity_status" not in validity_subset.columns:
        raise ValueError(
            "Unable to normalize validity column."
        )

    merged = profiles.merge(
        validity_subset,
        on="video_id",
        how="left",
        validate="many_to_one",
    )

    merged["validity_status"] = (
        merged["validity_status"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    valid = merged[
        merged["validity_status"].eq("VALID_SWING")
    ].copy()

    return merged, valid, validity_session


# =============================================================================
# PER-METRIC CONSISTENCY
# =============================================================================

def compute_metric_consistency(
    valid_df: pd.DataFrame,
) -> pd.DataFrame:

    rows: List[Dict] = []

    for config in METRICS:

        column = config["column"]

        if column not in valid_df.columns:
            rows.append(
                {
                    "consistency_version": VERSION,
                    "group": config["group"],
                    "metric": config["metric"],
                    "source_column": column,
                    "unit": config["unit"],
                    "n": 0,
                    "mean": np.nan,
                    "sd": np.nan,
                    "median": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "range": np.nan,
                    "cv_percent": np.nan,
                    "repeatability_index": np.nan,
                    "classification": "METRIC_NOT_AVAILABLE",
                }
            )

            continue

        values = pd.to_numeric(
            valid_df[column],
            errors="coerce",
        ).dropna()

        n = int(len(values))

        mean = safe_mean(values)
        sd = safe_sd(values)
        median = safe_median(values)
        minimum = safe_min(values)
        maximum = safe_max(values)

        if np.isfinite(minimum) and np.isfinite(maximum):
            value_range = maximum - minimum
        else:
            value_range = np.nan

        cv = coefficient_of_variation(
            mean,
            sd,
        )

        repeatability = variability_score(cv)

        classification = classify_cv(cv)

        rows.append(
            {
                "consistency_version": VERSION,
                "group": config["group"],
                "metric": config["metric"],
                "source_column": column,
                "unit": config["unit"],
                "n": n,
                "mean": mean,
                "sd": sd,
                "median": median,
                "min": minimum,
                "max": maximum,
                "range": value_range,
                "cv_percent": cv,
                "repeatability_index": repeatability,
                "classification": classification,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# PER-SWING DEVIATION
# =============================================================================

def compute_swing_deviations(
    valid_df: pd.DataFrame,
    metric_df: pd.DataFrame,
) -> pd.DataFrame:

    rows: List[Dict] = []

    metric_lookup = {
        row["source_column"]: row
        for _, row in metric_df.iterrows()
    }

    for _, swing in valid_df.iterrows():

        video_id = str(swing["video_id"])

        for config in METRICS:

            column = config["column"]

            if column not in valid_df.columns:
                continue

            value = safe_float(
                swing.get(column, np.nan)
            )

            metric_stats = metric_lookup.get(column)

            if metric_stats is None:
                continue

            mean = safe_float(
                metric_stats.get("mean", np.nan)
            )

            sd = safe_float(
                metric_stats.get("sd", np.nan)
            )

            if (
                np.isfinite(value)
                and np.isfinite(mean)
                and np.isfinite(sd)
                and sd > 1e-12
            ):
                z_score = (value - mean) / sd
            else:
                z_score = np.nan

            if (
                np.isfinite(value)
                and np.isfinite(mean)
                and abs(mean) > 1e-12
            ):
                difference_percent = (
                    (value - mean)
                    / abs(mean)
                    * 100.0
                )
            else:
                difference_percent = np.nan

            rows.append(
                {
                    "consistency_version": VERSION,
                    "video_id": video_id,
                    "player_id": swing["player_id"],
                    "session_id": swing["session_id"],
                    "validity_status": swing["validity_status"],
                    "group": config["group"],
                    "metric": config["metric"],
                    "source_column": column,
                    "unit": config["unit"],
                    "value": value,
                    "session_mean": mean,
                    "session_sd": sd,
                    "difference_percent": difference_percent,
                    "z_score": z_score,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# SUMMARY
# =============================================================================

def compute_summary(
    all_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    player_id: str,
    session_id: str,
) -> pd.DataFrame:

    available = metric_df[
        pd.to_numeric(
            metric_df["cv_percent"],
            errors="coerce",
        ).notna()
    ].copy()

    if available.empty:
        mean_cv = np.nan
        median_cv = np.nan
        mean_repeatability = np.nan

    else:
        mean_cv = float(
            available["cv_percent"].mean()
        )

        median_cv = float(
            available["cv_percent"].median()
        )

        mean_repeatability = float(
            available["repeatability_index"].mean()
        )

    counts = (
        metric_df["classification"]
        .value_counts()
        .to_dict()
    )

    pose_quality = np.nan

    if "mean_pose_quality" in valid_df.columns:
        pose_quality = safe_mean(
            valid_df["mean_pose_quality"]
        )

    summary = {
        "consistency_version": VERSION,
        "player_id": player_id,
        "session_id": session_id,
        "profiles_total": int(len(all_df)),
        "valid_swings": int(len(valid_df)),
        "excluded_profiles": int(
            len(all_df) - len(valid_df)
        ),
        "metrics_configured": int(len(METRICS)),
        "metrics_available": int(len(available)),
        "mean_cv_percent": mean_cv,
        "median_cv_percent": median_cv,
        "mean_repeatability_index": mean_repeatability,
        "very_consistent_metrics": int(
            counts.get("VERY_CONSISTENT", 0)
        ),
        "consistent_metrics": int(
            counts.get("CONSISTENT", 0)
        ),
        "moderate_variability_metrics": int(
            counts.get("MODERATE_VARIABILITY", 0)
        ),
        "high_variability_metrics": int(
            counts.get("HIGH_VARIABILITY", 0)
        ),
        "very_high_variability_metrics": int(
            counts.get("VERY_HIGH_VARIABILITY", 0)
        ),
        "mean_pose_quality": pose_quality,
        "calibrated_biomechanics": False,
    }

    return pd.DataFrame([summary])


# =============================================================================
# CONSOLE REPORT
# =============================================================================

def print_console_report(
    all_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    player_id: str,
    session_id: str,
    output_dir: Path,
) -> None:

    print_header(
        "BASEVISIONAI - SWING CONSISTENCY V1 COMPLETE"
    )

    print_section("SESSION")

    print(
        f"{'Player ID':26}: {player_id}"
    )
    print(
        f"{'Session ID':26}: {session_id}"
    )
    print(
        f"{'Profiles analyzed':26}: {len(all_df)}"
    )
    print(
        f"{'VALID_SWING profiles':26}: {len(valid_df)}"
    )
    print(
        f"{'Excluded profiles':26}: {len(all_df) - len(valid_df)}"
    )

    print_section("VALID SWINGS")

    for _, row in valid_df.iterrows():

        evidence = row.get(
            "evidence_quality",
            "N/A",
        )

        print(
            f"{str(row['video_id']):22} "
            f"{str(row['validity_status']):20} "
            f"{str(evidence)}"
        )

    current_group: Optional[str] = None

    print()
    print("CONSISTENCY BY METRIC")
    print("=" * 88)

    for _, row in metric_df.iterrows():

        group = row["group"]

        if group != current_group:
            print()
            print(group)
            print("-" * 88)
            current_group = group

        metric = str(row["metric"])
        n = int(row["n"])

        mean = safe_float(row["mean"])
        sd = safe_float(row["sd"])
        cv = safe_float(row["cv_percent"])
        repeatability = safe_float(
            row["repeatability_index"]
        )

        classification = str(
            row["classification"]
        )

        unit = str(row["unit"])

        print()
        print(metric)
        print(
            f"  Samples                 : {n}"
        )
        print(
            f"  Mean                    : "
            f"{format_number(mean)} {unit}"
        )
        print(
            f"  SD                      : "
            f"{format_number(sd)} {unit}"
        )

        if np.isfinite(cv):
            print(
                f"  CV                      : "
                f"{cv:.1f}%"
            )
        else:
            print(
                "  CV                      : N/A"
            )

        if np.isfinite(repeatability):
            print(
                f"  Repeatability index     : "
                f"{repeatability:.1f}/100"
            )
        else:
            print(
                "  Repeatability index     : N/A"
            )

        print(
            f"  Classification          : "
            f"{classification}"
        )

    summary = summary_df.iloc[0]

    print_section("SESSION SUMMARY")

    mean_cv = safe_float(
        summary["mean_cv_percent"]
    )

    median_cv = safe_float(
        summary["median_cv_percent"]
    )

    repeatability = safe_float(
        summary["mean_repeatability_index"]
    )

    if np.isfinite(mean_cv):
        print(
            f"{'Mean metric CV':26}: "
            f"{mean_cv:.1f}%"
        )
    else:
        print(
            f"{'Mean metric CV':26}: N/A"
        )

    if np.isfinite(median_cv):
        print(
            f"{'Median metric CV':26}: "
            f"{median_cv:.1f}%"
        )
    else:
        print(
            f"{'Median metric CV':26}: N/A"
        )

    if np.isfinite(repeatability):
        print(
            f"{'Mean repeatability index':26}: "
            f"{repeatability:.1f}/100"
        )
    else:
        print(
            f"{'Mean repeatability index':26}: N/A"
        )

    print()
    print(
        f"{'Very consistent':26}: "
        f"{int(summary['very_consistent_metrics'])}"
    )

    print(
        f"{'Consistent':26}: "
        f"{int(summary['consistent_metrics'])}"
    )

    print(
        f"{'Moderate variability':26}: "
        f"{int(summary['moderate_variability_metrics'])}"
    )

    print(
        f"{'High variability':26}: "
        f"{int(summary['high_variability_metrics'])}"
    )

    print(
        f"{'Very high variability':26}: "
        f"{int(summary['very_high_variability_metrics'])}"
    )

    print_section("OUTPUT FILES")

    print(
        f"{'Metrics':26}: "
        f"{output_dir / 'swing_consistency_v1_metrics.csv'}"
    )

    print(
        f"{'Per-swing deviations':26}: "
        f"{output_dir / 'swing_consistency_v1_swings.csv'}"
    )

    print(
        f"{'Summary':26}: "
        f"{output_dir / 'swing_consistency_v1_summary.csv'}"
    )

    print_section("IMPORTANT")

    print(
        "Consistency describes repeatability across the player's "
        "VALID_SWING profiles."
    )

    print(
        "Lower CV indicates lower session variability; "
        "it does not mean a swing is mechanically better."
    )

    print(
        "The repeatability index is derived from observed CV "
        "and is not a performance score."
    )

    print(
        "Projected rotation metrics remain comparative "
        "2D measurements."
    )

    print(
        "No calibrated biomechanical or performance score "
        "is generated."
    )

    print("=" * 88)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    args = parse_args()

    database_path = Path(args.database)
    validity_path = Path(args.validity)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        # ---------------------------------------------------------------------
        # LOAD
        # ---------------------------------------------------------------------

        database = load_database(
            database_path
        )

        validity, validity_column = load_validity(
            validity_path
        )

        # ---------------------------------------------------------------------
        # SELECT SESSION
        # ---------------------------------------------------------------------

        all_profiles, valid_profiles, _ = select_session(
            database=database,
            validity=validity,
            validity_column=validity_column,
            player_id=args.player_id,
            session_id=args.session_id,
        )

        if len(valid_profiles) < 2:
            raise ValueError(
                "At least 2 VALID_SWING profiles are required "
                "to compute session consistency."
            )

        # ---------------------------------------------------------------------
        # COMPUTE
        # ---------------------------------------------------------------------

        metric_df = compute_metric_consistency(
            valid_profiles
        )

        swing_df = compute_swing_deviations(
            valid_profiles,
            metric_df,
        )

        summary_df = compute_summary(
            all_df=all_profiles,
            valid_df=valid_profiles,
            metric_df=metric_df,
            player_id=args.player_id,
            session_id=args.session_id,
        )

        # ---------------------------------------------------------------------
        # OUTPUT PATHS
        # ---------------------------------------------------------------------

        metrics_output = (
            output_dir
            / "swing_consistency_v1_metrics.csv"
        )

        swings_output = (
            output_dir
            / "swing_consistency_v1_swings.csv"
        )

        summary_output = (
            output_dir
            / "swing_consistency_v1_summary.csv"
        )

        # ---------------------------------------------------------------------
        # SAVE
        # ---------------------------------------------------------------------

        metric_df.to_csv(
            metrics_output,
            index=False,
        )

        swing_df.to_csv(
            swings_output,
            index=False,
        )

        summary_df.to_csv(
            summary_output,
            index=False,
        )

        # ---------------------------------------------------------------------
        # REPORT
        # ---------------------------------------------------------------------

        print_console_report(
            all_df=all_profiles,
            valid_df=valid_profiles,
            metric_df=metric_df,
            summary_df=summary_df,
            player_id=args.player_id,
            session_id=args.session_id,
            output_dir=output_dir,
        )

    except Exception as exc:

        print_header(
            "BASEVISIONAI - SWING CONSISTENCY V1 FAILED"
        )

        print()
        print(f"ERROR: {exc}")
        print()

        raise SystemExit(1)


if __name__ == "__main__":
    main()