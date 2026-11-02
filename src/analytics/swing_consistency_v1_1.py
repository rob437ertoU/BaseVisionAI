"""
BaseVisionAI - Swing Consistency V1.1
=====================================

Session-level repeatability analysis using Player Swing Profile V1
and Swing Validity V1.1.

V1.1 improvements
-----------------
- Uses baseline_eligible from Swing Validity V1.1.
- Preserves Mean, SD and CV.
- Adds Median, MAD and IQR.
- Computes robust per-swing distance from the session median profile.
- Identifies the most representative baseline swing.
- Identifies the most different baseline swing.

IMPORTANT
---------
This module measures statistical repeatability of comparative
2D computer-vision features.

It does NOT generate:
- biomechanical quality scores
- coaching grades
- GOOD/BAD classifications
- calibrated physical measurements

"Representative" means closest to the session's robust central profile.
"Most different" means farthest from that profile.
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

VERSION = "Swing Consistency V1.1"


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

def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Swing Consistency V1.1"
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


def numeric_values(series):

    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=float)
    )


def coefficient_of_variation(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) < 2:
        return np.nan

    mean = np.mean(values)

    if abs(mean) < 1e-12:
        return np.nan

    sd = np.std(
        values,
        ddof=1,
    )

    return abs(sd / mean) * 100.0


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
            np.abs(
                values - median
            )
        )
    )


def robust_cv(values):

    """
    Robust coefficient of variation.

    Uses:
        1.4826 * MAD / |median| * 100

    1.4826 scales MAD to an SD-like estimate under a
    normal-distribution assumption.

    This remains a descriptive statistic, not a performance score.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) < 2:
        return np.nan

    median = np.median(values)

    if abs(median) < 1e-12:
        return np.nan

    mad = median_absolute_deviation(
        values
    )

    robust_sigma = (
        1.4826 * mad
    )

    return (
        abs(
            robust_sigma / median
        )
        * 100.0
    )


def robust_z_scores(values):

    """
    Robust z-score:

        z = 0.67448975 * (x - median) / MAD

    If MAD collapses to zero, conventional z-score is used
    as a fallback.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    output = np.full(
        values.shape,
        np.nan,
        dtype=float,
    )

    mask = np.isfinite(
        values
    )

    if mask.sum() == 0:
        return output

    x = values[mask]

    median = np.median(x)

    mad = median_absolute_deviation(
        x
    )

    if (
        np.isfinite(mad)
        and mad > 1e-12
    ):

        output[mask] = (
            0.67448975
            * (x - median)
            / mad
        )

        return output

    # -------------------------------------------------------------------------
    # Fallback if MAD = 0
    # -------------------------------------------------------------------------

    if len(x) >= 2:

        mean = np.mean(x)

        sd = np.std(
            x,
            ddof=1,
        )

        if sd > 1e-12:

            output[mask] = (
                (x - mean)
                / sd
            )

        else:

            output[mask] = 0.0

    else:

        output[mask] = 0.0

    return output


def classify_cv(cv):

    if not np.isfinite(cv):
        return "INSUFFICIENT_DATA"

    if cv <= 10:
        return "VERY_CONSISTENT"

    if cv <= 20:
        return "CONSISTENT"

    if cv <= 35:
        return "MODERATE_VARIABILITY"

    if cv <= 50:
        return "HIGH_VARIABILITY"

    return "VERY_HIGH_VARIABILITY"


def bool_from_value(value):

    if isinstance(value, bool):
        return value

    text = str(
        value
    ).strip().lower()

    return text in {
        "true",
        "1",
        "yes",
        "y",
    }


def format_value(value, decimals=3):

    value = safe_float(
        value
    )

    if not np.isfinite(value):
        return "N/A"

    return f"{value:.{decimals}f}"


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
# SESSION SELECTION
# =============================================================================

def select_session(
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
            f"No player profiles found for "
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
            f"No validity records found for "
            f"{player_id} / {session_id}"
        )

    validity_columns = [
        "video_id",
        "validity_status_v1_1",
        "baseline_eligible",
    ]

    optional_columns = [
        "evidence_quality_v1_1",
        "temporal_confidence",
        "validity_reason_v1_1",
    ]

    for column in optional_columns:

        if column in validity_session.columns:
            validity_columns.append(
                column
            )

    validity_subset = (
        validity_session[
            validity_columns
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

    merged["baseline_eligible"] = (
        merged["baseline_eligible"]
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

    baseline = merged[
        merged["baseline_eligible"]
    ].copy()

    return merged, baseline


# =============================================================================
# METRIC STATISTICS
# =============================================================================

def compute_metric_statistics(
    baseline,
):

    rows = []

    for config in METRICS:

        column = config[
            "column"
        ]

        if column not in baseline.columns:

            rows.append(
                {
                    "consistency_version":
                        VERSION,

                    "group":
                        config["group"],

                    "metric":
                        config["metric"],

                    "source_column":
                        column,

                    "unit":
                        config["unit"],

                    "n":
                        0,

                    "mean":
                        np.nan,

                    "sd":
                        np.nan,

                    "cv_percent":
                        np.nan,

                    "median":
                        np.nan,

                    "mad":
                        np.nan,

                    "robust_cv_percent":
                        np.nan,

                    "q1":
                        np.nan,

                    "q3":
                        np.nan,

                    "iqr":
                        np.nan,

                    "min":
                        np.nan,

                    "max":
                        np.nan,

                    "classification":
                        "METRIC_NOT_AVAILABLE",
                }
            )

            continue

        values = numeric_values(
            baseline[column]
        )

        n = len(values)

        if n == 0:

            mean = np.nan
            sd = np.nan
            cv = np.nan

            median = np.nan
            mad = np.nan
            rcv = np.nan

            q1 = np.nan
            q3 = np.nan
            iqr = np.nan

            minimum = np.nan
            maximum = np.nan

        else:

            mean = float(
                np.mean(values)
            )

            median = float(
                np.median(values)
            )

            minimum = float(
                np.min(values)
            )

            maximum = float(
                np.max(values)
            )

            q1 = float(
                np.percentile(
                    values,
                    25,
                )
            )

            q3 = float(
                np.percentile(
                    values,
                    75,
                )
            )

            iqr = (
                q3 - q1
            )

            mad = (
                median_absolute_deviation(
                    values
                )
            )

            if n >= 2:

                sd = float(
                    np.std(
                        values,
                        ddof=1,
                    )
                )

            else:

                sd = np.nan

            cv = (
                coefficient_of_variation(
                    values
                )
            )

            rcv = (
                robust_cv(
                    values
                )
            )

        rows.append(
            {
                "consistency_version":
                    VERSION,

                "group":
                    config["group"],

                "metric":
                    config["metric"],

                "source_column":
                    column,

                "unit":
                    config["unit"],

                "n":
                    n,

                "mean":
                    mean,

                "sd":
                    sd,

                "cv_percent":
                    cv,

                "median":
                    median,

                "mad":
                    mad,

                "robust_cv_percent":
                    rcv,

                "q1":
                    q1,

                "q3":
                    q3,

                "iqr":
                    iqr,

                "min":
                    minimum,

                "max":
                    maximum,

                "classification":
                    classify_cv(cv),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# ROBUST DISTANCES
# =============================================================================

def compute_robust_distances(
    baseline,
):

    usable_metrics = []

    z_table = pd.DataFrame(
        index=baseline.index
    )

    for config in METRICS:

        column = config[
            "column"
        ]

        if column not in baseline.columns:
            continue

        values = pd.to_numeric(
            baseline[column],
            errors="coerce",
        )

        if values.notna().sum() < 2:
            continue

        z = robust_z_scores(
            values.to_numpy(
                dtype=float
            )
        )

        z_table[column] = np.abs(
            z
        )

        usable_metrics.append(
            column
        )

    if not usable_metrics:

        raise ValueError(
            "No usable metrics available for "
            "robust swing-distance analysis."
        )

    rows = []

    for idx, swing in baseline.iterrows():

        distances = (
            z_table
            .loc[idx]
            .dropna()
            .to_numpy(dtype=float)
        )

        if len(distances) == 0:

            mean_distance = np.nan
            median_distance = np.nan
            rms_distance = np.nan
            max_distance = np.nan

        else:

            mean_distance = float(
                np.mean(
                    distances
                )
            )

            median_distance = float(
                np.median(
                    distances
                )
            )

            rms_distance = float(
                np.sqrt(
                    np.mean(
                        distances ** 2
                    )
                )
            )

            max_distance = float(
                np.max(
                    distances
                )
            )

        rows.append(
            {
                "consistency_version":
                    VERSION,

                "video_id":
                    swing["video_id"],

                "player_id":
                    swing["player_id"],

                "session_id":
                    swing["session_id"],

                "validity_status":
                    swing[
                        "validity_status_v1_1"
                    ],

                "baseline_eligible":
                    True,

                "metrics_used":
                    len(distances),

                "mean_robust_distance":
                    mean_distance,

                "median_robust_distance":
                    median_distance,

                "rms_robust_distance":
                    rms_distance,

                "max_robust_distance":
                    max_distance,
            }
        )

    result = pd.DataFrame(
        rows
    )

    result = result.sort_values(
        by=[
            "mean_robust_distance",
            "rms_robust_distance",
            "video_id",
        ],
        ascending=[
            True,
            True,
            True,
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    result[
        "representativeness_rank"
    ] = np.arange(
        1,
        len(result) + 1,
    )

    return result


# =============================================================================
# GROUP SUMMARY
# =============================================================================

def compute_group_summary(
    metrics,
):

    rows = []

    for group in metrics[
        "group"
    ].drop_duplicates():

        subset = metrics[
            metrics["group"] == group
        ].copy()

        cv = pd.to_numeric(
            subset["cv_percent"],
            errors="coerce",
        ).dropna()

        robust = pd.to_numeric(
            subset["robust_cv_percent"],
            errors="coerce",
        ).dropna()

        rows.append(
            {
                "consistency_version":
                    VERSION,

                "group":
                    group,

                "metrics_available":
                    int(
                        subset[
                            "cv_percent"
                        ].notna().sum()
                    ),

                "mean_cv_percent":
                    (
                        float(cv.mean())
                        if len(cv)
                        else np.nan
                    ),

                "median_cv_percent":
                    (
                        float(cv.median())
                        if len(cv)
                        else np.nan
                    ),

                "mean_robust_cv_percent":
                    (
                        float(robust.mean())
                        if len(robust)
                        else np.nan
                    ),

                "median_robust_cv_percent":
                    (
                        float(robust.median())
                        if len(robust)
                        else np.nan
                    ),

                "classification":
                    classify_cv(
                        float(cv.median())
                        if len(cv)
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# SESSION SUMMARY
# =============================================================================

def compute_session_summary(
    all_profiles,
    baseline,
    metrics,
    distances,
    player_id,
    session_id,
):

    usable_distances = (
        distances.dropna(
            subset=[
                "mean_robust_distance"
            ]
        )
    )

    representative_swing = ""
    representative_distance = np.nan

    most_different_swing = ""
    most_different_distance = np.nan

    if not usable_distances.empty:

        representative = (
            usable_distances.iloc[0]
        )

        different = (
            usable_distances.iloc[-1]
        )

        representative_swing = (
            representative["video_id"]
        )

        representative_distance = (
            representative[
                "mean_robust_distance"
            ]
        )

        most_different_swing = (
            different["video_id"]
        )

        most_different_distance = (
            different[
                "mean_robust_distance"
            ]
        )

    cv = pd.to_numeric(
        metrics["cv_percent"],
        errors="coerce",
    ).dropna()

    rcv = pd.to_numeric(
        metrics["robust_cv_percent"],
        errors="coerce",
    ).dropna()

    row = {
        "consistency_version":
            VERSION,

        "player_id":
            player_id,

        "session_id":
            session_id,

        "profiles_total":
            len(all_profiles),

        "baseline_eligible_swings":
            len(baseline),

        "excluded_profiles":
            len(all_profiles)
            - len(baseline),

        "metrics_configured":
            len(METRICS),

        "metrics_available":
            len(cv),

        "mean_cv_percent":
            (
                float(cv.mean())
                if len(cv)
                else np.nan
            ),

        "median_cv_percent":
            (
                float(cv.median())
                if len(cv)
                else np.nan
            ),

        "mean_robust_cv_percent":
            (
                float(rcv.mean())
                if len(rcv)
                else np.nan
            ),

        "median_robust_cv_percent":
            (
                float(rcv.median())
                if len(rcv)
                else np.nan
            ),

        "representative_swing":
            representative_swing,

        "representative_distance":
            representative_distance,

        "most_different_swing":
            most_different_swing,

        "most_different_distance":
            most_different_distance,

        "distance_method":
            (
                "mean absolute robust z-distance "
                "from session median profile"
            ),

        "baseline_filter":
            "baseline_eligible=True",

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
    all_profiles,
    baseline,
    metrics,
    groups,
    distances,
    summary,
    output_dir,
):

    print()
    separator("=")

    print(
        "BASEVISIONAI - SWING CONSISTENCY V1.1 COMPLETE"
    )

    separator("=")

    print()
    print("SESSION")
    separator()

    print(
        f"Player ID                       : "
        f"{summary.iloc[0]['player_id']}"
    )

    print(
        f"Session ID                      : "
        f"{summary.iloc[0]['session_id']}"
    )

    print(
        f"Profiles analyzed               : "
        f"{len(all_profiles)}"
    )

    print(
        f"Baseline eligible               : "
        f"{len(baseline)}"
    )

    print(
        f"Excluded from baseline          : "
        f"{len(all_profiles) - len(baseline)}"
    )

    print()
    print("BASELINE SWINGS")
    separator()

    for _, row in baseline.iterrows():

        print(
            f"{row['video_id']:<24}"
            f"{row['validity_status_v1_1']:<28}"
            f"baseline=True"
        )

    print()
    print("CONSISTENCY BY METRIC")
    separator("=")

    current_group = None

    for _, row in metrics.iterrows():

        if row["group"] != current_group:

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

        print(
            f"  Samples                       : "
            f"{row['n']}"
        )

        print(
            f"  Mean                          : "
            f"{format_value(row['mean'])} "
            f"{row['unit']}"
        )

        print(
            f"  SD                            : "
            f"{format_value(row['sd'])} "
            f"{row['unit']}"
        )

        print(
            f"  CV                            : "
            f"{format_value(row['cv_percent'], 1)}%"
        )

        print(
            f"  Median                        : "
            f"{format_value(row['median'])} "
            f"{row['unit']}"
        )

        print(
            f"  MAD                           : "
            f"{format_value(row['mad'])} "
            f"{row['unit']}"
        )

        print(
            f"  Robust CV                     : "
            f"{format_value(row['robust_cv_percent'], 1)}%"
        )

        print(
            f"  IQR                           : "
            f"{format_value(row['iqr'])} "
            f"{row['unit']}"
        )

        print(
            f"  Classification                : "
            f"{row['classification']}"
        )

    print()
    print("GROUP SUMMARY")
    separator()

    for _, row in groups.iterrows():

        print(
            f"{row['group']:<28}"
            f"Median CV="
            f"{format_value(row['median_cv_percent'], 1):>7}%"
            f" | Robust CV="
            f"{format_value(row['median_robust_cv_percent'], 1):>7}%"
        )

    print()
    print("SWING REPRESENTATIVENESS")
    separator()

    for _, row in distances.iterrows():

        print(
            f"Rank "
            f"{int(row['representativeness_rank']):02d}"
            f" | "
            f"{row['video_id']:<20}"
            f" | Mean robust distance = "
            f"{format_value(row['mean_robust_distance'])}"
            f" | Metrics = "
            f"{int(row['metrics_used'])}"
        )

    s = summary.iloc[0]

    print()
    print("SESSION PROFILE")
    separator()

    print(
        f"Most representative swing       : "
        f"{s['representative_swing']}"
    )

    print(
        f"Representative distance          : "
        f"{format_value(s['representative_distance'])}"
    )

    print(
        f"Most different swing             : "
        f"{s['most_different_swing']}"
    )

    print(
        f"Most different distance          : "
        f"{format_value(s['most_different_distance'])}"
    )

    print()
    print("SESSION STATISTICS")
    separator()

    print(
        f"Mean metric CV                   : "
        f"{format_value(s['mean_cv_percent'], 1)}%"
    )

    print(
        f"Median metric CV                 : "
        f"{format_value(s['median_cv_percent'], 1)}%"
    )

    print(
        f"Mean robust CV                   : "
        f"{format_value(s['mean_robust_cv_percent'], 1)}%"
    )

    print(
        f"Median robust CV                 : "
        f"{format_value(s['median_robust_cv_percent'], 1)}%"
    )

    print()
    print("OUTPUT FILES")
    separator()

    print(
        f"Metrics                         : "
        f"{output_dir / 'swing_consistency_v1_1_metrics.csv'}"
    )

    print(
        f"Groups                          : "
        f"{output_dir / 'swing_consistency_v1_1_groups.csv'}"
    )

    print(
        f"Swing distances                 : "
        f"{output_dir / 'swing_consistency_v1_1_swings.csv'}"
    )

    print(
        f"Summary                         : "
        f"{output_dir / 'swing_consistency_v1_1_summary.csv'}"
    )

    print()
    print("IMPORTANT")
    separator()

    print(
        "Only baseline_eligible=True swings from "
        "Swing Validity V1.1 define the session profile."
    )

    print(
        "Mean/SD/CV and Median/MAD/IQR are reported "
        "together for transparency."
    )

    print(
        "Representative means closest to the robust "
        "session profile; it does not mean best."
    )

    print(
        "Most different means farthest from the robust "
        "session profile; it does not mean worst."
    )

    print(
        "No biomechanical quality or performance score "
        "is generated."
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

        all_profiles, baseline = (
            select_session(
                database=database,
                validity=validity,
                player_id=args.player_id,
                session_id=args.session_id,
            )
        )

        if len(baseline) < 2:

            raise ValueError(
                "Swing Consistency V1.1 requires at least "
                "2 baseline_eligible swings."
            )

        # ---------------------------------------------------------------------
        # Statistics
        # ---------------------------------------------------------------------

        metrics = (
            compute_metric_statistics(
                baseline
            )
        )

        groups = (
            compute_group_summary(
                metrics
            )
        )

        distances = (
            compute_robust_distances(
                baseline
            )
        )

        summary = (
            compute_session_summary(
                all_profiles=all_profiles,
                baseline=baseline,
                metrics=metrics,
                distances=distances,
                player_id=args.player_id,
                session_id=args.session_id,
            )
        )

        # ---------------------------------------------------------------------
        # Output
        # ---------------------------------------------------------------------

        output_dir = Path(
            args.output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        metrics_path = (
            output_dir
            / "swing_consistency_v1_1_metrics.csv"
        )

        groups_path = (
            output_dir
            / "swing_consistency_v1_1_groups.csv"
        )

        swings_path = (
            output_dir
            / "swing_consistency_v1_1_swings.csv"
        )

        summary_path = (
            output_dir
            / "swing_consistency_v1_1_summary.csv"
        )

        metrics.to_csv(
            metrics_path,
            index=False,
        )

        groups.to_csv(
            groups_path,
            index=False,
        )

        distances.to_csv(
            swings_path,
            index=False,
        )

        summary.to_csv(
            summary_path,
            index=False,
        )

        # ---------------------------------------------------------------------
        # Console
        # ---------------------------------------------------------------------

        print_report(
            all_profiles=all_profiles,
            baseline=baseline,
            metrics=metrics,
            groups=groups,
            distances=distances,
            summary=summary,
            output_dir=output_dir,
        )

    except Exception as exc:

        print()
        separator("=")

        print(
            "BASEVISIONAI - SWING CONSISTENCY V1.1 FAILED"
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