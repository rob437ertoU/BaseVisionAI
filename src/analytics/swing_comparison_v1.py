from pathlib import Path
import argparse

import numpy as np
import pandas as pd


# ============================================================
# BASEVISIONAI - SWING COMPARISON V1
# ============================================================
#
# Purpose:
#   Compare multiple Player Swing Profile V1 records belonging
#   to the same player.
#
# Input:
#   player_swing_database_v1.csv
#
# Outputs:
#   - swing_comparison_v1_swings.csv
#   - swing_comparison_v1_summary.csv
#   - swing_comparison_v1_long.csv
#
# IMPORTANT:
#   - No model inference.
#   - No threshold tuning.
#   - No strike / ball / hit classification.
#   - Metrics remain comparative 2D computer-vision metrics.
# ============================================================


# ============================================================
# METRICS TO COMPARE
# ============================================================

METRICS = {

    # --------------------------------------------------------
    # Wrist motion
    # --------------------------------------------------------

    "left_wrist_peak_px_frame":
        "Left wrist peak",

    "right_wrist_peak_px_frame":
        "Right wrist peak",

    "left_wrist_peak_torsoscale_frame":
        "Left wrist peak normalized",

    "right_wrist_peak_torsoscale_frame":
        "Right wrist peak normalized",

    "right_minus_left_wrist_peak":
        "R-L wrist peak difference",

    "wrist_peak_time_gap_sec":
        "Wrist peak timing gap",

    # --------------------------------------------------------
    # Body motion
    # --------------------------------------------------------

    "hip_center_peak_px_frame":
        "Hip center peak",

    "shoulder_center_peak_px_frame":
        "Shoulder center peak",

    # --------------------------------------------------------
    # Projected rotation
    # --------------------------------------------------------

    "shoulder_rotation_projected_deg":
        "Shoulder projected rotation",

    "hip_rotation_projected_deg":
        "Hip projected rotation",

    "hip_shoulder_separation_projected_deg":
        "Hip-shoulder separation",

    "shoulder_minus_hip_rotation_deg":
        "Shoulder-Hip rotation difference",

    # --------------------------------------------------------
    # Pre-peak stability
    # --------------------------------------------------------

    "head_prepeak_torso":
        "Head pre-peak movement",

    "left_ankle_prepeak_torso":
        "Left ankle pre-peak movement",

    "right_ankle_prepeak_torso":
        "Right ankle pre-peak movement",

    "left_knee_prepeak_torso":
        "Left knee pre-peak movement",

    "right_knee_prepeak_torso":
        "Right knee pre-peak movement",

    # --------------------------------------------------------
    # Follow-through
    # --------------------------------------------------------

    "head_follow_torso":
        "Head follow-through movement",

    "left_ankle_follow_torso":
        "Left ankle follow-through movement",

    "right_ankle_follow_torso":
        "Right ankle follow-through movement",

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    "load_duration_sec":
        "Load duration",

    "swing_duration_sec":
        "Swing duration",

    "peak_duration_sec":
        "Peak duration",

    "follow_duration_sec":
        "Follow-through duration",

    "load_to_peak_sec":
        "Load to peak",

    "swing_to_peak_sec":
        "Swing to peak",

    # --------------------------------------------------------
    # Quality
    # --------------------------------------------------------

    "mean_pose_quality":
        "Mean pose quality",
}


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Swing Comparison V1"
    )

    parser.add_argument(
        "--database",
        required=True,
        help="Player Swing Database V1 CSV"
    )

    parser.add_argument(
        "--player-id",
        required=True,
        help="Player identifier to compare"
    )

    parser.add_argument(
        "--session-id",
        default=None,
        help=(
            "Optional session filter. "
            "If omitted, all sessions for the player are used."
        )
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory"
    )

    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def numeric_series(
    df,
    column
):

    if column not in df.columns:

        return pd.Series(
            dtype=float
        )

    return pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()


def safe_mean(
    values
):

    if len(values) == 0:
        return np.nan

    return float(
        values.mean()
    )


def safe_median(
    values
):

    if len(values) == 0:
        return np.nan

    return float(
        values.median()
    )


def safe_std(
    values
):

    if len(values) < 2:
        return np.nan

    return float(
        values.std(
            ddof=1
        )
    )


def safe_min(
    values
):

    if len(values) == 0:
        return np.nan

    return float(
        values.min()
    )


def safe_max(
    values
):

    if len(values) == 0:
        return np.nan

    return float(
        values.max()
    )


def safe_range(
    values
):

    if len(values) == 0:
        return np.nan

    return float(
        values.max()
        -
        values.min()
    )


def safe_cv(
    mean,
    std
):

    if not (
        np.isfinite(mean)
        and
        np.isfinite(std)
    ):
        return np.nan

    if abs(mean) < 1e-9:
        return np.nan

    return float(
        std
        /
        abs(mean)
    )


def safe_cv_percent(
    mean,
    std
):

    cv = safe_cv(
        mean,
        std
    )

    if not np.isfinite(
        cv
    ):
        return np.nan

    return float(
        cv * 100.0
    )


def fmt(
    value,
    digits=3
):

    if not np.isfinite(
        value
    ):
        return "N/A"

    return f"{value:.{digits}f}"


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    database_path = Path(
        args.database
    )

    output_dir = Path(
        args.output_dir
    )

    if not database_path.exists():

        raise FileNotFoundError(
            f"Database not found:\n"
            f"{database_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # LOAD DATABASE
    # ========================================================

    database_df = pd.read_csv(
        database_path
    )

    if len(
        database_df
    ) == 0:

        raise ValueError(
            "Swing database is empty."
        )

    required_columns = [
        "player_id",
        "session_id",
        "video_id",
    ]

    for column in required_columns:

        if column not in database_df.columns:

            raise ValueError(
                f"Database missing required column: {column}"
            )

    # ========================================================
    # FILTER PLAYER
    # ========================================================

    player_mask = (
        database_df[
            "player_id"
        ].astype(str)
        ==
        str(
            args.player_id
        )
    )

    comparison_df = database_df[
        player_mask
    ].copy()

    if len(
        comparison_df
    ) == 0:

        raise ValueError(
            f"No swings found for player_id={args.player_id}"
        )

    # ========================================================
    # OPTIONAL SESSION FILTER
    # ========================================================

    if args.session_id is not None:

        session_mask = (
            comparison_df[
                "session_id"
            ].astype(str)
            ==
            str(
                args.session_id
            )
        )

        comparison_df = comparison_df[
            session_mask
        ].copy()

        if len(
            comparison_df
        ) == 0:

            raise ValueError(
                "No swings found for "
                f"player_id={args.player_id} "
                f"and session_id={args.session_id}"
            )

    comparison_df = comparison_df.reset_index(
        drop=True
    )

    # ========================================================
    # ADD SWING NUMBER
    # ========================================================

    comparison_df.insert(
        0,
        "swing_number",
        np.arange(
            1,
            len(comparison_df) + 1
        )
    )

    # ========================================================
    # SAVE FILTERED SWINGS
    # ========================================================

    swings_output = (
        output_dir
        /
        "swing_comparison_v1_swings.csv"
    )

    comparison_df.to_csv(
        swings_output,
        index=False
    )

    # ========================================================
    # SUMMARY STATISTICS
    # ========================================================

    summary_rows = []

    for column, label in METRICS.items():

        values = numeric_series(
            comparison_df,
            column
        )

        n = int(
            len(values)
        )

        mean = safe_mean(
            values
        )

        median = safe_median(
            values
        )

        std = safe_std(
            values
        )

        minimum = safe_min(
            values
        )

        maximum = safe_max(
            values
        )

        value_range = safe_range(
            values
        )

        cv = safe_cv(
            mean,
            std
        )

        cv_percent = safe_cv_percent(
            mean,
            std
        )

        summary_rows.append(
            {
                "metric":
                    column,

                "label":
                    label,

                "n":
                    n,

                "mean":
                    mean,

                "median":
                    median,

                "std":
                    std,

                "min":
                    minimum,

                "max":
                    maximum,

                "range":
                    value_range,

                "cv":
                    cv,

                "cv_percent":
                    cv_percent,
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_output = (
        output_dir
        /
        "swing_comparison_v1_summary.csv"
    )

    summary_df.to_csv(
        summary_output,
        index=False
    )

    # ========================================================
    # LONG-FORM DATASET
    # ========================================================

    long_rows = []

    for _, swing_row in comparison_df.iterrows():

        for column, label in METRICS.items():

            if column not in swing_row.index:
                continue

            try:

                value = float(
                    swing_row[
                        column
                    ]
                )

            except Exception:
                continue

            if not np.isfinite(
                value
            ):
                continue

            long_rows.append(
                {
                    "player_id":
                        swing_row[
                            "player_id"
                        ],

                    "session_id":
                        swing_row[
                            "session_id"
                        ],

                    "video_id":
                        swing_row[
                            "video_id"
                        ],

                    "swing_number":
                        swing_row[
                            "swing_number"
                        ],

                    "metric":
                        column,

                    "label":
                        label,

                    "value":
                        value,
                }
            )

    long_df = pd.DataFrame(
        long_rows
    )

    long_output = (
        output_dir
        /
        "swing_comparison_v1_long.csv"
    )

    long_df.to_csv(
        long_output,
        index=False
    )

    # ========================================================
    # CONSOLE REPORT
    # ========================================================

    print()
    print("=" * 80)
    print(
        "SWING COMPARISON V1 COMPLETE"
    )
    print("=" * 80)

    print()
    print("SELECTION")
    print("-" * 80)

    print(
        f"Player ID                 : {args.player_id}"
    )

    print(
        f"Session filter            : "
        f"{args.session_id if args.session_id else 'ALL'}"
    )

    print(
        f"Swings compared           : {len(comparison_df)}"
    )

    print(
        f"Sessions represented      : "
        f"{comparison_df['session_id'].nunique()}"
    )

    # ========================================================
    # SINGLE-SWING WARNING
    # ========================================================

    if len(
        comparison_df
    ) < 2:

        print()
        print("CONSISTENCY")
        print("-" * 80)

        print(
            "Only one swing is available."
        )

        print(
            "Mean values can be reported, but variability "
            "and consistency cannot yet be estimated."
        )

    else:

        print()
        print("KEY COMPARISON METRICS")
        print("-" * 80)

        key_metrics = [
            "right_wrist_peak_px_frame",
            "left_wrist_peak_px_frame",
            "hip_center_peak_px_frame",
            "shoulder_center_peak_px_frame",
            "load_to_peak_sec",
            "swing_to_peak_sec",
            "head_prepeak_torso",
            "hip_shoulder_separation_projected_deg",
        ]

        for metric in key_metrics:

            row_match = summary_df[
                summary_df[
                    "metric"
                ]
                ==
                metric
            ]

            if len(
                row_match
            ) == 0:
                continue

            stats = row_match.iloc[
                0
            ]

            print(
                f"{stats['label']:<32} "
                f"Mean={fmt(stats['mean'])}  "
                f"SD={fmt(stats['std'])}  "
                f"CV={fmt(stats['cv_percent'], 1)}%"
            )

    # ========================================================
    # INDIVIDUAL SWINGS
    # ========================================================

    print()
    print("SWINGS")
    print("-" * 80)

    for _, row in comparison_df.iterrows():

        video_id = row.get(
            "video_id",
            "N/A"
        )

        session_id = row.get(
            "session_id",
            "N/A"
        )

        right_wrist = pd.to_numeric(
            pd.Series(
                [
                    row.get(
                        "right_wrist_peak_px_frame",
                        np.nan
                    )
                ]
            ),
            errors="coerce"
        ).iloc[0]

        load_peak = pd.to_numeric(
            pd.Series(
                [
                    row.get(
                        "load_to_peak_sec",
                        np.nan
                    )
                ]
            ),
            errors="coerce"
        ).iloc[0]

        print(
            f"Swing {int(row['swing_number']):02d} | "
            f"{video_id} | "
            f"{session_id} | "
            f"R wrist={fmt(right_wrist, 2)} px/f | "
            f"Load->Peak={fmt(load_peak, 3)} s"
        )

    # ========================================================
    # OUTPUTS
    # ========================================================

    print()
    print("OUTPUT FILES")
    print("-" * 80)

    print(
        f"Swings  : {swings_output.resolve()}"
    )

    print(
        f"Summary : {summary_output.resolve()}"
    )

    print(
        f"Long    : {long_output.resolve()}"
    )

    print()
    print("IMPORTANT")
    print("-" * 80)

    print(
        "CV represents statistical variability, not swing quality."
    )

    print(
        "Projected angles remain 2D comparative measurements."
    )

    print(
        "No calibrated biomechanical or performance score "
        "is generated."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()