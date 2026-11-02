from pathlib import Path
import argparse

import numpy as np
import pandas as pd


# ============================================================
# BASEVISIONAI - SWING COMPARISON V2
# ============================================================
#
# Compares:
#   1. ALL swing profiles
#   2. VALID_SWING only
#
# Swing Validity V1 is treated as a structural evidence filter,
# not as ground truth that a baseball swing actually occurred.
#
# ============================================================


METRICS = {
    "left_wrist_peak_px_frame": "Left wrist peak",
    "right_wrist_peak_px_frame": "Right wrist peak",

    "left_wrist_peak_torsoscale_frame":
        "Left wrist peak normalized",

    "right_wrist_peak_torsoscale_frame":
        "Right wrist peak normalized",

    "hip_center_peak_px_frame": "Hip center peak",
    "shoulder_center_peak_px_frame": "Shoulder center peak",

    "shoulder_rotation_projected_deg":
        "Shoulder projected rotation",

    "hip_rotation_projected_deg":
        "Hip projected rotation",

    "hip_shoulder_separation_projected_deg":
        "Hip-shoulder separation",

    "head_prepeak_torso":
        "Head pre-peak movement",

    "left_ankle_prepeak_torso":
        "Left ankle pre-peak movement",

    "right_ankle_prepeak_torso":
        "Right ankle pre-peak movement",

    "load_duration_sec":
        "Load duration",

    "swing_duration_sec":
        "Swing duration",

    "load_to_peak_sec":
        "Load to peak",

    "swing_to_peak_sec":
        "Swing to peak",

    "mean_pose_quality":
        "Mean pose quality",
}


def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Swing Comparison V2"
    )

    parser.add_argument(
        "--database",
        required=True
    )

    parser.add_argument(
        "--validity",
        required=True
    )

    parser.add_argument(
        "--player-id",
        required=True
    )

    parser.add_argument(
        "--session-id",
        default=None
    )

    parser.add_argument(
        "--output-dir",
        required=True
    )

    return parser.parse_args()


def numeric_values(df, column):

    if column not in df.columns:
        return np.array([], dtype=float)

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).to_numpy(dtype=float)

    return values[
        np.isfinite(values)
    ]


def calculate_stats(df, column):

    values = numeric_values(
        df,
        column
    )

    n = len(values)

    if n == 0:

        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "cv_percent": np.nan,
        }

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

    if n >= 2:

        std = float(
            np.std(
                values,
                ddof=1
            )
        )

    else:

        std = np.nan

    if (
        np.isfinite(std)
        and abs(mean) > 1e-9
    ):

        cv_percent = (
            std
            /
            abs(mean)
            *
            100.0
        )

    else:

        cv_percent = np.nan

    return {
        "n": n,
        "mean": mean,
        "median": median,
        "std": std,
        "min": minimum,
        "max": maximum,
        "cv_percent": cv_percent,
    }


def fmt(value, digits=3):

    if not np.isfinite(value):
        return "N/A"

    return f"{value:.{digits}f}"


def main():

    args = parse_args()

    database_path = Path(
        args.database
    )

    validity_path = Path(
        args.validity
    )

    output_dir = Path(
        args.output_dir
    )

    if not database_path.exists():

        raise FileNotFoundError(
            f"Database not found:\n{database_path}"
        )

    if not validity_path.exists():

        raise FileNotFoundError(
            f"Validity file not found:\n{validity_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    database_df = pd.read_csv(
        database_path
    )

    validity_df = pd.read_csv(
        validity_path
    )

    # ========================================================
    # FILTER PLAYER / SESSION
    # ========================================================

    database_df = database_df[
        database_df["player_id"].astype(str)
        ==
        str(args.player_id)
    ].copy()

    validity_df = validity_df[
        validity_df["player_id"].astype(str)
        ==
        str(args.player_id)
    ].copy()

    if args.session_id is not None:

        database_df = database_df[
            database_df["session_id"].astype(str)
            ==
            str(args.session_id)
        ].copy()

        validity_df = validity_df[
            validity_df["session_id"].astype(str)
            ==
            str(args.session_id)
        ].copy()

    if len(database_df) == 0:

        raise ValueError(
            "No swing profiles found for selection."
        )

    # ========================================================
    # MERGE VALIDITY
    # ========================================================

    merge_keys = [
        "video_id",
        "player_id",
        "session_id",
    ]

    validity_columns = merge_keys + [
        "validity_status",
        "evidence_quality",
        "reason",
    ]

    merged_df = database_df.merge(
        validity_df[
            validity_columns
        ],
        on=merge_keys,
        how="left",
        validate="one_to_one"
    )

    merged_df[
        "validity_status"
    ] = merged_df[
        "validity_status"
    ].fillna(
        "NOT_EVALUATED"
    )

    # ========================================================
    # GROUPS
    # ========================================================

    all_df = merged_df.copy()

    valid_df = merged_df[
        merged_df[
            "validity_status"
        ]
        ==
        "VALID_SWING"
    ].copy()

    # ========================================================
    # COMPARISON
    # ========================================================

    rows = []

    for metric, label in METRICS.items():

        all_stats = calculate_stats(
            all_df,
            metric
        )

        valid_stats = calculate_stats(
            valid_df,
            metric
        )

        mean_change = np.nan
        cv_change = np.nan

        if (
            np.isfinite(all_stats["mean"])
            and
            np.isfinite(valid_stats["mean"])
        ):

            mean_change = (
                valid_stats["mean"]
                -
                all_stats["mean"]
            )

        if (
            np.isfinite(all_stats["cv_percent"])
            and
            np.isfinite(valid_stats["cv_percent"])
        ):

            cv_change = (
                valid_stats["cv_percent"]
                -
                all_stats["cv_percent"]
            )

        rows.append(
            {
                "metric":
                    metric,

                "label":
                    label,

                "all_n":
                    all_stats["n"],

                "all_mean":
                    all_stats["mean"],

                "all_median":
                    all_stats["median"],

                "all_std":
                    all_stats["std"],

                "all_min":
                    all_stats["min"],

                "all_max":
                    all_stats["max"],

                "all_cv_percent":
                    all_stats["cv_percent"],

                "valid_n":
                    valid_stats["n"],

                "valid_mean":
                    valid_stats["mean"],

                "valid_median":
                    valid_stats["median"],

                "valid_std":
                    valid_stats["std"],

                "valid_min":
                    valid_stats["min"],

                "valid_max":
                    valid_stats["max"],

                "valid_cv_percent":
                    valid_stats["cv_percent"],

                "valid_minus_all_mean":
                    mean_change,

                "valid_minus_all_cv_percent":
                    cv_change,
            }
        )

    comparison_df = pd.DataFrame(
        rows
    )

    # ========================================================
    # SAVE
    # ========================================================

    merged_output = (
        output_dir
        /
        "swing_comparison_v2_swings.csv"
    )

    comparison_output = (
        output_dir
        /
        "swing_comparison_v2_summary.csv"
    )

    valid_output = (
        output_dir
        /
        "swing_comparison_v2_valid_swings.csv"
    )

    merged_df.to_csv(
        merged_output,
        index=False
    )

    comparison_df.to_csv(
        comparison_output,
        index=False
    )

    valid_df.to_csv(
        valid_output,
        index=False
    )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 90)
    print(
        "SWING COMPARISON V2 COMPLETE"
    )
    print("=" * 90)

    print()
    print("SELECTION")
    print("-" * 90)

    print(
        f"Player ID                 : {args.player_id}"
    )

    print(
        f"Session                    : "
        f"{args.session_id if args.session_id else 'ALL'}"
    )

    print(
        f"All profiles               : {len(all_df)}"
    )

    print(
        f"VALID_SWING profiles       : {len(valid_df)}"
    )

    print(
        f"Excluded from valid subset : "
        f"{len(all_df) - len(valid_df)}"
    )

    print()
    print("VALIDITY")
    print("-" * 90)

    for _, row in merged_df.iterrows():

        print(
            f"{str(row['video_id']):<20} "
            f"{str(row['validity_status']):<24} "
            f"{str(row['evidence_quality']):<12}"
        )

    print()
    print("ALL vs VALID_SWING")
    print("-" * 90)

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

        match = comparison_df[
            comparison_df["metric"]
            ==
            metric
        ]

        if len(match) == 0:
            continue

        row = match.iloc[0]

        print()
        print(
            f"{row['label']}"
        )

        print(
            f"  ALL   : "
            f"Mean={fmt(row['all_mean'])}  "
            f"SD={fmt(row['all_std'])}  "
            f"CV={fmt(row['all_cv_percent'], 1)}%"
        )

        print(
            f"  VALID : "
            f"Mean={fmt(row['valid_mean'])}  "
            f"SD={fmt(row['valid_std'])}  "
            f"CV={fmt(row['valid_cv_percent'], 1)}%"
        )

    print()
    print("OUTPUT FILES")
    print("-" * 90)

    print(
        f"All swings     : {merged_output.resolve()}"
    )

    print(
        f"Valid swings   : {valid_output.resolve()}"
    )

    print(
        f"Comparison     : {comparison_output.resolve()}"
    )

    print()
    print("IMPORTANT")
    print("-" * 90)

    print(
        "VALID_SWING is a structural comparison filter, "
        "not proof that a true baseball swing occurred."
    )

    print(
        "Excluded profiles remain preserved in the ALL dataset."
    )

    print(
        "CV measures variability, not performance quality."
    )

    print("=" * 90)


if __name__ == "__main__":
    main()