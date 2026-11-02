from pathlib import Path
import argparse

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "results"
    / "batter_pose_v2"
    / "baseball_train10_batter_pose_v2.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "batter_pose_v2"
)


# ============================================================
# CONFIG
# ============================================================

# Keypoints below this confidence are considered unreliable.
KEYPOINT_CONF_THRESHOLD = 0.30

# Only interpolate short missing gaps.
MAX_INTERPOLATION_GAP = 3

# EMA:
# Higher alpha = follows movement more closely
# Lower alpha = smoother but slower
EMA_ALPHA = 0.60


# ============================================================
# COCO KEYPOINTS
# ============================================================

KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Pose Smoothing V2"
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Input batter pose CSV"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory"
    )

    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=KEYPOINT_CONF_THRESHOLD,
        help="Minimum keypoint confidence"
    )

    parser.add_argument(
        "--max-gap",
        type=int,
        default=MAX_INTERPOLATION_GAP,
        help="Maximum interpolation gap in frames"
    )

    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=EMA_ALPHA,
        help="EMA smoothing factor"
    )

    return parser.parse_args()


# ============================================================
# EMA
# ============================================================

def ema_series(series, alpha):
    """
    Applies EMA only over valid values.

    Missing values remain missing until a valid sample
    is available again.
    """

    values = series.to_numpy(
        dtype=float
    )

    output = np.full(
        len(values),
        np.nan,
        dtype=float
    )

    previous = np.nan

    for i, value in enumerate(values):

        if np.isnan(value):
            continue

        if np.isnan(previous):
            smoothed = value

        else:
            smoothed = (
                alpha * value
                + (1.0 - alpha) * previous
            )

        output[i] = smoothed
        previous = smoothed

    return pd.Series(
        output,
        index=series.index
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    input_path = Path(
        args.input
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    if not input_path.exists():

        raise FileNotFoundError(
            f"Input CSV not found:\n"
            f"{input_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        / f"{input_path.stem}_smoothed.csv"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = pd.read_csv(
        input_path
    )

    print()
    print("=" * 72)
    print("BaseVisionAI - Pose Smoothing V2")
    print("=" * 72)

    print(
        f"Input              : "
        f"{input_path}"
    )

    print(
        f"Frames             : "
        f"{len(df)}"
    )

    print(
        f"Confidence threshold: "
        f"{args.conf_threshold:.2f}"
    )

    print(
        f"Max interp gap     : "
        f"{args.max_gap} frames"
    )

    print(
        f"EMA alpha          : "
        f"{args.ema_alpha:.2f}"
    )

    print("=" * 72)
    print()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_points = 0
    low_conf_filtered = 0
    interpolated_points = 0

    valid_after_smoothing = []

    # ========================================================
    # KEYPOINT LOOP
    # ========================================================

    for kp in KEYPOINT_NAMES:

        x_col = f"{kp}_x"
        y_col = f"{kp}_y"
        conf_col = f"{kp}_conf"

        # ----------------------------------------------------
        # Validate required columns
        # ----------------------------------------------------

        if (
            x_col not in df.columns
            or y_col not in df.columns
            or conf_col not in df.columns
        ):

            print(
                f"WARNING: missing columns for {kp}"
            )

            continue

        # ----------------------------------------------------
        # Preserve raw data
        # ----------------------------------------------------

        df[f"{kp}_x_raw"] = df[x_col]
        df[f"{kp}_y_raw"] = df[y_col]
        df[f"{kp}_conf_raw"] = df[conf_col]

        # Convert to numeric safely
        x = pd.to_numeric(
            df[x_col],
            errors="coerce"
        )

        y = pd.to_numeric(
            df[y_col],
            errors="coerce"
        )

        conf = pd.to_numeric(
            df[conf_col],
            errors="coerce"
        )

        total_points += len(df)

        # ----------------------------------------------------
        # Confidence filtering
        # ----------------------------------------------------

        low_conf_mask = (
            conf.isna()
            | (conf < args.conf_threshold)
        )

        low_conf_filtered += int(
            low_conf_mask.sum()
        )

        x_filtered = x.copy()
        y_filtered = y.copy()

        x_filtered[
            low_conf_mask
        ] = np.nan

        y_filtered[
            low_conf_mask
        ] = np.nan

        df[f"{kp}_x_filtered"] = (
            x_filtered
        )

        df[f"{kp}_y_filtered"] = (
            y_filtered
        )

        # ----------------------------------------------------
        # Short-gap interpolation
        # ----------------------------------------------------

        before_valid = (
            x_filtered.notna()
            & y_filtered.notna()
        )

        x_interp = (
            x_filtered
            .interpolate(
                method="linear",
                limit=args.max_gap,
                limit_area="inside"
            )
        )

        y_interp = (
            y_filtered
            .interpolate(
                method="linear",
                limit=args.max_gap,
                limit_area="inside"
            )
        )

        after_valid = (
            x_interp.notna()
            & y_interp.notna()
        )

        newly_interpolated = (
            after_valid
            & ~before_valid
        )

        interpolated_points += int(
            newly_interpolated.sum()
        )

        df[f"{kp}_x_interp"] = (
            x_interp
        )

        df[f"{kp}_y_interp"] = (
            y_interp
        )

        df[f"{kp}_interpolated"] = (
            newly_interpolated.astype(int)
        )

        # ----------------------------------------------------
        # EMA
        # ----------------------------------------------------

        x_smooth = ema_series(
            x_interp,
            args.ema_alpha
        )

        y_smooth = ema_series(
            y_interp,
            args.ema_alpha
        )

        df[f"{kp}_x_smooth"] = (
            x_smooth
        )

        df[f"{kp}_y_smooth"] = (
            y_smooth
        )

        kp_valid = (
            x_smooth.notna()
            & y_smooth.notna()
        )

        df[f"{kp}_valid"] = (
            kp_valid.astype(int)
        )

        valid_after_smoothing.append(
            kp_valid
        )

    # ========================================================
    # POSE QUALITY
    # ========================================================

    if valid_after_smoothing:

        valid_matrix = pd.concat(
            valid_after_smoothing,
            axis=1
        )

        valid_count = (
            valid_matrix.sum(axis=1)
        )

        pose_quality = (
            valid_count
            / len(valid_after_smoothing)
        )

        df["valid_keypoints"] = (
            valid_count
        )

        df["pose_quality"] = (
            pose_quality
        )

        def quality_label(value):

            if value >= 0.85:
                return "HIGH"

            if value >= 0.65:
                return "MEDIUM"

            return "LOW"

        df["pose_quality_label"] = (
            df["pose_quality"]
            .apply(
                quality_label
            )
        )

    else:

        df["valid_keypoints"] = 0
        df["pose_quality"] = 0.0
        df["pose_quality_label"] = "LOW"

    # ========================================================
    # SAVE
    # ========================================================

    df.to_csv(
        output_path,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    high_count = int(
        (
            df["pose_quality_label"]
            == "HIGH"
        ).sum()
    )

    medium_count = int(
        (
            df["pose_quality_label"]
            == "MEDIUM"
        ).sum()
    )

    low_count = int(
        (
            df["pose_quality_label"]
            == "LOW"
        ).sum()
    )

    avg_quality = float(
        df["pose_quality"].mean()
    )

    print()
    print("=" * 72)
    print("POSE SMOOTHING V2 SUMMARY")
    print("=" * 72)

    print(
        f"Frames total           : "
        f"{len(df)}"
    )

    print(
        f"Raw keypoint samples   : "
        f"{total_points}"
    )

    print(
        f"Low-confidence filtered: "
        f"{low_conf_filtered}"
    )

    print(
        f"Interpolated samples   : "
        f"{interpolated_points}"
    )

    print()

    print(
        f"HIGH quality frames    : "
        f"{high_count}"
    )

    print(
        f"MEDIUM quality frames  : "
        f"{medium_count}"
    )

    print(
        f"LOW quality frames     : "
        f"{low_count}"
    )

    print(
        f"Average pose quality   : "
        f"{avg_quality:.3f}"
    )

    print()

    print(
        f"Output CSV             : "
        f"{output_path}"
    )

    print("=" * 72)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()