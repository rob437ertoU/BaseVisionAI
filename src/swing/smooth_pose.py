from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "batter_pose_v1.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "batter_pose_smoothed_v1.csv"
)


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
    "right_ankle"
]


# ============================================================
# PARAMETERS
# ============================================================

# Keypoints below this confidence are considered unreliable.
KEYPOINT_CONF_THRESHOLD = 0.30

# Maximum number of consecutive missing frames that we allow
# interpolation to fill.
MAX_INTERPOLATION_GAP = 3

# Exponential Moving Average:
#
# higher alpha -> follows movement faster
# lower alpha  -> smoother
#
# Baseball swings are fast, so we don't want excessive
# smoothing.
SMOOTHING_ALPHA = 0.60


# ============================================================
# FUNCTIONS
# ============================================================

def confidence_filter(df, keypoint):
    """
    Remove x/y observations when keypoint confidence is below
    the threshold.

    Original values remain available in raw_* columns.
    """

    x_col = f"{keypoint}_x"
    y_col = f"{keypoint}_y"
    conf_col = f"{keypoint}_conf"

    valid = (
        df[conf_col].notna()
        &
        (df[conf_col] >= KEYPOINT_CONF_THRESHOLD)
    )

    filtered_x = df[x_col].where(valid, np.nan)
    filtered_y = df[y_col].where(valid, np.nan)

    return filtered_x, filtered_y


def interpolate_short_gaps(series):
    """
    Fill only short gaps surrounded by valid observations.

    limit_area='inside' prevents extrapolation before the
    first or after the final real observation.
    """

    return series.interpolate(
        method="linear",
        limit=MAX_INTERPOLATION_GAP,
        limit_direction="both",
        limit_area="inside"
    )


def exponential_smoothing(series):
    """
    Apply exponential moving average while preserving NaNs.

    The smoothing is performed separately for each continuous
    valid segment. This avoids carrying an old position across
    a long missing-data gap.
    """

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float
    )

    valid_indices = series.index[
        series.notna()
    ].tolist()

    if not valid_indices:
        return result

    previous_index = None
    previous_smoothed = None

    for index in valid_indices:

        current_value = float(
            series.loc[index]
        )

        # Start a new smoothing segment if there is a gap.
        if (
            previous_index is None
            or
            index != previous_index + 1
        ):
            smoothed_value = current_value

        else:
            smoothed_value = (
                SMOOTHING_ALPHA
                * current_value
                +
                (1.0 - SMOOTHING_ALPHA)
                * previous_smoothed
            )

        result.loc[index] = smoothed_value

        previous_smoothed = smoothed_value
        previous_index = index

    return result


def process_keypoint(df, keypoint):
    """
    Complete processing pipeline for one keypoint:

    raw
     ↓
    confidence filtering
     ↓
    short-gap interpolation
     ↓
    temporal smoothing
    """

    x_col = f"{keypoint}_x"
    y_col = f"{keypoint}_y"
    conf_col = f"{keypoint}_conf"

    # ========================================================
    # PRESERVE RAW VALUES
    # ========================================================

    df[f"raw_{x_col}"] = df[x_col]
    df[f"raw_{y_col}"] = df[y_col]
    df[f"raw_{conf_col}"] = df[conf_col]

    # ========================================================
    # CONFIDENCE FILTER
    # ========================================================

    filtered_x, filtered_y = confidence_filter(
        df,
        keypoint
    )

    df[f"{keypoint}_filtered_x"] = filtered_x
    df[f"{keypoint}_filtered_y"] = filtered_y

    # ========================================================
    # INTERPOLATION
    # ========================================================

    interpolated_x = interpolate_short_gaps(
        filtered_x
    )

    interpolated_y = interpolate_short_gaps(
        filtered_y
    )

    df[f"{keypoint}_interpolated_x"] = (
        interpolated_x
    )

    df[f"{keypoint}_interpolated_y"] = (
        interpolated_y
    )

    # Mark which frames were reconstructed rather than
    # directly accepted from YOLO Pose.
    interpolated_flag = (
        filtered_x.isna()
        &
        interpolated_x.notna()
    )

    df[f"{keypoint}_interpolated"] = (
        interpolated_flag
    )

    # ========================================================
    # SMOOTHING
    # ========================================================

    smoothed_x = exponential_smoothing(
        interpolated_x
    )

    smoothed_y = exponential_smoothing(
        interpolated_y
    )

    df[f"{keypoint}_smooth_x"] = (
        smoothed_x
    )

    df[f"{keypoint}_smooth_y"] = (
        smoothed_y
    )

    # ========================================================
    # VALID FLAG
    # ========================================================

    df[f"{keypoint}_valid"] = (
        smoothed_x.notna()
        &
        smoothed_y.notna()
    )

    return df


# ============================================================
# QUALITY STATISTICS
# ============================================================

def print_keypoint_statistics(df):
    """
    Print quality information for each body keypoint.
    """

    print("\n" + "=" * 80)
    print("KEYPOINT QUALITY")
    print("=" * 80)

    total_frames = len(df)

    print(
        f"\nTotal frames: {total_frames}"
    )

    print(
        f"Confidence threshold: "
        f"{KEYPOINT_CONF_THRESHOLD}"
    )

    print(
        f"Max interpolation gap: "
        f"{MAX_INTERPOLATION_GAP}"
    )

    print(
        f"Smoothing alpha: "
        f"{SMOOTHING_ALPHA}"
    )

    print("\n" + "-" * 80)

    for keypoint in KEYPOINT_NAMES:

        conf_col = (
            f"raw_{keypoint}_conf"
        )

        valid_col = (
            f"{keypoint}_valid"
        )

        interpolation_col = (
            f"{keypoint}_interpolated"
        )

        raw_valid = (
            df[conf_col].notna()
            &
            (
                df[conf_col]
                >=
                KEYPOINT_CONF_THRESHOLD
            )
        ).sum()

        final_valid = (
            df[valid_col].sum()
        )

        interpolated = (
            df[
                interpolation_col
            ].sum()
        )

        raw_percentage = (
            raw_valid
            /
            total_frames
            *
            100
        )

        final_percentage = (
            final_valid
            /
            total_frames
            *
            100
        )

        print(
            f"{keypoint:<18} | "
            f"raw valid: "
            f"{raw_valid:3d} "
            f"({raw_percentage:5.1f}%) | "
            f"interpolated: "
            f"{interpolated:3d} | "
            f"final: "
            f"{final_valid:3d} "
            f"({final_percentage:5.1f}%)"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("BaseVisionAI - Batter Pose Temporal Filtering V1")
    print("=" * 80)

    print(
        f"\nInput:\n{INPUT_CSV}"
    )

    # ========================================================
    # CHECK FILE
    # ========================================================

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Pose CSV not found:\n"
            f"{INPUT_CSV}"
        )

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = pd.read_csv(
        INPUT_CSV
    )

    print(
        f"\nRows loaded: {len(df)}"
    )

    # ========================================================
    # VALIDATE COLUMNS
    # ========================================================

    required_columns = [
        "frame"
    ]

    for keypoint in KEYPOINT_NAMES:

        required_columns.extend(
            [
                f"{keypoint}_x",
                f"{keypoint}_y",
                f"{keypoint}_conf"
            ]
        )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns in pose CSV:\n"
            +
            "\n".join(
                missing_columns
            )
        )

    # ========================================================
    # SORT
    # ========================================================

    df = df.sort_values(
        "frame"
    ).reset_index(
        drop=True
    )

    # ========================================================
    # PROCESS EACH KEYPOINT
    # ========================================================

    for keypoint in KEYPOINT_NAMES:

        print(
            f"Processing: "
            f"{keypoint}"
        )

        df = process_keypoint(
            df,
            keypoint
        )

    # ========================================================
    # GLOBAL POSE QUALITY
    # ========================================================

    valid_columns = [
        f"{keypoint}_valid"
        for keypoint in KEYPOINT_NAMES
    ]

    df["valid_keypoints"] = (
        df[
            valid_columns
        ]
        .sum(
            axis=1
        )
    )

    df["pose_quality"] = (
        df["valid_keypoints"]
        /
        len(KEYPOINT_NAMES)
    )

    # Important body parts for swing analysis
    swing_keypoints = [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee"
    ]

    swing_valid_columns = [
        f"{keypoint}_valid"
        for keypoint
        in swing_keypoints
    ]

    df["valid_swing_keypoints"] = (
        df[
            swing_valid_columns
        ]
        .sum(
            axis=1
        )
    )

    df["swing_pose_quality"] = (
        df["valid_swing_keypoints"]
        /
        len(swing_keypoints)
    )

    # ========================================================
    # QUALITY LABEL
    # ========================================================

    def quality_label(value):

        if value >= 0.85:
            return "HIGH"

        if value >= 0.65:
            return "MEDIUM"

        if value >= 0.40:
            return "LOW"

        return "UNRELIABLE"

    df["swing_pose_quality_label"] = (
        df[
            "swing_pose_quality"
        ].apply(
            quality_label
        )
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    print_keypoint_statistics(
        df
    )

    print("\n" + "=" * 80)
    print("SWING POSE QUALITY")
    print("=" * 80)

    quality_counts = (
        df[
            "swing_pose_quality_label"
        ]
        .value_counts()
    )

    print(
        quality_counts.to_string()
    )

    mean_quality = float(
        df[
            "swing_pose_quality"
        ].mean()
    )

    print(
        f"\nAverage swing pose quality: "
        f"{mean_quality:.3f}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("\n" + "=" * 80)
    print("POSE FILTERING COMPLETED")
    print("=" * 80)

    print(
        f"\nOutput:\n{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()