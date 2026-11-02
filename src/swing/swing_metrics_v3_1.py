from pathlib import Path
import argparse

import numpy as np
import pandas as pd


# ============================================================
# BASEVISIONAI - SWING METRICS V3.1
# ============================================================
#
# Improvements over V3:
#
# 1. Body orientation is treated as an AXIS modulo 180 deg.
#    This avoids false ~300 deg excursions caused by left/right
#    keypoint direction changes.
#
# 2. Robust motion speed uses a centered 3-frame displacement.
#    This reduces single-frame pose jitter.
#
# 3. Head stability is measured relative to hip center.
#
# 4. Leg displacement is measured relative to hip center.
#
# IMPORTANT:
# - All distances are still 2D image-space measurements.
# - px/frame is NOT m/s or mph.
# - projected angles are NOT true 3D biomechanics.
#
# ============================================================


KEYPOINTS = [
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
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Swing Metrics V3.1"
    )

    parser.add_argument(
        "--pose-csv",
        required=True,
        help="Pose Smoothing V2 CSV"
    )

    parser.add_argument(
        "--swing-csv",
        required=True,
        help="Swing Analysis V2.2 CSV"
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video FPS"
    )

    parser.add_argument(
        "--output-dir",
        default="results/swing_metrics_v3_1",
        help="Output directory"
    )

    return parser.parse_args()


# ============================================================
# BASIC HELPERS
# ============================================================

def finite(value):

    return (
        value is not None
        and np.isfinite(value)
    )


def nanmax_safe(values):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        np.max(values)
    )


def nanmean_safe(values):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        np.mean(values)
    )


def nanmedian_safe(values):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        np.median(values)
    )


def frame_of_max(
    frames,
    values
):

    frames = np.asarray(
        frames
    )

    values = np.asarray(
        values,
        dtype=float
    )

    valid = np.isfinite(
        values
    )

    if not np.any(valid):
        return None

    values_valid = values[
        valid
    ]

    frames_valid = frames[
        valid
    ]

    index = int(
        np.argmax(
            values_valid
        )
    )

    return int(
        frames_valid[index]
    )


# ============================================================
# POSE COLUMN HANDLING
# ============================================================

def coordinate_candidates(
    keypoint,
    axis
):

    return [
        f"{keypoint}_{axis}_smooth",
        f"{keypoint}_{axis}_smoothed",
        f"{keypoint}_{axis}_ema",
        f"{keypoint}_{axis}_interp",
        f"{keypoint}_{axis}_interpolated",
        f"{keypoint}_{axis}_filtered",
        f"{keypoint}_{axis}",
    ]


def find_coordinate_column(
    df,
    keypoint,
    axis
):

    for column in coordinate_candidates(
        keypoint,
        axis
    ):

        if column in df.columns:
            return column

    return None


def get_coordinate(
    df,
    keypoint,
    axis
):

    column = find_coordinate_column(
        df,
        keypoint,
        axis
    )

    if column is None:

        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float
        )

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# GEOMETRY
# ============================================================

def distance_series(
    x1,
    y1,
    x2,
    y2
):

    return np.sqrt(
        (x2 - x1) ** 2
        +
        (y2 - y1) ** 2
    )


def center_series(
    x1,
    y1,
    x2,
    y2
):

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    )


# ============================================================
# ROBUST 3-FRAME MOTION
# ============================================================

def robust_speed_3frame(
    x,
    y
):

    # --------------------------------------------------------
    # Centered displacement:
    #
    # frame t speed =
    # distance(position[t+1], position[t-1]) / 2
    #
    # This reduces one-frame jitter.
    # --------------------------------------------------------

    dx = (
        x.shift(-1)
        - x.shift(1)
    )

    dy = (
        y.shift(-1)
        - y.shift(1)
    )

    speed = np.sqrt(
        dx ** 2
        +
        dy ** 2
    ) / 2.0

    return speed


# ============================================================
# AXIS ORIENTATION
# ============================================================

def body_axis_angle(
    x1,
    y1,
    x2,
    y2
):

    dx = (
        x2 - x1
    ).to_numpy(
        dtype=float
    )

    dy = (
        y2 - y1
    ).to_numpy(
        dtype=float
    )

    raw = np.degrees(
        np.arctan2(
            dy,
            dx
        )
    )

    # --------------------------------------------------------
    # Line orientation, not directed vector:
    #
    # 10 deg == 190 deg
    #
    # Map to [0, 180)
    # --------------------------------------------------------

    axis_angle = np.mod(
        raw,
        180.0
    )

    invalid = (
        ~np.isfinite(dx)
        | ~np.isfinite(dy)
    )

    axis_angle[
        invalid
    ] = np.nan

    return pd.Series(
        axis_angle,
        index=x1.index
    )


def unwrap_axis_angle(
    angle_series_input
):

    values = angle_series_input.to_numpy(
        dtype=float
    )

    output = np.full(
        len(values),
        np.nan,
        dtype=float
    )

    valid_indices = np.where(
        np.isfinite(values)
    )[0]

    if len(valid_indices) == 0:

        return pd.Series(
            output,
            index=angle_series_input.index
        )

    # --------------------------------------------------------
    # For an unoriented axis:
    #
    # angle is periodic every 180 deg.
    #
    # Doubling transforms:
    # 0..180  -> 0..360
    #
    # then normal unwrap works.
    # --------------------------------------------------------

    doubled = np.radians(
        values[
            valid_indices
        ] * 2.0
    )

    unwrapped = np.unwrap(
        doubled
    )

    output[
        valid_indices
    ] = np.degrees(
        unwrapped
    ) / 2.0

    return pd.Series(
        output,
        index=angle_series_input.index
    )


def axis_difference_deg(
    a,
    b
):

    # --------------------------------------------------------
    # Smallest difference between two body axes.
    #
    # Result in [-90, +90].
    # --------------------------------------------------------

    difference = a - b

    difference = (
        difference + 90.0
    ) % 180.0 - 90.0

    return difference


# ============================================================
# BODY-RELATIVE POSITION
# ============================================================

def relative_coordinates(
    x,
    y,
    center_x,
    center_y
):

    return (
        x - center_x,
        y - center_y
    )


def displacement_from_initial(
    x,
    y
):

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    if not np.any(
        valid
    ):

        return pd.Series(
            np.nan,
            index=x.index
        )

    positions = np.where(
        valid.to_numpy()
    )[0]

    first_index = int(
        positions[0]
    )

    x0 = float(
        x.iloc[
            first_index
        ]
    )

    y0 = float(
        y.iloc[
            first_index
        ]
    )

    return np.sqrt(
        (x - x0) ** 2
        +
        (y - y0) ** 2
    )


# ============================================================
# PHASE HANDLING
# ============================================================

def detect_phase_column(
    swing_df
):

    candidates = [
        "phase",
        "swing_phase",
        "phase_name",
        "predicted_phase",
    ]

    for column in candidates:

        if column in swing_df.columns:
            return column

    return None


def normalize_phase(
    value
):

    if pd.isna(value):
        return "UNKNOWN"

    value = str(
        value
    ).strip().upper()

    value = value.replace(
        " ",
        "_"
    )

    value = value.replace(
        "-",
        "_"
    )

    aliases = {
        "PEAK": "PEAK_SWING",
        "PEAKSWING": "PEAK_SWING",
        "FOLLOW": "FOLLOW_THROUGH",
        "FOLLOWTHROUGH": "FOLLOW_THROUGH",
    }

    return aliases.get(
        value,
        value
    )


def phase_stats(
    df,
    fps
):

    results = {}

    phases = [
        "SETUP",
        "LOAD",
        "SWING",
        "PEAK_SWING",
        "FOLLOW_THROUGH",
    ]

    for phase in phases:

        subset = df[
            df["phase_normalized"]
            == phase
        ]

        if len(subset) == 0:
            continue

        start_frame = int(
            subset[
                "frame"
            ].min()
        )

        end_frame = int(
            subset[
                "frame"
            ].max()
        )

        frame_count = (
            end_frame
            - start_frame
            + 1
        )

        duration_sec = (
            frame_count
            / fps
        )

        key = phase.lower()

        results[
            f"{key}_start_frame"
        ] = start_frame

        results[
            f"{key}_end_frame"
        ] = end_frame

        results[
            f"{key}_frames"
        ] = frame_count

        results[
            f"{key}_duration_sec"
        ] = duration_sec

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    pose_path = Path(
        args.pose_csv
    )

    swing_path = Path(
        args.swing_csv
    )

    output_dir = Path(
        args.output_dir
    )

    fps = float(
        args.fps
    )

    # --------------------------------------------------------
    # CHECK INPUTS
    # --------------------------------------------------------

    if not pose_path.exists():

        raise FileNotFoundError(
            f"Pose CSV not found:\n"
            f"{pose_path}"
        )

    if not swing_path.exists():

        raise FileNotFoundError(
            f"Swing CSV not found:\n"
            f"{swing_path}"
        )

    if fps <= 0:

        raise ValueError(
            "FPS must be greater than zero."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    pose_df = pd.read_csv(
        pose_path
    )

    swing_df = pd.read_csv(
        swing_path
    )

    if "frame" not in pose_df.columns:

        raise ValueError(
            "Pose CSV requires a 'frame' column."
        )

    if "frame" not in swing_df.columns:

        raise ValueError(
            "Swing CSV requires a 'frame' column."
        )

    for df in [
        pose_df,
        swing_df,
    ]:

        df["frame"] = pd.to_numeric(
            df["frame"],
            errors="coerce"
        )

        df.dropna(
            subset=["frame"],
            inplace=True
        )

        df["frame"] = (
            df["frame"]
            .astype(int)
        )

        df.sort_values(
            "frame",
            inplace=True
        )

        df.drop_duplicates(
            subset=["frame"],
            keep="first",
            inplace=True
        )

        df.reset_index(
            drop=True,
            inplace=True
        )

    # ========================================================
    # BASE METRICS TABLE
    # ========================================================

    metrics = pd.DataFrame({
        "frame": pose_df[
            "frame"
        ]
    })

    metrics[
        "time_sec"
    ] = (
        metrics["frame"] - 1
    ) / fps

    # ========================================================
    # PHASE
    # ========================================================

    phase_column = detect_phase_column(
        swing_df
    )

    if phase_column is not None:

        phase_table = swing_df[
            [
                "frame",
                phase_column
            ]
        ].copy()

        phase_table.rename(
            columns={
                phase_column:
                "phase_raw"
            },
            inplace=True
        )

        metrics = metrics.merge(
            phase_table,
            on="frame",
            how="left"
        )

        metrics[
            "phase_normalized"
        ] = metrics[
            "phase_raw"
        ].apply(
            normalize_phase
        )

    else:

        metrics[
            "phase_raw"
        ] = "UNKNOWN"

        metrics[
            "phase_normalized"
        ] = "UNKNOWN"

    # ========================================================
    # SWING MOTION SCORE
    # ========================================================

    score_column = None

    for candidate in [
        "swing_score",
        "combined_score",
        "motion_score",
        "smoothed_swing_score",
    ]:

        if candidate in swing_df.columns:

            score_column = candidate
            break

    if score_column is not None:

        score_table = swing_df[
            [
                "frame",
                score_column
            ]
        ].copy()

        score_table.rename(
            columns={
                score_column:
                "swing_motion_score"
            },
            inplace=True
        )

        metrics = metrics.merge(
            score_table,
            on="frame",
            how="left"
        )

    else:

        metrics[
            "swing_motion_score"
        ] = np.nan

    # ========================================================
    # LOAD COORDINATES
    # ========================================================

    coordinates = {}

    print()
    print("=" * 78)
    print(
        "BaseVisionAI - Swing Metrics V3.1"
    )
    print("=" * 78)

    print(
        f"Pose CSV      : {pose_path}"
    )

    print(
        f"Swing CSV     : {swing_path}"
    )

    print(
        f"FPS           : {fps:.2f}"
    )

    print(
        f"Pose frames   : {len(pose_df)}"
    )

    print(
        f"Swing frames  : {len(swing_df)}"
    )

    print(
        f"Phase column  : {phase_column}"
    )

    print("-" * 78)
    print(
        "Pose coordinate mapping:"
    )

    for keypoint in KEYPOINTS:

        x_column = find_coordinate_column(
            pose_df,
            keypoint,
            "x"
        )

        y_column = find_coordinate_column(
            pose_df,
            keypoint,
            "y"
        )

        coordinates[
            f"{keypoint}_x"
        ] = get_coordinate(
            pose_df,
            keypoint,
            "x"
        )

        coordinates[
            f"{keypoint}_y"
        ] = get_coordinate(
            pose_df,
            keypoint,
            "y"
        )

        if (
            x_column is not None
            and y_column is not None
        ):

            print(
                f"  {keypoint:<16}"
                f"{x_column} / {y_column}"
            )

    # ========================================================
    # SHORT VARIABLES
    # ========================================================

    nosex = coordinates[
        "nose_x"
    ]

    nosey = coordinates[
        "nose_y"
    ]

    lsx = coordinates[
        "left_shoulder_x"
    ]

    lsy = coordinates[
        "left_shoulder_y"
    ]

    rsx = coordinates[
        "right_shoulder_x"
    ]

    rsy = coordinates[
        "right_shoulder_y"
    ]

    lwx = coordinates[
        "left_wrist_x"
    ]

    lwy = coordinates[
        "left_wrist_y"
    ]

    rwx = coordinates[
        "right_wrist_x"
    ]

    rwy = coordinates[
        "right_wrist_y"
    ]

    lhx = coordinates[
        "left_hip_x"
    ]

    lhy = coordinates[
        "left_hip_y"
    ]

    rhx = coordinates[
        "right_hip_x"
    ]

    rhy = coordinates[
        "right_hip_y"
    ]

    lkx = coordinates[
        "left_knee_x"
    ]

    lky = coordinates[
        "left_knee_y"
    ]

    rkx = coordinates[
        "right_knee_x"
    ]

    rky = coordinates[
        "right_knee_y"
    ]

    lax = coordinates[
        "left_ankle_x"
    ]

    lay = coordinates[
        "left_ankle_y"
    ]

    rax = coordinates[
        "right_ankle_x"
    ]

    ray = coordinates[
        "right_ankle_y"
    ]

    # ========================================================
    # BODY SCALE
    # ========================================================

    shoulder_width = distance_series(
        lsx,
        lsy,
        rsx,
        rsy
    )

    hip_width = distance_series(
        lhx,
        lhy,
        rhx,
        rhy
    )

    median_shoulder_width = nanmedian_safe(
        shoulder_width
    )

    median_hip_width = nanmedian_safe(
        hip_width
    )

    metrics[
        "shoulder_width_px"
    ] = shoulder_width.to_numpy()

    metrics[
        "hip_width_px"
    ] = hip_width.to_numpy()

    # ========================================================
    # BODY CENTERS
    # ========================================================

    shoulder_cx, shoulder_cy = center_series(
        lsx,
        lsy,
        rsx,
        rsy
    )

    hip_cx, hip_cy = center_series(
        lhx,
        lhy,
        rhx,
        rhy
    )

    metrics[
        "shoulder_center_x"
    ] = shoulder_cx.to_numpy()

    metrics[
        "shoulder_center_y"
    ] = shoulder_cy.to_numpy()

    metrics[
        "hip_center_x"
    ] = hip_cx.to_numpy()

    metrics[
        "hip_center_y"
    ] = hip_cy.to_numpy()

    # ========================================================
    # ROBUST MOTION SPEEDS
    # ========================================================

    left_wrist_speed = robust_speed_3frame(
        lwx,
        lwy
    )

    right_wrist_speed = robust_speed_3frame(
        rwx,
        rwy
    )

    hip_center_speed = robust_speed_3frame(
        hip_cx,
        hip_cy
    )

    shoulder_center_speed = robust_speed_3frame(
        shoulder_cx,
        shoulder_cy
    )

    head_speed = robust_speed_3frame(
        nosex,
        nosey
    )

    metrics[
        "left_wrist_speed_robust_px_frame"
    ] = left_wrist_speed.to_numpy()

    metrics[
        "right_wrist_speed_robust_px_frame"
    ] = right_wrist_speed.to_numpy()

    metrics[
        "hip_center_speed_robust_px_frame"
    ] = hip_center_speed.to_numpy()

    metrics[
        "shoulder_center_speed_robust_px_frame"
    ] = shoulder_center_speed.to_numpy()

    metrics[
        "head_speed_robust_px_frame"
    ] = head_speed.to_numpy()

    # ========================================================
    # NORMALIZED SPEEDS
    # ========================================================

    if (
        finite(
            median_shoulder_width
        )
        and median_shoulder_width > 1e-6
    ):

        metrics[
            "left_wrist_speed_bodywidth_frame"
        ] = (
            left_wrist_speed
            / median_shoulder_width
        ).to_numpy()

        metrics[
            "right_wrist_speed_bodywidth_frame"
        ] = (
            right_wrist_speed
            / median_shoulder_width
        ).to_numpy()

        metrics[
            "hip_center_speed_bodywidth_frame"
        ] = (
            hip_center_speed
            / median_shoulder_width
        ).to_numpy()

        metrics[
            "shoulder_center_speed_bodywidth_frame"
        ] = (
            shoulder_center_speed
            / median_shoulder_width
        ).to_numpy()

    else:

        metrics[
            "left_wrist_speed_bodywidth_frame"
        ] = np.nan

        metrics[
            "right_wrist_speed_bodywidth_frame"
        ] = np.nan

        metrics[
            "hip_center_speed_bodywidth_frame"
        ] = np.nan

        metrics[
            "shoulder_center_speed_bodywidth_frame"
        ] = np.nan

    # ========================================================
    # PROJECTED BODY AXIS ROTATION
    # ========================================================

    shoulder_axis_raw = body_axis_angle(
        lsx,
        lsy,
        rsx,
        rsy
    )

    hip_axis_raw = body_axis_angle(
        lhx,
        lhy,
        rhx,
        rhy
    )

    shoulder_axis = unwrap_axis_angle(
        shoulder_axis_raw
    )

    hip_axis = unwrap_axis_angle(
        hip_axis_raw
    )

    separation = axis_difference_deg(
        shoulder_axis,
        hip_axis
    )

    metrics[
        "shoulder_axis_projected_deg"
    ] = shoulder_axis.to_numpy()

    metrics[
        "hip_axis_projected_deg"
    ] = hip_axis.to_numpy()

    metrics[
        "hip_shoulder_separation_projected_deg"
    ] = separation.to_numpy()

    metrics[
        "abs_hip_shoulder_separation_deg"
    ] = np.abs(
        separation.to_numpy()
    )

    # --------------------------------------------------------
    # Robust angular speed
    # --------------------------------------------------------

    metrics[
        "shoulder_angular_speed_robust_deg_frame"
    ] = np.abs(
        (
            shoulder_axis.shift(-1)
            -
            shoulder_axis.shift(1)
        ) / 2.0
    ).to_numpy()

    metrics[
        "hip_angular_speed_robust_deg_frame"
    ] = np.abs(
        (
            hip_axis.shift(-1)
            -
            hip_axis.shift(1)
        ) / 2.0
    ).to_numpy()

    # ========================================================
    # BODY-RELATIVE HEAD STABILITY
    # ========================================================

    head_rel_x, head_rel_y = relative_coordinates(
        nosex,
        nosey,
        hip_cx,
        hip_cy
    )

    head_relative_displacement = displacement_from_initial(
        head_rel_x,
        head_rel_y
    )

    metrics[
        "head_relative_x_px"
    ] = head_rel_x.to_numpy()

    metrics[
        "head_relative_y_px"
    ] = head_rel_y.to_numpy()

    metrics[
        "head_relative_displacement_px"
    ] = head_relative_displacement.to_numpy()

    if (
        finite(
            median_shoulder_width
        )
        and median_shoulder_width > 1e-6
    ):

        metrics[
            "head_relative_displacement_bodywidth"
        ] = (
            head_relative_displacement
            / median_shoulder_width
        ).to_numpy()

    else:

        metrics[
            "head_relative_displacement_bodywidth"
        ] = np.nan

    # ========================================================
    # BODY-RELATIVE LEG MOVEMENT
    # ========================================================

    left_ankle_rel_x, left_ankle_rel_y = relative_coordinates(
        lax,
        lay,
        hip_cx,
        hip_cy
    )

    right_ankle_rel_x, right_ankle_rel_y = relative_coordinates(
        rax,
        ray,
        hip_cx,
        hip_cy
    )

    left_knee_rel_x, left_knee_rel_y = relative_coordinates(
        lkx,
        lky,
        hip_cx,
        hip_cy
    )

    right_knee_rel_x, right_knee_rel_y = relative_coordinates(
        rkx,
        rky,
        hip_cx,
        hip_cy
    )

    left_ankle_relative_displacement = displacement_from_initial(
        left_ankle_rel_x,
        left_ankle_rel_y
    )

    right_ankle_relative_displacement = displacement_from_initial(
        right_ankle_rel_x,
        right_ankle_rel_y
    )

    left_knee_relative_displacement = displacement_from_initial(
        left_knee_rel_x,
        left_knee_rel_y
    )

    right_knee_relative_displacement = displacement_from_initial(
        right_knee_rel_x,
        right_knee_rel_y
    )

    metrics[
        "left_ankle_relative_displacement_px"
    ] = left_ankle_relative_displacement.to_numpy()

    metrics[
        "right_ankle_relative_displacement_px"
    ] = right_ankle_relative_displacement.to_numpy()

    metrics[
        "left_knee_relative_displacement_px"
    ] = left_knee_relative_displacement.to_numpy()

    metrics[
        "right_knee_relative_displacement_px"
    ] = right_knee_relative_displacement.to_numpy()

    if (
        finite(
            median_shoulder_width
        )
        and median_shoulder_width > 1e-6
    ):

        metrics[
            "left_ankle_relative_displacement_bodywidth"
        ] = (
            left_ankle_relative_displacement
            / median_shoulder_width
        ).to_numpy()

        metrics[
            "right_ankle_relative_displacement_bodywidth"
        ] = (
            right_ankle_relative_displacement
            / median_shoulder_width
        ).to_numpy()

        metrics[
            "left_knee_relative_displacement_bodywidth"
        ] = (
            left_knee_relative_displacement
            / median_shoulder_width
        ).to_numpy()

        metrics[
            "right_knee_relative_displacement_bodywidth"
        ] = (
            right_knee_relative_displacement
            / median_shoulder_width
        ).to_numpy()

    else:

        metrics[
            "left_ankle_relative_displacement_bodywidth"
        ] = np.nan

        metrics[
            "right_ankle_relative_displacement_bodywidth"
        ] = np.nan

        metrics[
            "left_knee_relative_displacement_bodywidth"
        ] = np.nan

        metrics[
            "right_knee_relative_displacement_bodywidth"
        ] = np.nan

    # ========================================================
    # POSE QUALITY
    # ========================================================

    pose_quality_column = None

    for candidate in [
        "pose_quality",
        "quality",
        "mean_confidence",
        "avg_confidence",
    ]:

        if candidate in pose_df.columns:

            pose_quality_column = candidate
            break

    if pose_quality_column is not None:

        metrics[
            "pose_quality"
        ] = pd.to_numeric(
            pose_df[
                pose_quality_column
            ],
            errors="coerce"
        ).to_numpy()

    else:

        metrics[
            "pose_quality"
        ] = np.nan

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {}

    summary[
        "version"
    ] = "Swing Metrics V3.1"

    summary[
        "frames_analyzed"
    ] = int(
        len(metrics)
    )

    summary[
        "fps"
    ] = fps

    summary[
        "duration_sec"
    ] = (
        len(metrics)
        / fps
    )

    summary[
        "median_shoulder_width_px"
    ] = median_shoulder_width

    summary[
        "median_hip_width_px"
    ] = median_hip_width

    # --------------------------------------------------------
    # WRIST SPEED
    # --------------------------------------------------------

    summary[
        "left_wrist_peak_speed_robust_px_frame"
    ] = nanmax_safe(
        metrics[
            "left_wrist_speed_robust_px_frame"
        ]
    )

    summary[
        "left_wrist_peak_frame"
    ] = frame_of_max(
        metrics["frame"],
        metrics[
            "left_wrist_speed_robust_px_frame"
        ]
    )

    summary[
        "right_wrist_peak_speed_robust_px_frame"
    ] = nanmax_safe(
        metrics[
            "right_wrist_speed_robust_px_frame"
        ]
    )

    summary[
        "right_wrist_peak_frame"
    ] = frame_of_max(
        metrics["frame"],
        metrics[
            "right_wrist_speed_robust_px_frame"
        ]
    )

    summary[
        "left_wrist_peak_speed_bodywidth_frame"
    ] = nanmax_safe(
        metrics[
            "left_wrist_speed_bodywidth_frame"
        ]
    )

    summary[
        "right_wrist_peak_speed_bodywidth_frame"
    ] = nanmax_safe(
        metrics[
            "right_wrist_speed_bodywidth_frame"
        ]
    )

    # --------------------------------------------------------
    # BODY CENTER SPEED
    # --------------------------------------------------------

    summary[
        "hip_center_peak_speed_robust_px_frame"
    ] = nanmax_safe(
        metrics[
            "hip_center_speed_robust_px_frame"
        ]
    )

    summary[
        "hip_center_peak_frame"
    ] = frame_of_max(
        metrics["frame"],
        metrics[
            "hip_center_speed_robust_px_frame"
        ]
    )

    summary[
        "shoulder_center_peak_speed_robust_px_frame"
    ] = nanmax_safe(
        metrics[
            "shoulder_center_speed_robust_px_frame"
        ]
    )

    summary[
        "shoulder_center_peak_frame"
    ] = frame_of_max(
        metrics["frame"],
        metrics[
            "shoulder_center_speed_robust_px_frame"
        ]
    )

    # --------------------------------------------------------
    # ROTATION
    # --------------------------------------------------------

    shoulder_values = pd.to_numeric(
        metrics[
            "shoulder_axis_projected_deg"
        ],
        errors="coerce"
    ).to_numpy()

    shoulder_values = shoulder_values[
        np.isfinite(
            shoulder_values
        )
    ]

    hip_values = pd.to_numeric(
        metrics[
            "hip_axis_projected_deg"
        ],
        errors="coerce"
    ).to_numpy()

    hip_values = hip_values[
        np.isfinite(
            hip_values
        )
    ]

    if len(
        shoulder_values
    ) >= 2:

        summary[
            "shoulder_axis_excursion_projected_deg"
        ] = float(
            np.max(
                shoulder_values
            )
            -
            np.min(
                shoulder_values
            )
        )

    else:

        summary[
            "shoulder_axis_excursion_projected_deg"
        ] = np.nan

    if len(
        hip_values
    ) >= 2:

        summary[
            "hip_axis_excursion_projected_deg"
        ] = float(
            np.max(
                hip_values
            )
            -
            np.min(
                hip_values
            )
        )

    else:

        summary[
            "hip_axis_excursion_projected_deg"
        ] = np.nan

    summary[
        "max_hip_shoulder_separation_projected_deg"
    ] = nanmax_safe(
        metrics[
            "abs_hip_shoulder_separation_deg"
        ]
    )

    summary[
        "max_separation_frame"
    ] = frame_of_max(
        metrics["frame"],
        metrics[
            "abs_hip_shoulder_separation_deg"
        ]
    )

    summary[
        "shoulder_peak_angular_speed_robust_deg_frame"
    ] = nanmax_safe(
        metrics[
            "shoulder_angular_speed_robust_deg_frame"
        ]
    )

    summary[
        "hip_peak_angular_speed_robust_deg_frame"
    ] = nanmax_safe(
        metrics[
            "hip_angular_speed_robust_deg_frame"
        ]
    )

    # --------------------------------------------------------
    # HEAD RELATIVE STABILITY
    # --------------------------------------------------------

    summary[
        "max_head_relative_displacement_px"
    ] = nanmax_safe(
        metrics[
            "head_relative_displacement_px"
        ]
    )

    summary[
        "max_head_relative_displacement_bodywidth"
    ] = nanmax_safe(
        metrics[
            "head_relative_displacement_bodywidth"
        ]
    )

    # --------------------------------------------------------
    # LEG RELATIVE MOVEMENT
    # --------------------------------------------------------

    summary[
        "left_ankle_max_relative_displacement_px"
    ] = nanmax_safe(
        metrics[
            "left_ankle_relative_displacement_px"
        ]
    )

    summary[
        "right_ankle_max_relative_displacement_px"
    ] = nanmax_safe(
        metrics[
            "right_ankle_relative_displacement_px"
        ]
    )

    summary[
        "left_ankle_max_relative_displacement_bodywidth"
    ] = nanmax_safe(
        metrics[
            "left_ankle_relative_displacement_bodywidth"
        ]
    )

    summary[
        "right_ankle_max_relative_displacement_bodywidth"
    ] = nanmax_safe(
        metrics[
            "right_ankle_relative_displacement_bodywidth"
        ]
    )

    summary[
        "left_knee_max_relative_displacement_bodywidth"
    ] = nanmax_safe(
        metrics[
            "left_knee_relative_displacement_bodywidth"
        ]
    )

    summary[
        "right_knee_max_relative_displacement_bodywidth"
    ] = nanmax_safe(
        metrics[
            "right_knee_relative_displacement_bodywidth"
        ]
    )

    # --------------------------------------------------------
    # POSE QUALITY
    # --------------------------------------------------------

    summary[
        "mean_pose_quality"
    ] = nanmean_safe(
        metrics[
            "pose_quality"
        ]
    )

    # --------------------------------------------------------
    # PHASE TIMINGS
    # --------------------------------------------------------

    summary.update(
        phase_stats(
            metrics,
            fps
        )
    )

    load_start = summary.get(
        "load_start_frame"
    )

    swing_start = summary.get(
        "swing_start_frame"
    )

    peak_start = summary.get(
        "peak_swing_start_frame"
    )

    if (
        load_start is not None
        and peak_start is not None
    ):

        summary[
            "load_to_peak_frames"
        ] = (
            peak_start
            - load_start
        )

        summary[
            "load_to_peak_sec"
        ] = (
            peak_start
            - load_start
        ) / fps

    else:

        summary[
            "load_to_peak_frames"
        ] = np.nan

        summary[
            "load_to_peak_sec"
        ] = np.nan

    if (
        swing_start is not None
        and peak_start is not None
    ):

        summary[
            "swing_to_peak_frames"
        ] = (
            peak_start
            - swing_start
        )

        summary[
            "swing_to_peak_sec"
        ] = (
            peak_start
            - swing_start
        ) / fps

    else:

        summary[
            "swing_to_peak_frames"
        ] = np.nan

        summary[
            "swing_to_peak_sec"
        ] = np.nan

    # ========================================================
    # SAVE
    # ========================================================

    frame_output = (
        output_dir
        / "swing_metrics_v3_1_frames.csv"
    )

    summary_output = (
        output_dir
        / "swing_metrics_v3_1_summary.csv"
    )

    metrics.to_csv(
        frame_output,
        index=False
    )

    pd.DataFrame(
        [
            summary
        ]
    ).to_csv(
        summary_output,
        index=False
    )

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print()
    print("=" * 78)
    print(
        "SWING METRICS V3.1 COMPLETE"
    )
    print("=" * 78)

    print()
    print("BODY SCALE")
    print("-" * 78)

    if finite(
        median_shoulder_width
    ):

        print(
            f"Median shoulder width     : "
            f"{median_shoulder_width:.2f} px"
        )

    if finite(
        median_hip_width
    ):

        print(
            f"Median hip width          : "
            f"{median_hip_width:.2f} px"
        )

    print()
    print("ROBUST WRIST SPEED")
    print("-" * 78)

    left_peak = summary[
        "left_wrist_peak_speed_robust_px_frame"
    ]

    right_peak = summary[
        "right_wrist_peak_speed_robust_px_frame"
    ]

    if finite(
        left_peak
    ):

        print(
            f"Left wrist peak           : "
            f"{left_peak:.2f} px/frame "
            f"@ frame "
            f"{summary['left_wrist_peak_frame']}"
        )

    if finite(
        right_peak
    ):

        print(
            f"Right wrist peak          : "
            f"{right_peak:.2f} px/frame "
            f"@ frame "
            f"{summary['right_wrist_peak_frame']}"
        )

    print()
    print("ROBUST BODY CENTER SPEED")
    print("-" * 78)

    hip_peak = summary[
        "hip_center_peak_speed_robust_px_frame"
    ]

    shoulder_peak = summary[
        "shoulder_center_peak_speed_robust_px_frame"
    ]

    if finite(
        hip_peak
    ):

        print(
            f"Hip center peak           : "
            f"{hip_peak:.2f} px/frame "
            f"@ frame "
            f"{summary['hip_center_peak_frame']}"
        )

    if finite(
        shoulder_peak
    ):

        print(
            f"Shoulder center peak      : "
            f"{shoulder_peak:.2f} px/frame "
            f"@ frame "
            f"{summary['shoulder_center_peak_frame']}"
        )

    print()
    print("PROJECTED BODY AXIS")
    print("-" * 78)

    shoulder_excursion = summary[
        "shoulder_axis_excursion_projected_deg"
    ]

    hip_excursion = summary[
        "hip_axis_excursion_projected_deg"
    ]

    separation_peak = summary[
        "max_hip_shoulder_separation_projected_deg"
    ]

    if finite(
        shoulder_excursion
    ):

        print(
            f"Shoulder axis excursion   : "
            f"{shoulder_excursion:.2f} deg projected"
        )

    if finite(
        hip_excursion
    ):

        print(
            f"Hip axis excursion        : "
            f"{hip_excursion:.2f} deg projected"
        )

    if finite(
        separation_peak
    ):

        print(
            f"Max hip-shoulder sep.     : "
            f"{separation_peak:.2f} deg projected "
            f"@ frame "
            f"{summary['max_separation_frame']}"
        )

    print()
    print("BODY-RELATIVE STABILITY")
    print("-" * 78)

    head_relative = summary[
        "max_head_relative_displacement_px"
    ]

    head_bodywidth = summary[
        "max_head_relative_displacement_bodywidth"
    ]

    if finite(
        head_relative
    ):

        print(
            f"Head relative movement    : "
            f"{head_relative:.2f} px"
        )

    if finite(
        head_bodywidth
    ):

        print(
            f"Head movement normalized  : "
            f"{head_bodywidth:.3f} body widths"
        )

    left_ankle_bw = summary[
        "left_ankle_max_relative_displacement_bodywidth"
    ]

    right_ankle_bw = summary[
        "right_ankle_max_relative_displacement_bodywidth"
    ]

    if finite(
        left_ankle_bw
    ):

        print(
            f"Left ankle movement       : "
            f"{left_ankle_bw:.3f} body widths"
        )

    if finite(
        right_ankle_bw
    ):

        print(
            f"Right ankle movement      : "
            f"{right_ankle_bw:.3f} body widths"
        )

    print()
    print("SWING TIMING")
    print("-" * 78)

    for phase in [
        "SETUP",
        "LOAD",
        "SWING",
        "PEAK_SWING",
        "FOLLOW_THROUGH",
    ]:

        key = phase.lower()

        start_key = (
            f"{key}_start_frame"
        )

        end_key = (
            f"{key}_end_frame"
        )

        duration_key = (
            f"{key}_duration_sec"
        )

        if start_key in summary:

            print(
                f"{phase:<20}: "
                f"{summary[start_key]} -> "
                f"{summary[end_key]} "
                f"("
                f"{summary[duration_key]:.3f} s"
                f")"
            )

    if finite(
        summary[
            "load_to_peak_sec"
        ]
    ):

        print(
            f"\nLOAD -> PEAK          : "
            f"{summary['load_to_peak_frames']} frames "
            f"("
            f"{summary['load_to_peak_sec']:.3f} s"
            f")"
        )

    if finite(
        summary[
            "swing_to_peak_sec"
        ]
    ):

        print(
            f"SWING -> PEAK         : "
            f"{summary['swing_to_peak_frames']} frames "
            f"("
            f"{summary['swing_to_peak_sec']:.3f} s"
            f")"
        )

    print()
    print("QUALITY")
    print("-" * 78)

    quality = summary[
        "mean_pose_quality"
    ]

    if finite(
        quality
    ):

        print(
            f"Mean pose quality         : "
            f"{quality:.3f}"
        )

    else:

        print(
            "Mean pose quality         : N/A"
        )

    print()
    print("IMPORTANT")
    print("-" * 78)

    print(
        "Speeds use centered 3-frame 2D displacement."
    )

    print(
        "Body rotation is represented as an unoriented 2D axis."
    )

    print(
        "Head and leg movement are relative to the hip center."
    )

    print(
        "These are comparative 2D vision metrics, not calibrated biomechanics."
    )

    print()
    print("OUTPUT FILES")
    print("-" * 78)

    print(
        f"Frame metrics : {frame_output.resolve()}"
    )

    print(
        f"Summary       : {summary_output.resolve()}"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()