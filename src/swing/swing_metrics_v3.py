from pathlib import Path
import argparse
import math

import numpy as np
import pandas as pd


# ============================================================
# BASEVISIONAI - SWING METRICS V3
# ============================================================
#
# Inputs:
#   1. Pose Smoothing V2 CSV
#   2. Swing Analysis V2.2 CSV
#
# Outputs:
#   1. swing_metrics_v3_frames.csv
#   2. swing_metrics_v3_summary.csv
#
# IMPORTANT:
#   - Speeds are 2D image-space measurements.
#   - px/frame is NOT m/s or mph.
#   - projected rotation is NOT true 3D biomechanical rotation.
#   - normalized speeds are relative to median shoulder width.
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
        description="BaseVisionAI - Swing Metrics V3"
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
        help="Video FPS. Default: 30"
    )

    parser.add_argument(
        "--output-dir",
        default="results/swing_metrics_v3",
        help="Output directory"
    )

    return parser.parse_args()


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_float(value):

    try:

        if pd.isna(value):
            return np.nan

        value = float(value)

        if not np.isfinite(value):
            return np.nan

        return value

    except Exception:
        return np.nan


def safe_int(value):

    try:

        if pd.isna(value):
            return None

        return int(round(float(value)))

    except Exception:
        return None


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

    mask = np.isfinite(values)

    if not np.any(mask):
        return None

    valid_values = values[mask]
    valid_frames = frames[mask]

    index = int(
        np.argmax(
            valid_values
        )
    )

    return int(
        valid_frames[index]
    )


# ============================================================
# POSE COLUMN DETECTION
# ============================================================

def coordinate_candidates(
    keypoint,
    axis,
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
    axis,
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
    axis,
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
    y2,
):

    dx = x2 - x1
    dy = y2 - y1

    return np.sqrt(
        dx ** 2
        + dy ** 2
    )


def center_series(
    x1,
    y1,
    x2,
    y2,
):

    cx = (
        x1 + x2
    ) / 2.0

    cy = (
        y1 + y2
    ) / 2.0

    return cx, cy


def speed_series(
    x,
    y,
):

    dx = x.diff()
    dy = y.diff()

    speed = np.sqrt(
        dx ** 2
        + dy ** 2
    )

    return speed


def angle_series(
    x1,
    y1,
    x2,
    y2,
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

    angle = np.degrees(
        np.arctan2(
            dy,
            dx
        )
    )

    invalid = (
        ~np.isfinite(dx)
        | ~np.isfinite(dy)
    )

    angle[
        invalid
    ] = np.nan

    return pd.Series(
        angle,
        index=x1.index
    )


def unwrap_angles_deg(
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

    valid_angles = np.radians(
        values[
            valid_indices
        ]
    )

    unwrapped = np.unwrap(
        valid_angles
    )

    output[
        valid_indices
    ] = np.degrees(
        unwrapped
    )

    return pd.Series(
        output,
        index=angle_series_input.index
    )


def angle_difference_deg(
    a,
    b,
):

    diff = a - b

    diff = (
        diff + 180.0
    ) % 360.0 - 180.0

    return diff


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
    phase
):

    if pd.isna(phase):
        return "UNKNOWN"

    phase = str(
        phase
    ).strip().upper()

    phase = phase.replace(
        " ",
        "_"
    )

    phase = phase.replace(
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
        phase,
        phase
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
        "FOLLOW",
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
            if fps > 0
            else np.nan
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
# RANGE / ROTATION HELPERS
# ============================================================

def rotation_range(
    series
):

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).to_numpy(
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) < 2:
        return np.nan

    return float(
        np.max(values)
        - np.min(values)
    )


def displacement_from_reference(
    x,
    y,
):

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    if not np.any(valid):
        return pd.Series(
            np.nan,
            index=x.index
        )

    first_index = np.where(
        valid.to_numpy()
    )[0][0]

    reference_x = float(
        x.iloc[
            first_index
        ]
    )

    reference_y = float(
        y.iloc[
            first_index
        ]
    )

    displacement = np.sqrt(
        (
            x - reference_x
        ) ** 2
        +
        (
            y - reference_y
        ) ** 2
    )

    return displacement


# ============================================================
# MAIN ANALYSIS
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
    # INPUT CHECK
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

    pose_df["frame"] = pd.to_numeric(
        pose_df["frame"],
        errors="coerce"
    )

    swing_df["frame"] = pd.to_numeric(
        swing_df["frame"],
        errors="coerce"
    )

    pose_df = pose_df.dropna(
        subset=["frame"]
    ).copy()

    swing_df = swing_df.dropna(
        subset=["frame"]
    ).copy()

    pose_df["frame"] = (
        pose_df["frame"]
        .astype(int)
    )

    swing_df["frame"] = (
        swing_df["frame"]
        .astype(int)
    )

    pose_df = (
        pose_df
        .sort_values("frame")
        .drop_duplicates(
            subset="frame",
            keep="first"
        )
        .reset_index(drop=True)
    )

    swing_df = (
        swing_df
        .sort_values("frame")
        .drop_duplicates(
            subset="frame",
            keep="first"
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # BASE FRAME TABLE
    # --------------------------------------------------------

    metrics = pd.DataFrame({
        "frame": pose_df["frame"]
    })

    metrics[
        "time_sec"
    ] = (
        metrics["frame"] - 1
    ) / fps

    # --------------------------------------------------------
    # PHASE
    # --------------------------------------------------------

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
                phase_column: "phase_raw"
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

    # --------------------------------------------------------
    # COPY SWING SCORE IF AVAILABLE
    # --------------------------------------------------------

    swing_score_column = None

    for candidate in [
        "swing_score",
        "combined_score",
        "motion_score",
        "smoothed_swing_score",
    ]:

        if candidate in swing_df.columns:

            swing_score_column = candidate
            break

    if swing_score_column is not None:

        score_table = swing_df[
            [
                "frame",
                swing_score_column
            ]
        ].copy()

        score_table.rename(
            columns={
                swing_score_column:
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
    # LOAD POSE COORDINATES
    # ========================================================

    coordinates = {}

    print()
    print("=" * 78)
    print(
        "BaseVisionAI - Swing Metrics V3"
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
    print("Pose coordinate mapping:")

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
                f"  {keypoint:<16} "
                f"{x_column} / {y_column}"
            )

    # ========================================================
    # SHORT VARIABLES
    # ========================================================

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

    lex = coordinates[
        "left_elbow_x"
    ]
    ley = coordinates[
        "left_elbow_y"
    ]

    rex = coordinates[
        "right_elbow_x"
    ]
    rey = coordinates[
        "right_elbow_y"
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

    nosex = coordinates[
        "nose_x"
    ]
    nosey = coordinates[
        "nose_y"
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
    # POINT SPEEDS
    # ========================================================

    left_wrist_speed = speed_series(
        lwx,
        lwy
    )

    right_wrist_speed = speed_series(
        rwx,
        rwy
    )

    left_elbow_speed = speed_series(
        lex,
        ley
    )

    right_elbow_speed = speed_series(
        rex,
        rey
    )

    shoulder_center_speed = speed_series(
        shoulder_cx,
        shoulder_cy
    )

    hip_center_speed = speed_series(
        hip_cx,
        hip_cy
    )

    nose_speed = speed_series(
        nosex,
        nosey
    )

    left_ankle_speed = speed_series(
        lax,
        lay
    )

    right_ankle_speed = speed_series(
        rax,
        ray
    )

    left_knee_speed = speed_series(
        lkx,
        lky
    )

    right_knee_speed = speed_series(
        rkx,
        rky
    )

    metrics[
        "left_wrist_speed_px_frame"
    ] = left_wrist_speed.to_numpy()

    metrics[
        "right_wrist_speed_px_frame"
    ] = right_wrist_speed.to_numpy()

    metrics[
        "left_elbow_speed_px_frame"
    ] = left_elbow_speed.to_numpy()

    metrics[
        "right_elbow_speed_px_frame"
    ] = right_elbow_speed.to_numpy()

    metrics[
        "shoulder_center_speed_px_frame"
    ] = shoulder_center_speed.to_numpy()

    metrics[
        "hip_center_speed_px_frame"
    ] = hip_center_speed.to_numpy()

    metrics[
        "head_speed_px_frame"
    ] = nose_speed.to_numpy()

    metrics[
        "left_ankle_speed_px_frame"
    ] = left_ankle_speed.to_numpy()

    metrics[
        "right_ankle_speed_px_frame"
    ] = right_ankle_speed.to_numpy()

    metrics[
        "left_knee_speed_px_frame"
    ] = left_knee_speed.to_numpy()

    metrics[
        "right_knee_speed_px_frame"
    ] = right_knee_speed.to_numpy()

    # ========================================================
    # NORMALIZED SPEEDS
    # ========================================================
    #
    # body-width/frame:
    # movement relative to median shoulder width.
    #
    # This helps comparison inside similar camera views but
    # is still NOT a physical velocity.
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

        metrics[
            "head_speed_bodywidth_frame"
        ] = (
            nose_speed
            / median_shoulder_width
        ).to_numpy()

    else:

        for column in [
            "left_wrist_speed_bodywidth_frame",
            "right_wrist_speed_bodywidth_frame",
            "hip_center_speed_bodywidth_frame",
            "shoulder_center_speed_bodywidth_frame",
            "head_speed_bodywidth_frame",
        ]:

            metrics[
                column
            ] = np.nan

    # ========================================================
    # PROJECTED BODY ROTATION
    # ========================================================

    shoulder_angle_raw = angle_series(
        lsx,
        lsy,
        rsx,
        rsy
    )

    hip_angle_raw = angle_series(
        lhx,
        lhy,
        rhx,
        rhy
    )

    shoulder_angle = unwrap_angles_deg(
        shoulder_angle_raw
    )

    hip_angle = unwrap_angles_deg(
        hip_angle_raw
    )

    separation = angle_difference_deg(
        shoulder_angle,
        hip_angle
    )

    metrics[
        "shoulder_angle_projected_deg"
    ] = shoulder_angle.to_numpy()

    metrics[
        "hip_angle_projected_deg"
    ] = hip_angle.to_numpy()

    metrics[
        "hip_shoulder_separation_projected_deg"
    ] = separation.to_numpy()

    metrics[
        "abs_hip_shoulder_separation_deg"
    ] = np.abs(
        separation.to_numpy()
    )

    # --------------------------------------------------------
    # ANGULAR SPEED PER FRAME
    # --------------------------------------------------------

    metrics[
        "shoulder_angular_speed_deg_frame"
    ] = np.abs(
        shoulder_angle.diff()
    ).to_numpy()

    metrics[
        "hip_angular_speed_deg_frame"
    ] = np.abs(
        hip_angle.diff()
    ).to_numpy()

    # ========================================================
    # HEAD DISPLACEMENT
    # ========================================================

    head_displacement = displacement_from_reference(
        nosex,
        nosey
    )

    metrics[
        "head_displacement_px"
    ] = head_displacement.to_numpy()

    if (
        finite(
            median_shoulder_width
        )
        and median_shoulder_width > 1e-6
    ):

        metrics[
            "head_displacement_bodywidth"
        ] = (
            head_displacement
            / median_shoulder_width
        ).to_numpy()

    else:

        metrics[
            "head_displacement_bodywidth"
        ] = np.nan

    # ========================================================
    # LEG MOVEMENT
    # ========================================================

    left_ankle_displacement = displacement_from_reference(
        lax,
        lay
    )

    right_ankle_displacement = displacement_from_reference(
        rax,
        ray
    )

    left_knee_displacement = displacement_from_reference(
        lkx,
        lky
    )

    right_knee_displacement = displacement_from_reference(
        rkx,
        rky
    )

    metrics[
        "left_ankle_displacement_px"
    ] = left_ankle_displacement.to_numpy()

    metrics[
        "right_ankle_displacement_px"
    ] = right_ankle_displacement.to_numpy()

    metrics[
        "left_knee_displacement_px"
    ] = left_knee_displacement.to_numpy()

    metrics[
        "right_knee_displacement_px"
    ] = right_knee_displacement.to_numpy()

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
    ] = "Swing Metrics V3"

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
    # WRIST PEAKS
    # --------------------------------------------------------

    summary[
        "left_wrist_peak_speed_px_frame"
    ] = nanmax_safe(
        metrics[
            "left_wrist_speed_px_frame"
        ]
    )

    summary[
        "left_wrist_peak_frame"
    ] = frame_of_max(
        metrics["frame"],
        metrics[
            "left_wrist_speed_px_frame"
        ]
    )

    summary[
        "right_wrist_peak_speed_px_frame"
    ] = nanmax_safe(
        metrics[
            "right_wrist_speed_px_frame"
        ]
    )

    summary[
        "right_wrist_peak_frame"
    ] = frame_of_max(
        metrics["frame"],
        metrics[
            "right_wrist_speed_px_frame"
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
    # HIP / SHOULDER CENTER SPEED
    # --------------------------------------------------------

    summary[
        "hip_center_peak_speed_px_frame"
    ] = nanmax_safe(
        metrics[
            "hip_center_speed_px_frame"
        ]
    )

    summary[
        "hip_center_peak_frame"
    ] = frame_of_max(
        metrics["frame"],
        metrics[
            "hip_center_speed_px_frame"
        ]
    )

    summary[
        "shoulder_center_peak_speed_px_frame"
    ] = nanmax_safe(
        metrics[
            "shoulder_center_speed_px_frame"
        ]
    )

    summary[
        "shoulder_center_peak_frame"
    ] = frame_of_max(
        metrics["frame"],
        metrics[
            "shoulder_center_speed_px_frame"
        ]
    )

    # --------------------------------------------------------
    # ROTATION
    # --------------------------------------------------------

    summary[
        "shoulder_rotation_range_projected_deg"
    ] = rotation_range(
        metrics[
            "shoulder_angle_projected_deg"
        ]
    )

    summary[
        "hip_rotation_range_projected_deg"
    ] = rotation_range(
        metrics[
            "hip_angle_projected_deg"
        ]
    )

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
        "shoulder_peak_angular_speed_deg_frame"
    ] = nanmax_safe(
        metrics[
            "shoulder_angular_speed_deg_frame"
        ]
    )

    summary[
        "shoulder_peak_angular_speed_frame"
    ] = frame_of_max(
        metrics["frame"],
        metrics[
            "shoulder_angular_speed_deg_frame"
        ]
    )

    summary[
        "hip_peak_angular_speed_deg_frame"
    ] = nanmax_safe(
        metrics[
            "hip_angular_speed_deg_frame"
        ]
    )

    summary[
        "hip_peak_angular_speed_frame"
    ] = frame_of_max(
        metrics["frame"],
        metrics[
            "hip_angular_speed_deg_frame"
        ]
    )

    # --------------------------------------------------------
    # HEAD
    # --------------------------------------------------------

    summary[
        "max_head_displacement_px"
    ] = nanmax_safe(
        metrics[
            "head_displacement_px"
        ]
    )

    summary[
        "max_head_displacement_bodywidth"
    ] = nanmax_safe(
        metrics[
            "head_displacement_bodywidth"
        ]
    )

    summary[
        "head_peak_speed_px_frame"
    ] = nanmax_safe(
        metrics[
            "head_speed_px_frame"
        ]
    )

    # --------------------------------------------------------
    # LEG MOVEMENT
    # --------------------------------------------------------

    summary[
        "left_ankle_max_displacement_px"
    ] = nanmax_safe(
        metrics[
            "left_ankle_displacement_px"
        ]
    )

    summary[
        "right_ankle_max_displacement_px"
    ] = nanmax_safe(
        metrics[
            "right_ankle_displacement_px"
        ]
    )

    summary[
        "left_knee_max_displacement_px"
    ] = nanmax_safe(
        metrics[
            "left_knee_displacement_px"
        ]
    )

    summary[
        "right_knee_max_displacement_px"
    ] = nanmax_safe(
        metrics[
            "right_knee_displacement_px"
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

    summary[
        "median_pose_quality"
    ] = nanmedian_safe(
        metrics[
            "pose_quality"
        ]
    )

    # --------------------------------------------------------
    # SWING MOTION SCORE
    # --------------------------------------------------------

    summary[
        "peak_swing_motion_score"
    ] = nanmax_safe(
        metrics[
            "swing_motion_score"
        ]
    )

    summary[
        "peak_swing_motion_frame"
    ] = frame_of_max(
        metrics["frame"],
        metrics[
            "swing_motion_score"
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

    # --------------------------------------------------------
    # LOAD -> PEAK TIMING
    # --------------------------------------------------------

    load_start = summary.get(
        "load_start_frame"
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

    # --------------------------------------------------------
    # SWING START -> PEAK
    # --------------------------------------------------------

    swing_start = summary.get(
        "swing_start_frame"
    )

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
        / "swing_metrics_v3_frames.csv"
    )

    summary_output = (
        output_dir
        / "swing_metrics_v3_summary.csv"
    )

    metrics.to_csv(
        frame_output,
        index=False
    )

    summary_df = pd.DataFrame(
        [
            summary
        ]
    )

    summary_df.to_csv(
        summary_output,
        index=False
    )

    # ========================================================
    # CONSOLE REPORT
    # ========================================================

    print()
    print("=" * 78)
    print(
        "SWING METRICS V3 COMPLETE"
    )
    print("=" * 78)

    print()
    print("BODY SCALE")
    print("-" * 78)

    print(
        f"Median shoulder width     : "
        f"{median_shoulder_width:.2f} px"
        if finite(median_shoulder_width)
        else
        "Median shoulder width     : N/A"
    )

    print(
        f"Median hip width          : "
        f"{median_hip_width:.2f} px"
        if finite(median_hip_width)
        else
        "Median hip width          : N/A"
    )

    print()
    print("WRIST SPEED")
    print("-" * 78)

    left_peak = summary[
        "left_wrist_peak_speed_px_frame"
    ]

    right_peak = summary[
        "right_wrist_peak_speed_px_frame"
    ]

    print(
        f"Left wrist peak           : "
        f"{left_peak:.2f} px/frame "
        f"@ frame "
        f"{summary['left_wrist_peak_frame']}"
        if finite(left_peak)
        else
        "Left wrist peak           : N/A"
    )

    print(
        f"Right wrist peak          : "
        f"{right_peak:.2f} px/frame "
        f"@ frame "
        f"{summary['right_wrist_peak_frame']}"
        if finite(right_peak)
        else
        "Right wrist peak          : N/A"
    )

    print()
    print("BODY CENTER SPEED")
    print("-" * 78)

    hip_peak = summary[
        "hip_center_peak_speed_px_frame"
    ]

    shoulder_peak = summary[
        "shoulder_center_peak_speed_px_frame"
    ]

    print(
        f"Hip center peak           : "
        f"{hip_peak:.2f} px/frame "
        f"@ frame "
        f"{summary['hip_center_peak_frame']}"
        if finite(hip_peak)
        else
        "Hip center peak           : N/A"
    )

    print(
        f"Shoulder center peak      : "
        f"{shoulder_peak:.2f} px/frame "
        f"@ frame "
        f"{summary['shoulder_center_peak_frame']}"
        if finite(shoulder_peak)
        else
        "Shoulder center peak      : N/A"
    )

    print()
    print("PROJECTED ROTATION")
    print("-" * 78)

    shoulder_rotation = summary[
        "shoulder_rotation_range_projected_deg"
    ]

    hip_rotation = summary[
        "hip_rotation_range_projected_deg"
    ]

    max_separation = summary[
        "max_hip_shoulder_separation_projected_deg"
    ]

    print(
        f"Shoulder rotation range   : "
        f"{shoulder_rotation:.2f} deg projected"
        if finite(shoulder_rotation)
        else
        "Shoulder rotation range   : N/A"
    )

    print(
        f"Hip rotation range        : "
        f"{hip_rotation:.2f} deg projected"
        if finite(hip_rotation)
        else
        "Hip rotation range        : N/A"
    )

    print(
        f"Max hip-shoulder sep.     : "
        f"{max_separation:.2f} deg projected "
        f"@ frame "
        f"{summary['max_separation_frame']}"
        if finite(max_separation)
        else
        "Max hip-shoulder sep.     : N/A"
    )

    print()
    print("STABILITY / LOWER BODY")
    print("-" * 78)

    head_disp = summary[
        "max_head_displacement_px"
    ]

    print(
        f"Max head displacement     : "
        f"{head_disp:.2f} px"
        if finite(head_disp)
        else
        "Max head displacement     : N/A"
    )

    print(
        f"Left ankle displacement   : "
        f"{summary['left_ankle_max_displacement_px']:.2f} px"
        if finite(
            summary[
                "left_ankle_max_displacement_px"
            ]
        )
        else
        "Left ankle displacement   : N/A"
    )

    print(
        f"Right ankle displacement  : "
        f"{summary['right_ankle_max_displacement_px']:.2f} px"
        if finite(
            summary[
                "right_ankle_max_displacement_px"
            ]
        )
        else
        "Right ankle displacement  : N/A"
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
        "FOLLOW",
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
                f"({summary[duration_key]:.3f} s)"
            )

    if finite(
        summary[
            "load_to_peak_sec"
        ]
    ):

        print(
            f"\nLOAD -> PEAK          : "
            f"{summary['load_to_peak_frames']} frames "
            f"({summary['load_to_peak_sec']:.3f} s)"
        )

    if finite(
        summary[
            "swing_to_peak_sec"
        ]
    ):

        print(
            f"SWING -> PEAK         : "
            f"{summary['swing_to_peak_frames']} frames "
            f"({summary['swing_to_peak_sec']:.3f} s)"
        )

    print()
    print("QUALITY")
    print("-" * 78)

    mean_quality = summary[
        "mean_pose_quality"
    ]

    if finite(mean_quality):

        print(
            f"Mean pose quality         : "
            f"{mean_quality:.3f}"
        )

    else:

        print(
            "Mean pose quality         : N/A"
        )

    print()
    print("IMPORTANT")
    print("-" * 78)

    print(
        "Speeds are 2D image-space measurements."
    )

    print(
        "px/frame must NOT be interpreted as mph or m/s."
    )

    print(
        "Rotation values are projected 2D angles, not true 3D biomechanics."
    )

    print(
        "Body-width normalized values are relative comparison metrics."
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