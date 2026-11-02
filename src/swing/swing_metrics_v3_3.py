from pathlib import Path
import argparse

import numpy as np
import pandas as pd


# ============================================================
# BASEVISIONAI - SWING METRICS V3.3
# ============================================================
#
# Changes from V3.2:
#
# 1. Normalization uses median torso length as the main
#    body scale instead of median([shoulders, hips, torso]).
#
# 2. Stability metrics are measured only from:
#       LOAD + SWING + PEAK_SWING
#
# 3. FOLLOW_THROUGH displacement is reported separately.
#
# Retained from V3.2:
# - robust centered 3-frame speeds
# - phase-restricted motion peaks
# - LOAD-relative projected rotation
# - late-SETUP baseline for body-relative displacement
#
# IMPORTANT:
# - All measurements remain 2D vision metrics.
# - px/frame is NOT physical velocity.
# - torso-length values are relative image measurements.
# - projected rotation is NOT calibrated 3D biomechanics.
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
        description="BaseVisionAI - Swing Metrics V3.3"
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
        default="results/swing_metrics_v3_3",
        help="Output directory"
    )

    parser.add_argument(
        "--setup-baseline-frames",
        type=int,
        default=10,
        help="Final SETUP frames used for baseline"
    )

    return parser.parse_args()


# ============================================================
# BASIC HELPERS
# ============================================================

def finite(value):
    return value is not None and np.isfinite(value)


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

    mask = np.isfinite(
        values
    )

    if not np.any(mask):
        return None

    valid_frames = frames[
        mask
    ]

    valid_values = values[
        mask
    ]

    idx = int(
        np.argmax(
            valid_values
        )
    )

    return int(
        valid_frames[
            idx
        ]
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


def robust_speed_3frame(
    x,
    y
):

    dx = (
        x.shift(-1)
        -
        x.shift(1)
    )

    dy = (
        y.shift(-1)
        -
        y.shift(1)
    )

    return np.sqrt(
        dx ** 2
        +
        dy ** 2
    ) / 2.0


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

    angle = np.degrees(
        np.arctan2(
            dy,
            dx
        )
    )

    angle = np.mod(
        angle,
        180.0
    )

    invalid = (
        ~np.isfinite(dx)
        |
        ~np.isfinite(dy)
    )

    angle[
        invalid
    ] = np.nan

    return pd.Series(
        angle,
        index=x1.index
    )


def axis_difference_from_baseline(
    angle,
    baseline
):

    if not finite(
        baseline
    ):

        return pd.Series(
            np.nan,
            index=angle.index
        )

    return (
        angle
        -
        baseline
        +
        90.0
    ) % 180.0 - 90.0


def axis_difference_between(
    a,
    b
):

    return (
        a
        -
        b
        +
        90.0
    ) % 180.0 - 90.0


# ============================================================
# PHASE HANDLING
# ============================================================

def detect_phase_column(
    swing_df
):

    for column in [
        "phase",
        "swing_phase",
        "phase_name",
        "predicted_phase",
    ]:

        if column in swing_df.columns:
            return column

    return None


def normalize_phase(
    value
):

    if pd.isna(
        value
    ):
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

    for phase in [
        "SETUP",
        "LOAD",
        "SWING",
        "PEAK_SWING",
        "FOLLOW_THROUGH",
    ]:

        subset = df[
            df[
                "phase_normalized"
            ] == phase
        ]

        if len(
            subset
        ) == 0:
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
            -
            start_frame
            +
            1
        )

        duration_sec = (
            frame_count
            /
            fps
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
# BASELINES
# ============================================================

def last_phase_frames(
    df,
    phase,
    count
):

    subset = df[
        df[
            "phase_normalized"
        ] == phase
    ].copy()

    if len(
        subset
    ) == 0:

        return subset

    subset = subset.sort_values(
        "frame"
    )

    return subset.tail(
        count
    )


def baseline_median(
    values,
    mask
):

    selected = values[
        mask
    ]

    return nanmedian_safe(
        selected
    )


def relative_position(
    point_x,
    point_y,
    center_x,
    center_y
):

    return (
        point_x - center_x,
        point_y - center_y
    )


def displacement_from_baseline(
    x,
    y,
    baseline_x,
    baseline_y
):

    if (
        not finite(
            baseline_x
        )
        or
        not finite(
            baseline_y
        )
    ):

        return pd.Series(
            np.nan,
            index=x.index
        )

    return np.sqrt(
        (
            x - baseline_x
        ) ** 2
        +
        (
            y - baseline_y
        ) ** 2
    )


# ============================================================
# PHASE FILTERING
# ============================================================

def phase_mask(
    metrics,
    phases
):

    return metrics[
        "phase_normalized"
    ].isin(
        phases
    )


def restricted_peak(
    metrics,
    value_column,
    phases
):

    mask = phase_mask(
        metrics,
        phases
    )

    subset = metrics[
        mask
    ]

    if len(
        subset
    ) == 0:

        return np.nan, None

    peak = nanmax_safe(
        subset[
            value_column
        ]
    )

    frame = frame_of_max(
        subset[
            "frame"
        ],
        subset[
            value_column
        ]
    )

    return peak, frame


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

    setup_baseline_frames = int(
        args.setup_baseline_frames
    )

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

    if setup_baseline_frames < 1:

        raise ValueError(
            "--setup-baseline-frames must be >= 1"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # LOAD
    # ========================================================

    pose_df = pd.read_csv(
        pose_path
    )

    swing_df = pd.read_csv(
        swing_path
    )

    for name, df in [
        ("pose", pose_df),
        ("swing", swing_df),
    ]:

        if "frame" not in df.columns:

            raise ValueError(
                f"{name} CSV requires a 'frame' column."
            )

        df[
            "frame"
        ] = pd.to_numeric(
            df[
                "frame"
            ],
            errors="coerce"
        )

        df.dropna(
            subset=["frame"],
            inplace=True
        )

        df[
            "frame"
        ] = df[
            "frame"
        ].astype(
            int
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
    # BASE METRICS
    # ========================================================

    metrics = pd.DataFrame({
        "frame":
            pose_df[
                "frame"
            ]
    })

    metrics[
        "time_sec"
    ] = (
        metrics[
            "frame"
        ] - 1
    ) / fps

    # ========================================================
    # PHASE MERGE
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
    # OPTIONAL SWING SCORE
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
    # POSE COORDINATES
    # ========================================================

    coordinates = {}

    print()
    print("=" * 80)
    print(
        "BaseVisionAI - Swing Metrics V3.3"
    )
    print("=" * 80)

    print(
        f"Pose CSV              : {pose_path}"
    )

    print(
        f"Swing CSV             : {swing_path}"
    )

    print(
        f"FPS                   : {fps:.2f}"
    )

    print(
        f"Setup baseline frames : {setup_baseline_frames}"
    )

    print("-" * 80)

    for keypoint in KEYPOINTS:

        x_col = find_coordinate_column(
            pose_df,
            keypoint,
            "x"
        )

        y_col = find_coordinate_column(
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
            x_col is not None
            and y_col is not None
        ):

            print(
                f"{keypoint:<18}"
                f"{x_col} / {y_col}"
            )

    # ========================================================
    # SHORT VARIABLES
    # ========================================================

    nosex = coordinates["nose_x"]
    nosey = coordinates["nose_y"]

    lsx = coordinates["left_shoulder_x"]
    lsy = coordinates["left_shoulder_y"]

    rsx = coordinates["right_shoulder_x"]
    rsy = coordinates["right_shoulder_y"]

    lwx = coordinates["left_wrist_x"]
    lwy = coordinates["left_wrist_y"]

    rwx = coordinates["right_wrist_x"]
    rwy = coordinates["right_wrist_y"]

    lhx = coordinates["left_hip_x"]
    lhy = coordinates["left_hip_y"]

    rhx = coordinates["right_hip_x"]
    rhy = coordinates["right_hip_y"]

    lkx = coordinates["left_knee_x"]
    lky = coordinates["left_knee_y"]

    rkx = coordinates["right_knee_x"]
    rky = coordinates["right_knee_y"]

    lax = coordinates["left_ankle_x"]
    lay = coordinates["left_ankle_y"]

    rax = coordinates["right_ankle_x"]
    ray = coordinates["right_ankle_y"]

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

    torso_length = distance_series(
        shoulder_cx,
        shoulder_cy,
        hip_cx,
        hip_cy
    )

    median_shoulder_width = nanmedian_safe(
        shoulder_width
    )

    median_hip_width = nanmedian_safe(
        hip_width
    )

    median_torso_length = nanmedian_safe(
        torso_length
    )

    # --------------------------------------------------------
    # V3.3 CHANGE:
    # torso length is now the normalization reference
    # --------------------------------------------------------

    body_scale_px = median_torso_length

    metrics[
        "shoulder_width_px"
    ] = shoulder_width.to_numpy()

    metrics[
        "hip_width_px"
    ] = hip_width.to_numpy()

    metrics[
        "torso_length_px"
    ] = torso_length.to_numpy()

    metrics[
        "body_scale_px"
    ] = body_scale_px

    # ========================================================
    # ROBUST SPEEDS
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

    if (
        finite(
            body_scale_px
        )
        and body_scale_px > 1e-6
    ):

        metrics[
            "left_wrist_speed_torsoscale_frame"
        ] = (
            left_wrist_speed
            / body_scale_px
        ).to_numpy()

        metrics[
            "right_wrist_speed_torsoscale_frame"
        ] = (
            right_wrist_speed
            / body_scale_px
        ).to_numpy()

        metrics[
            "hip_center_speed_torsoscale_frame"
        ] = (
            hip_center_speed
            / body_scale_px
        ).to_numpy()

        metrics[
            "shoulder_center_speed_torsoscale_frame"
        ] = (
            shoulder_center_speed
            / body_scale_px
        ).to_numpy()

    else:

        metrics[
            "left_wrist_speed_torsoscale_frame"
        ] = np.nan

        metrics[
            "right_wrist_speed_torsoscale_frame"
        ] = np.nan

        metrics[
            "hip_center_speed_torsoscale_frame"
        ] = np.nan

        metrics[
            "shoulder_center_speed_torsoscale_frame"
        ] = np.nan

    # ========================================================
    # BODY AXIS ROTATION
    # ========================================================

    shoulder_axis = body_axis_angle(
        lsx,
        lsy,
        rsx,
        rsy
    )

    hip_axis = body_axis_angle(
        lhx,
        lhy,
        rhx,
        rhy
    )

    metrics[
        "shoulder_axis_raw_projected_deg"
    ] = shoulder_axis.to_numpy()

    metrics[
        "hip_axis_raw_projected_deg"
    ] = hip_axis.to_numpy()

    load_mask = (
        metrics[
            "phase_normalized"
        ] == "LOAD"
    ).to_numpy()

    shoulder_load_baseline = baseline_median(
        shoulder_axis,
        load_mask
    )

    hip_load_baseline = baseline_median(
        hip_axis,
        load_mask
    )

    shoulder_rotation = axis_difference_from_baseline(
        shoulder_axis,
        shoulder_load_baseline
    )

    hip_rotation = axis_difference_from_baseline(
        hip_axis,
        hip_load_baseline
    )

    separation = axis_difference_between(
        shoulder_axis,
        hip_axis
    )

    metrics[
        "shoulder_rotation_from_load_deg"
    ] = shoulder_rotation.to_numpy()

    metrics[
        "hip_rotation_from_load_deg"
    ] = hip_rotation.to_numpy()

    metrics[
        "hip_shoulder_separation_projected_deg"
    ] = separation.to_numpy()

    metrics[
        "abs_hip_shoulder_separation_deg"
    ] = np.abs(
        separation.to_numpy()
    )

    # ========================================================
    # LATE SETUP BASELINE
    # ========================================================

    setup_subset = last_phase_frames(
        metrics,
        "SETUP",
        setup_baseline_frames
    )

    setup_frames = set(
        setup_subset[
            "frame"
        ].astype(
            int
        ).tolist()
    )

    setup_mask = metrics[
        "frame"
    ].isin(
        setup_frames
    ).to_numpy()

    # ========================================================
    # HEAD RELATIVE TO HIP CENTER
    # ========================================================

    head_rel_x, head_rel_y = relative_position(
        nosex,
        nosey,
        hip_cx,
        hip_cy
    )

    head_base_x = baseline_median(
        head_rel_x,
        setup_mask
    )

    head_base_y = baseline_median(
        head_rel_y,
        setup_mask
    )

    head_disp = displacement_from_baseline(
        head_rel_x,
        head_rel_y,
        head_base_x,
        head_base_y
    )

    metrics[
        "head_displacement_from_setup_px"
    ] = head_disp.to_numpy()

    # ========================================================
    # LEGS RELATIVE TO HIP CENTER
    # ========================================================

    left_ankle_rel_x, left_ankle_rel_y = relative_position(
        lax,
        lay,
        hip_cx,
        hip_cy
    )

    right_ankle_rel_x, right_ankle_rel_y = relative_position(
        rax,
        ray,
        hip_cx,
        hip_cy
    )

    left_knee_rel_x, left_knee_rel_y = relative_position(
        lkx,
        lky,
        hip_cx,
        hip_cy
    )

    right_knee_rel_x, right_knee_rel_y = relative_position(
        rkx,
        rky,
        hip_cx,
        hip_cy
    )

    left_ankle_base_x = baseline_median(
        left_ankle_rel_x,
        setup_mask
    )

    left_ankle_base_y = baseline_median(
        left_ankle_rel_y,
        setup_mask
    )

    right_ankle_base_x = baseline_median(
        right_ankle_rel_x,
        setup_mask
    )

    right_ankle_base_y = baseline_median(
        right_ankle_rel_y,
        setup_mask
    )

    left_knee_base_x = baseline_median(
        left_knee_rel_x,
        setup_mask
    )

    left_knee_base_y = baseline_median(
        left_knee_rel_y,
        setup_mask
    )

    right_knee_base_x = baseline_median(
        right_knee_rel_x,
        setup_mask
    )

    right_knee_base_y = baseline_median(
        right_knee_rel_y,
        setup_mask
    )

    left_ankle_disp = displacement_from_baseline(
        left_ankle_rel_x,
        left_ankle_rel_y,
        left_ankle_base_x,
        left_ankle_base_y
    )

    right_ankle_disp = displacement_from_baseline(
        right_ankle_rel_x,
        right_ankle_rel_y,
        right_ankle_base_x,
        right_ankle_base_y
    )

    left_knee_disp = displacement_from_baseline(
        left_knee_rel_x,
        left_knee_rel_y,
        left_knee_base_x,
        left_knee_base_y
    )

    right_knee_disp = displacement_from_baseline(
        right_knee_rel_x,
        right_knee_rel_y,
        right_knee_base_x,
        right_knee_base_y
    )

    metrics[
        "left_ankle_displacement_from_setup_px"
    ] = left_ankle_disp.to_numpy()

    metrics[
        "right_ankle_displacement_from_setup_px"
    ] = right_ankle_disp.to_numpy()

    metrics[
        "left_knee_displacement_from_setup_px"
    ] = left_knee_disp.to_numpy()

    metrics[
        "right_knee_displacement_from_setup_px"
    ] = right_knee_disp.to_numpy()

    # ========================================================
    # TORSO NORMALIZATION
    # ========================================================

    if (
        finite(
            body_scale_px
        )
        and body_scale_px > 1e-6
    ):

        metrics[
            "head_displacement_torsoscale"
        ] = (
            head_disp
            / body_scale_px
        ).to_numpy()

        metrics[
            "left_ankle_displacement_torsoscale"
        ] = (
            left_ankle_disp
            / body_scale_px
        ).to_numpy()

        metrics[
            "right_ankle_displacement_torsoscale"
        ] = (
            right_ankle_disp
            / body_scale_px
        ).to_numpy()

        metrics[
            "left_knee_displacement_torsoscale"
        ] = (
            left_knee_disp
            / body_scale_px
        ).to_numpy()

        metrics[
            "right_knee_displacement_torsoscale"
        ] = (
            right_knee_disp
            / body_scale_px
        ).to_numpy()

    else:

        metrics[
            "head_displacement_torsoscale"
        ] = np.nan

        metrics[
            "left_ankle_displacement_torsoscale"
        ] = np.nan

        metrics[
            "right_ankle_displacement_torsoscale"
        ] = np.nan

        metrics[
            "left_knee_displacement_torsoscale"
        ] = np.nan

        metrics[
            "right_knee_displacement_torsoscale"
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

    summary = {
        "version":
            "Swing Metrics V3.3",

        "frames_analyzed":
            int(
                len(metrics)
            ),

        "fps":
            fps,

        "duration_sec":
            len(metrics) / fps,

        "median_shoulder_width_px":
            median_shoulder_width,

        "median_hip_width_px":
            median_hip_width,

        "median_torso_length_px":
            median_torso_length,

        "body_scale_px":
            body_scale_px,

        "shoulder_load_baseline_deg":
            shoulder_load_baseline,

        "hip_load_baseline_deg":
            hip_load_baseline,
    }

    # ========================================================
    # MOTION PEAKS
    # ========================================================

    left_peak, left_peak_frame = restricted_peak(
        metrics,
        "left_wrist_speed_robust_px_frame",
        [
            "SWING",
            "PEAK_SWING",
        ]
    )

    right_peak, right_peak_frame = restricted_peak(
        metrics,
        "right_wrist_speed_robust_px_frame",
        [
            "SWING",
            "PEAK_SWING",
        ]
    )

    hip_peak, hip_peak_frame = restricted_peak(
        metrics,
        "hip_center_speed_robust_px_frame",
        [
            "LOAD",
            "SWING",
            "PEAK_SWING",
        ]
    )

    shoulder_peak, shoulder_peak_frame = restricted_peak(
        metrics,
        "shoulder_center_speed_robust_px_frame",
        [
            "LOAD",
            "SWING",
            "PEAK_SWING",
        ]
    )

    summary[
        "left_wrist_peak_swing_px_frame"
    ] = left_peak

    summary[
        "left_wrist_peak_swing_frame"
    ] = left_peak_frame

    summary[
        "right_wrist_peak_swing_px_frame"
    ] = right_peak

    summary[
        "right_wrist_peak_swing_frame"
    ] = right_peak_frame

    summary[
        "hip_center_peak_load_to_peak_px_frame"
    ] = hip_peak

    summary[
        "hip_center_peak_load_to_peak_frame"
    ] = hip_peak_frame

    summary[
        "shoulder_center_peak_load_to_peak_px_frame"
    ] = shoulder_peak

    summary[
        "shoulder_center_peak_load_to_peak_frame"
    ] = shoulder_peak_frame

    # ========================================================
    # NORMALIZED WRIST PEAKS
    # ========================================================

    left_peak_norm, _ = restricted_peak(
        metrics,
        "left_wrist_speed_torsoscale_frame",
        [
            "SWING",
            "PEAK_SWING",
        ]
    )

    right_peak_norm, _ = restricted_peak(
        metrics,
        "right_wrist_speed_torsoscale_frame",
        [
            "SWING",
            "PEAK_SWING",
        ]
    )

    summary[
        "left_wrist_peak_torsoscale_frame"
    ] = left_peak_norm

    summary[
        "right_wrist_peak_torsoscale_frame"
    ] = right_peak_norm

    # ========================================================
    # ROTATION SUMMARY
    # ========================================================

    rotation_mask = phase_mask(
        metrics,
        [
            "LOAD",
            "SWING",
            "PEAK_SWING",
        ]
    )

    rotation_subset = metrics[
        rotation_mask
    ]

    summary[
        "max_abs_shoulder_rotation_from_load_deg"
    ] = nanmax_safe(
        np.abs(
            rotation_subset[
                "shoulder_rotation_from_load_deg"
            ]
        )
    )

    summary[
        "max_abs_hip_rotation_from_load_deg"
    ] = nanmax_safe(
        np.abs(
            rotation_subset[
                "hip_rotation_from_load_deg"
            ]
        )
    )

    summary[
        "max_hip_shoulder_separation_deg"
    ] = nanmax_safe(
        rotation_subset[
            "abs_hip_shoulder_separation_deg"
        ]
    )

    summary[
        "max_separation_frame"
    ] = frame_of_max(
        rotation_subset[
            "frame"
        ],
        rotation_subset[
            "abs_hip_shoulder_separation_deg"
        ]
    )

    # ========================================================
    # PRE-PEAK STABILITY
    # ========================================================

    pre_peak_mask = phase_mask(
        metrics,
        [
            "LOAD",
            "SWING",
            "PEAK_SWING",
        ]
    )

    pre_peak = metrics[
        pre_peak_mask
    ]

    summary[
        "prepeak_head_max_px"
    ] = nanmax_safe(
        pre_peak[
            "head_displacement_from_setup_px"
        ]
    )

    summary[
        "prepeak_head_max_torsoscale"
    ] = nanmax_safe(
        pre_peak[
            "head_displacement_torsoscale"
        ]
    )

    summary[
        "prepeak_left_ankle_max_torsoscale"
    ] = nanmax_safe(
        pre_peak[
            "left_ankle_displacement_torsoscale"
        ]
    )

    summary[
        "prepeak_right_ankle_max_torsoscale"
    ] = nanmax_safe(
        pre_peak[
            "right_ankle_displacement_torsoscale"
        ]
    )

    summary[
        "prepeak_left_knee_max_torsoscale"
    ] = nanmax_safe(
        pre_peak[
            "left_knee_displacement_torsoscale"
        ]
    )

    summary[
        "prepeak_right_knee_max_torsoscale"
    ] = nanmax_safe(
        pre_peak[
            "right_knee_displacement_torsoscale"
        ]
    )

    # ========================================================
    # FOLLOW THROUGH MOVEMENT
    # ========================================================

    follow_mask = phase_mask(
        metrics,
        [
            "FOLLOW_THROUGH",
        ]
    )

    follow = metrics[
        follow_mask
    ]

    summary[
        "follow_head_max_torsoscale"
    ] = nanmax_safe(
        follow[
            "head_displacement_torsoscale"
        ]
    )

    summary[
        "follow_left_ankle_max_torsoscale"
    ] = nanmax_safe(
        follow[
            "left_ankle_displacement_torsoscale"
        ]
    )

    summary[
        "follow_right_ankle_max_torsoscale"
    ] = nanmax_safe(
        follow[
            "right_ankle_displacement_torsoscale"
        ]
    )

    # ========================================================
    # QUALITY
    # ========================================================

    summary[
        "mean_pose_quality"
    ] = nanmean_safe(
        metrics[
            "pose_quality"
        ]
    )

    # ========================================================
    # PHASE TIMING
    # ========================================================

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
            -
            load_start
        )

        summary[
            "load_to_peak_sec"
        ] = (
            peak_start
            -
            load_start
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
            -
            swing_start
        )

        summary[
            "swing_to_peak_sec"
        ] = (
            peak_start
            -
            swing_start
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
        /
        "swing_metrics_v3_3_frames.csv"
    )

    summary_output = (
        output_dir
        /
        "swing_metrics_v3_3_summary.csv"
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
    # CONSOLE REPORT
    # ========================================================

    print()
    print("=" * 80)
    print(
        "SWING METRICS V3.3 COMPLETE"
    )
    print("=" * 80)

    print()
    print("BODY SCALE")
    print("-" * 80)

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

    if finite(
        median_torso_length
    ):

        print(
            f"Median torso length       : "
            f"{median_torso_length:.2f} px"
        )

    if finite(
        body_scale_px
    ):

        print(
            f"Normalization scale       : "
            f"{body_scale_px:.2f} px "
            f"(torso length)"
        )

    print()
    print("PHASE-RESTRICTED MOTION")
    print("-" * 80)

    if finite(
        left_peak
    ):

        print(
            f"Left wrist peak           : "
            f"{left_peak:.2f} px/frame "
            f"@ frame {left_peak_frame}"
        )

    if finite(
        right_peak
    ):

        print(
            f"Right wrist peak          : "
            f"{right_peak:.2f} px/frame "
            f"@ frame {right_peak_frame}"
        )

    if finite(
        hip_peak
    ):

        print(
            f"Hip center peak           : "
            f"{hip_peak:.2f} px/frame "
            f"@ frame {hip_peak_frame}"
        )

    if finite(
        shoulder_peak
    ):

        print(
            f"Shoulder center peak      : "
            f"{shoulder_peak:.2f} px/frame "
            f"@ frame {shoulder_peak_frame}"
        )

    print()
    print("PROJECTED ROTATION")
    print("-" * 80)

    shoulder_rot = summary[
        "max_abs_shoulder_rotation_from_load_deg"
    ]

    hip_rot = summary[
        "max_abs_hip_rotation_from_load_deg"
    ]

    sep = summary[
        "max_hip_shoulder_separation_deg"
    ]

    if finite(
        shoulder_load_baseline
    ):

        print(
            f"Shoulder LOAD baseline    : "
            f"{shoulder_load_baseline:.2f} deg"
        )

    if finite(
        hip_load_baseline
    ):

        print(
            f"Hip LOAD baseline         : "
            f"{hip_load_baseline:.2f} deg"
        )

    if finite(
        shoulder_rot
    ):

        print(
            f"Max shoulder deviation    : "
            f"{shoulder_rot:.2f} deg projected"
        )

    if finite(
        hip_rot
    ):

        print(
            f"Max hip deviation         : "
            f"{hip_rot:.2f} deg projected"
        )

    if finite(
        sep
    ):

        print(
            f"Max hip-shoulder sep.     : "
            f"{sep:.2f} deg projected "
            f"@ frame "
            f"{summary['max_separation_frame']}"
        )

    print()
    print("PRE-PEAK STABILITY")
    print("-" * 80)

    pre_head_px = summary[
        "prepeak_head_max_px"
    ]

    pre_head_norm = summary[
        "prepeak_head_max_torsoscale"
    ]

    pre_left_ankle = summary[
        "prepeak_left_ankle_max_torsoscale"
    ]

    pre_right_ankle = summary[
        "prepeak_right_ankle_max_torsoscale"
    ]

    if finite(
        pre_head_px
    ):

        print(
            f"Head movement             : "
            f"{pre_head_px:.2f} px"
        )

    if finite(
        pre_head_norm
    ):

        print(
            f"Head normalized           : "
            f"{pre_head_norm:.3f} torso lengths"
        )

    if finite(
        pre_left_ankle
    ):

        print(
            f"Left ankle movement       : "
            f"{pre_left_ankle:.3f} torso lengths"
        )

    if finite(
        pre_right_ankle
    ):

        print(
            f"Right ankle movement      : "
            f"{pre_right_ankle:.3f} torso lengths"
        )

    print()
    print("FOLLOW-THROUGH MOVEMENT")
    print("-" * 80)

    follow_head = summary[
        "follow_head_max_torsoscale"
    ]

    follow_left = summary[
        "follow_left_ankle_max_torsoscale"
    ]

    follow_right = summary[
        "follow_right_ankle_max_torsoscale"
    ]

    if finite(
        follow_head
    ):

        print(
            f"Head max                 : "
            f"{follow_head:.3f} torso lengths"
        )

    if finite(
        follow_left
    ):

        print(
            f"Left ankle max           : "
            f"{follow_left:.3f} torso lengths"
        )

    if finite(
        follow_right
    ):

        print(
            f"Right ankle max          : "
            f"{follow_right:.3f} torso lengths"
        )

    print()
    print("SWING TIMING")
    print("-" * 80)

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
    print("-" * 80)

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
    print("-" * 80)

    print(
        "Motion uses centered 3-frame 2D displacement."
    )

    print(
        "Normalization uses median torso length."
    )

    print(
        "Pre-peak stability excludes follow-through."
    )

    print(
        "Follow-through displacement is reported separately."
    )

    print(
        "These remain comparative 2D vision metrics."
    )

    print()
    print("OUTPUT FILES")
    print("-" * 80)

    print(
        f"Frame metrics : {frame_output.resolve()}"
    )

    print(
        f"Summary       : {summary_output.resolve()}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()