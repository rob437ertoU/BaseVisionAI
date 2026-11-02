from pathlib import Path
from collections import deque
import argparse

import cv2
import numpy as np
import pandas as pd


# ============================================================
# BASEVISIONAI - FULL ANALYSIS VISUALIZATION V3
# ============================================================
#
# Integrates:
#   - Pose Smoothing V2
#   - Swing Analysis V2.2
#   - Ball Tracking V3.2
#   - Swing Metrics V3.3
#
# Visual principles:
#   - YOLO / RECOVERED = real detector observations
#   - KALMAN = prediction only
#   - LOST = no active ball track
#   - Ball trail uses ONLY real detector observations
#   - Swing metrics are clearly labeled as projected 2D
#
# IMPORTANT:
#   - No strike / ball / hit classification here.
#   - No mph / m/s claims.
#   - No 3D biomechanics claims.
#
# ============================================================


# ------------------------------------------------------------
# COCO 17 KEYPOINTS
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Skeleton connections
# ------------------------------------------------------------

SKELETON = [
    ("left_shoulder", "right_shoulder"),

    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),

    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),

    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),

    ("left_hip", "right_hip"),

    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),

    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Full Analysis Visualization V3"
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Input video"
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
        "--ball-csv",
        required=True,
        help="Ball Tracking V3.2 CSV"
    )

    parser.add_argument(
        "--metrics-frames-csv",
        required=True,
        help="Swing Metrics V3.3 frame-level CSV"
    )

    parser.add_argument(
        "--metrics-summary-csv",
        required=True,
        help="Swing Metrics V3.3 summary CSV"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output video path"
    )

    parser.add_argument(
        "--trail-length",
        type=int,
        default=18,
        help="Maximum real-observation ball trail points"
    )

    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def finite(value):

    try:
        return np.isfinite(float(value))
    except Exception:
        return False


def safe_float(
    value,
    default=np.nan
):

    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


def safe_int(
    value,
    default=None
):

    try:
        value = float(value)

        if np.isfinite(value):
            return int(round(value))

    except Exception:
        pass

    return default


def safe_text(
    value,
    default="N/A"
):

    if value is None:
        return default

    if pd.isna(value):
        return default

    text = str(value).strip()

    if not text:
        return default

    return text


def get_frame_row(
    lookup,
    frame_number
):

    return lookup.get(
        int(frame_number)
    )


def make_lookup(
    df
):

    if "frame" not in df.columns:

        raise ValueError(
            "CSV requires a 'frame' column."
        )

    df = df.copy()

    df[
        "frame"
    ] = pd.to_numeric(
        df["frame"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["frame"]
    )

    df[
        "frame"
    ] = df[
        "frame"
    ].astype(
        int
    )

    df = df.drop_duplicates(
        subset=["frame"],
        keep="first"
    )

    return {
        int(row["frame"]): row
        for _, row in df.iterrows()
    }


# ============================================================
# COORDINATE COLUMN HANDLING
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


def build_pose_column_map(
    pose_df
):

    mapping = {}

    for kp in KEYPOINTS:

        mapping[
            f"{kp}_x"
        ] = find_coordinate_column(
            pose_df,
            kp,
            "x"
        )

        mapping[
            f"{kp}_y"
        ] = find_coordinate_column(
            pose_df,
            kp,
            "y"
        )

    return mapping


def get_pose_point(
    pose_row,
    mapping,
    keypoint
):

    if pose_row is None:
        return None

    x_col = mapping.get(
        f"{keypoint}_x"
    )

    y_col = mapping.get(
        f"{keypoint}_y"
    )

    if (
        x_col is None
        or y_col is None
    ):

        return None

    x = safe_float(
        pose_row.get(
            x_col,
            np.nan
        )
    )

    y = safe_float(
        pose_row.get(
            y_col,
            np.nan
        )
    )

    if not (
        finite(x)
        and finite(y)
    ):

        return None

    return (
        int(round(x)),
        int(round(y))
    )


# ============================================================
# DRAWING HELPERS
# ============================================================

def draw_text(
    image,
    text,
    origin,
    scale=0.55,
    thickness=1,
    color=(255, 255, 255)
):

    cv2.putText(
        image,
        str(text),
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_panel(
    image,
    x1,
    y1,
    x2,
    y2,
    alpha=0.72
):

    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (15, 15, 15),
        -1
    )

    cv2.addWeighted(
        overlay,
        alpha,
        image,
        1.0 - alpha,
        0,
        image
    )

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (180, 180, 180),
        1
    )


def draw_pose(
    frame,
    pose_row,
    mapping
):

    if pose_row is None:
        return

    points = {}

    for kp in KEYPOINTS:

        point = get_pose_point(
            pose_row,
            mapping,
            kp
        )

        points[
            kp
        ] = point

    # Skeleton lines
    for a, b in SKELETON:

        pa = points.get(a)
        pb = points.get(b)

        if (
            pa is None
            or pb is None
        ):
            continue

        cv2.line(
            frame,
            pa,
            pb,
            (0, 220, 255),
            3,
            cv2.LINE_AA
        )

    # Keypoints
    for kp, point in points.items():

        if point is None:
            continue

        cv2.circle(
            frame,
            point,
            4,
            (255, 255, 255),
            -1,
            cv2.LINE_AA
        )

        cv2.circle(
            frame,
            point,
            6,
            (0, 180, 255),
            1,
            cv2.LINE_AA
        )


# ============================================================
# BALL VISUALIZATION
# ============================================================

def normalize_ball_state(
    ball_row
):

    if ball_row is None:
        return "NONE"

    state = safe_text(
        ball_row.get(
            "state",
            ""
        ),
        default=""
    ).upper()

    source = safe_text(
        ball_row.get(
            "detection_source",
            ""
        ),
        default=""
    ).upper()

    if state in {
        "TRACKING",
        "RECOVERED",
        "KALMAN",
        "LOST",
    }:

        return state

    if "RECOVER" in source:
        return "RECOVERED"

    if "YOLO" in source:
        return "TRACKING"

    if "KALMAN" in source:
        return "KALMAN"

    return state if state else "NONE"


def get_ball_real_point(
    ball_row
):

    if ball_row is None:
        return None

    x = safe_float(
        ball_row.get(
            "observed_x",
            np.nan
        )
    )

    y = safe_float(
        ball_row.get(
            "observed_y",
            np.nan
        )
    )

    if finite(x) and finite(y):

        return (
            int(round(x)),
            int(round(y))
        )

    return None


def get_ball_filtered_point(
    ball_row
):

    if ball_row is None:
        return None

    x = safe_float(
        ball_row.get(
            "filtered_x",
            np.nan
        )
    )

    y = safe_float(
        ball_row.get(
            "filtered_y",
            np.nan
        )
    )

    if finite(x) and finite(y):

        return (
            int(round(x)),
            int(round(y))
        )

    return None


def draw_ball(
    frame,
    ball_row,
    trail
):

    state = normalize_ball_state(
        ball_row
    )

    label = "NO ACTIVE BALL TRACK"

    # --------------------------------------------------------
    # TRACKING = real YOLO observation
    # --------------------------------------------------------

    if state == "TRACKING":

        real_point = get_ball_real_point(
            ball_row
        )

        if real_point is not None:

            trail.append(
                real_point
            )

            cv2.circle(
                frame,
                real_point,
                11,
                (0, 255, 0),
                3,
                cv2.LINE_AA
            )

            cv2.circle(
                frame,
                real_point,
                3,
                (255, 255, 255),
                -1
            )

        label = "REAL YOLO OBSERVATION"

    # --------------------------------------------------------
    # RECOVERED = also real detector output
    # --------------------------------------------------------

    elif state == "RECOVERED":

        real_point = get_ball_real_point(
            ball_row
        )

        if real_point is not None:

            trail.append(
                real_point
            )

            cv2.circle(
                frame,
                real_point,
                11,
                (255, 200, 0),
                3,
                cv2.LINE_AA
            )

        label = "REAL RECOVERED OBSERVATION"

    # --------------------------------------------------------
    # KALMAN = prediction only
    # --------------------------------------------------------

    elif state == "KALMAN":

        point = get_ball_filtered_point(
            ball_row
        )

        if point is not None:

            x, y = point

            cv2.line(
                frame,
                (x - 10, y - 10),
                (x + 10, y + 10),
                (0, 255, 255),
                3,
                cv2.LINE_AA
            )

            cv2.line(
                frame,
                (x - 10, y + 10),
                (x + 10, y - 10),
                (0, 255, 255),
                3,
                cv2.LINE_AA
            )

        label = "KALMAN PREDICTION ONLY"

    # --------------------------------------------------------
    # LOST = clear trail
    # --------------------------------------------------------

    elif state == "LOST":

        trail.clear()

        label = "NO ACTIVE BALL TRACK"

    # --------------------------------------------------------
    # Draw REAL-observation trail only
    # --------------------------------------------------------

    if len(trail) >= 2:

        points = list(
            trail
        )

        for i in range(
            1,
            len(points)
        ):

            cv2.line(
                frame,
                points[i - 1],
                points[i],
                (0, 220, 0),
                2,
                cv2.LINE_AA
            )

    return state, label


# ============================================================
# SWING PHASE
# ============================================================

def detect_phase_column(
    swing_df
):

    for candidate in [
        "phase",
        "swing_phase",
        "phase_name",
        "predicted_phase",
    ]:

        if candidate in swing_df.columns:
            return candidate

    return None


def normalize_phase(
    phase
):

    if phase is None:
        return "UNKNOWN"

    if pd.isna(
        phase
    ):
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


# ============================================================
# METRIC ACCESS
# ============================================================

def metric_value(
    row,
    key,
    default=np.nan
):

    if row is None:
        return default

    return safe_float(
        row.get(
            key,
            default
        ),
        default=default
    )


def summary_value(
    summary_row,
    key,
    default=np.nan
):

    if summary_row is None:
        return default

    return safe_float(
        summary_row.get(
            key,
            default
        ),
        default=default
    )


# ============================================================
# METRIC PANEL
# ============================================================

def draw_metrics_panel(
    frame,
    frame_number,
    phase,
    metrics_row,
    summary_row,
    pose_quality
):

    h, w = frame.shape[
        :2
    ]

    panel_w = min(
        520,
        int(
            w * 0.30
        )
    )

    x1 = (
        w
        - panel_w
        - 20
    )

    y1 = 105

    x2 = (
        w
        - 20
    )

    y2 = min(
        h - 20,
        875
    )

    draw_panel(
        frame,
        x1,
        y1,
        x2,
        y2
    )

    tx = x1 + 20
    y = y1 + 32

    draw_text(
        frame,
        "BASEVISION AI - SWING ANALYSIS",
        (tx, y),
        scale=0.65,
        thickness=2
    )

    y += 34

    draw_text(
        frame,
        "2D PROJECTED / COMPARATIVE METRICS",
        (tx, y),
        scale=0.48,
        thickness=1
    )

    y += 22

    draw_text(
        frame,
        "NOT CALIBRATED BIOMECHANICS",
        (tx, y),
        scale=0.46,
        thickness=1
    )

    y += 34

    # --------------------------------------------------------
    # PHASE
    # --------------------------------------------------------

    draw_text(
        frame,
        f"PHASE: {phase}",
        (tx, y),
        scale=0.62,
        thickness=2
    )

    y += 34

    # --------------------------------------------------------
    # CURRENT MOTION
    # --------------------------------------------------------

    draw_text(
        frame,
        "MOTION - CURRENT FRAME",
        (tx, y),
        scale=0.50,
        thickness=2
    )

    y += 25

    left_wrist = metric_value(
        metrics_row,
        "left_wrist_speed_robust_px_frame"
    )

    right_wrist = metric_value(
        metrics_row,
        "right_wrist_speed_robust_px_frame"
    )

    hip_speed = metric_value(
        metrics_row,
        "hip_center_speed_robust_px_frame"
    )

    shoulder_speed = metric_value(
        metrics_row,
        "shoulder_center_speed_robust_px_frame"
    )

    current_motion_lines = [
        (
            "L Wrist",
            left_wrist,
            "px/f"
        ),
        (
            "R Wrist",
            right_wrist,
            "px/f"
        ),
        (
            "Hip ctr",
            hip_speed,
            "px/f"
        ),
        (
            "Shoulder ctr",
            shoulder_speed,
            "px/f"
        ),
    ]

    for label, value, unit in current_motion_lines:

        value_text = (
            f"{value:.2f}"
            if finite(value)
            else "N/A"
        )

        draw_text(
            frame,
            f"{label:<14} {value_text:>7} {unit}",
            (tx, y),
            scale=0.48
        )

        y += 23

    y += 10

    # --------------------------------------------------------
    # CURRENT ROTATION
    # --------------------------------------------------------

    draw_text(
        frame,
        "ROTATION - PROJECTED 2D",
        (tx, y),
        scale=0.50,
        thickness=2
    )

    y += 25

    shoulder_rot = metric_value(
        metrics_row,
        "shoulder_rotation_from_load_deg"
    )

    hip_rot = metric_value(
        metrics_row,
        "hip_rotation_from_load_deg"
    )

    separation = metric_value(
        metrics_row,
        "hip_shoulder_separation_projected_deg"
    )

    rotation_lines = [
        (
            "Shoulder",
            shoulder_rot
        ),
        (
            "Hip",
            hip_rot
        ),
        (
            "Separation",
            separation
        ),
    ]

    for label, value in rotation_lines:

        value_text = (
            f"{value:+.1f}"
            if finite(value)
            else "N/A"
        )

        draw_text(
            frame,
            f"{label:<14} {value_text:>7} deg",
            (tx, y),
            scale=0.48
        )

        y += 23

    y += 10

    # --------------------------------------------------------
    # STABILITY
    # --------------------------------------------------------

    draw_text(
        frame,
        "STABILITY FROM SETUP",
        (tx, y),
        scale=0.50,
        thickness=2
    )

    y += 25

    head = metric_value(
        metrics_row,
        "head_displacement_torsoscale"
    )

    left_ankle = metric_value(
        metrics_row,
        "left_ankle_displacement_torsoscale"
    )

    right_ankle = metric_value(
        metrics_row,
        "right_ankle_displacement_torsoscale"
    )

    stability_lines = [
        (
            "Head",
            head
        ),
        (
            "L ankle",
            left_ankle
        ),
        (
            "R ankle",
            right_ankle
        ),
    ]

    for label, value in stability_lines:

        value_text = (
            f"{value:.3f}"
            if finite(value)
            else "N/A"
        )

        draw_text(
            frame,
            f"{label:<14} {value_text:>7} torso",
            (tx, y),
            scale=0.48
        )

        y += 23

    y += 10

    # --------------------------------------------------------
    # SWING PROFILE SUMMARY
    # --------------------------------------------------------

    draw_text(
        frame,
        "SWING PROFILE",
        (tx, y),
        scale=0.50,
        thickness=2
    )

    y += 25

    peak_r = summary_value(
        summary_row,
        "right_wrist_peak_swing_px_frame"
    )

    peak_r_frame = safe_int(
        summary_row.get(
            "right_wrist_peak_swing_frame",
            np.nan
        )
        if summary_row is not None
        else np.nan
    )

    peak_l = summary_value(
        summary_row,
        "left_wrist_peak_swing_px_frame"
    )

    peak_l_frame = safe_int(
        summary_row.get(
            "left_wrist_peak_swing_frame",
            np.nan
        )
        if summary_row is not None
        else np.nan
    )

    load_to_peak = summary_value(
        summary_row,
        "load_to_peak_sec"
    )

    swing_to_peak = summary_value(
        summary_row,
        "swing_to_peak_sec"
    )

    profile_lines = [
        (
            "Peak R wrist",
            (
                f"{peak_r:.2f} @ F{peak_r_frame}"
                if finite(peak_r)
                and peak_r_frame is not None
                else "N/A"
            )
        ),
        (
            "Peak L wrist",
            (
                f"{peak_l:.2f} @ F{peak_l_frame}"
                if finite(peak_l)
                and peak_l_frame is not None
                else "N/A"
            )
        ),
        (
            "Load->Peak",
            (
                f"{load_to_peak:.3f} s"
                if finite(load_to_peak)
                else "N/A"
            )
        ),
        (
            "Swing->Peak",
            (
                f"{swing_to_peak:.3f} s"
                if finite(swing_to_peak)
                else "N/A"
            )
        ),
    ]

    for label, value_text in profile_lines:

        draw_text(
            frame,
            f"{label:<14} {value_text}",
            (tx, y),
            scale=0.47
        )

        y += 23

    y += 10

    # --------------------------------------------------------
    # QUALITY
    # --------------------------------------------------------

    draw_text(
        frame,
        "POSE QUALITY",
        (tx, y),
        scale=0.50,
        thickness=2
    )

    y += 25

    quality_text = (
        f"{pose_quality:.3f}"
        if finite(pose_quality)
        else "N/A"
    )

    draw_text(
        frame,
        f"Mean/current quality: {quality_text}",
        (tx, y),
        scale=0.47
    )

    y += 28

    draw_text(
        frame,
        f"Frame: {frame_number}",
        (tx, y),
        scale=0.46
    )


# ============================================================
# TOP STATUS PANEL
# ============================================================

def draw_top_panel(
    frame,
    frame_number,
    total_frames,
    phase,
    ball_label,
    ball_state
):

    h, w = frame.shape[
        :2
    ]

    draw_panel(
        frame,
        20,
        20,
        w - 20,
        90
    )

    draw_text(
        frame,
        f"FRAME {frame_number}/{total_frames}",
        (40, 50),
        scale=0.65,
        thickness=2
    )

    draw_text(
        frame,
        f"SWING PHASE: {phase}",
        (320, 50),
        scale=0.65,
        thickness=2
    )

    draw_text(
        frame,
        f"BALL: {ball_label}",
        (720, 50),
        scale=0.58,
        thickness=2
    )

    draw_text(
        frame,
        f"STATE: {ball_state}",
        (720, 76),
        scale=0.46,
        thickness=1
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    video_path = Path(
        args.video
    )

    pose_path = Path(
        args.pose_csv
    )

    swing_path = Path(
        args.swing_csv
    )

    ball_path = Path(
        args.ball_csv
    )

    metrics_frames_path = Path(
        args.metrics_frames_csv
    )

    metrics_summary_path = Path(
        args.metrics_summary_csv
    )

    output_path = Path(
        args.output
    )

    for path in [
        video_path,
        pose_path,
        swing_path,
        ball_path,
        metrics_frames_path,
        metrics_summary_path,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Input not found:\n{path}"
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # LOAD CSVs
    # ========================================================

    pose_df = pd.read_csv(
        pose_path
    )

    swing_df = pd.read_csv(
        swing_path
    )

    ball_df = pd.read_csv(
        ball_path
    )

    metrics_df = pd.read_csv(
        metrics_frames_path
    )

    summary_df = pd.read_csv(
        metrics_summary_path
    )

    summary_row = (
        summary_df.iloc[0]
        if len(summary_df) > 0
        else None
    )

    pose_lookup = make_lookup(
        pose_df
    )

    swing_lookup = make_lookup(
        swing_df
    )

    ball_lookup = make_lookup(
        ball_df
    )

    metrics_lookup = make_lookup(
        metrics_df
    )

    pose_mapping = build_pose_column_map(
        pose_df
    )

    phase_column = detect_phase_column(
        swing_df
    )

    # ========================================================
    # VIDEO
    # ========================================================

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n{video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:

        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (
            width,
            height
        )
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            f"Could not create output video:\n{output_path}"
        )

    trail = deque(
        maxlen=max(
            2,
            int(
                args.trail_length
            )
        )
    )

    frame_number = 0

    state_counts = {
        "TRACKING": 0,
        "RECOVERED": 0,
        "KALMAN": 0,
        "LOST": 0,
        "NONE": 0,
    }

    # ========================================================
    # PROCESS
    # ========================================================

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        frame_number += 1

        pose_row = get_frame_row(
            pose_lookup,
            frame_number
        )

        swing_row = get_frame_row(
            swing_lookup,
            frame_number
        )

        ball_row = get_frame_row(
            ball_lookup,
            frame_number
        )

        metrics_row = get_frame_row(
            metrics_lookup,
            frame_number
        )

        # ----------------------------------------------------
        # Swing phase
        # ----------------------------------------------------

        if (
            swing_row is not None
            and phase_column is not None
        ):

            phase = normalize_phase(
                swing_row.get(
                    phase_column,
                    "UNKNOWN"
                )
            )

        elif metrics_row is not None:

            phase = normalize_phase(
                metrics_row.get(
                    "phase_normalized",
                    "UNKNOWN"
                )
            )

        else:

            phase = "UNKNOWN"

        # ----------------------------------------------------
        # Pose
        # ----------------------------------------------------

        draw_pose(
            frame,
            pose_row,
            pose_mapping
        )

        # ----------------------------------------------------
        # Ball
        # ----------------------------------------------------

        ball_state, ball_label = draw_ball(
            frame,
            ball_row,
            trail
        )

        if ball_state not in state_counts:
            state_counts[
                ball_state
            ] = 0

        state_counts[
            ball_state
        ] += 1

        # ----------------------------------------------------
        # Pose quality
        # ----------------------------------------------------

        pose_quality = np.nan

        if metrics_row is not None:

            pose_quality = metric_value(
                metrics_row,
                "pose_quality"
            )

        if not finite(
            pose_quality
        ):

            for candidate in [
                "pose_quality",
                "quality",
                "mean_confidence",
                "avg_confidence",
            ]:

                if (
                    pose_row is not None
                    and candidate in pose_row.index
                ):

                    pose_quality = safe_float(
                        pose_row.get(
                            candidate,
                            np.nan
                        )
                    )

                    if finite(
                        pose_quality
                    ):
                        break

        # ----------------------------------------------------
        # Panels
        # ----------------------------------------------------

        draw_top_panel(
            frame,
            frame_number,
            total_frames,
            phase,
            ball_label,
            ball_state
        )

        draw_metrics_panel(
            frame,
            frame_number,
            phase,
            metrics_row,
            summary_row,
            pose_quality
        )

        # ----------------------------------------------------
        # Footer
        # ----------------------------------------------------

        draw_text(
            frame,
            "BaseVisionAI | 2D computer-vision analysis | "
            "Not calibrated physical biomechanics",
            (
                30,
                height - 25
            ),
            scale=0.45,
            thickness=1
        )

        writer.write(
            frame
        )

    # ========================================================
    # CLOSE
    # ========================================================

    cap.release()
    writer.release()

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 80)
    print(
        "FULL ANALYSIS VISUALIZATION V3 COMPLETE"
    )
    print("=" * 80)

    print(
        f"Frames processed          : {frame_number}"
    )

    print(
        f"Video FPS                 : {fps:.2f}"
    )

    print(
        f"Resolution                : {width}x{height}"
    )

    print(
        f"Pose rows                 : {len(pose_df)}"
    )

    print(
        f"Swing rows                : {len(swing_df)}"
    )

    print(
        f"Metrics rows              : {len(metrics_df)}"
    )

    print(
        f"Real YOLO ball frames     : "
        f"{state_counts.get('TRACKING', 0)}"
    )

    print(
        f"Recovered ball frames     : "
        f"{state_counts.get('RECOVERED', 0)}"
    )

    print(
        f"Kalman prediction frames  : "
        f"{state_counts.get('KALMAN', 0)}"
    )

    print(
        f"LOST frames               : "
        f"{state_counts.get('LOST', 0)}"
    )

    print(
        f"No-state frames            : "
        f"{state_counts.get('NONE', 0)}"
    )

    print()
    print(
        f"Output : {output_path.resolve()}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()