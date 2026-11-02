from pathlib import Path
from collections import deque
import argparse

import cv2
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_VIDEO = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "baseball_train10.mp4"
)

DEFAULT_SWING_CSV = (
    PROJECT_ROOT
    / "results"
    / "swing_analysis_v2"
    / "baseball_train10_batter_pose_v2_smoothed_swing_analysis_v2_2.csv"
)

DEFAULT_BALL_CSV = (
    PROJECT_ROOT
    / "results"
    / "ball_tracking_v3_1"
    / "baseball_train10"
    / "ball_tracking_v3_1.csv"
)

DEFAULT_FUSION_CSV = (
    PROJECT_ROOT
    / "results"
    / "contact_fusion_v2_1"
    / "contact_fusion_v2_1.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "contact_fusion_v2_1"
)


# ============================================================
# DISPLAY CONFIGURATION
# ============================================================

KEYPOINT_CONF_THRESHOLD = 0.30
BALL_TRAIL_LENGTH = 12

# OpenCV = BGR
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
BLUE = (255, 150, 0)
CYAN = (255, 255, 0)
GRAY = (170, 170, 170)
PURPLE = (255, 0, 255)


PHASE_COLORS = {
    "SETUP": GRAY,
    "LOAD": YELLOW,
    "SWING": ORANGE,
    "PEAK_SWING": RED,
    "FOLLOW_THROUGH": BLUE,
}


# ============================================================
# COCO POSE
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
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "BaseVisionAI - "
            "Contact Fusion Visualization V2"
        )
    )

    parser.add_argument(
        "--video",
        type=str,
        default=str(DEFAULT_VIDEO)
    )

    parser.add_argument(
        "--swing-csv",
        type=str,
        default=str(DEFAULT_SWING_CSV)
    )

    parser.add_argument(
        "--ball-csv",
        type=str,
        default=str(DEFAULT_BALL_CSV)
    )

    parser.add_argument(
        "--fusion-csv",
        type=str,
        default=str(DEFAULT_FUSION_CSV)
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR)
    )

    parser.add_argument(
        "--trail-length",
        type=int,
        default=BALL_TRAIL_LENGTH
    )

    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=np.nan):

    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


def safe_int(value, default=None):

    try:

        if pd.isna(value):
            return default

        return int(value)

    except Exception:
        return default


def normalize_bool(value):

    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y"
    }


def draw_text_box(
    frame,
    text,
    x,
    y,
    font_scale=0.55,
    text_color=WHITE,
    background_color=BLACK,
    thickness=1,
    padding=5
):

    font = cv2.FONT_HERSHEY_SIMPLEX

    (w, h), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    cv2.rectangle(
        frame,
        (
            x - padding,
            y - h - padding
        ),
        (
            x + w + padding,
            y + baseline + padding
        ),
        background_color,
        -1
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# POSE
# ============================================================

def get_pose_point(row, name):

    x_col = f"{name}_x_smooth"
    y_col = f"{name}_y_smooth"

    valid_col = f"{name}_valid"
    conf_col = f"{name}_conf_raw"

    if x_col not in row.index:
        return None

    if y_col not in row.index:
        return None

    x = safe_float(
        row.get(
            x_col,
            np.nan
        )
    )

    y = safe_float(
        row.get(
            y_col,
            np.nan
        )
    )

    valid = safe_int(
        row.get(
            valid_col,
            1
        ),
        1
    )

    conf = safe_float(
        row.get(
            conf_col,
            1.0
        ),
        1.0
    )

    interpolated = safe_int(
        row.get(
            f"{name}_interpolated",
            0
        ),
        0
    )

    if not (
        np.isfinite(x)
        and np.isfinite(y)
    ):
        return None

    if valid != 1:
        return None

    if (
        interpolated != 1
        and conf < KEYPOINT_CONF_THRESHOLD
    ):
        return None

    return (
        int(round(x)),
        int(round(y))
    )


def draw_pose(frame, row):

    points = {}

    for name in KEYPOINT_NAMES:

        point = get_pose_point(
            row,
            name
        )

        if point is not None:

            points[name] = point

    # Skeleton
    for a, b in SKELETON:

        if (
            a in points
            and b in points
        ):

            cv2.line(
                frame,
                points[a],
                points[b],
                GREEN,
                3,
                cv2.LINE_AA
            )

    # Keypoints
    for name, point in points.items():

        if "wrist" in name:

            color = RED
            radius = 6

        elif "elbow" in name:

            color = YELLOW
            radius = 5

        else:

            color = WHITE
            radius = 4

        cv2.circle(
            frame,
            point,
            radius,
            color,
            -1,
            cv2.LINE_AA
        )


# ============================================================
# BALL
# ============================================================

def get_ball_position(ball_row):

    if ball_row is None:
        return None, None

    yolo_detected = normalize_bool(
        ball_row.get(
            "yolo_detected",
            False
        )
    )

    state = str(
        ball_row.get(
            "state",
            "WAITING"
        )
    )

    if yolo_detected:

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

        if (
            np.isfinite(x)
            and np.isfinite(y)
        ):

            return (
                int(round(x)),
                int(round(y))
            ), "YOLO"

    if state in {
        "KALMAN",
        "LOST"
    }:

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

        if (
            np.isfinite(x)
            and np.isfinite(y)
        ):

            return (
                int(round(x)),
                int(round(y))
            ), "KALMAN"

    return None, None


def draw_ball(
    frame,
    ball_row,
    trail
):

    position, source = get_ball_position(
        ball_row
    )

    if position is None:
        return

    # -----------------------------------------
    # Add to trail
    # -----------------------------------------

    trail.append(
        (
            position,
            source
        )
    )

    # -----------------------------------------
    # Trail
    # -----------------------------------------

    previous = None

    for point, trail_source in trail:

        if previous is not None:

            cv2.line(
                frame,
                previous,
                point,
                CYAN,
                2,
                cv2.LINE_AA
            )

        previous = point

    # -----------------------------------------
    # Current point
    # -----------------------------------------

    if source == "YOLO":

        cv2.circle(
            frame,
            position,
            10,
            GREEN,
            3,
            cv2.LINE_AA
        )

        cv2.circle(
            frame,
            position,
            3,
            GREEN,
            -1,
            cv2.LINE_AA
        )

        label = "BALL YOLO"

    else:

        cv2.circle(
            frame,
            position,
            10,
            YELLOW,
            2,
            cv2.LINE_AA
        )

        cv2.line(
            frame,
            (
                position[0] - 7,
                position[1] - 7
            ),
            (
                position[0] + 7,
                position[1] + 7
            ),
            YELLOW,
            2
        )

        cv2.line(
            frame,
            (
                position[0] + 7,
                position[1] - 7
            ),
            (
                position[0] - 7,
                position[1] + 7
            ),
            YELLOW,
            2
        )

        label = "BALL KALMAN"

    draw_text_box(
        frame,
        label,
        position[0] + 15,
        position[1] - 15,
        font_scale=0.45,
        background_color=BLACK
    )


# ============================================================
# TIMELINE
# ============================================================

def draw_timeline(
    frame,
    current_frame,
    total_frames,
    swing_start,
    swing_end,
    peak_start,
    peak_end,
    candidate_frame,
    swing_peak_frame,
    last_yolo_frame,
    first_lost_frame
):

    h, w = frame.shape[:2]

    x_start = 80
    x_end = w - 80

    y = h - 45

    timeline_width = (
        x_end - x_start
    )

    def frame_to_x(frame_number):

        return (
            x_start
            + int(
                (
                    (frame_number - 1)
                    / max(
                        total_frames - 1,
                        1
                    )
                )
                * timeline_width
            )
        )

    # Base
    cv2.line(
        frame,
        (x_start, y),
        (x_end, y),
        GRAY,
        8
    )

    # SWING
    swing_x1 = frame_to_x(
        swing_start
    )

    swing_x2 = frame_to_x(
        swing_end
    )

    cv2.line(
        frame,
        (swing_x1, y),
        (swing_x2, y),
        ORANGE,
        10
    )

    # PEAK
    peak_x1 = frame_to_x(
        peak_start
    )

    peak_x2 = frame_to_x(
        peak_end
    )

    cv2.line(
        frame,
        (peak_x1, y),
        (peak_x2, y),
        RED,
        10
    )

    # Last YOLO
    if last_yolo_frame is not None:

        x = frame_to_x(
            last_yolo_frame
        )

        cv2.circle(
            frame,
            (x, y),
            8,
            GREEN,
            -1
        )

    # Candidate
    if candidate_frame is not None:

        x = frame_to_x(
            candidate_frame
        )

        cv2.drawMarker(
            frame,
            (x, y),
            PURPLE,
            cv2.MARKER_STAR,
            22,
            3
        )

    # Swing peak
    if swing_peak_frame is not None:

        x = frame_to_x(
            swing_peak_frame
        )

        cv2.drawMarker(
            frame,
            (x, y),
            WHITE,
            cv2.MARKER_TRIANGLE_UP,
            20,
            2
        )

    # LOST
    if first_lost_frame is not None:

        x = frame_to_x(
            first_lost_frame
        )

        cv2.drawMarker(
            frame,
            (x, y),
            RED,
            cv2.MARKER_TILTED_CROSS,
            18,
            2
        )

    # Current position
    current_x = frame_to_x(
        current_frame
    )

    cv2.line(
        frame,
        (
            current_x,
            y - 18
        ),
        (
            current_x,
            y + 18
        ),
        WHITE,
        3
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    video_path = Path(
        args.video
    ).resolve()

    swing_path = Path(
        args.swing_csv
    ).resolve()

    ball_path = Path(
        args.ball_csv
    ).resolve()

    fusion_path = Path(
        args.fusion_csv
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    # ========================================================
    # VALIDATE
    # ========================================================

    required_paths = [
        video_path,
        swing_path,
        ball_path,
        fusion_path
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n"
                f"{path}"
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_video = (
        output_dir
        / (
            f"{video_path.stem}"
            "_contact_fusion_v2.mp4"
        )
    )

    # ========================================================
    # LOAD CSV
    # ========================================================

    swing_df = pd.read_csv(
        swing_path
    )

    ball_df = pd.read_csv(
        ball_path
    )

    fusion_df = pd.read_csv(
        fusion_path
    )

    if len(fusion_df) == 0:

        raise RuntimeError(
            "Fusion CSV is empty."
        )

    fusion = fusion_df.iloc[0]

    # Maps by 1-based video frame
    swing_map = {
        int(row["frame"]): row
        for _, row
        in swing_df.iterrows()
    }

    ball_map = {
        int(row["frame"]): row
        for _, row
        in ball_df.iterrows()
    }

    # ========================================================
    # FUSION VARIABLES
    # ========================================================

    classification = str(
        fusion[
            "event_classification"
        ]
    )

    evidence_quality = str(
        fusion[
            "evidence_quality"
        ]
    )

    fusion_score = safe_float(
        fusion[
            "fusion_evidence_score"
        ],
        0.0
    )

    candidate_frame = safe_int(
        fusion[
            "candidate_event_frame"
        ]
    )

    swing_peak_frame = safe_int(
        fusion[
            "swing_peak_frame"
        ]
    )

    swing_start = safe_int(
        fusion[
            "swing_window_start"
        ]
    )

    swing_end = safe_int(
        fusion[
            "swing_window_end"
        ]
    )

    peak_start = safe_int(
        fusion[
            "peak_window_start"
        ]
    )

    peak_end = safe_int(
        fusion[
            "peak_window_end"
        ]
    )

    last_yolo_frame = safe_int(
        fusion[
            "last_yolo_frame"
        ]
    )

    first_lost_frame = safe_int(
        fusion[
            "first_lost_frame"
        ]
    )

    trajectory_status = str(
        fusion[
            "trajectory_status"
        ]
    )

    # ========================================================
    # VIDEO
    # ========================================================

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n"
            f"{video_path}"
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

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_video),
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
            "Could not initialize output video."
        )

    # ========================================================
    # BALL TRAIL
    # ========================================================

    trail = deque(
        maxlen=args.trail_length
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 78)
    print(
        "BaseVisionAI - "
        "Contact Fusion Visualization V2"
    )
    print("=" * 78)

    print(
        f"Video                : "
        f"{video_path}"
    )

    print(
        f"Frames               : "
        f"{total_frames}"
    )

    print(
        f"FPS                  : "
        f"{fps:.2f}"
    )

    print(
        f"Candidate event      : "
        f"{candidate_frame}"
    )

    print(
        f"Swing peak           : "
        f"{swing_peak_frame}"
    )

    print(
        f"Last YOLO ball       : "
        f"{last_yolo_frame}"
    )

    print(
        f"First LOST           : "
        f"{first_lost_frame}"
    )

    print(
        f"Evidence quality     : "
        f"{evidence_quality}"
    )

    print("=" * 78)

    # ========================================================
    # FRAME LOOP
    # ========================================================

    frame_number = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        swing_row = swing_map.get(
            frame_number
        )

        ball_row = ball_map.get(
            frame_number
        )

        # ====================================================
        # POSE
        # ====================================================

        if swing_row is not None:

            draw_pose(
                frame,
                swing_row
            )

        # ====================================================
        # BALL
        # ====================================================

        if ball_row is not None:

            draw_ball(
                frame,
                ball_row,
                trail
            )

        # ====================================================
        # SWING INFO
        # ====================================================

        if swing_row is not None:

            phase = str(
                swing_row.get(
                    "swing_phase",
                    "UNKNOWN"
                )
            )

            phase_color = (
                PHASE_COLORS.get(
                    phase,
                    WHITE
                )
            )

            draw_text_box(
                frame,
                f"SWING PHASE: {phase}",
                30,
                45,
                font_scale=0.75,
                thickness=2,
                background_color=phase_color
            )

            swing_energy = safe_float(
                swing_row.get(
                    "swing_motion_energy",
                    np.nan
                )
            )

            pose_quality = safe_float(
                swing_row.get(
                    "pose_quality",
                    np.nan
                )
            )

            draw_text_box(
                frame,
                (
                    f"Swing evidence signal: "
                    f"{swing_energy:.3f}"
                ),
                30,
                85
            )

            draw_text_box(
                frame,
                (
                    f"Pose quality: "
                    f"{pose_quality:.3f}"
                ),
                30,
                120
            )

        # ====================================================
        # BALL STATUS
        # ====================================================

        if ball_row is not None:

            ball_state = str(
                ball_row.get(
                    "state",
                    "UNKNOWN"
                )
            )

            ball_conf = safe_float(
                ball_row.get(
                    "confidence",
                    np.nan
                )
            )

            yolo_detected = normalize_bool(
                ball_row.get(
                    "yolo_detected",
                    False
                )
            )

            draw_text_box(
                frame,
                (
                    f"BALL: {ball_state} | "
                    f"YOLO={yolo_detected}"
                ),
                30,
                160
            )

            if np.isfinite(ball_conf):

                draw_text_box(
                    frame,
                    (
                        f"Ball confidence: "
                        f"{ball_conf:.3f}"
                    ),
                    30,
                    195
                )

        # ====================================================
        # FUSION PANEL
        # ====================================================

        panel_x = max(
            width - 580,
            30
        )

        draw_text_box(
            frame,
            classification,
            panel_x,
            45,
            font_scale=0.75,
            thickness=2,
            background_color=PURPLE
        )

        draw_text_box(
            frame,
            (
                f"Evidence quality: "
                f"{evidence_quality}"
            ),
            panel_x,
            85
        )

        draw_text_box(
            frame,
            (
                f"Evidence score: "
                f"{fusion_score:.3f}"
            ),
            panel_x,
            120
        )

        draw_text_box(
            frame,
            (
                f"Trajectory: "
                f"{trajectory_status}"
            ),
            panel_x,
            155,
            font_scale=0.43
        )

        # ====================================================
        # SPECIAL EVENTS
        # ====================================================

        if (
            last_yolo_frame is not None
            and frame_number
            == last_yolo_frame
        ):

            draw_text_box(
                frame,
                "LAST REAL YOLO BALL OBSERVATION",
                width // 2 - 220,
                70,
                font_scale=0.7,
                thickness=2,
                background_color=GREEN
            )

        if (
            candidate_frame is not None
            and frame_number
            == candidate_frame
        ):

            draw_text_box(
                frame,
                "CONTACT CANDIDATE",
                width // 2 - 180,
                70,
                font_scale=1.0,
                thickness=2,
                background_color=PURPLE
            )

            draw_text_box(
                frame,
                "HEURISTIC EVENT FRAME - NOT CONFIRMED CONTACT",
                width // 2 - 280,
                110,
                font_scale=0.55,
                thickness=1,
                background_color=BLACK
            )

        if (
            swing_peak_frame is not None
            and frame_number
            == swing_peak_frame
        ):

            draw_text_box(
                frame,
                "PEAK SWING CANDIDATE",
                width // 2 - 180,
                70,
                font_scale=0.9,
                thickness=2,
                background_color=RED
            )

        if (
            first_lost_frame is not None
            and frame_number
            == first_lost_frame
        ):

            draw_text_box(
                frame,
                "BALL TRACK LOST",
                width // 2 - 140,
                70,
                font_scale=0.9,
                thickness=2,
                background_color=RED
            )

        # ====================================================
        # FRAME / TIME
        # ====================================================

        time_sec = (
            (frame_number - 1)
            / fps
            if fps > 0
            else 0.0
        )

        draw_text_box(
            frame,
            (
                f"Frame {frame_number}/{total_frames} | "
                f"{time_sec:.3f} s"
            ),
            30,
            height - 80,
            font_scale=0.48
        )

        # ====================================================
        # TIMELINE
        # ====================================================

        draw_timeline(
            frame,
            current_frame=frame_number,
            total_frames=total_frames,
            swing_start=swing_start,
            swing_end=swing_end,
            peak_start=peak_start,
            peak_end=peak_end,
            candidate_frame=candidate_frame,
            swing_peak_frame=swing_peak_frame,
            last_yolo_frame=last_yolo_frame,
            first_lost_frame=first_lost_frame
        )

        # ====================================================
        # DISCLAIMER
        # ====================================================

        cv2.putText(
            frame,
            (
                "Contact candidate = temporal evidence only | "
                "not confirmed bat-ball contact"
            ),
            (
                width - 700,
                height - 78
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            WHITE,
            1,
            cv2.LINE_AA
        )

        writer.write(
            frame
        )

        if (
            frame_number % 30 == 0
            or frame_number == total_frames
        ):

            print(
                f"Processed frame "
                f"{frame_number}/{total_frames}"
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    writer.release()

    print()
    print("=" * 78)
    print(
        "CONTACT FUSION VISUALIZATION COMPLETE"
    )
    print("=" * 78)

    print(
        f"Frames processed       : "
        f"{frame_number}"
    )

    print(
        f"Last real YOLO ball    : "
        f"{last_yolo_frame}"
    )

    print(
        f"Candidate event frame  : "
        f"{candidate_frame}"
    )

    print(
        f"Swing peak frame       : "
        f"{swing_peak_frame}"
    )

    print(
        f"First LOST frame       : "
        f"{first_lost_frame}"
    )

    print(
        f"Classification         : "
        f"{classification}"
    )

    print(
        f"Evidence quality       : "
        f"{evidence_quality}"
    )

    print()

    print(
        f"Output video           : "
        f"{output_video}"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()