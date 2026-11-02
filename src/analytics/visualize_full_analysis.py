from pathlib import Path
from collections import deque

import cv2
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "baseball_test.mp4"
)

BALL_CSV = (
    PROJECT_ROOT
    / "results"
    / "ball_tracking_v3_1.csv"
)

POSE_CSV = (
    PROJECT_ROOT
    / "results"
    / "batter_pose_smoothed_v1.csv"
)

SWING_PHASES_CSV = (
    PROJECT_ROOT
    / "results"
    / "swing_phases_v1_1.csv"
)

CONTACT_FUSION_CSV = (
    PROJECT_ROOT
    / "results"
    / "contact_fusion_v1_1.csv"
)

CONTACT_SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "contact_fusion_summary_v1_1.csv"
)

OUTPUT_VIDEO = (
    PROJECT_ROOT
    / "results"
    / "basevision_full_analysis_v1_1.mp4"
)


# ============================================================
# DISPLAY SETTINGS
# ============================================================

TRAIL_LENGTH = 18
CONTACT_WINDOW = 2

POSE_POINT_RADIUS = 4
POSE_LINE_THICKNESS = 2


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


POSE_CONNECTIONS = [
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
    ("right_knee", "right_ankle")
]


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=np.nan):

    try:

        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def safe_bool(value):

    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)

    text = str(value).strip().lower()

    return text in {
        "true",
        "1",
        "yes",
        "y"
    }


def draw_text(
    image,
    text,
    xy,
    scale=0.65,
    thickness=2,
    color=(255, 255, 255)
):

    cv2.putText(
        image,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_panel(
    frame,
    x1,
    y1,
    x2,
    y2,
    alpha=0.65
):

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (0, 0, 0),
        -1
    )

    frame[:] = cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0
    )


def build_frame_lookup(df):

    lookup = {}

    if "frame" not in df.columns:
        return lookup

    for _, row in df.iterrows():

        try:
            frame_number = int(row["frame"])

        except Exception:
            continue

        lookup[frame_number] = row

    return lookup


# ============================================================
# POSE HELPERS
# ============================================================

def get_xy_columns(df, keypoint):

    candidates = [
        (
            f"{keypoint}_x_smooth",
            f"{keypoint}_y_smooth"
        ),
        (
            f"{keypoint}_x_smoothed",
            f"{keypoint}_y_smoothed"
        ),
        (
            f"{keypoint}_x_final",
            f"{keypoint}_y_final"
        ),
        (
            f"{keypoint}_x",
            f"{keypoint}_y"
        )
    ]

    for x_col, y_col in candidates:

        if (
            x_col in df.columns
            and
            y_col in df.columns
        ):
            return x_col, y_col

    return None, None


def build_pose_column_map(pose_df):

    mapping = {}

    for keypoint in KEYPOINT_NAMES:

        x_col, y_col = get_xy_columns(
            pose_df,
            keypoint
        )

        if x_col is not None:

            mapping[keypoint] = (
                x_col,
                y_col
            )

    return mapping


def draw_pose(
    frame,
    pose_row,
    pose_column_map
):

    if pose_row is None:
        return

    points = {}

    for keypoint, (
        x_col,
        y_col
    ) in pose_column_map.items():

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

        if (
            np.isfinite(x)
            and
            np.isfinite(y)
        ):

            points[keypoint] = (
                int(round(x)),
                int(round(y))
            )

    # Draw skeleton
    for a, b in POSE_CONNECTIONS:

        if (
            a in points
            and
            b in points
        ):

            cv2.line(
                frame,
                points[a],
                points[b],
                (255, 255, 255),
                POSE_LINE_THICKNESS,
                cv2.LINE_AA
            )

    # Draw keypoints
    for point in points.values():

        cv2.circle(
            frame,
            point,
            POSE_POINT_RADIUS,
            (0, 255, 255),
            -1,
            cv2.LINE_AA
        )


# ============================================================
# BALL HELPERS
# ============================================================

def find_ball_xy_columns(ball_df):

    candidates = [
        ("filtered_x", "filtered_y"),
        ("observed_x", "observed_y"),
        ("kalman_predicted_x", "kalman_predicted_y"),
        ("x", "y"),
        ("center_x", "center_y"),
        ("ball_x", "ball_y"),
        ("cx", "cy")
    ]

    for x_col, y_col in candidates:

        if (
            x_col in ball_df.columns
            and
            y_col in ball_df.columns
        ):
            return x_col, y_col

    raise ValueError(
        "Could not find ball X/Y columns.\n"
        f"Available columns:\n{list(ball_df.columns)}"
    )


def valid_ball_point(
    row,
    x_col,
    y_col
):

    if row is None:
        return None

    state = str(
        row.get(
            "state",
            ""
        )
    ).upper()

    if state in {
        "WAITING",
        "LOST",
        "RESET"
    }:
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

    if (
        not np.isfinite(x)
        or
        not np.isfinite(y)
    ):
        return None

    return (
        int(round(x)),
        int(round(y))
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("BaseVisionAI - Full Analysis Visualization V1.1")
    print("=" * 80)

    required_files = [
        VIDEO_PATH,
        BALL_CSV,
        POSE_CSV,
        SWING_PHASES_CSV,
        CONTACT_FUSION_CSV,
        CONTACT_SUMMARY_CSV
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing file:\n{path}"
            )

    # ========================================================
    # LOAD DATA
    # ========================================================

    ball_df = pd.read_csv(
        BALL_CSV
    )

    pose_df = pd.read_csv(
        POSE_CSV
    )

    swing_df = pd.read_csv(
        SWING_PHASES_CSV
    )

    fusion_df = pd.read_csv(
        CONTACT_FUSION_CSV
    )

    contact_summary = pd.read_csv(
        CONTACT_SUMMARY_CSV
    )

    print(
        f"\nBall rows: {len(ball_df)}"
    )

    print(
        f"Pose rows: {len(pose_df)}"
    )

    print(
        f"Swing rows: {len(swing_df)}"
    )

    print(
        f"Fusion rows: {len(fusion_df)}"
    )

    # ========================================================
    # CONTACT RESULT
    # ========================================================

    predicted_contact_frame = int(
        contact_summary.iloc[0][
            "predicted_contact_frame"
        ]
    )

    predicted_contact_score = float(
        contact_summary.iloc[0][
            "fusion_score"
        ]
    )

    print(
        f"\nPredicted contact frame: "
        f"{predicted_contact_frame}"
    )

    print(
        f"Contact fusion score: "
        f"{predicted_contact_score:.3f}"
    )

    # ========================================================
    # LOOKUPS
    # ========================================================

    ball_lookup = build_frame_lookup(
        ball_df
    )

    pose_lookup = build_frame_lookup(
        pose_df
    )

    swing_lookup = build_frame_lookup(
        swing_df
    )

    fusion_lookup = build_frame_lookup(
        fusion_df
    )

    # ========================================================
    # POSE COLUMN MAP
    # ========================================================

    pose_column_map = build_pose_column_map(
        pose_df
    )

    print(
        f"\nPose keypoints available: "
        f"{len(pose_column_map)} / 17"
    )

    # ========================================================
    # BALL COLUMNS
    # ========================================================

    ball_x_col, ball_y_col = (
        find_ball_xy_columns(
            ball_df
        )
    )

    print(
        f"Ball coordinates: "
        f"{ball_x_col}, {ball_y_col}"
    )

    # ========================================================
    # OPEN VIDEO
    # ========================================================

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open video."
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

    OUTPUT_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (
            width,
            height
        )
    )

    if not writer.isOpened():

        raise RuntimeError(
            "Could not create output video."
        )

    print(
        f"\nFPS: {fps:.2f}"
    )

    print(
        f"Resolution: "
        f"{width}x{height}"
    )

    print(
        f"Frames: {total_frames}"
    )

    # ========================================================
    # TRACKING STATE
    # ========================================================

    ball_trail = deque(
        maxlen=TRAIL_LENGTH
    )

    previous_track_id = None

    frame_number = 0

    # ========================================================
    # VIDEO LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # ----------------------------------------------------
        # Get current rows
        # ----------------------------------------------------

        ball_row = ball_lookup.get(
            frame_number
        )

        pose_row = pose_lookup.get(
            frame_number
        )

        swing_row = swing_lookup.get(
            frame_number
        )

        fusion_row = fusion_lookup.get(
            frame_number
        )

        # ====================================================
        # DRAW BATTER POSE
        # ====================================================

        draw_pose(
            frame,
            pose_row,
            pose_column_map
        )

        # ====================================================
        # BALL STATE
        # ====================================================

        ball_state = "UNKNOWN"
        current_track_id = None

        if ball_row is not None:

            ball_state = str(
                ball_row.get(
                    "state",
                    "UNKNOWN"
                )
            ).upper()

            track_value = ball_row.get(
                "track_id",
                np.nan
            )

            if pd.notna(track_value):

                try:
                    current_track_id = int(
                        float(track_value)
                    )

                except Exception:
                    current_track_id = None

        # ====================================================
        # RESET TRAIL WHEN TRACK CHANGES
        # ====================================================

        if (
            current_track_id is not None
            and
            previous_track_id is not None
            and
            current_track_id != previous_track_id
        ):

            ball_trail.clear()

        # ====================================================
        # RESET TRAIL ON LOST / RESET
        # ====================================================

        if ball_state in {
            "LOST",
            "RESET"
        }:

            ball_trail.clear()

        # ====================================================
        # BALL POSITION
        # ====================================================

        ball_point = valid_ball_point(
            ball_row,
            ball_x_col,
            ball_y_col
        )

        if ball_point is not None:

            ball_trail.append(
                ball_point
            )

            # -----------------------------------------------
            # Was YOLO actually detecting the ball?
            # -----------------------------------------------

            yolo_detected = safe_bool(
                ball_row.get(
                    "yolo_detected",
                    False
                )
            )

            # -----------------------------------------------
            # YOLO observation = GREEN
            # -----------------------------------------------

            if yolo_detected:

                cv2.circle(
                    frame,
                    ball_point,
                    9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                cv2.circle(
                    frame,
                    ball_point,
                    3,
                    (0, 255, 0),
                    -1,
                    cv2.LINE_AA
                )

            # -----------------------------------------------
            # Kalman prediction = YELLOW
            # -----------------------------------------------

            else:

                cv2.circle(
                    frame,
                    ball_point,
                    9,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.circle(
                    frame,
                    ball_point,
                    3,
                    (0, 255, 255),
                    -1,
                    cv2.LINE_AA
                )

        # ====================================================
        # STORE TRACK ID
        # ====================================================

        if current_track_id is not None:

            previous_track_id = (
                current_track_id
            )

        # ====================================================
        # DRAW BALL TRAIL
        # ====================================================

        if len(ball_trail) >= 2:

            trail_points = list(
                ball_trail
            )

            for i in range(
                1,
                len(trail_points)
            ):

                cv2.line(
                    frame,
                    trail_points[i - 1],
                    trail_points[i],
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

        # ====================================================
        # LEFT PANEL
        # ====================================================

        draw_panel(
            frame,
            20,
            20,
            530,
            320
        )

        draw_text(
            frame,
            "BaseVision AI",
            (40, 55),
            scale=0.9
        )

        draw_text(
            frame,
            f"Frame: {frame_number}",
            (40, 90)
        )

        # ====================================================
        # SWING INFORMATION
        # ====================================================

        phase = "UNKNOWN"
        swing_score = np.nan
        leg_score = np.nan

        if swing_row is not None:

            phase = str(
                swing_row.get(
                    "swing_phase",
                    "UNKNOWN"
                )
            )

            swing_score = safe_float(
                swing_row.get(
                    "swing_activity_score",
                    np.nan
                )
            )

            leg_score = safe_float(
                swing_row.get(
                    "leg_activity_score",
                    np.nan
                )
            )

        draw_text(
            frame,
            f"Phase: {phase}",
            (40, 125),
            scale=0.75
        )

        if np.isfinite(
            swing_score
        ):

            draw_text(
                frame,
                f"Swing activity: "
                f"{swing_score:.2f}",
                (40, 160)
            )

        if np.isfinite(
            leg_score
        ):

            draw_text(
                frame,
                f"Leg activity: "
                f"{leg_score:.2f}",
                (40, 195)
            )

        # ====================================================
        # BALL TRACK INFORMATION
        # ====================================================

        draw_text(
            frame,
            f"Ball: {ball_state}",
            (40, 230)
        )

        track_label = (
            "-"
            if current_track_id is None
            else str(current_track_id)
        )

        draw_text(
            frame,
            f"Track ID: {track_label}",
            (40, 265)
        )

        # ====================================================
        # PER-FRAME CONTACT FUSION
        # ====================================================

        current_fusion = np.nan

        if fusion_row is not None:

            current_fusion = safe_float(
                fusion_row.get(
                    "fusion_score",
                    np.nan
                )
            )

        if np.isfinite(
            current_fusion
        ):

            draw_text(
                frame,
                f"Contact fusion: "
                f"{current_fusion:.2f}",
                (250, 265),
                scale=0.60
            )

        # ====================================================
        # PERMANENT CONTACT PREDICTION
        # ====================================================

        draw_text(
            frame,
            f"Pred. contact: "
            f"{predicted_contact_frame}",
            (250, 230),
            scale=0.60
        )

        # ====================================================
        # LEGEND
        # ====================================================

        cv2.circle(
            frame,
            (270, 300),
            6,
            (0, 255, 0),
            -1
        )

        draw_text(
            frame,
            "YOLO",
            (285, 307),
            scale=0.50,
            thickness=1
        )

        cv2.circle(
            frame,
            (380, 300),
            6,
            (0, 255, 255),
            -1
        )

        draw_text(
            frame,
            "Kalman",
            (395, 307),
            scale=0.50,
            thickness=1
        )

        # ====================================================
        # CONTACT WINDOW PANEL
        # ====================================================

        frame_distance = abs(
            frame_number
            -
            predicted_contact_frame
        )

        if (
            frame_distance
            <= CONTACT_WINDOW
        ):

            panel_x1 = max(
                width - 470,
                20
            )

            panel_y1 = 35

            panel_x2 = (
                width - 35
            )

            panel_y2 = 155

            cv2.rectangle(
                frame,
                (
                    panel_x1,
                    panel_y1
                ),
                (
                    panel_x2,
                    panel_y2
                ),
                (0, 0, 255),
                -1
            )

            draw_text(
                frame,
                "CONTACT CANDIDATE",
                (
                    panel_x1 + 25,
                    panel_y1 + 45
                ),
                scale=0.8,
                thickness=3
            )

            draw_text(
                frame,
                (
                    f"Predicted frame: "
                    f"{predicted_contact_frame}"
                ),
                (
                    panel_x1 + 25,
                    panel_y1 + 80
                ),
                scale=0.60
            )

            draw_text(
                frame,
                (
                    f"Fusion score: "
                    f"{predicted_contact_score:.2f}"
                ),
                (
                    panel_x1 + 25,
                    panel_y1 + 108
                ),
                scale=0.60
            )

        # ====================================================
        # EXACT CONTACT FRAME
        # ====================================================

        if (
            frame_number
            ==
            predicted_contact_frame
        ):

            cv2.rectangle(
                frame,
                (4, 4),
                (
                    width - 5,
                    height - 5
                ),
                (0, 0, 255),
                6
            )

        # ====================================================
        # WRITE FRAME
        # ====================================================

        writer.write(
            frame
        )

        if (
            frame_number % 20
            == 0
        ):

            print(
                f"Processed "
                f"{frame_number}/"
                f"{total_frames}"
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    writer.release()

    print(
        "\n"
        +
        "=" * 80
    )

    print(
        "FULL ANALYSIS VISUALIZATION V1.1 COMPLETED"
    )

    print(
        "=" * 80
    )

    print(
        f"\nOutput:\n"
        f"{OUTPUT_VIDEO}"
    )


if __name__ == "__main__":
    main()