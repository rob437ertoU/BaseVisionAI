from pathlib import Path
from collections import deque
import argparse

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


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

# Usamos el CSV que ya contiene la pose suavizada + swing
DEFAULT_POSE_CSV = (
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

DEFAULT_ROLE_MODEL = (
    PROJECT_ROOT
    / "runs"
    / "player_roles"
    / "player_roles_v2_yolo11n"
    / "weights"
    / "best.pt"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "strike_zone_v1"
)


# ============================================================
# CONFIG
# ============================================================

ROLE_CONF = 0.25
ROLE_IMGSZ = 1280
DEVICE = 0

# player_roles_v2
BATTER_CLASS = 0
CATCHER_CLASS = 1
PITCHER_CLASS = 2
UMPIRE_CLASS = 3

POSE_VALID_THRESHOLD = 0.30

# Suavizado temporal de la zona
ZONE_EMA_ALPHA = 0.35

# Ancho de zona aproximado respecto al ancho de hombros
#
# Esto NO representa todavía el ancho reglamentario real del plato.
ZONE_WIDTH_SHOULDER_FACTOR = 0.80

# Si perdemos al catcher, mantenemos su centro unos frames
MAX_CATCHER_HOLD = 8


# ============================================================
# COLORS - BGR
# ============================================================

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
GRAY = (160, 160, 160)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Strike Zone Estimation V1"
    )

    parser.add_argument(
        "--video",
        type=str,
        default=str(DEFAULT_VIDEO)
    )

    parser.add_argument(
        "--pose-csv",
        type=str,
        default=str(DEFAULT_POSE_CSV)
    )

    parser.add_argument(
        "--ball-csv",
        type=str,
        default=str(DEFAULT_BALL_CSV)
    )

    parser.add_argument(
        "--role-model",
        type=str,
        default=str(DEFAULT_ROLE_MODEL)
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR)
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
    scale=0.55,
    color=WHITE,
    background=BLACK,
    thickness=1
):

    font = cv2.FONT_HERSHEY_SIMPLEX

    (tw, th), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness
    )

    cv2.rectangle(
        frame,
        (x - 5, y - th - 5),
        (x + tw + 5, y + baseline + 5),
        background,
        -1
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# POSE HELPERS
# ============================================================

def get_keypoint(row, name):

    x_columns = [
        f"{name}_x_smooth",
        f"{name}_x_smoothed",
        f"{name}_x"
    ]

    y_columns = [
        f"{name}_y_smooth",
        f"{name}_y_smoothed",
        f"{name}_y"
    ]

    x = np.nan
    y = np.nan

    for col in x_columns:

        if col in row.index:

            value = safe_float(
                row[col]
            )

            if np.isfinite(value):
                x = value
                break

    for col in y_columns:

        if col in row.index:

            value = safe_float(
                row[col]
            )

            if np.isfinite(value):
                y = value
                break

    if (
        not np.isfinite(x)
        or not np.isfinite(y)
    ):
        return None

    valid_col = f"{name}_valid"

    if valid_col in row.index:

        try:
            if int(row[valid_col]) != 1:
                return None
        except Exception:
            pass

    return np.array(
        [x, y],
        dtype=np.float32
    )


def midpoint(p1, p2):

    if p1 is None or p2 is None:
        return None

    return (
        p1 + p2
    ) / 2.0


# ============================================================
# VERTICAL STRIKE ZONE FROM BATTER
# ============================================================

def estimate_vertical_zone(pose_row):

    left_shoulder = get_keypoint(
        pose_row,
        "left_shoulder"
    )

    right_shoulder = get_keypoint(
        pose_row,
        "right_shoulder"
    )

    left_hip = get_keypoint(
        pose_row,
        "left_hip"
    )

    right_hip = get_keypoint(
        pose_row,
        "right_hip"
    )

    left_knee = get_keypoint(
        pose_row,
        "left_knee"
    )

    right_knee = get_keypoint(
        pose_row,
        "right_knee"
    )

    shoulder_mid = midpoint(
        left_shoulder,
        right_shoulder
    )

    hip_mid = midpoint(
        left_hip,
        right_hip
    )

    knee_mid = midpoint(
        left_knee,
        right_knee
    )

    if (
        shoulder_mid is None
        or hip_mid is None
        or knee_mid is None
    ):

        return None

    # --------------------------------------------------------
    # Top:
    # Approximation of midpoint between shoulders
    # and top of pants/hips.
    # --------------------------------------------------------

    zone_top = (
        shoulder_mid[1]
        + hip_mid[1]
    ) / 2.0

    # --------------------------------------------------------
    # Bottom:
    # Knee level proxy.
    # --------------------------------------------------------

    zone_bottom = knee_mid[1]

    if zone_bottom <= zone_top:
        return None

    shoulder_width = np.linalg.norm(
        left_shoulder
        - right_shoulder
    )

    if shoulder_width < 10:
        return None

    return {
        "top": float(zone_top),
        "bottom": float(zone_bottom),
        "shoulder_width": float(shoulder_width),
        "shoulder_mid_x": float(shoulder_mid[0]),
        "hip_mid_x": float(hip_mid[0])
    }


# ============================================================
# CATCHER DETECTION
# ============================================================

def detect_catcher(model, frame):

    results = model.predict(
        source=frame,
        conf=ROLE_CONF,
        imgsz=ROLE_IMGSZ,
        device=DEVICE,
        verbose=False
    )

    if len(results) == 0:
        return None

    result = results[0]

    if result.boxes is None:
        return None

    candidates = []

    for box in result.boxes:

        cls = int(
            box.cls[0].item()
        )

        conf = float(
            box.conf[0].item()
        )

        if cls != CATCHER_CLASS:
            continue

        xyxy = (
            box.xyxy[0]
            .detach()
            .cpu()
            .numpy()
        )

        x1, y1, x2, y2 = xyxy

        candidates.append(
            {
                "conf": conf,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "cx": float(
                    (x1 + x2) / 2.0
                ),
                "cy": float(
                    (y1 + y2) / 2.0
                )
            }
        )

    if not candidates:
        return None

    # V1:
    # tomamos el catcher con mayor confianza.
    candidates.sort(
        key=lambda item: item["conf"],
        reverse=True
    )

    return candidates[0]


# ============================================================
# ZONE EMA
# ============================================================

def smooth_value(previous, current, alpha):

    if current is None:
        return previous

    if previous is None:
        return current

    return (
        alpha * current
        + (1.0 - alpha) * previous
    )


# ============================================================
# BALL
# ============================================================

def get_real_ball(ball_row):

    if ball_row is None:
        return None

    detected = normalize_bool(
        ball_row.get(
            "yolo_detected",
            False
        )
    )

    if not detected:
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

    if (
        not np.isfinite(x)
        or not np.isfinite(y)
    ):
        return None

    return (
        float(x),
        float(y)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    video_path = Path(
        args.video
    ).resolve()

    pose_path = Path(
        args.pose_csv
    ).resolve()

    ball_path = Path(
        args.ball_csv
    ).resolve()

    role_model_path = Path(
        args.role_model
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    required = [
        video_path,
        pose_path,
        ball_path,
        role_model_path
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_csv = (
        output_dir
        / "baseball_train10_strike_zone_v1.csv"
    )

    output_video = (
        output_dir
        / "baseball_train10_strike_zone_v1.mp4"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    pose_df = pd.read_csv(
        pose_path
    )

    ball_df = pd.read_csv(
        ball_path
    )

    pose_map = {
        int(row["frame"]): row
        for _, row in pose_df.iterrows()
    }

    ball_map = {
        int(row["frame"]): row
        for _, row in ball_df.iterrows()
    }

    # ========================================================
    # ROLE MODEL
    # ========================================================

    print()
    print(
        f"Loading role detector:\n"
        f"{role_model_path}"
    )

    role_model = YOLO(
        str(role_model_path)
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

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width, height)
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "Could not initialize output video."
        )

    # ========================================================
    # STATE
    # ========================================================

    smooth_center_x = None
    smooth_top = None
    smooth_bottom = None
    smooth_width = None

    catcher_hold_frames = 0
    last_catcher = None

    rows = []

    print()
    print("=" * 78)
    print(
        "BaseVisionAI - Strike Zone Estimation V1"
    )
    print("=" * 78)

    print(
        f"Video          : {video_path}"
    )

    print(
        f"Frames         : {total_frames}"
    )

    print(
        f"FPS            : {fps:.2f}"
    )

    print()
    print(
        "IMPORTANT: This is an ESTIMATED 2D strike zone."
    )

    print(
        "It is not a calibrated MLB strike-zone measurement."
    )

    print("=" * 78)

    # ========================================================
    # LOOP
    # ========================================================

    frame_number = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        pose_row = pose_map.get(
            frame_number
        )

        ball_row = ball_map.get(
            frame_number
        )

        # ====================================================
        # BATTER VERTICAL ZONE
        # ====================================================

        vertical = None

        if pose_row is not None:

            vertical = estimate_vertical_zone(
                pose_row
            )

        # ====================================================
        # CATCHER
        # ====================================================

        catcher = detect_catcher(
            role_model,
            frame
        )

        catcher_source = "NONE"

        if catcher is not None:

            last_catcher = catcher
            catcher_hold_frames = 0
            catcher_source = "YOLO"

        elif (
            last_catcher is not None
            and catcher_hold_frames
            < MAX_CATCHER_HOLD
        ):

            catcher = last_catcher

            catcher_hold_frames += 1

            catcher_source = "HOLD"

        else:

            catcher = None
            last_catcher = None

        # ====================================================
        # BUILD ZONE
        # ====================================================

        zone_available = False

        zone_left = np.nan
        zone_right = np.nan
        zone_top = np.nan
        zone_bottom = np.nan
        zone_center_x = np.nan

        if (
            vertical is not None
            and catcher is not None
        ):

            raw_center_x = catcher[
                "cx"
            ]

            raw_top = vertical[
                "top"
            ]

            raw_bottom = vertical[
                "bottom"
            ]

            raw_width = (
                vertical[
                    "shoulder_width"
                ]
                * ZONE_WIDTH_SHOULDER_FACTOR
            )

            smooth_center_x = smooth_value(
                smooth_center_x,
                raw_center_x,
                ZONE_EMA_ALPHA
            )

            smooth_top = smooth_value(
                smooth_top,
                raw_top,
                ZONE_EMA_ALPHA
            )

            smooth_bottom = smooth_value(
                smooth_bottom,
                raw_bottom,
                ZONE_EMA_ALPHA
            )

            smooth_width = smooth_value(
                smooth_width,
                raw_width,
                ZONE_EMA_ALPHA
            )

            zone_center_x = (
                smooth_center_x
            )

            zone_top = (
                smooth_top
            )

            zone_bottom = (
                smooth_bottom
            )

            zone_left = (
                zone_center_x
                - smooth_width / 2.0
            )

            zone_right = (
                zone_center_x
                + smooth_width / 2.0
            )

            zone_available = True

        # ====================================================
        # REAL YOLO BALL
        # ====================================================

        ball = get_real_ball(
            ball_row
        )

        ball_inside_zone = False

        if (
            zone_available
            and ball is not None
        ):

            bx, by = ball

            ball_inside_zone = (
                zone_left <= bx <= zone_right
                and
                zone_top <= by <= zone_bottom
            )

        # ====================================================
        # DRAW CATCHER
        # ====================================================

        if catcher is not None:

            x1 = int(
                catcher["x1"]
            )

            y1 = int(
                catcher["y1"]
            )

            x2 = int(
                catcher["x2"]
            )

            y2 = int(
                catcher["y2"]
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                CYAN,
                2
            )

            draw_text_box(
                frame,
                (
                    f"CATCHER {catcher_source} "
                    f"{catcher['conf']:.2f}"
                ),
                x1,
                max(
                    25,
                    y1 - 8
                ),
                scale=0.45,
                background=BLACK
            )

            cv2.circle(
                frame,
                (
                    int(catcher["cx"]),
                    int(catcher["cy"])
                ),
                5,
                CYAN,
                -1
            )

        # ====================================================
        # DRAW STRIKE ZONE
        # ====================================================

        if zone_available:

            zl = int(
                round(zone_left)
            )

            zr = int(
                round(zone_right)
            )

            zt = int(
                round(zone_top)
            )

            zb = int(
                round(zone_bottom)
            )

            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (zl, zt),
                (zr, zb),
                GREEN,
                -1
            )

            cv2.addWeighted(
                overlay,
                0.14,
                frame,
                0.86,
                0,
                frame
            )

            cv2.rectangle(
                frame,
                (zl, zt),
                (zr, zb),
                GREEN,
                3
            )

            draw_text_box(
                frame,
                "ESTIMATED 2D STRIKE ZONE",
                zl,
                max(
                    30,
                    zt - 10
                ),
                scale=0.5,
                background=GREEN
            )

        # ====================================================
        # DRAW BALL
        # ====================================================

        if ball is not None:

            bx = int(
                round(ball[0])
            )

            by = int(
                round(ball[1])
            )

            color = (
                RED
                if ball_inside_zone
                else YELLOW
            )

            cv2.circle(
                frame,
                (bx, by),
                10,
                color,
                3
            )

            cv2.circle(
                frame,
                (bx, by),
                3,
                color,
                -1
            )

            if zone_available:

                label = (
                    "BALL POINT: INSIDE ZONE"
                    if ball_inside_zone
                    else
                    "BALL POINT: OUTSIDE ZONE"
                )

            else:

                label = (
                    "BALL YOLO - ZONE UNAVAILABLE"
                )

            draw_text_box(
                frame,
                label,
                bx + 15,
                by - 15,
                scale=0.45
            )

        # ====================================================
        # INFO PANEL
        # ====================================================

        draw_text_box(
            frame,
            "BaseVisionAI - Strike Zone V1",
            30,
            40,
            scale=0.7,
            background=PURPLE,
            thickness=2
        )

        draw_text_box(
            frame,
            f"Frame: {frame_number}/{total_frames}",
            30,
            80
        )

        draw_text_box(
            frame,
            (
                "Zone available: "
                f"{zone_available}"
            ),
            30,
            115
        )

        if zone_available:

            draw_text_box(
                frame,
                (
                    f"Zone X: "
                    f"{zone_left:.0f} -> "
                    f"{zone_right:.0f}"
                ),
                30,
                150
            )

            draw_text_box(
                frame,
                (
                    f"Zone Y: "
                    f"{zone_top:.0f} -> "
                    f"{zone_bottom:.0f}"
                ),
                30,
                185
            )

        draw_text_box(
            frame,
            (
                "2D ESTIMATE - "
                "NOT CALIBRATED BALL/STRIKE CALL"
            ),
            30,
            height - 35,
            scale=0.50,
            background=BLACK
        )

        # ====================================================
        # SAVE ROW
        # ====================================================

        rows.append(
            {
                "frame":
                    frame_number,

                "zone_available":
                    zone_available,

                "zone_left":
                    zone_left,

                "zone_right":
                    zone_right,

                "zone_top":
                    zone_top,

                "zone_bottom":
                    zone_bottom,

                "zone_center_x":
                    zone_center_x,

                "catcher_detected":
                    catcher_source == "YOLO",

                "catcher_source":
                    catcher_source,

                "catcher_confidence":
                    (
                        catcher["conf"]
                        if catcher is not None
                        else np.nan
                    ),

                "ball_yolo_detected":
                    ball is not None,

                "ball_x":
                    (
                        ball[0]
                        if ball is not None
                        else np.nan
                    ),

                "ball_y":
                    (
                        ball[1]
                        if ball is not None
                        else np.nan
                    ),

                "ball_inside_estimated_zone":
                    (
                        ball_inside_zone
                        if (
                            ball is not None
                            and zone_available
                        )
                        else False
                    )
            }
        )

        writer.write(
            frame
        )

        if (
            frame_number % 20 == 0
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

    output_df = pd.DataFrame(
        rows
    )

    output_df.to_csv(
        output_csv,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    zone_frames = int(
        output_df[
            "zone_available"
        ].sum()
    )

    real_ball_frames = int(
        output_df[
            "ball_yolo_detected"
        ].sum()
    )

    ball_zone_rows = output_df[
        (
            output_df[
                "ball_yolo_detected"
            ]
        )
        &
        (
            output_df[
                "zone_available"
            ]
        )
    ]

    inside_frames = int(
        ball_zone_rows[
            "ball_inside_estimated_zone"
        ].sum()
    )

    print()
    print("=" * 78)
    print(
        "STRIKE ZONE V1 COMPLETE"
    )
    print("=" * 78)

    print(
        f"Frames processed          : "
        f"{frame_number}"
    )

    print(
        f"Strike-zone frames        : "
        f"{zone_frames}"
    )

    print(
        f"Real YOLO ball frames     : "
        f"{real_ball_frames}"
    )

    print(
        f"Ball + zone overlap       : "
        f"{len(ball_zone_rows)}"
    )

    print(
        f"Ball points inside zone   : "
        f"{inside_frames}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "A ball point inside this rectangle "
        "does NOT yet equal a confirmed strike."
    )

    print(
        "We still need to estimate the pitch crossing "
        "near the home-plate plane."
    )

    print()
    print(
        f"Output CSV                : "
        f"{output_csv}"
    )

    print(
        f"Output video              : "
        f"{output_video}"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()