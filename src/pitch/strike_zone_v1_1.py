from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_VIDEO = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "baseball_train10.mp4"
)

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
    / "strike_zone_v1_1"
)


# ============================================================
# CONFIG
# ============================================================

ROLE_CONF = 0.25
ROLE_IMGSZ = 1280
DEVICE = 0

CATCHER_CLASS = 1

ZONE_EMA_ALPHA = 0.35

# Estimated strike-zone width from batter shoulder width.
ZONE_WIDTH_SHOULDER_FACTOR = 0.80

# Frames used to establish a robust horizontal plate reference.
PLATE_REFERENCE_START = 1
PLATE_REFERENCE_END = 80

# Minimum number of catcher detections before we trust reference.
MIN_PLATE_REFERENCE_SAMPLES = 5


# ============================================================
# COLORS
# OpenCV BGR
# ============================================================

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
PURPLE = (255, 0, 255)
ORANGE = (0, 165, 255)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Strike Zone Estimation V1.1"
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


def ema(previous, current, alpha):

    if current is None:
        return previous

    if previous is None:
        return float(current)

    return (
        alpha * float(current)
        + (1.0 - alpha) * float(previous)
    )


# ============================================================
# POSE
# ============================================================

def get_keypoint(row, name):

    x_candidates = [
        f"{name}_x_smooth",
        f"{name}_x_smoothed",
        f"{name}_x"
    ]

    y_candidates = [
        f"{name}_y_smooth",
        f"{name}_y_smoothed",
        f"{name}_y"
    ]

    x = np.nan
    y = np.nan

    for col in x_candidates:

        if col in row.index:

            value = safe_float(
                row[col]
            )

            if np.isfinite(value):
                x = value
                break

    for col in y_candidates:

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


def midpoint(a, b):

    if a is None or b is None:
        return None

    return (a + b) / 2.0


def estimate_vertical_zone(row):

    left_shoulder = get_keypoint(
        row,
        "left_shoulder"
    )

    right_shoulder = get_keypoint(
        row,
        "right_shoulder"
    )

    left_hip = get_keypoint(
        row,
        "left_hip"
    )

    right_hip = get_keypoint(
        row,
        "right_hip"
    )

    left_knee = get_keypoint(
        row,
        "left_knee"
    )

    right_knee = get_keypoint(
        row,
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

    zone_top = (
        shoulder_mid[1]
        + hip_mid[1]
    ) / 2.0

    zone_bottom = knee_mid[1]

    shoulder_width = np.linalg.norm(
        left_shoulder
        - right_shoulder
    )

    if (
        zone_bottom <= zone_top
        or shoulder_width < 10
    ):
        return None

    return {
        "top": float(zone_top),
        "bottom": float(zone_bottom),
        "shoulder_width": float(shoulder_width),
    }


# ============================================================
# CATCHER
# ============================================================

def detect_catcher(model, frame):

    results = model.predict(
        source=frame,
        conf=ROLE_CONF,
        imgsz=ROLE_IMGSZ,
        device=DEVICE,
        verbose=False
    )

    if not results:
        return None

    result = results[0]

    if result.boxes is None:
        return None

    candidates = []

    for box in result.boxes:

        cls = int(
            box.cls[0].item()
        )

        if cls != CATCHER_CLASS:
            continue

        conf = float(
            box.conf[0].item()
        )

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

    candidates.sort(
        key=lambda x: x["conf"],
        reverse=True
    )

    return candidates[0]


# ============================================================
# BALL
# ============================================================

def get_real_ball(row):

    if row is None:
        return None

    if not normalize_bool(
        row.get(
            "yolo_detected",
            False
        )
    ):
        return None

    x = safe_float(
        row.get(
            "observed_x",
            np.nan
        )
    )

    y = safe_float(
        row.get(
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
# PASS 1
# FIND STABLE PLATE REFERENCE
# ============================================================

def estimate_plate_reference(
    video_path,
    model
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open video for plate-reference pass."
        )

    catcher_x_samples = []
    catcher_conf_samples = []

    frame_number = 0

    print()
    print("PASS 1 - Estimating horizontal home-plate reference")
    print("-" * 78)

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        frame_number += 1

        if frame_number < PLATE_REFERENCE_START:
            continue

        if frame_number > PLATE_REFERENCE_END:
            break

        catcher = detect_catcher(
            model,
            frame
        )

        if catcher is not None:

            catcher_x_samples.append(
                catcher["cx"]
            )

            catcher_conf_samples.append(
                catcher["conf"]
            )

    cap.release()

    if (
        len(catcher_x_samples)
        < MIN_PLATE_REFERENCE_SAMPLES
    ):

        raise RuntimeError(
            "Insufficient catcher detections to establish "
            "a stable home-plate proxy.\n"
            f"Samples found: {len(catcher_x_samples)}"
        )

    # Median is much more robust than mean against wrong detections.
    plate_x = float(
        np.median(
            catcher_x_samples
        )
    )

    median_conf = float(
        np.median(
            catcher_conf_samples
        )
    )

    print(
        f"Catcher samples           : "
        f"{len(catcher_x_samples)}"
    )

    print(
        f"Median catcher confidence : "
        f"{median_conf:.3f}"
    )

    print(
        f"Frozen plate X reference  : "
        f"{plate_x:.1f} px"
    )

    return (
        plate_x,
        len(catcher_x_samples),
        median_conf
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

    for path in [
        video_path,
        pose_path,
        ball_path,
        role_model_path
    ]:

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
        / "baseball_train10_strike_zone_v1_1.csv"
    )

    output_video = (
        output_dir
        / "baseball_train10_strike_zone_v1_1.mp4"
    )

    # ========================================================
    # LOAD CSV
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
    # MODEL
    # ========================================================

    print()
    print(
        f"Loading role model:\n"
        f"{role_model_path}"
    )

    model = YOLO(
        str(role_model_path)
    )

    # ========================================================
    # PASS 1
    # ========================================================

    (
        plate_x,
        plate_reference_samples,
        plate_reference_conf
    ) = estimate_plate_reference(
        video_path,
        model
    )

    # ========================================================
    # PASS 2
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

        raise RuntimeError(
            "Could not create output video."
        )

    print()
    print("=" * 78)
    print(
        "BaseVisionAI - Strike Zone Estimation V1.1"
    )
    print("=" * 78)

    print(
        f"Frames                    : {total_frames}"
    )

    print(
        f"FPS                       : {fps:.2f}"
    )

    print(
        f"Frozen plate X            : {plate_x:.1f}"
    )

    print()
    print(
        "PASS 2 - Building strike zone from "
        "fixed plate reference + dynamic batter pose"
    )

    print("=" * 78)

    smooth_top = None
    smooth_bottom = None
    smooth_width = None

    rows = []

    frame_number = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        frame_number += 1

        pose_row = pose_map.get(
            frame_number
        )

        ball_row = ball_map.get(
            frame_number
        )

        # ====================================================
        # VERTICAL ZONE
        # ====================================================

        vertical = None

        if pose_row is not None:

            vertical = estimate_vertical_zone(
                pose_row
            )

        zone_available = False

        zone_left = np.nan
        zone_right = np.nan
        zone_top = np.nan
        zone_bottom = np.nan

        if vertical is not None:

            smooth_top = ema(
                smooth_top,
                vertical["top"],
                ZONE_EMA_ALPHA
            )

            smooth_bottom = ema(
                smooth_bottom,
                vertical["bottom"],
                ZONE_EMA_ALPHA
            )

            raw_width = (
                vertical["shoulder_width"]
                * ZONE_WIDTH_SHOULDER_FACTOR
            )

            smooth_width = ema(
                smooth_width,
                raw_width,
                ZONE_EMA_ALPHA
            )

            zone_left = (
                plate_x
                - smooth_width / 2.0
            )

            zone_right = (
                plate_x
                + smooth_width / 2.0
            )

            zone_top = smooth_top
            zone_bottom = smooth_bottom

            zone_available = True

        # ====================================================
        # BALL
        # ====================================================

        ball = get_real_ball(
            ball_row
        )

        ball_inside_zone = False

        horizontal_inside = False
        vertical_inside = False

        if (
            ball is not None
            and zone_available
        ):

            bx, by = ball

            horizontal_inside = (
                zone_left
                <= bx
                <= zone_right
            )

            vertical_inside = (
                zone_top
                <= by
                <= zone_bottom
            )

            ball_inside_zone = (
                horizontal_inside
                and vertical_inside
            )

        # ====================================================
        # DRAW PLATE REFERENCE
        # ====================================================

        px = int(
            round(plate_x)
        )

        cv2.line(
            frame,
            (px, 0),
            (px, height),
            CYAN,
            1
        )

        draw_text_box(
            frame,
            "FROZEN PLATE-X PROXY",
            px + 8,
            100,
            scale=0.43,
            background=BLACK
        )

        # ====================================================
        # DRAW ZONE
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
                0.13,
                frame,
                0.87,
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
                    zt - 10,
                    30
                ),
                scale=0.47,
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

            if (
                zone_available
                and ball_inside_zone
            ):
                ball_color = RED

            else:
                ball_color = YELLOW

            cv2.circle(
                frame,
                (bx, by),
                12,
                ball_color,
                3
            )

            cv2.circle(
                frame,
                (bx, by),
                3,
                ball_color,
                -1
            )

            draw_text_box(
                frame,
                "REAL YOLO BALL",
                bx + 15,
                by - 15,
                scale=0.45
            )

        # ====================================================
        # PANEL
        # ====================================================

        draw_text_box(
            frame,
            "BaseVisionAI - Strike Zone V1.1",
            30,
            40,
            scale=0.70,
            background=PURPLE,
            thickness=2
        )

        draw_text_box(
            frame,
            f"Frame {frame_number}/{total_frames}",
            30,
            80
        )

        draw_text_box(
            frame,
            (
                f"Plate X proxy: "
                f"{plate_x:.1f}px "
                f"(frozen)"
            ),
            30,
            115
        )

        draw_text_box(
            frame,
            (
                f"Zone available: "
                f"{zone_available}"
            ),
            30,
            150
        )

        if ball is not None:

            draw_text_box(
                frame,
                (
                    f"Ball + zone: "
                    f"{zone_available}"
                ),
                30,
                185
            )

            if zone_available:

                draw_text_box(
                    frame,
                    (
                        f"Horizontal inside: "
                        f"{horizontal_inside}"
                    ),
                    30,
                    220
                )

                draw_text_box(
                    frame,
                    (
                        f"Vertical inside: "
                        f"{vertical_inside}"
                    ),
                    30,
                    255
                )

        draw_text_box(
            frame,
            (
                "2D ESTIMATE ONLY - "
                "NOT A CONFIRMED BALL/STRIKE CALL"
            ),
            30,
            height - 30,
            scale=0.50
        )

        # ====================================================
        # CSV
        # ====================================================

        rows.append(
            {
                "frame":
                    frame_number,

                "plate_x_proxy":
                    plate_x,

                "plate_reference_samples":
                    plate_reference_samples,

                "plate_reference_median_conf":
                    plate_reference_conf,

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

                "horizontal_inside":
                    (
                        horizontal_inside
                        if (
                            ball is not None
                            and zone_available
                        )
                        else False
                    ),

                "vertical_inside":
                    (
                        vertical_inside
                        if (
                            ball is not None
                            and zone_available
                        )
                        else False
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

    cap.release()
    writer.release()

    # ========================================================
    # OUTPUT
    # ========================================================

    output_df = pd.DataFrame(
        rows
    )

    output_df.to_csv(
        output_csv,
        index=False
    )

    zone_frames = int(
        output_df[
            "zone_available"
        ].sum()
    )

    ball_frames = int(
        output_df[
            "ball_yolo_detected"
        ].sum()
    )

    overlap = output_df[
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

    inside_count = int(
        overlap[
            "ball_inside_estimated_zone"
        ].sum()
    )

    horizontal_inside_count = int(
        overlap[
            "horizontal_inside"
        ].sum()
    )

    vertical_inside_count = int(
        overlap[
            "vertical_inside"
        ].sum()
    )

    print()
    print("=" * 78)
    print(
        "STRIKE ZONE V1.1 COMPLETE"
    )
    print("=" * 78)

    print(
        f"Frames processed          : "
        f"{frame_number}"
    )

    print(
        f"Plate reference samples   : "
        f"{plate_reference_samples}"
    )

    print(
        f"Frozen plate X            : "
        f"{plate_x:.1f} px"
    )

    print(
        f"Strike-zone frames        : "
        f"{zone_frames}"
    )

    print(
        f"Real YOLO ball frames     : "
        f"{ball_frames}"
    )

    print(
        f"Ball + zone overlap       : "
        f"{len(overlap)}"
    )

    print(
        f"Horizontal inside points  : "
        f"{horizontal_inside_count}"
    )

    print(
        f"Vertical inside points    : "
        f"{vertical_inside_count}"
    )

    print(
        f"Full-zone inside points   : "
        f"{inside_count}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Inside/outside refers only to "
        "2D image-plane overlap."
    )

    print(
        "It is NOT yet a confirmed BALL or STRIKE."
    )

    print(
        "Pitch-crossing estimation comes next."
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