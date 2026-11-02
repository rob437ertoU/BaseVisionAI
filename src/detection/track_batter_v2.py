from pathlib import Path
import argparse
import csv
import math

import cv2
from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = (
    PROJECT_ROOT
    / "runs"
    / "player_roles"
    / "player_roles_v2_yolo26n"
    / "weights"
    / "best.pt"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "batter_tracking_v2"
)


# ============================================================
# CONFIG
# ============================================================

BATTER_CLASS_ID = 0

DEFAULT_CONF = 0.25
DEFAULT_IMGSZ = 1280

EMA_ALPHA = 0.45

MAX_MISSING_FRAMES = 5

MAX_CENTER_JUMP = 220.0


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Batter Tracking V2"
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Input video path"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help="YOLO26 player roles model"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help="Detection confidence threshold"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help="Inference image size"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0"
    )

    return parser.parse_args()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def bbox_center(box):

    x1, y1, x2, y2 = box

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    return cx, cy


def center_distance(box_a, box_b):

    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)

    return math.sqrt(
        (ax - bx) ** 2
        + (ay - by) ** 2
    )


def ema_bbox(previous_box, current_box, alpha):

    if previous_box is None:
        return current_box

    smoothed = []

    for previous, current in zip(
        previous_box,
        current_box
    ):

        value = (
            alpha * current
            + (1.0 - alpha) * previous
        )

        smoothed.append(value)

    return smoothed


# ============================================================
# SELECT BATTER
# ============================================================

def select_batter(
    detections,
    previous_box
):

    if not detections:
        return None

    # --------------------------------------------------------
    # First frame / no previous track
    # --------------------------------------------------------

    if previous_box is None:

        return max(
            detections,
            key=lambda d: d["confidence"]
        )

    # --------------------------------------------------------
    # Spatial consistency
    # --------------------------------------------------------

    candidates = []

    for detection in detections:

        distance = center_distance(
            previous_box,
            detection["bbox"]
        )

        if distance <= MAX_CENTER_JUMP:

            candidates.append(
                (
                    distance,
                    detection
                )
            )

    # --------------------------------------------------------
    # If there are spatially valid candidates,
    # choose nearest one
    # --------------------------------------------------------

    if candidates:

        candidates.sort(
            key=lambda item: item[0]
        )

        return candidates[0][1]

    # --------------------------------------------------------
    # No spatial candidate
    #
    # Reject jump instead of switching identity.
    # --------------------------------------------------------

    return None


# ============================================================
# MAIN TRACKING
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    video_path = Path(args.video).resolve()
    model_path = Path(args.model).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found:\n{video_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    video_name = video_path.stem

    output_video = (
        output_dir
        / f"{video_name}_batter_tracking_v2.mp4"
    )

    output_csv = (
        output_dir
        / f"{video_name}_batter_tracking_v2.csv"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BaseVisionAI - Batter Tracking V2")
    print("=" * 70)

    print(f"Video : {video_path}")
    print(f"Model : {model_path}")
    print(f"Conf  : {args.conf}")
    print(f"ImgSz : {args.imgsz}")

    print("=" * 70)
    print()

    model = YOLO(str(model_path))

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

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

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        fps,
        (width, height)
    )

    # --------------------------------------------------------
    # Tracking state
    # --------------------------------------------------------

    previous_box = None
    smoothed_box = None

    missing_frames = 0

    csv_rows = []

    frame_number = 0

    # ========================================================
    # FRAME LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        results = model.predict(
            source=frame,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False
        )

        result = results[0]

        batter_detections = []

        # ----------------------------------------------------
        # Collect batter detections only
        # ----------------------------------------------------

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls[0].item()
                )

                if class_id != BATTER_CLASS_ID:
                    continue

                confidence = float(
                    box.conf[0].item()
                )

                xyxy = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .tolist()
                )

                x1, y1, x2, y2 = xyxy

                batter_detections.append({
                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2
                    ],
                    "confidence": confidence
                })

        # ----------------------------------------------------
        # Select candidate
        # ----------------------------------------------------

        selected = select_batter(
            batter_detections,
            previous_box
        )

        # ====================================================
        # TRACKING
        # ====================================================

        if selected is not None:

            raw_box = selected["bbox"]
            confidence = selected["confidence"]

            smoothed_box = ema_bbox(
                smoothed_box,
                raw_box,
                EMA_ALPHA
            )

            previous_box = smoothed_box

            missing_frames = 0

            state = "TRACKING"

        else:

            confidence = None

            # ------------------------------------------------
            # Short hold
            # ------------------------------------------------

            if (
                smoothed_box is not None
                and missing_frames
                < MAX_MISSING_FRAMES
            ):

                missing_frames += 1

                state = "HOLD"

            else:

                missing_frames += 1

                state = "LOST"

                previous_box = None
                smoothed_box = None

        # ----------------------------------------------------
        # Draw
        # ----------------------------------------------------

        if smoothed_box is not None:

            x1, y1, x2, y2 = [
                int(v)
                for v in smoothed_box
            ]

            if state == "TRACKING":

                label = (
                    f"BATTER "
                    f"{confidence:.2f}"
                )

            else:

                label = (
                    f"BATTER {state}"
                )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(30, y1 - 10)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cx, cy = bbox_center(
                smoothed_box
            )

        else:

            x1 = y1 = x2 = y2 = None
            cx = cy = None

        # ----------------------------------------------------
        # HUD
        # ----------------------------------------------------

        hud = (
            f"Frame {frame_number}/{total_frames} "
            f"| State: {state} "
            f"| Batter detections: "
            f"{len(batter_detections)}"
        )

        cv2.putText(
            frame,
            hud,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        writer.write(frame)

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        csv_rows.append({
            "frame": frame_number,

            "time_sec":
                (frame_number - 1)
                / fps,

            "state": state,

            "num_batter_detections":
                len(batter_detections),

            "confidence":
                confidence
                if confidence is not None
                else "",

            "x1":
                x1
                if x1 is not None
                else "",

            "y1":
                y1
                if y1 is not None
                else "",

            "x2":
                x2
                if x2 is not None
                else "",

            "y2":
                y2
                if y2 is not None
                else "",

            "center_x":
                cx
                if cx is not None
                else "",

            "center_y":
                cy
                if cy is not None
                else "",
        })

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if frame_number % 30 == 0:

            print(
                f"\rProcessing "
                f"{frame_number}/{total_frames}",
                end="",
                flush=True
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    writer.release()

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer_csv = csv.DictWriter(
            f,
            fieldnames=csv_rows[0].keys()
        )

        writer_csv.writeheader()
        writer_csv.writerows(
            csv_rows
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    tracking_frames = sum(
        row["state"] == "TRACKING"
        for row in csv_rows
    )

    hold_frames = sum(
        row["state"] == "HOLD"
        for row in csv_rows
    )

    lost_frames = sum(
        row["state"] == "LOST"
        for row in csv_rows
    )

    print()
    print()
    print("=" * 70)
    print("TRACKING SUMMARY")
    print("=" * 70)

    print(
        f"Frames total : "
        f"{frame_number}"
    )

    print(
        f"TRACKING     : "
        f"{tracking_frames}"
    )

    print(
        f"HOLD         : "
        f"{hold_frames}"
    )

    print(
        f"LOST         : "
        f"{lost_frames}"
    )

    print()
    print(
        f"Video saved  : "
        f"{output_video}"
    )

    print(
        f"CSV saved    : "
        f"{output_csv}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()