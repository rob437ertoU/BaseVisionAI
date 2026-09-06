from ultralytics import YOLO
from pathlib import Path
import cv2
import csv


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

OUTPUT_VIDEO = (
    PROJECT_ROOT
    / "results"
    / "player_detection_v1.mp4"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "player_detection_v1.csv"
)


# ============================================================
# MODEL
# ============================================================

# Generic YOLO model
MODEL_NAME = "yolo26n.pt"

# Confidence for person detection
CONF_THRESHOLD = 0.35

# COCO class id:
# person = 0
PERSON_CLASS_ID = 0


# ============================================================
# ROLE PARAMETERS
# ============================================================
#
# IMPORTANT:
# These rules are designed for the current broadcast camera:
#
# pitcher in the lower/central field
# batter near home plate
# catcher/umpire behind batter
#
# They are deliberately simple for V1.
# ============================================================


def assign_role(
    center_x,
    center_y,
    box_width,
    box_height,
    frame_width,
    frame_height
):
    """
    Assign an approximate baseball role using image position.

    Returns:
        PITCHER
        BATTER
        CATCHER_UMPIRE
        OTHER_PLAYER
    """

    # Convert pixel coordinates to normalized image coordinates
    nx = center_x / frame_width
    ny = center_y / frame_height

    # ========================================================
    # PITCHER
    # ========================================================
    #
    # In this camera, pitcher is typically:
    # - around center horizontally
    # - relatively low in the image
    # ========================================================

    if (
        0.30 <= nx <= 0.65
        and
        0.45 <= ny <= 0.90
    ):
        return "PITCHER"

    # ========================================================
    # BATTER / HOME PLATE AREA
    # ========================================================
    #
    # Batter appears near the upper-center/right portion
    # depending on handedness/camera framing.
    # ========================================================

    if (
        0.45 <= nx <= 0.78
        and
        0.15 <= ny <= 0.55
    ):
        return "BATTER"

    # ========================================================
    # CATCHER / UMPIRE
    # ========================================================
    #
    # Both can overlap heavily in broadcast footage.
    # For V1 we keep them together.
    # ========================================================

    if (
        0.30 <= nx <= 0.60
        and
        0.12 <= ny <= 0.50
    ):
        return "CATCHER_UMPIRE"

    return "OTHER_PLAYER"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BaseVisionAI - Player Detection V1")
    print("=" * 70)

    print(f"Video: {VIDEO_PATH}")
    print(f"Model: {MODEL_NAME}")

    # ========================================================
    # CHECK VIDEO
    # ========================================================

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found:\n{VIDEO_PATH}"
        )

    # ========================================================
    # LOAD YOLO
    # ========================================================

    model = YOLO(MODEL_NAME)

    # ========================================================
    # OPEN VIDEO
    # ========================================================

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{VIDEO_PATH}"
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

    print(f"FPS: {fps:.2f}")
    print(f"Resolution: {width} x {height}")
    print(f"Frames: {total_frames}")

    # ========================================================
    # OUTPUT VIDEO
    # ========================================================

    OUTPUT_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height)
    )

    # ========================================================
    # CSV
    # ========================================================

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        csv_writer = csv.writer(
            csv_file
        )

        csv_writer.writerow(
            [
                "frame",
                "detection_id",
                "role",
                "confidence",

                "x1",
                "y1",
                "x2",
                "y2",

                "center_x",
                "center_y",

                "box_width",
                "box_height",

                "normalized_x",
                "normalized_y"
            ]
        )

        # ====================================================
        # FRAME LOOP
        # ====================================================

        frame_number = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            # =================================================
            # YOLO INFERENCE
            # =================================================

            results = model.predict(
                frame,
                imgsz=1280,
                conf=CONF_THRESHOLD,
                device=0,
                classes=[PERSON_CLASS_ID],
                verbose=False
            )

            boxes = results[0].boxes

            detections = []

            # =================================================
            # PROCESS PERSON BOXES
            # =================================================

            if boxes is not None:

                for detection_id, box in enumerate(
                    boxes,
                    start=1
                ):

                    confidence = float(
                        box.conf[0]
                    )

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )

                    center_x = int(
                        (x1 + x2) / 2
                    )

                    center_y = int(
                        (y1 + y2) / 2
                    )

                    box_width = (
                        x2 - x1
                    )

                    box_height = (
                        y2 - y1
                    )

                    normalized_x = (
                        center_x / width
                    )

                    normalized_y = (
                        center_y / height
                    )

                    role = assign_role(
                        center_x,
                        center_y,
                        box_width,
                        box_height,
                        width,
                        height
                    )

                    detections.append(
                        {
                            "id": detection_id,
                            "role": role,
                            "confidence": confidence,
                            "bbox": (
                                x1,
                                y1,
                                x2,
                                y2
                            ),
                            "center": (
                                center_x,
                                center_y
                            ),
                            "box_width": box_width,
                            "box_height": box_height,
                            "normalized_x":
                                normalized_x,
                            "normalized_y":
                                normalized_y
                        }
                    )

                    # =========================================
                    # WRITE CSV
                    # =========================================

                    csv_writer.writerow(
                        [
                            frame_number,
                            detection_id,
                            role,
                            confidence,

                            x1,
                            y1,
                            x2,
                            y2,

                            center_x,
                            center_y,

                            box_width,
                            box_height,

                            normalized_x,
                            normalized_y
                        ]
                    )

            # =================================================
            # DRAW DETECTIONS
            # =================================================

            for detection in detections:

                x1, y1, x2, y2 = (
                    detection["bbox"]
                )

                center_x, center_y = (
                    detection["center"]
                )

                role = (
                    detection["role"]
                )

                confidence = (
                    detection["confidence"]
                )

                # ------------------------------------------------
                # Bounding box
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )

                # ------------------------------------------------
                # Center point
                # ------------------------------------------------

                cv2.circle(
                    frame,
                    (center_x, center_y),
                    4,
                    (255, 255, 255),
                    -1
                )

                # ------------------------------------------------
                # Label
                # ------------------------------------------------

                label = (
                    f"{role} "
                    f"{confidence:.2f}"
                )

                cv2.putText(
                    frame,
                    label,
                    (
                        x1,
                        max(
                            25,
                            y1 - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

            # =================================================
            # FRAME INFORMATION
            # =================================================

            cv2.putText(
                frame,
                f"Frame: {frame_number}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"People: {len(detections)}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # =================================================
            # ROLE COUNTS
            # =================================================

            pitcher_count = sum(
                1
                for d in detections
                if d["role"] == "PITCHER"
            )

            batter_count = sum(
                1
                for d in detections
                if d["role"] == "BATTER"
            )

            catcher_count = sum(
                1
                for d in detections
                if d["role"]
                == "CATCHER_UMPIRE"
            )

            cv2.putText(
                frame,
                f"Pitcher: {pitcher_count}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Batter: {batter_count}",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Catcher/Umpire: {catcher_count}",
                (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            # =================================================
            # WRITE VIDEO
            # =================================================

            writer.write(
                frame
            )

            # =================================================
            # TERMINAL
            # =================================================

            print(
                f"Frame {frame_number:03d}/"
                f"{total_frames} | "
                f"People: {len(detections)} | "
                f"P: {pitcher_count} | "
                f"B: {batter_count} | "
                f"C/U: {catcher_count}"
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    writer.release()

    print("\n" + "=" * 70)
    print("PLAYER DETECTION V1 COMPLETED")
    print("=" * 70)

    print("\nVideo:")
    print(OUTPUT_VIDEO)

    print("\nCSV:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()