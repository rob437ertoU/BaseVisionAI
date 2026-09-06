from ultralytics import YOLO
from pathlib import Path
import cv2
import csv
import math


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "baseball"
    / "baseball_v1-3"
    / "weights"
    / "best.pt"
)

VIDEO_PATH = PROJECT_ROOT / "data" / "raw" / "baseball_test.mp4"

OUTPUT_VIDEO = PROJECT_ROOT / "results" / "ball_tracking_v2.mp4"
OUTPUT_CSV = PROJECT_ROOT / "results" / "ball_tracking_v2.csv"


# -----------------------------
# Tracking parameters
# -----------------------------

CONF_THRESHOLD = 0.15

# Distancia máxima permitida entre posiciones consecutivas
MAX_DISTANCE = 150

# Cuántos frames permitimos perder la pelota antes de reiniciar el track
MAX_MISSING_FRAMES = 5

# Número máximo de puntos visibles en la trayectoria
TRAIL_LENGTH = 30


def euclidean_distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def predict_next_position(history):
    """
    Predicción básica usando velocidad entre los últimos dos puntos.
    """

    if len(history) < 2:
        return history[-1] if history else None

    x1, y1 = history[-2]
    x2, y2 = history[-1]

    velocity_x = x2 - x1
    velocity_y = y2 - y1

    predicted_x = int(x2 + velocity_x)
    predicted_y = int(y2 + velocity_y)

    return predicted_x, predicted_y


def main():

    print("=" * 60)
    print("BaseVisionAI - Ball Tracker V2")
    print("=" * 60)

    print(f"Model: {MODEL_PATH}")
    print(f"Video: {VIDEO_PATH}")

    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")
    print(f"Frames: {total_frames}")

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

    trajectory = []

    frame_number = 0

    missing_frames = 0

    tracking_active = False

    previous_position = None

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        csv_writer = csv.writer(csv_file)

        csv_writer.writerow(
            [
                "frame",
                "detected",
                "x_center",
                "y_center",
                "confidence",
                "predicted_x",
                "predicted_y",
                "distance_to_prediction",
                "missing_frames"
            ]
        )

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            # -----------------------------
            # Predict next ball position
            # -----------------------------

            predicted_position = predict_next_position(
                trajectory
            )

            # -----------------------------
            # YOLO detection
            # -----------------------------

            results = model.predict(
                frame,
                imgsz=1280,
                conf=CONF_THRESHOLD,
                device=0,
                verbose=False
            )

            boxes = results[0].boxes

            candidates = []

            if boxes is not None:

                for box in boxes:

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

                    candidates.append(
                        {
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
                            "confidence": confidence
                        }
                    )

            selected = None
            selected_distance = None

            # -----------------------------
            # Choose candidate
            # -----------------------------

            if candidates:

                # Si todavía no tenemos track,
                # iniciamos con la detección
                # de mayor confianza

                if not tracking_active:

                    selected = max(
                        candidates,
                        key=lambda x: x["confidence"]
                    )

                    tracking_active = True

                else:

                    reference_position = (
                        predicted_position
                        if predicted_position is not None
                        else previous_position
                    )

                    best_distance = float("inf")

                    for candidate in candidates:

                        distance = euclidean_distance(
                            reference_position,
                            candidate["center"]
                        )

                        if (
                            distance < best_distance
                            and distance <= MAX_DISTANCE
                        ):

                            best_distance = distance
                            selected = candidate

                    if selected is not None:
                        selected_distance = best_distance

            # -----------------------------
            # Detection found
            # -----------------------------

            if selected is not None:

                x1, y1, x2, y2 = selected["bbox"]

                center_x, center_y = selected["center"]

                confidence = selected["confidence"]

                current_position = (
                    center_x,
                    center_y
                )

                trajectory.append(
                    current_position
                )

                previous_position = (
                    current_position
                )

                missing_frames = 0

                # Bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )

                # Ball center
                cv2.circle(
                    frame,
                    current_position,
                    5,
                    (255, 255, 255),
                    -1
                )

                # Label
                cv2.putText(
                    frame,
                    f"Baseball {confidence:.2f}",
                    (
                        x1,
                        max(20, y1 - 10)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                csv_writer.writerow(
                    [
                        frame_number,
                        True,
                        center_x,
                        center_y,
                        confidence,
                        (
                            predicted_position[0]
                            if predicted_position
                            else ""
                        ),
                        (
                            predicted_position[1]
                            if predicted_position
                            else ""
                        ),
                        (
                            selected_distance
                            if selected_distance is not None
                            else ""
                        ),
                        missing_frames
                    ]
                )

            # -----------------------------
            # No detection
            # -----------------------------

            else:

                missing_frames += 1

                csv_writer.writerow(
                    [
                        frame_number,
                        False,
                        "",
                        "",
                        "",
                        (
                            predicted_position[0]
                            if predicted_position
                            else ""
                        ),
                        (
                            predicted_position[1]
                            if predicted_position
                            else ""
                        ),
                        "",
                        missing_frames
                    ]
                )

                # Reset track after too many missing frames
                if missing_frames > MAX_MISSING_FRAMES:

                    tracking_active = False

                    previous_position = None

                    trajectory = []

            # -----------------------------
            # Draw prediction
            # -----------------------------

            if predicted_position is not None:

                cv2.circle(
                    frame,
                    predicted_position,
                    6,
                    (200, 200, 200),
                    1
                )

            # -----------------------------
            # Draw trajectory
            # -----------------------------

            visible_trajectory = trajectory[
                -TRAIL_LENGTH:
            ]

            for i in range(
                1,
                len(visible_trajectory)
            ):

                cv2.line(
                    frame,
                    visible_trajectory[i - 1],
                    visible_trajectory[i],
                    (255, 255, 255),
                    2
                )

            # -----------------------------
            # Status text
            # -----------------------------

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
                f"Missing: {missing_frames}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            writer.write(frame)

            print(
                f"Frame {frame_number}/{total_frames} | "
                f"Candidates: {len(candidates)} | "
                f"Selected: {selected is not None} | "
                f"Missing: {missing_frames}"
            )

    cap.release()

    writer.release()

    print("\n" + "=" * 60)
    print("BALL TRACKER V2 COMPLETED")
    print("=" * 60)

    print(f"Video: {OUTPUT_VIDEO}")
    print(f"CSV:   {OUTPUT_CSV}")


if __name__ == "__main__":
    main()