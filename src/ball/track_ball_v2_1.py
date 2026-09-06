
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

OUTPUT_VIDEO = PROJECT_ROOT / "results" / "ball_tracking_v2_1.mp4"
OUTPUT_CSV = PROJECT_ROOT / "results" / "ball_tracking_v2_1.csv"


# ============================================================
# TRACKING PARAMETERS
# ============================================================

CONF_THRESHOLD = 0.15

# Ya no es una barrera absoluta.
# Se usa para normalizar la penalización por distancia.
MAX_DISTANCE = 250

# Frames sin detección antes de considerar perdido el track
MAX_MISSING_FRAMES = 5

# Longitud máxima de la trayectoria que se dibuja
TRAIL_LENGTH = 30


# ============================================================
# AUXILIARY FUNCTIONS
# ============================================================

def euclidean_distance(p1, p2):
    """
    Calculates Euclidean distance between two points.
    """

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def predict_next_position(history):
    """
    Simple constant-velocity prediction.

    Uses the last two detected positions to estimate
    where the baseball should appear in the next frame.
    """

    if len(history) == 0:
        return None

    if len(history) == 1:
        return history[-1]

    x1, y1 = history[-2]
    x2, y2 = history[-1]

    velocity_x = x2 - x1
    velocity_y = y2 - y1

    predicted_x = int(x2 + velocity_x)
    predicted_y = int(y2 + velocity_y)

    return predicted_x, predicted_y


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("BaseVisionAI - Ball Tracker V2.1")
    print("=" * 60)

    print(f"Model: {MODEL_PATH}")
    print(f"Video: {VIDEO_PATH}")

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found:\n{VIDEO_PATH}"
        )

    # --------------------------------------------------------
    # Load YOLO
    # --------------------------------------------------------

    model = YOLO(str(MODEL_PATH))

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{VIDEO_PATH}"
        )

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

    print(f"FPS: {fps:.2f}")
    print(f"Resolution: {width}x{height}")
    print(f"Frames: {total_frames}")

    # --------------------------------------------------------
    # Output preparation
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Tracking state
    # --------------------------------------------------------

    trajectory = []

    previous_position = None

    frame_number = 0

    missing_frames = 0

    tracking_active = False

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

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
                "candidate_score",
                "missing_frames"
            ]
        )

        # ====================================================
        # VIDEO LOOP
        # ====================================================

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            # ------------------------------------------------
            # Predict next position
            # ------------------------------------------------

            predicted_position = predict_next_position(
                trajectory
            )

            # ------------------------------------------------
            # YOLO inference
            # ------------------------------------------------

            results = model.predict(
                frame,
                imgsz=1280,
                conf=CONF_THRESHOLD,
                device=0,
                verbose=False
            )

            boxes = results[0].boxes

            candidates = []

            # ------------------------------------------------
            # Convert YOLO boxes into candidates
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Select candidate
            # ------------------------------------------------

            selected = None

            selected_distance = None

            selected_score = None

            if candidates:

                # ============================================
                # CASE 1:
                # Tracking has not started yet
                # ============================================

                if not tracking_active:

                    selected = max(
                        candidates,
                        key=lambda candidate:
                            candidate["confidence"]
                    )

                    tracking_active = True

                    selected_score = (
                        1.0 -
                        selected["confidence"]
                    )

                # ============================================
                # CASE 2:
                # Tracking is already active
                # ============================================

                else:

                    reference_position = (
                        predicted_position
                        if predicted_position is not None
                        else previous_position
                    )

                    scored_candidates = []

                    for candidate in candidates:

                        distance = euclidean_distance(
                            reference_position,
                            candidate["center"]
                        )

                        confidence = (
                            candidate["confidence"]
                        )

                        # ------------------------------------
                        # Distance normalization
                        # ------------------------------------

                        distance_score = min(
                            distance / MAX_DISTANCE,
                            2.0
                        )

                        # ------------------------------------
                        # Combined score
                        #
                        # Lower = better
                        #
                        # 65% detection confidence
                        # 35% trajectory consistency
                        # ------------------------------------

                        score = (
                            distance_score * 0.35
                            +
                            (1.0 - confidence) * 0.65
                        )

                        scored_candidates.append(
                            (
                                score,
                                distance,
                                candidate
                            )
                        )

                    # Sort from best to worst
                    scored_candidates.sort(
                        key=lambda item: item[0]
                    )

                    (
                        selected_score,
                        selected_distance,
                        selected
                    ) = scored_candidates[0]

            # ------------------------------------------------
            # Detection selected
            # ------------------------------------------------

            if selected is not None:

                x1, y1, x2, y2 = (
                    selected["bbox"]
                )

                center_x, center_y = (
                    selected["center"]
                )

                confidence = (
                    selected["confidence"]
                )

                current_position = (
                    center_x,
                    center_y
                )

                # Add point to trajectory
                trajectory.append(
                    current_position
                )

                previous_position = (
                    current_position
                )

                missing_frames = 0

                # --------------------------------------------
                # Bounding box
                # --------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )

                # --------------------------------------------
                # Center of baseball
                # --------------------------------------------

                cv2.circle(
                    frame,
                    current_position,
                    5,
                    (255, 255, 255),
                    -1
                )

                # --------------------------------------------
                # Label
                # --------------------------------------------

                cv2.putText(
                    frame,
                    f"Baseball {confidence:.2f}",
                    (
                        x1,
                        max(25, y1 - 10)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                # --------------------------------------------
                # CSV
                # --------------------------------------------

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
                            round(
                                selected_distance,
                                2
                            )
                            if selected_distance
                            is not None
                            else ""
                        ),

                        (
                            round(
                                selected_score,
                                4
                            )
                            if selected_score
                            is not None
                            else ""
                        ),

                        missing_frames
                    ]
                )

            # ------------------------------------------------
            # No detection
            # ------------------------------------------------

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
                        "",
                        missing_frames
                    ]
                )

                # --------------------------------------------
                # If ball has been lost for too long,
                # reset tracking
                # --------------------------------------------

                if (
                    missing_frames
                    >
                    MAX_MISSING_FRAMES
                ):

                    tracking_active = False

                    previous_position = None

                    trajectory = []

            # ------------------------------------------------
            # Draw predicted position
            # ------------------------------------------------

            if predicted_position is not None:

                predicted_x, predicted_y = (
                    predicted_position
                )

                # Only draw if prediction is inside frame
                if (
                    0 <= predicted_x < width
                    and
                    0 <= predicted_y < height
                ):

                    cv2.circle(
                        frame,
                        predicted_position,
                        7,
                        (200, 200, 200),
                        1
                    )

            # ------------------------------------------------
            # Draw trajectory
            # ------------------------------------------------

            visible_trajectory = (
                trajectory[-TRAIL_LENGTH:]
            )

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

            # ------------------------------------------------
            # Debug information
            # ------------------------------------------------

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
                f"Candidates: {len(candidates)}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Missing: {missing_frames}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # ------------------------------------------------
            # Write processed frame
            # ------------------------------------------------

            writer.write(frame)

            # ------------------------------------------------
            # Terminal output
            # ------------------------------------------------

            if selected is not None:

                print(
                    f"Frame {frame_number}/{total_frames} | "
                    f"Candidates: {len(candidates)} | "
                    f"Baseball: YES | "
                    f"Conf: {selected['confidence']:.2f} | "
                    f"Distance: "
                    f"{selected_distance if selected_distance is not None else '-'}"
                )

            else:

                print(
                    f"Frame {frame_number}/{total_frames} | "
                    f"Candidates: {len(candidates)} | "
                    f"Baseball: NO | "
                    f"Missing: {missing_frames}"
                )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    writer.release()

    print("\n" + "=" * 60)
    print("BALL TRACKER V2.1 COMPLETED")
    print("=" * 60)

    print(f"Video saved to:")
    print(OUTPUT_VIDEO)

    print("\nCSV saved to:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()