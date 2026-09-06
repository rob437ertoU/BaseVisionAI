from ultralytics import YOLO
from pathlib import Path
import cv2
import csv
import math
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "baseball"
    / "baseball_v1-3"
    / "weights"
    / "best.pt"
)

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "baseball_test.mp4"
)

OUTPUT_VIDEO = (
    PROJECT_ROOT
    / "results"
    / "ball_tracking_v3_kalman.mp4"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "ball_tracking_v3_kalman.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

CONF_THRESHOLD = 0.15

# Distancia utilizada como referencia para puntuar candidatos.
# NO elimina automáticamente detecciones.
DISTANCE_NORMALIZER = 250.0

# Máximo número de frames donde permitimos que Kalman
# dibuje una posición sin tener medición YOLO.
MAX_KALMAN_GAP = 3

# Después de esta cantidad de frames sin YOLO,
# damos el track por perdido.
RESET_AFTER_MISSING = 6

# Longitud visual de trayectoria
TRAIL_LENGTH = 30


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def euclidean_distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2
        +
        (p1[1] - p2[1]) ** 2
    )


def create_kalman_filter():
    """
    State:
        [x, y, vx, vy]

    Measurement:
        [x, y]
    """

    kalman = cv2.KalmanFilter(4, 2)

    # --------------------------------------------------------
    # State transition matrix
    #
    # x_next = x + vx
    # y_next = y + vy
    # vx_next = vx
    # vy_next = vy
    # --------------------------------------------------------

    kalman.transitionMatrix = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Measurement matrix
    #
    # We measure only x and y
    # --------------------------------------------------------

    kalman.measurementMatrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Process noise
    #
    # Larger values let Kalman react more quickly
    # to direction changes.
    # --------------------------------------------------------

    kalman.processNoiseCov = np.array(
        [
            [1e-2, 0,    0,    0],
            [0,    1e-2, 0,    0],
            [0,    0,    5e-2, 0],
            [0,    0,    0,    5e-2],
        ],
        dtype=np.float32
    )

    # YOLO measurement noise
    kalman.measurementNoiseCov = np.array(
        [
            [5e-1, 0],
            [0, 5e-1],
        ],
        dtype=np.float32
    )

    kalman.errorCovPost = np.eye(
        4,
        dtype=np.float32
    )

    return kalman


def initialize_kalman(kalman, x, y):
    """
    Initialize tracker with first YOLO detection.
    """

    state = np.array(
        [
            [float(x)],
            [float(y)],
            [0.0],
            [0.0],
        ],
        dtype=np.float32
    )

    kalman.statePre = state.copy()
    kalman.statePost = state.copy()

    kalman.errorCovPost = np.eye(
        4,
        dtype=np.float32
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BaseVisionAI - Ball Tracker V3 + Kalman Filter")
    print("=" * 70)

    print(f"Model: {MODEL_PATH}")
    print(f"Video: {VIDEO_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found:\n{VIDEO_PATH}"
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = YOLO(str(MODEL_PATH))

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
    # OUTPUT
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
    # TRACKING STATE
    # ========================================================

    kalman = create_kalman_filter()

    kalman_initialized = False

    tracking_active = False

    missing_frames = 0

    frame_number = 0

    # Filtered trajectory used for drawing
    filtered_trajectory = []

    # Keep track of which points were real detections
    trajectory_sources = []

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
                "source",
                "yolo_detected",
                "confidence",

                "observed_x",
                "observed_y",

                "kalman_predicted_x",
                "kalman_predicted_y",

                "filtered_x",
                "filtered_y",

                "distance_to_prediction",
                "candidate_score",

                "missing_frames"
            ]
        )

        # ====================================================
        # FRAME LOOP
        # ====================================================

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            # =================================================
            # KALMAN PREDICTION
            # =================================================

            predicted_position = None

            if kalman_initialized:

                prediction = kalman.predict()

                predicted_x = int(
                    prediction[0, 0]
                )

                predicted_y = int(
                    prediction[1, 0]
                )

                predicted_position = (
                    predicted_x,
                    predicted_y
                )

            # =================================================
            # YOLO
            # =================================================

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

                            "confidence":
                                confidence
                        }
                    )

            # =================================================
            # CANDIDATE SELECTION
            # =================================================

            selected = None

            selected_distance = None

            selected_score = None

            if candidates:

                # ---------------------------------------------
                # No active track:
                # choose highest confidence
                # ---------------------------------------------

                if (
                    not tracking_active
                    or
                    predicted_position is None
                ):

                    selected = max(
                        candidates,
                        key=lambda candidate:
                            candidate["confidence"]
                    )

                # ---------------------------------------------
                # Active track:
                # combine confidence + distance to Kalman
                # ---------------------------------------------

                else:

                    scored_candidates = []

                    for candidate in candidates:

                        distance = euclidean_distance(
                            predicted_position,
                            candidate["center"]
                        )

                        confidence = (
                            candidate["confidence"]
                        )

                        distance_score = min(
                            distance
                            /
                            DISTANCE_NORMALIZER,
                            2.0
                        )

                        # Lower score = better
                        #
                        # Confidence has greater influence than
                        # Kalman distance because the ball can
                        # change direction after contact.
                        score = (
                            distance_score * 0.30
                            +
                            (1.0 - confidence) * 0.70
                        )

                        scored_candidates.append(
                            (
                                score,
                                distance,
                                candidate
                            )
                        )

                    scored_candidates.sort(
                        key=lambda item:
                            item[0]
                    )

                    (
                        selected_score,
                        selected_distance,
                        selected
                    ) = scored_candidates[0]

            # =================================================
            # YOLO DETECTION AVAILABLE
            # =================================================

            if selected is not None:

                x1, y1, x2, y2 = (
                    selected["bbox"]
                )

                observed_x, observed_y = (
                    selected["center"]
                )

                confidence = (
                    selected["confidence"]
                )

                # ---------------------------------------------
                # Initialize Kalman on first measurement
                # ---------------------------------------------

                if not kalman_initialized:

                    initialize_kalman(
                        kalman,
                        observed_x,
                        observed_y
                    )

                    kalman_initialized = True

                    filtered_x = observed_x
                    filtered_y = observed_y

                else:

                    # -----------------------------------------
                    # Correct prediction using YOLO measurement
                    # -----------------------------------------

                    measurement = np.array(
                        [
                            [np.float32(observed_x)],
                            [np.float32(observed_y)]
                        ]
                    )

                    corrected = kalman.correct(
                        measurement
                    )

                    filtered_x = int(
                        corrected[0, 0]
                    )

                    filtered_y = int(
                        corrected[1, 0]
                    )

                filtered_position = (
                    filtered_x,
                    filtered_y
                )

                tracking_active = True

                missing_frames = 0

                # ---------------------------------------------
                # Save trajectory
                # ---------------------------------------------

                filtered_trajectory.append(
                    filtered_position
                )

                trajectory_sources.append(
                    "YOLO"
                )

                # ---------------------------------------------
                # YOLO bounding box
                # ---------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )

                cv2.circle(
                    frame,
                    (observed_x, observed_y),
                    5,
                    (255, 255, 255),
                    -1
                )

                cv2.putText(
                    frame,
                    f"YOLO {confidence:.2f}",
                    (
                        x1,
                        max(25, y1 - 10)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                # ---------------------------------------------
                # Filtered Kalman position
                # ---------------------------------------------

                cv2.circle(
                    frame,
                    filtered_position,
                    7,
                    (180, 180, 180),
                    2
                )

                source = "YOLO+KALMAN"

                csv_writer.writerow(
                    [
                        frame_number,
                        source,
                        True,
                        confidence,

                        observed_x,
                        observed_y,

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

                        filtered_x,
                        filtered_y,

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

            # =================================================
            # NO YOLO DETECTION
            # =================================================

            else:

                missing_frames += 1

                confidence = ""

                observed_x = ""
                observed_y = ""

                filtered_x = ""
                filtered_y = ""

                source = "LOST"

                # ---------------------------------------------
                # Short gap:
                # use Kalman prediction
                # ---------------------------------------------

                if (
                    kalman_initialized
                    and
                    predicted_position is not None
                    and
                    missing_frames <= MAX_KALMAN_GAP
                ):

                    filtered_x = (
                        predicted_position[0]
                    )

                    filtered_y = (
                        predicted_position[1]
                    )

                    filtered_position = (
                        filtered_x,
                        filtered_y
                    )

                    filtered_trajectory.append(
                        filtered_position
                    )

                    trajectory_sources.append(
                        "KALMAN"
                    )

                    source = "KALMAN"

                    # Dashed-style visual representation
                    cv2.circle(
                        frame,
                        filtered_position,
                        8,
                        (150, 150, 150),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Kalman prediction",
                        (
                            filtered_x + 10,
                            filtered_y
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (180, 180, 180),
                        2
                    )

                # ---------------------------------------------
                # Track lost for too long
                # ---------------------------------------------

                if (
                    missing_frames
                    >=
                    RESET_AFTER_MISSING
                ):

                    tracking_active = False

                    kalman_initialized = False

                    kalman = create_kalman_filter()

                    # Start a new visual segment
                    filtered_trajectory.append(
                        None
                    )

                    trajectory_sources.append(
                        "RESET"
                    )

                    source = "RESET"

                csv_writer.writerow(
                    [
                        frame_number,
                        source,
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

                        filtered_x,
                        filtered_y,

                        "",
                        "",

                        missing_frames
                    ]
                )

            # =================================================
            # DRAW KALMAN PREDICTED POSITION
            # =================================================

            if predicted_position is not None:

                px, py = predicted_position

                if (
                    0 <= px < width
                    and
                    0 <= py < height
                ):

                    cv2.circle(
                        frame,
                        predicted_position,
                        5,
                        (120, 120, 120),
                        1
                    )

            # =================================================
            # DRAW TRAJECTORY
            # =================================================

            visible_trajectory = (
                filtered_trajectory[
                    -TRAIL_LENGTH:
                ]
            )

            visible_sources = (
                trajectory_sources[
                    -TRAIL_LENGTH:
                ]
            )

            for i in range(
                1,
                len(visible_trajectory)
            ):

                point_previous = (
                    visible_trajectory[
                        i - 1
                    ]
                )

                point_current = (
                    visible_trajectory[i]
                )

                if (
                    point_previous is None
                    or
                    point_current is None
                ):
                    continue

                source_current = (
                    visible_sources[i]
                )

                # ---------------------------------------------
                # Solid line = YOLO corrected
                # Thinner line = Kalman-only prediction
                # ---------------------------------------------

                if source_current == "KALMAN":

                    thickness = 1

                else:

                    thickness = 2

                cv2.line(
                    frame,
                    point_previous,
                    point_current,
                    (255, 255, 255),
                    thickness
                )

            # =================================================
            # STATUS
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

            cv2.putText(
                frame,
                f"Source: {source}",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # =================================================
            # SAVE FRAME
            # =================================================

            writer.write(
                frame
            )

            print(
                f"Frame {frame_number:03d}/{total_frames} | "
                f"Candidates: {len(candidates)} | "
                f"Source: {source} | "
                f"Missing: {missing_frames}"
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    writer.release()

    print("\n" + "=" * 70)
    print("BALL TRACKER V3 COMPLETED")
    print("=" * 70)

    print("\nVideo:")
    print(OUTPUT_VIDEO)

    print("\nCSV:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()