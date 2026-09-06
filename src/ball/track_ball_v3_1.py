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
    / "ball_tracking_v3_1.mp4"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "ball_tracking_v3_1.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

CONF_THRESHOLD = 0.15

# Used only to score candidate proximity.
DISTANCE_NORMALIZER = 250.0

# Kalman predictions allowed without YOLO detection
MAX_KALMAN_GAP = 3

# Number of missed frames before track is reset
RESET_AFTER_MISSING = 6

# Number of trajectory points displayed
TRAIL_LENGTH = 30


# ============================================================
# TRACK STATES
# ============================================================

WAITING = "WAITING"
TRACKING = "TRACKING"
KALMAN = "KALMAN"
LOST = "LOST"
RESET = "RESET"


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def euclidean_distance(p1, p2):
    """
    Calculate Euclidean distance between two points.
    """

    return math.sqrt(
        (p1[0] - p2[0]) ** 2
        +
        (p1[1] - p2[1]) ** 2
    )


def create_kalman_filter():
    """
    Kalman state:
        [x, y, vx, vy]

    Measurements:
        [x, y]
    """

    kalman = cv2.KalmanFilter(4, 2)

    # --------------------------------------------------------
    # State transition matrix
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

    # --------------------------------------------------------
    # Measurement noise
    # --------------------------------------------------------

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
    Initialize Kalman state using first YOLO measurement.
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
    print("BaseVisionAI - Ball Tracker V3.1")
    print("=" * 70)

    print(f"Model: {MODEL_PATH}")
    print(f"Video: {VIDEO_PATH}")

    # ========================================================
    # VALIDATE PATHS
    # ========================================================

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found:\n{VIDEO_PATH}"
        )

    # ========================================================
    # LOAD YOLO MODEL
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
    # TRACKING VARIABLES
    # ========================================================

    kalman = create_kalman_filter()

    kalman_initialized = False

    current_state = WAITING

    missing_frames = 0

    frame_number = 0

    track_id = 0

    reset_logged = False

    # --------------------------------------------------------
    # Global visual trajectory
    #
    # Each item:
    # {
    #   "point": (x, y) or None,
    #   "source": "YOLO" / "KALMAN" / "RESET",
    #   "track_id": int
    # }
    # --------------------------------------------------------

    trajectory = []

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
                "track_id",
                "state",

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
            # YOLO DETECTION
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

                            "confidence": confidence
                        }
                    )

            # =================================================
            # SELECT BEST CANDIDATE
            # =================================================

            selected = None

            selected_distance = None

            selected_score = None

            if candidates:

                # ---------------------------------------------
                # No active Kalman track
                # ---------------------------------------------

                if (
                    not kalman_initialized
                    or
                    predicted_position is None
                ):

                    selected = max(
                        candidates,
                        key=lambda candidate:
                            candidate["confidence"]
                    )

                # ---------------------------------------------
                # Active track
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

                        # Lower score is better
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
            # CASE 1: YOLO DETECTION
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
                # Start a NEW TRACK if needed
                # ---------------------------------------------

                if not kalman_initialized:

                    track_id += 1

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
                    # Correct Kalman using YOLO measurement
                    # -----------------------------------------

                    measurement = np.array(
                        [
                            [np.float32(observed_x)],
                            [np.float32(observed_y)]
                        ],
                        dtype=np.float32
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

                # ---------------------------------------------
                # State update
                # ---------------------------------------------

                current_state = TRACKING

                missing_frames = 0

                reset_logged = False

                # ---------------------------------------------
                # Store trajectory
                # ---------------------------------------------

                trajectory.append(
                    {
                        "point": filtered_position,
                        "source": "YOLO",
                        "track_id": track_id
                    }
                )

                # ---------------------------------------------
                # Draw YOLO bounding box
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
                    f"Baseball {confidence:.2f}",
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
                # Draw filtered point
                # ---------------------------------------------

                cv2.circle(
                    frame,
                    filtered_position,
                    7,
                    (180, 180, 180),
                    2
                )

                # ---------------------------------------------
                # CSV
                # ---------------------------------------------

                csv_writer.writerow(
                    [
                        frame_number,
                        track_id,
                        current_state,

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
            # CASE 2: NO YOLO DETECTION
            # =================================================

            else:

                # ---------------------------------------------
                # No track exists yet
                # ---------------------------------------------

                if not kalman_initialized:

                    current_state = WAITING

                    missing_frames = 0

                    csv_writer.writerow(
                        [
                            frame_number,
                            0,
                            current_state,

                            False,
                            "",

                            "",
                            "",

                            "",
                            "",

                            "",
                            "",

                            "",
                            "",

                            0
                        ]
                    )

                # ---------------------------------------------
                # Active track but YOLO missed
                # ---------------------------------------------

                else:

                    missing_frames += 1

                    # =========================================
                    # KALMAN prediction gap
                    # =========================================

                    if (
                        predicted_position is not None
                        and
                        missing_frames <= MAX_KALMAN_GAP
                    ):

                        current_state = KALMAN

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

                        trajectory.append(
                            {
                                "point": filtered_position,
                                "source": "KALMAN",
                                "track_id": track_id
                            }
                        )

                        # Draw Kalman prediction
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

                        csv_writer.writerow(
                            [
                                frame_number,
                                track_id,
                                current_state,

                                False,
                                "",

                                "",
                                "",

                                predicted_position[0],
                                predicted_position[1],

                                filtered_x,
                                filtered_y,

                                "",
                                "",

                                missing_frames
                            ]
                        )

                    # =========================================
                    # LOST
                    # =========================================

                    elif (
                        missing_frames
                        <
                        RESET_AFTER_MISSING
                    ):

                        current_state = LOST

                        csv_writer.writerow(
                            [
                                frame_number,
                                track_id,
                                current_state,

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

                                "",
                                "",

                                missing_frames
                            ]
                        )

                    # =========================================
                    # RESET
                    # =========================================

                    else:

                        current_state = RESET

                        if not reset_logged:

                            trajectory.append(
                                {
                                    "point": None,
                                    "source": "RESET",
                                    "track_id": track_id
                                }
                            )

                            reset_logged = True

                            csv_writer.writerow(
                                [
                                    frame_number,
                                    track_id,
                                    current_state,

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

                                    "",
                                    "",

                                    missing_frames
                                ]
                            )

                        # -------------------------------------
                        # Fully reset Kalman
                        # -------------------------------------

                        kalman = create_kalman_filter()

                        kalman_initialized = False

                        missing_frames = 0

                        current_state = WAITING

            # =================================================
            # DRAW PREDICTED POINT
            # =================================================

            if (
                predicted_position is not None
                and
                kalman_initialized
            ):

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
                trajectory[-TRAIL_LENGTH:]
            )

            for i in range(
                1,
                len(visible_trajectory)
            ):

                previous_item = (
                    visible_trajectory[
                        i - 1
                    ]
                )

                current_item = (
                    visible_trajectory[i]
                )

                previous_point = (
                    previous_item["point"]
                )

                current_point = (
                    current_item["point"]
                )

                # ---------------------------------------------
                # Break line after RESET
                # ---------------------------------------------

                if (
                    previous_point is None
                    or
                    current_point is None
                ):
                    continue

                # ---------------------------------------------
                # Never connect different track IDs
                # ---------------------------------------------

                if (
                    previous_item["track_id"]
                    !=
                    current_item["track_id"]
                ):
                    continue

                # ---------------------------------------------
                # Thin line for Kalman-only
                # ---------------------------------------------

                if (
                    current_item["source"]
                    ==
                    "KALMAN"
                ):
                    thickness = 1

                else:
                    thickness = 2

                cv2.line(
                    frame,
                    previous_point,
                    current_point,
                    (255, 255, 255),
                    thickness
                )

            # =================================================
            # STATUS INFORMATION
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
                f"Track ID: {track_id if kalman_initialized else '-'}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"State: {current_state}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Candidates: {len(candidates)}",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Missing: {missing_frames}",
                (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # =================================================
            # WRITE FRAME
            # =================================================

            writer.write(
                frame
            )

            # =================================================
            # TERMINAL OUTPUT
            # =================================================

            print(
                f"Frame {frame_number:03d}/{total_frames} | "
                f"Track: "
                f"{track_id if kalman_initialized else '-'} | "
                f"State: {current_state:<8} | "
                f"Candidates: {len(candidates)} | "
                f"Missing: {missing_frames}"
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    writer.release()

    print("\n" + "=" * 70)
    print("BALL TRACKER V3.1 COMPLETED")
    print("=" * 70)

    print("\nVideo:")
    print(OUTPUT_VIDEO)

    print("\nCSV:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()