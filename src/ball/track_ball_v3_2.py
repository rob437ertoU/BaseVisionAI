from pathlib import Path
import argparse
import math

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = (
    PROJECT_ROOT
    / "runs"
    / "baseball"
    / "baseball_v1-3"
    / "weights"
    / "best.pt"
)

DEFAULT_VIDEO = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "baseball_train10.mp4"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "ball_tracking_v3_2"
    / "baseball_train10"
)


# ============================================================
# DETECTOR CONFIG
# ============================================================

DEVICE = 0
IMGSZ = 1280

# Normal full-frame detector confidence.
NORMAL_CONF = 0.25

# Recovery confidence is intentionally lower,
# but it is used ONLY inside the predicted ROI.
RECOVERY_CONF = 0.08

# ============================================================
# TRACK CONFIG
# ============================================================

MAX_NORMAL_JUMP = 170.0

MAX_RECOVERY_DISTANCE = 90.0

MAX_MISSING_FRAMES = 6

RECOVERY_ROI_RADIUS = 110

# Maximum candidate-size change relative to recent ball size.
MIN_SIZE_RATIO = 0.35
MAX_SIZE_RATIO = 2.8


# ============================================================
# KALMAN CONFIG
# State: [x, y, vx, vy]
# ============================================================

PROCESS_NOISE = 5.0
MEASUREMENT_NOISE = 8.0


# ============================================================
# VISUAL
# ============================================================

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
CYAN = (255, 255, 0)
PURPLE = (255, 0, 255)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Ball Tracking V3.2 Local Recovery"
    )

    parser.add_argument(
        "--video",
        type=str,
        default=str(DEFAULT_VIDEO)
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL)
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

def euclidean(a, b):

    return math.sqrt(
        (a[0] - b[0]) ** 2
        +
        (a[1] - b[1]) ** 2
    )


def draw_text_box(
    frame,
    text,
    x,
    y,
    scale=0.5,
    background=BLACK,
    color=WHITE,
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
# KALMAN
# ============================================================

def create_kalman():

    kf = cv2.KalmanFilter(
        4,
        2
    )

    kf.transitionMatrix = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32
    )

    kf.measurementMatrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.float32
    )

    kf.processNoiseCov = (
        np.eye(
            4,
            dtype=np.float32
        )
        * PROCESS_NOISE
    )

    kf.measurementNoiseCov = (
        np.eye(
            2,
            dtype=np.float32
        )
        * MEASUREMENT_NOISE
    )

    kf.errorCovPost = np.eye(
        4,
        dtype=np.float32
    )

    return kf


def initialize_kalman(kf, x, y):

    kf.statePost = np.array(
        [
            [x],
            [y],
            [0],
            [0],
        ],
        dtype=np.float32
    )


# ============================================================
# DETECTOR
# ============================================================

def detect_full_frame(
    model,
    frame
):

    results = model.predict(
        source=frame,
        conf=NORMAL_CONF,
        imgsz=IMGSZ,
        device=DEVICE,
        verbose=False
    )

    detections = []

    if not results:
        return detections

    result = results[0]

    if result.boxes is None:
        return detections

    for box in result.boxes:

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

        cx = (
            x1 + x2
        ) / 2.0

        cy = (
            y1 + y2
        ) / 2.0

        width = (
            x2 - x1
        )

        height = (
            y2 - y1
        )

        detections.append(
            {
                "confidence": conf,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "cx": float(cx),
                "cy": float(cy),
                "width": float(width),
                "height": float(height),
                "size": float(
                    max(
                        width,
                        height
                    )
                )
            }
        )

    return detections


# ============================================================
# RECOVERY DETECTOR
# ============================================================

def detect_recovery_roi(
    model,
    frame,
    predicted_x,
    predicted_y,
    reference_size
):

    h, w = frame.shape[:2]

    radius = RECOVERY_ROI_RADIUS

    x1 = max(
        0,
        int(
            predicted_x - radius
        )
    )

    y1 = max(
        0,
        int(
            predicted_y - radius
        )
    )

    x2 = min(
        w,
        int(
            predicted_x + radius
        )
    )

    y2 = min(
        h,
        int(
            predicted_y + radius
        )
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):
        return None, (
            x1,
            y1,
            x2,
            y2
        )

    crop = frame[
        y1:y2,
        x1:x2
    ]

    results = model.predict(
        source=crop,
        conf=RECOVERY_CONF,
        imgsz=640,
        device=DEVICE,
        verbose=False
    )

    candidates = []

    if results:

        result = results[0]

        if result.boxes is not None:

            for box in result.boxes:

                conf = float(
                    box.conf[0].item()
                )

                xyxy = (
                    box.xyxy[0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                bx1, by1, bx2, by2 = xyxy

                cx_local = (
                    bx1 + bx2
                ) / 2.0

                cy_local = (
                    by1 + by2
                ) / 2.0

                cx = (
                    x1 + cx_local
                )

                cy = (
                    y1 + cy_local
                )

                width = (
                    bx2 - bx1
                )

                height = (
                    by2 - by1
                )

                size = float(
                    max(
                        width,
                        height
                    )
                )

                distance = euclidean(
                    (
                        cx,
                        cy
                    ),
                    (
                        predicted_x,
                        predicted_y
                    )
                )

                if (
                    distance
                    > MAX_RECOVERY_DISTANCE
                ):
                    continue

                if (
                    reference_size is not None
                    and reference_size > 1
                ):

                    size_ratio = (
                        size
                        / reference_size
                    )

                    if (
                        size_ratio
                        < MIN_SIZE_RATIO
                        or size_ratio
                        > MAX_SIZE_RATIO
                    ):
                        continue

                else:

                    size_ratio = np.nan

                # Higher score is better.
                #
                # Confidence helps,
                # distance to prediction helps more.
                distance_score = (
                    1.0
                    - min(
                        distance
                        / MAX_RECOVERY_DISTANCE,
                        1.0
                    )
                )

                candidate_score = (
                    0.65
                    * distance_score
                    +
                    0.35
                    * conf
                )

                candidates.append(
                    {
                        "confidence":
                            conf,

                        "cx":
                            float(cx),

                        "cy":
                            float(cy),

                        "width":
                            float(width),

                        "height":
                            float(height),

                        "size":
                            size,

                        "distance":
                            float(distance),

                        "size_ratio":
                            float(size_ratio)
                            if np.isfinite(size_ratio)
                            else np.nan,

                        "candidate_score":
                            float(candidate_score)
                    }
                )

    if not candidates:

        return None, (
            x1,
            y1,
            x2,
            y2
        )

    candidates.sort(
        key=lambda item:
            item["candidate_score"],
        reverse=True
    )

    return (
        candidates[0],
        (
            x1,
            y1,
            x2,
            y2
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    video_path = Path(
        args.video
    ).resolve()

    model_path = Path(
        args.model
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    for path in [
        video_path,
        model_path
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
        / "ball_tracking_v3_2.csv"
    )

    output_video = (
        output_dir
        / (
            f"{video_path.stem}"
            "_ball_tracking_v3_2.mp4"
        )
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print(
        f"Loading baseball detector:\n"
        f"{model_path}"
    )

    model = YOLO(
        str(model_path)
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
        (
            width,
            height
        )
    )

    if not writer.isOpened():

        raise RuntimeError(
            "Could not initialize output video."
        )

    # ========================================================
    # STATE
    # ========================================================

    kf = create_kalman()

    initialized = False

    track_id = 0

    missing_frames = 0

    last_observed = None

    last_size = None

    trail = []

    rows = []

    print()
    print("=" * 78)
    print(
        "BaseVisionAI - Ball Tracking V3.2 Local Recovery"
    )
    print("=" * 78)

    print(
        f"Frames          : {total_frames}"
    )

    print(
        f"FPS             : {fps:.2f}"
    )

    print(
        f"Normal conf     : {NORMAL_CONF}"
    )

    print(
        f"Recovery conf   : {RECOVERY_CONF}"
    )

    print(
        f"Recovery radius : {RECOVERY_ROI_RADIUS}px"
    )

    print("=" * 78)

    # ========================================================
    # LOOP
    # ========================================================

    frame_number = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        frame_number += 1

        # ----------------------------------------------------
        # KALMAN PREDICTION
        # ----------------------------------------------------

        predicted_x = np.nan
        predicted_y = np.nan

        if initialized:

            prediction = kf.predict()

            predicted_x = float(
                prediction[0, 0]
            )

            predicted_y = float(
                prediction[1, 0]
            )

        # ----------------------------------------------------
        # FULL-FRAME DETECTION
        # ----------------------------------------------------

        normal_detections = detect_full_frame(
            model,
            frame
        )

        chosen = None

        detection_source = None

        # ----------------------------------------------------
        # WAITING / INITIALIZE
        # ----------------------------------------------------

        if not initialized:

            if normal_detections:

                normal_detections.sort(
                    key=lambda d:
                        d["confidence"],
                    reverse=True
                )

                chosen = normal_detections[0]

                track_id += 1

                initialize_kalman(
                    kf,
                    chosen["cx"],
                    chosen["cy"]
                )

                initialized = True

                missing_frames = 0

                last_observed = (
                    chosen["cx"],
                    chosen["cy"]
                )

                last_size = (
                    chosen["size"]
                )

                state = "TRACKING"

                detection_source = "YOLO"

            else:

                state = "WAITING"

        # ----------------------------------------------------
        # ACTIVE TRACK
        # ----------------------------------------------------

        else:

            # =================================================
            # NORMAL DETECTION ASSOCIATION
            # =================================================

            if normal_detections:

                valid_normal = []

                for det in normal_detections:

                    distance = euclidean(
                        (
                            det["cx"],
                            det["cy"]
                        ),
                        (
                            predicted_x,
                            predicted_y
                        )
                    )

                    if (
                        distance
                        <= MAX_NORMAL_JUMP
                    ):

                        det[
                            "distance"
                        ] = distance

                        valid_normal.append(
                            det
                        )

                if valid_normal:

                    valid_normal.sort(
                        key=lambda d: (
                            d["distance"],
                            -d["confidence"]
                        )
                    )

                    chosen = (
                        valid_normal[0]
                    )

                    detection_source = (
                        "YOLO"
                    )

            # =================================================
            # LOCAL RECOVERY
            # =================================================

            recovery_roi = None

            if chosen is None:

                (
                    recovery,
                    recovery_roi
                ) = detect_recovery_roi(
                    model,
                    frame,
                    predicted_x,
                    predicted_y,
                    last_size
                )

                if recovery is not None:

                    chosen = recovery

                    detection_source = (
                        "RECOVERED"
                    )

            # =================================================
            # UPDATE
            # =================================================

            if chosen is not None:

                measurement = np.array(
                    [
                        [
                            chosen["cx"]
                        ],
                        [
                            chosen["cy"]
                        ]
                    ],
                    dtype=np.float32
                )

                corrected = kf.correct(
                    measurement
                )

                filtered_x = float(
                    corrected[0, 0]
                )

                filtered_y = float(
                    corrected[1, 0]
                )

                last_observed = (
                    chosen["cx"],
                    chosen["cy"]
                )

                last_size = (
                    chosen["size"]
                )

                missing_frames = 0

                if (
                    detection_source
                    == "YOLO"
                ):

                    state = (
                        "TRACKING"
                    )

                else:

                    state = (
                        "RECOVERED"
                    )

            else:

                missing_frames += 1

                filtered_x = (
                    predicted_x
                )

                filtered_y = (
                    predicted_y
                )

                if (
                    missing_frames
                    <= MAX_MISSING_FRAMES
                ):

                    state = (
                        "KALMAN"
                    )

                else:

                    state = (
                        "LOST"
                    )

                    initialized = False

                    last_observed = None
                    last_size = None

                    missing_frames = 0

                    trail.clear()

        # ----------------------------------------------------
        # OUTPUT COORDINATES
        # ----------------------------------------------------

        if chosen is not None:

            observed_x = (
                chosen["cx"]
            )

            observed_y = (
                chosen["cy"]
            )

            confidence = (
                chosen[
                    "confidence"
                ]
            )

            candidate_score = (
                chosen.get(
                    "candidate_score",
                    np.nan
                )
            )

            distance_to_prediction = (
                chosen.get(
                    "distance",
                    np.nan
                )
            )

            if not np.isfinite(
                distance_to_prediction
            ):

                if (
                    np.isfinite(
                        predicted_x
                    )
                    and np.isfinite(
                        predicted_y
                    )
                ):

                    distance_to_prediction = (
                        euclidean(
                            (
                                observed_x,
                                observed_y
                            ),
                            (
                                predicted_x,
                                predicted_y
                            )
                        )
                    )

            yolo_detected = (
                detection_source
                == "YOLO"
            )

            recovered_detected = (
                detection_source
                == "RECOVERED"
            )

        else:

            observed_x = np.nan
            observed_y = np.nan
            confidence = np.nan
            candidate_score = np.nan
            distance_to_prediction = np.nan

            yolo_detected = False
            recovered_detected = False

        # ----------------------------------------------------
        # FILTERED POSITION
        # ----------------------------------------------------

        if initialized:

            if chosen is not None:

                if (
                    'filtered_x'
                    not in locals()
                ):

                    filtered_x = (
                        observed_x
                    )

                    filtered_y = (
                        observed_y
                    )

            else:

                filtered_x = (
                    predicted_x
                )

                filtered_y = (
                    predicted_y
                )

        else:

            filtered_x = np.nan
            filtered_y = np.nan

        # ----------------------------------------------------
        # TRAIL
        # ----------------------------------------------------

        if (
            initialized
            and np.isfinite(
                filtered_x
            )
            and np.isfinite(
                filtered_y
            )
        ):

            trail.append(
                (
                    int(
                        round(
                            filtered_x
                        )
                    ),
                    int(
                        round(
                            filtered_y
                        )
                    )
                )
            )

            trail = trail[
                -15:
            ]

        # ====================================================
        # DRAW
        # ====================================================

        draw_text_box(
            frame,
            "BaseVisionAI - Ball Tracking V3.2",
            30,
            40,
            scale=0.7,
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
            f"State: {state}",
            30,
            115
        )

        if initialized:

            draw_text_box(
                frame,
                f"Track ID: {track_id}",
                30,
                150
            )

        # -----------------------------------------
        # trail
        # -----------------------------------------

        for i in range(
            1,
            len(trail)
        ):

            cv2.line(
                frame,
                trail[i - 1],
                trail[i],
                CYAN,
                2,
                cv2.LINE_AA
            )

        # -----------------------------------------
        # predicted
        # -----------------------------------------

        if (
            np.isfinite(
                predicted_x
            )
            and np.isfinite(
                predicted_y
            )
        ):

            px = int(
                round(
                    predicted_x
                )
            )

            py = int(
                round(
                    predicted_y
                )
            )

            cv2.drawMarker(
                frame,
                (
                    px,
                    py
                ),
                YELLOW,
                cv2.MARKER_TILTED_CROSS,
                15,
                2
            )

        # -----------------------------------------
        # current observation
        # -----------------------------------------

        if chosen is not None:

            cx = int(
                round(
                    observed_x
                )
            )

            cy = int(
                round(
                    observed_y
                )
            )

            if (
                detection_source
                == "YOLO"
            ):

                color = GREEN
                label = (
                    f"YOLO {confidence:.2f}"
                )

            else:

                color = ORANGE
                label = (
                    f"RECOVERED {confidence:.2f}"
                )

            cv2.circle(
                frame,
                (
                    cx,
                    cy
                ),
                11,
                color,
                3
            )

            cv2.circle(
                frame,
                (
                    cx,
                    cy
                ),
                3,
                color,
                -1
            )

            draw_text_box(
                frame,
                label,
                cx + 15,
                cy - 15,
                scale=0.45
            )

        # ====================================================
        # CSV
        # ====================================================

        rows.append(
            {
                "frame":
                    frame_number,

                "track_id":
                    track_id
                    if initialized
                    else np.nan,

                "state":
                    state,

                "detection_source":
                    (
                        detection_source
                        if detection_source
                        is not None
                        else "NONE"
                    ),

                "yolo_detected":
                    yolo_detected,

                "recovered_detected":
                    recovered_detected,

                "confidence":
                    confidence,

                "observed_x":
                    observed_x,

                "observed_y":
                    observed_y,

                "kalman_predicted_x":
                    predicted_x,

                "kalman_predicted_y":
                    predicted_y,

                "filtered_x":
                    filtered_x,

                "filtered_y":
                    filtered_y,

                "distance_to_prediction":
                    distance_to_prediction,

                "candidate_score":
                    candidate_score,

                "missing_frames":
                    missing_frames
            }
        )

        writer.write(
            frame
        )

        # ----------------------------------------------------
        # LOG IMPORTANT FRAMES
        # ----------------------------------------------------

        if (
            state
            in {
                "TRACKING",
                "RECOVERED",
                "KALMAN",
                "LOST"
            }
        ):

            print(
                f"Frame {frame_number:03d} | "
                f"{state:10s} | "
                f"source="
                f"{detection_source or 'NONE':9s}"
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    writer.release()

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        output_csv,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    yolo_count = int(
        df[
            "yolo_detected"
        ].sum()
    )

    recovery_count = int(
        df[
            "recovered_detected"
        ].sum()
    )

    kalman_count = int(
        (
            df["state"]
            == "KALMAN"
        ).sum()
    )

    lost_count = int(
        (
            df["state"]
            == "LOST"
        ).sum()
    )

    print()
    print("=" * 78)
    print(
        "BALL TRACKING V3.2 COMPLETE"
    )
    print("=" * 78)

    print(
        f"Frames processed          : "
        f"{frame_number}"
    )

    print(
        f"Normal YOLO observations  : "
        f"{yolo_count}"
    )

    print(
        f"Recovered observations    : "
        f"{recovery_count}"
    )

    print(
        f"Kalman-only frames        : "
        f"{kalman_count}"
    )

    print(
        f"LOST frames               : "
        f"{lost_count}"
    )

    print()
    print(
        "RECOVERED is an actual detector output from "
        "a localized ROI."
    )

    print(
        "KALMAN remains prediction-only and is not "
        "treated as a real observation."
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