from pathlib import Path
import argparse
import pandas as pd
import cv2
import math


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# PARAMETERS
# ============================================================

# Weight of new observation in exponential smoothing.
#
# Higher:
#   follows detections faster
#
# Lower:
#   smoother but slower
SMOOTHING_ALPHA = 0.45

# Maximum frames where the last known batter box is retained
# if detection disappears temporarily.
MAX_MISSING_FRAMES = 4

# Reject very large sudden center jumps.
MAX_CENTER_JUMP_PX = 180

# Minimum YOLO confidence for batter observation.
MIN_CONFIDENCE = 0.25


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Batter Tracking V1"
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Input video path"
    )

    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Player detection CSV path"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where results will be saved"
    )

    return parser.parse_args()


# ============================================================
# FUNCTIONS
# ============================================================

def center_of_box(box):
    """
    Return center point of bounding box.
    """

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    )


def box_distance(box1, box2):
    """
    Distance between box centers.
    """

    c1 = center_of_box(box1)
    c2 = center_of_box(box2)

    return math.sqrt(
        (c1[0] - c2[0]) ** 2
        +
        (c1[1] - c2[1]) ** 2
    )


def smooth_value(previous, current, alpha):
    """
    Exponential moving average.
    """

    return (
        alpha * current
        +
        (1.0 - alpha) * previous
    )


def smooth_box(previous_box, current_box):
    """
    Apply exponential smoothing to bbox coordinates.
    """

    if previous_box is None:
        return tuple(
            float(v)
            for v in current_box
        )

    return (
        smooth_value(
            previous_box[0],
            current_box[0],
            SMOOTHING_ALPHA
        ),
        smooth_value(
            previous_box[1],
            current_box[1],
            SMOOTHING_ALPHA
        ),
        smooth_value(
            previous_box[2],
            current_box[2],
            SMOOTHING_ALPHA
        ),
        smooth_value(
            previous_box[3],
            current_box[3],
            SMOOTHING_ALPHA
        )
    )


def clip_box(
    box,
    frame_width,
    frame_height
):
    """
    Keep bbox inside frame.
    """

    x1, y1, x2, y2 = box

    x1 = max(
        0,
        min(
            int(round(x1)),
            frame_width - 1
        )
    )

    y1 = max(
        0,
        min(
            int(round(y1)),
            frame_height - 1
        )
    )

    x2 = max(
        0,
        min(
            int(round(x2)),
            frame_width - 1
        )
    )

    y2 = max(
        0,
        min(
            int(round(y2)),
            frame_height - 1
        )
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


def get_batter_detection(
    batter_rows,
    frame_number
):
    """
    Get batter detection for one frame.

    If multiple batter rows exist, take highest confidence.
    """

    frame_rows = batter_rows[
        batter_rows["frame"]
        ==
        frame_number
    ]

    if frame_rows.empty:
        return None

    frame_rows = (
        frame_rows[
            frame_rows["confidence"]
            >=
            MIN_CONFIDENCE
        ]
    )

    if frame_rows.empty:
        return None

    best_row = frame_rows.loc[
        frame_rows[
            "confidence"
        ].idxmax()
    ]

    return best_row


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # ========================================================
    # RESOLVE INPUT PATHS
    # ========================================================

    video_path = Path(args.video)

    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path

    video_path = video_path.resolve()

    player_csv = Path(args.input_csv)

    if not player_csv.is_absolute():
        player_csv = PROJECT_ROOT / player_csv

    player_csv = player_csv.resolve()

    # ========================================================
    # RESOLVE OUTPUT PATHS
    # ========================================================

    output_dir = Path(args.output_dir)

    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    output_dir = output_dir.resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_video = (
        output_dir
        / "batter_tracking_v1.mp4"
    )

    output_csv = (
        output_dir
        / "batter_tracking_v1.csv"
    )

    # ========================================================
    # HEADER
    # ========================================================

    print("=" * 70)
    print("BaseVisionAI - Batter Tracking V1")
    print("=" * 70)

    print(f"Video: {video_path}")
    print(f"Player CSV: {player_csv}")
    print(f"Output directory: {output_dir}")

    # ========================================================
    # VALIDATE INPUTS
    # ========================================================

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video not found:\n"
            f"{video_path}"
        )

    if not player_csv.exists():

        raise FileNotFoundError(
            f"Player CSV not found:\n"
            f"{player_csv}"
        )

    # ========================================================
    # LOAD PLAYER DATA
    # ========================================================

    player_df = pd.read_csv(
        player_csv
    )

    required_columns = {
        "frame",
        "role",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2"
    }

    missing_columns = (
        required_columns
        -
        set(player_df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Player CSV is missing required columns: "
            +
            ", ".join(
                sorted(missing_columns)
            )
        )

    batter_df = player_df[
        player_df["role"]
        ==
        "BATTER"
    ].copy()

    print(
        f"Batter rows found: "
        f"{len(batter_df)}"
    )

    # ========================================================
    # OPEN VIDEO
    # ========================================================

    cap = cv2.VideoCapture(
        str(
            video_path
        )
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n"
            f"{video_path}"
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

    print(
        f"FPS: {fps:.2f}"
    )

    print(
        f"Resolution: "
        f"{width}x{height}"
    )

    print(
        f"Frames: "
        f"{total_frames}"
    )

    # ========================================================
    # VIDEO OUTPUT
    # ========================================================

    fourcc = (
        cv2.VideoWriter_fourcc(
            *"mp4v"
        )
    )

    writer = cv2.VideoWriter(
        str(
            output_video
        ),
        fourcc,
        fps,
        (
            width,
            height
        )
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            f"Could not create output video:\n"
            f"{output_video}"
        )

    # ========================================================
    # TRACK STATE
    # ========================================================

    previous_box = None
    smoothed_box = None

    missing_frames = 0

    track_active = False

    frame_number = 0

    # ========================================================
    # OUTPUT CSV
    # ========================================================

    output_rows = []

    # ========================================================
    # FRAME LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        detection = (
            get_batter_detection(
                batter_df,
                frame_number
            )
        )

        detected = False
        confidence = None

        raw_box = None

        state = "WAITING"

        # ====================================================
        # BATTER DETECTED
        # ====================================================

        if detection is not None:

            confidence = float(
                detection[
                    "confidence"
                ]
            )

            raw_box = (
                float(
                    detection["x1"]
                ),
                float(
                    detection["y1"]
                ),
                float(
                    detection["x2"]
                ),
                float(
                    detection["y2"]
                )
            )

            # =================================================
            # FIRST DETECTION
            # =================================================

            if previous_box is None:

                smoothed_box = raw_box

                track_active = True

                missing_frames = 0

                detected = True

                state = "TRACKING"

            # =================================================
            # EXISTING TRACK
            # =================================================

            else:

                jump = box_distance(
                    previous_box,
                    raw_box
                )

                # ---------------------------------------------
                # Accept reasonable movement
                # ---------------------------------------------

                if (
                    jump
                    <=
                    MAX_CENTER_JUMP_PX
                ):

                    smoothed_box = (
                        smooth_box(
                            smoothed_box,
                            raw_box
                        )
                    )

                    previous_box = raw_box

                    missing_frames = 0

                    detected = True

                    track_active = True

                    state = "TRACKING"

                # ---------------------------------------------
                # Suspicious jump
                # ---------------------------------------------

                else:

                    missing_frames += 1

                    state = "REJECTED_JUMP"

        # ====================================================
        # NO DETECTION
        # ====================================================

        else:

            if track_active:

                missing_frames += 1

                if (
                    missing_frames
                    <=
                    MAX_MISSING_FRAMES
                ):

                    state = "HOLD"

                else:

                    track_active = False
                    previous_box = None
                    smoothed_box = None

                    state = "LOST"

            else:

                state = "WAITING"

        # ====================================================
        # IMPORTANT
        # ====================================================
        #
        # For first detection, previous_box still needs update.
        # ====================================================

        if (
            detected
            and
            raw_box is not None
        ):

            previous_box = raw_box

        # ====================================================
        # DRAW TRACKED BATTER
        # ====================================================

        output_box = None

        if (
            smoothed_box is not None
            and
            track_active
        ):

            output_box = clip_box(
                smoothed_box,
                width,
                height
            )

            x1, y1, x2, y2 = (
                output_box
            )

            # ---------------------------------------------
            # Main bbox
            # ---------------------------------------------

            cv2.rectangle(
                frame,
                (
                    x1,
                    y1
                ),
                (
                    x2,
                    y2
                ),
                (
                    255,
                    255,
                    255
                ),
                3
            )

            # ---------------------------------------------
            # Center
            # ---------------------------------------------

            center_x = int(
                (
                    x1 + x2
                )
                /
                2
            )

            center_y = int(
                (
                    y1 + y2
                )
                /
                2
            )

            cv2.circle(
                frame,
                (
                    center_x,
                    center_y
                ),
                5,
                (
                    255,
                    255,
                    255
                ),
                -1
            )

            # ---------------------------------------------
            # Label
            # ---------------------------------------------

            if confidence is not None:

                label = (
                    f"BATTER TRACK "
                    f"{confidence:.2f}"
                )

            else:

                label = (
                    "BATTER TRACK"
                )

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        30,
                        y1 - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (
                    255,
                    255,
                    255
                ),
                2
            )

        # ====================================================
        # STATUS
        # ====================================================

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (
                20,
                35
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (
                255,
                255,
                255
            ),
            2
        )

        cv2.putText(
            frame,
            f"Batter state: {state}",
            (
                20,
                65
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (
                255,
                255,
                255
            ),
            2
        )

        cv2.putText(
            frame,
            f"Missing: {missing_frames}",
            (
                20,
                95
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (
                255,
                255,
                255
            ),
            2
        )

        # ====================================================
        # CSV ROW
        # ====================================================

        if output_box is not None:

            sx1, sy1, sx2, sy2 = (
                output_box
            )

            center_x = int(
                (
                    sx1 + sx2
                )
                /
                2
            )

            center_y = int(
                (
                    sy1 + sy2
                )
                /
                2
            )

        else:

            sx1 = None
            sy1 = None
            sx2 = None
            sy2 = None

            center_x = None
            center_y = None

        if raw_box is not None:

            rx1, ry1, rx2, ry2 = (
                raw_box
            )

        else:

            rx1 = None
            ry1 = None
            rx2 = None
            ry2 = None

        output_rows.append(
            {
                "frame":
                    frame_number,

                "state":
                    state,

                "detected":
                    detected,

                "confidence":
                    confidence,

                "raw_x1":
                    rx1,

                "raw_y1":
                    ry1,

                "raw_x2":
                    rx2,

                "raw_y2":
                    ry2,

                "tracked_x1":
                    sx1,

                "tracked_y1":
                    sy1,

                "tracked_x2":
                    sx2,

                "tracked_y2":
                    sy2,

                "center_x":
                    center_x,

                "center_y":
                    center_y,

                "missing_frames":
                    missing_frames
            }
        )

        # ====================================================
        # SAVE FRAME
        # ====================================================

        writer.write(
            frame
        )

        # ====================================================
        # TERMINAL
        # ====================================================

        print(
            f"Frame "
            f"{frame_number:03d}/"
            f"{total_frames} | "
            f"State: {state:<13} | "
            f"Detected: {detected} | "
            f"Missing: {missing_frames}"
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    writer.release()

    # ========================================================
    # SAVE CSV
    # ========================================================

    output_df = pd.DataFrame(
        output_rows
    )

    output_df.to_csv(
        output_csv,
        index=False
    )

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "BATTER TRACKING V1 COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        "\nVideo:"
    )

    print(
        output_video
    )

    print(
        "\nCSV:"
    )

    print(
        output_csv
    )


if __name__ == "__main__":
    main()