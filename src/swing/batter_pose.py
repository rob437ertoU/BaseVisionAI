from ultralytics import YOLO
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import cv2
import csv


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# MODEL
# ============================================================

POSE_MODEL = "yolo11n-pose.pt"

CONF_THRESHOLD = 0.25

IMG_SIZE = 640


# ============================================================
# ROI SETTINGS
# ============================================================

# Expands the batter bounding box slightly so arms/bat-side
# movements are less likely to be cropped.
ROI_PADDING_X = 0.20
ROI_PADDING_Y = 0.15


# ============================================================
# COCO 17 KEYPOINTS
# ============================================================

KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle"
]


# ============================================================
# SKELETON CONNECTIONS
# ============================================================

SKELETON = [
    ("left_shoulder", "right_shoulder"),

    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),

    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),

    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),

    ("left_hip", "right_hip"),

    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),

    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle")
]


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Batter Pose V1"
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
        help="Batter tracking CSV path"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where results will be saved"
    )

    return parser.parse_args()


# ============================================================
# AUXILIARY FUNCTIONS
# ============================================================

def expand_roi(
    x1,
    y1,
    x2,
    y2,
    frame_width,
    frame_height
):
    """
    Expand batter bbox with padding.
    """

    box_width = x2 - x1
    box_height = y2 - y1

    pad_x = box_width * ROI_PADDING_X
    pad_y = box_height * ROI_PADDING_Y

    new_x1 = int(
        max(
            0,
            x1 - pad_x
        )
    )

    new_y1 = int(
        max(
            0,
            y1 - pad_y
        )
    )

    new_x2 = int(
        min(
            frame_width - 1,
            x2 + pad_x
        )
    )

    new_y2 = int(
        min(
            frame_height - 1,
            y2 + pad_y
        )
    )

    return (
        new_x1,
        new_y1,
        new_x2,
        new_y2
    )


def get_batter_row(
    batter_df,
    frame_number
):
    """
    Get tracked batter row for current frame.
    """

    rows = batter_df[
        batter_df["frame"]
        ==
        frame_number
    ]

    if rows.empty:
        return None

    row = rows.iloc[0]

    if pd.isna(
        row["tracked_x1"]
    ):
        return None

    return row


def select_best_pose(result):
    """
    YOLO Pose can sometimes return more than one person
    inside the crop.

    Select the highest-confidence person.
    """

    if result.boxes is None:
        return None

    if len(result.boxes) == 0:
        return None

    confidences = (
        result.boxes.conf
        .cpu()
        .numpy()
    )

    best_index = int(
        np.argmax(confidences)
    )

    return best_index


def draw_pose(
    frame,
    keypoints_dict
):
    """
    Draw keypoints and skeleton.
    """

    # --------------------------------------------------------
    # Draw skeleton
    # --------------------------------------------------------

    for (
        point_a,
        point_b
    ) in SKELETON:

        kp_a = keypoints_dict.get(
            point_a
        )

        kp_b = keypoints_dict.get(
            point_b
        )

        if (
            kp_a is None
            or
            kp_b is None
        ):
            continue

        if (
            kp_a["confidence"] < 0.20
            or
            kp_b["confidence"] < 0.20
        ):
            continue

        cv2.line(
            frame,
            (
                int(kp_a["x"]),
                int(kp_a["y"])
            ),
            (
                int(kp_b["x"]),
                int(kp_b["y"])
            ),
            (
                255,
                255,
                255
            ),
            2
        )

    # --------------------------------------------------------
    # Draw joints
    # --------------------------------------------------------

    for name, kp in (
        keypoints_dict.items()
    ):

        if (
            kp["confidence"]
            <
            0.20
        ):
            continue

        cv2.circle(
            frame,
            (
                int(kp["x"]),
                int(kp["y"])
            ),
            5,
            (
                255,
                255,
                255
            ),
            -1
        )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # ========================================================
    # RESOLVE INPUT VIDEO
    # ========================================================

    video_path = Path(
        args.video
    )

    if not video_path.is_absolute():

        video_path = (
            PROJECT_ROOT
            /
            video_path
        )

    video_path = (
        video_path.resolve()
    )

    # ========================================================
    # RESOLVE BATTER TRACK CSV
    # ========================================================

    batter_track_csv = Path(
        args.input_csv
    )

    if not batter_track_csv.is_absolute():

        batter_track_csv = (
            PROJECT_ROOT
            /
            batter_track_csv
        )

    batter_track_csv = (
        batter_track_csv.resolve()
    )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    output_dir = Path(
        args.output_dir
    )

    if not output_dir.is_absolute():

        output_dir = (
            PROJECT_ROOT
            /
            output_dir
        )

    output_dir = (
        output_dir.resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_video = (
        output_dir
        /
        "batter_pose_v1.mp4"
    )

    output_csv = (
        output_dir
        /
        "batter_pose_v1.csv"
    )

    # ========================================================
    # HEADER
    # ========================================================

    print("=" * 70)
    print("BaseVisionAI - Batter Pose V1")
    print("=" * 70)

    print(
        f"Video: "
        f"{video_path}"
    )

    print(
        f"Batter tracking CSV: "
        f"{batter_track_csv}"
    )

    print(
        f"Output directory: "
        f"{output_dir}"
    )

    # ========================================================
    # CHECK INPUTS
    # ========================================================

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video not found:\n"
            f"{video_path}"
        )

    if not batter_track_csv.exists():

        raise FileNotFoundError(
            f"Batter tracking CSV not found:\n"
            f"{batter_track_csv}"
        )

    # ========================================================
    # LOAD BATTER TRACK
    # ========================================================

    batter_df = pd.read_csv(
        batter_track_csv
    )

    required_columns = {
        "frame",
        "tracked_x1",
        "tracked_y1",
        "tracked_x2",
        "tracked_y2"
    }

    missing_columns = (
        required_columns
        -
        set(
            batter_df.columns
        )
    )

    if missing_columns:

        raise ValueError(
            "Batter tracking CSV is missing required columns: "
            +
            ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    # ========================================================
    # LOAD POSE MODEL
    # ========================================================

    print(
        f"\nLoading pose model: "
        f"{POSE_MODEL}"
    )

    model = YOLO(
        POSE_MODEL
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
    # CSV OUTPUT
    # ========================================================

    csv_file = open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8"
    )

    csv_writer = csv.writer(
        csv_file
    )

    header = [
        "frame",
        "pose_detected",
        "pose_confidence",
        "roi_x1",
        "roi_y1",
        "roi_x2",
        "roi_y2"
    ]

    for keypoint_name in KEYPOINT_NAMES:

        header.extend(
            [
                f"{keypoint_name}_x",
                f"{keypoint_name}_y",
                f"{keypoint_name}_conf"
            ]
        )

    csv_writer.writerow(
        header
    )

    # ========================================================
    # FRAME LOOP
    # ========================================================

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        batter_row = get_batter_row(
            batter_df,
            frame_number
        )

        pose_detected = False
        pose_confidence = None

        roi_x1 = None
        roi_y1 = None
        roi_x2 = None
        roi_y2 = None

        keypoints_dict = {}

        # ====================================================
        # BATTER ROI AVAILABLE
        # ====================================================

        if batter_row is not None:

            x1 = int(
                batter_row[
                    "tracked_x1"
                ]
            )

            y1 = int(
                batter_row[
                    "tracked_y1"
                ]
            )

            x2 = int(
                batter_row[
                    "tracked_x2"
                ]
            )

            y2 = int(
                batter_row[
                    "tracked_y2"
                ]
            )

            (
                roi_x1,
                roi_y1,
                roi_x2,
                roi_y2
            ) = expand_roi(
                x1,
                y1,
                x2,
                y2,
                width,
                height
            )

            roi = frame[
                roi_y1:roi_y2,
                roi_x1:roi_x2
            ]

            # ---------------------------------------------
            # Safety check
            # ---------------------------------------------

            if (
                roi.size
                >
                0
            ):

                results = model.predict(
                    roi,
                    imgsz=IMG_SIZE,
                    conf=CONF_THRESHOLD,
                    device=0,
                    verbose=False
                )

                result = results[0]

                best_index = (
                    select_best_pose(
                        result
                    )
                )

                # =========================================
                # POSE FOUND
                # =========================================

                if (
                    best_index
                    is not None
                ):

                    pose_detected = True

                    pose_confidence = float(
                        result.boxes.conf[
                            best_index
                        ]
                    )

                    keypoints = (
                        result.keypoints
                    )

                    xy = (
                        keypoints.xy[
                            best_index
                        ]
                        .cpu()
                        .numpy()
                    )

                    # Some Ultralytics versions provide
                    # per-keypoint confidence
                    if (
                        keypoints.conf
                        is not None
                    ):

                        kp_conf = (
                            keypoints.conf[
                                best_index
                            ]
                            .cpu()
                            .numpy()
                        )

                    else:

                        kp_conf = np.ones(
                            len(
                                KEYPOINT_NAMES
                            ),
                            dtype=float
                        )

                    # =====================================
                    # Convert ROI coordinates -> full frame
                    # =====================================

                    for i, name in enumerate(
                        KEYPOINT_NAMES
                    ):

                        local_x = float(
                            xy[i][0]
                        )

                        local_y = float(
                            xy[i][1]
                        )

                        confidence = float(
                            kp_conf[i]
                        )

                        full_x = (
                            local_x
                            +
                            roi_x1
                        )

                        full_y = (
                            local_y
                            +
                            roi_y1
                        )

                        keypoints_dict[
                            name
                        ] = {
                            "x":
                                full_x,

                            "y":
                                full_y,

                            "confidence":
                                confidence
                        }

                    # =====================================
                    # DRAW POSE
                    # =====================================

                    draw_pose(
                        frame,
                        keypoints_dict
                    )

            # =================================================
            # DRAW BATTER ROI
            # =================================================

            cv2.rectangle(
                frame,
                (
                    roi_x1,
                    roi_y1
                ),
                (
                    roi_x2,
                    roi_y2
                ),
                (
                    150,
                    150,
                    150
                ),
                1
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

        if pose_detected:

            status_text = (
                f"BATTER POSE "
                f"{pose_confidence:.2f}"
            )

        else:

            status_text = (
                "BATTER POSE: NOT FOUND"
            )

        cv2.putText(
            frame,
            status_text,
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

        # ====================================================
        # CSV ROW
        # ====================================================

        row = [
            frame_number,
            pose_detected,
            pose_confidence,
            roi_x1,
            roi_y1,
            roi_x2,
            roi_y2
        ]

        for keypoint_name in KEYPOINT_NAMES:

            kp = keypoints_dict.get(
                keypoint_name
            )

            if kp is None:

                row.extend(
                    [
                        "",
                        "",
                        ""
                    ]
                )

            else:

                row.extend(
                    [
                        kp["x"],
                        kp["y"],
                        kp["confidence"]
                    ]
                )

        csv_writer.writerow(
            row
        )

        # ====================================================
        # SAVE VIDEO
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
            f"Pose: "
            f"{pose_detected} | "
            f"Confidence: "
            f"{pose_confidence if pose_confidence is not None else '-'}"
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    writer.release()
    csv_file.close()

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "BATTER POSE V1 COMPLETED"
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