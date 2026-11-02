from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
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
    / "batter_pose_v2"
    / "baseball_train10_batter_pose_v2_smoothed.csv"
)

DEFAULT_ANALYSIS_CSV = (
    PROJECT_ROOT
    / "results"
    / "swing_analysis_v2"
    / "baseball_train10_batter_pose_v2_smoothed_swing_analysis_v2_2.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "swing_visualization_v2"
)


# ============================================================
# COCO KEYPOINTS
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
    "right_ankle",
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
    ("right_knee", "right_ankle"),
]


# ============================================================
# PHASE DISPLAY
# ============================================================

PHASE_LABELS = {
    "SETUP": "SETUP",
    "LOAD": "LOAD",
    "SWING": "SWING",
    "PEAK_SWING": "PEAK SWING",
    "FOLLOW_THROUGH": "FOLLOW THROUGH",
}


# OpenCV uses BGR
PHASE_COLORS = {
    "SETUP": (200, 200, 200),
    "LOAD": (0, 215, 255),
    "SWING": (0, 165, 255),
    "PEAK_SWING": (0, 0, 255),
    "FOLLOW_THROUGH": (255, 180, 0),
}


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Swing Visualization V2"
    )

    parser.add_argument(
        "--video",
        type=str,
        default=str(DEFAULT_VIDEO),
        help="Original input video"
    )

    parser.add_argument(
        "--pose-csv",
        type=str,
        default=str(DEFAULT_POSE_CSV),
        help="Smoothed pose CSV"
    )

    parser.add_argument(
        "--analysis-csv",
        type=str,
        default=str(DEFAULT_ANALYSIS_CSV),
        help="Swing Analysis V2.2 CSV"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory"
    )

    parser.add_argument(
        "--keypoint-conf",
        type=float,
        default=0.30,
        help="Minimum original keypoint confidence for display"
    )

    return parser.parse_args()


# ============================================================
# UTILITIES
# ============================================================

def safe_float(value, default=np.nan):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


def safe_int(value, default=None):

    try:

        value = int(value)
        return value

    except Exception:
        return default


def draw_text_with_background(
    frame,
    text,
    origin,
    font_scale=0.65,
    thickness=2,
    text_color=(255, 255, 255),
    background_color=(0, 0, 0),
    padding=6
):

    font = cv2.FONT_HERSHEY_SIMPLEX

    (text_width, text_height), baseline = (
        cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness
        )
    )

    x, y = origin

    x1 = x - padding
    y1 = y - text_height - padding

    x2 = x + text_width + padding
    y2 = y + baseline + padding

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        background_color,
        -1
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )


def get_keypoint(
    row,
    keypoint,
    confidence_threshold
):

    x_col = f"{keypoint}_x_smooth"
    y_col = f"{keypoint}_y_smooth"

    # Use original confidence when available
    conf_col = f"{keypoint}_conf_raw"

    if conf_col not in row.index:
        conf_col = f"{keypoint}_conf"

    x = safe_float(
        row.get(
            x_col,
            np.nan
        )
    )

    y = safe_float(
        row.get(
            y_col,
            np.nan
        )
    )

    conf = safe_float(
        row.get(
            conf_col,
            np.nan
        )
    )

    # If point was interpolated, original confidence can be low.
    interpolated_col = (
        f"{keypoint}_interpolated"
    )

    interpolated = safe_int(
        row.get(
            interpolated_col,
            0
        ),
        0
    )

    valid_col = (
        f"{keypoint}_valid"
    )

    valid = safe_int(
        row.get(
            valid_col,
            1
        ),
        1
    )

    if not (
        np.isfinite(x)
        and np.isfinite(y)
    ):
        return None

    if valid != 1:
        return None

    # Permit interpolated points even if original confidence
    # was under the visualization threshold.
    if (
        interpolated != 1
        and np.isfinite(conf)
        and conf < confidence_threshold
    ):
        return None

    return (
        int(round(x)),
        int(round(y))
    )


# ============================================================
# POSE DRAWING
# ============================================================

def draw_pose(
    frame,
    pose_row,
    confidence_threshold
):

    points = {}

    for name in KEYPOINT_NAMES:

        point = get_keypoint(
            pose_row,
            name,
            confidence_threshold
        )

        if point is not None:
            points[name] = point

    # --------------------------------------------------------
    # Skeleton
    # --------------------------------------------------------

    for start_name, end_name in SKELETON:

        if (
            start_name in points
            and end_name in points
        ):

            cv2.line(
                frame,
                points[start_name],
                points[end_name],
                (0, 255, 0),
                3,
                cv2.LINE_AA
            )

    # --------------------------------------------------------
    # Keypoints
    # --------------------------------------------------------

    for name, point in points.items():

        # Make wrists slightly more visible
        if "wrist" in name:

            radius = 6

            cv2.circle(
                frame,
                point,
                radius,
                (0, 0, 255),
                -1,
                cv2.LINE_AA
            )

        elif "elbow" in name:

            cv2.circle(
                frame,
                point,
                5,
                (0, 255, 255),
                -1,
                cv2.LINE_AA
            )

        else:

            cv2.circle(
                frame,
                point,
                4,
                (255, 255, 255),
                -1,
                cv2.LINE_AA
            )


# ============================================================
# TIMELINE
# ============================================================

def draw_timeline(
    frame,
    current_frame,
    total_frames,
    phase_ranges
):

    height, width = frame.shape[:2]

    margin_x = 60

    timeline_y = (
        height - 45
    )

    timeline_width = (
        width - 2 * margin_x
    )

    # --------------------------------------------------------
    # Background bar
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (
            margin_x,
            timeline_y - 8
        ),
        (
            margin_x + timeline_width,
            timeline_y + 8
        ),
        (50, 50, 50),
        -1
    )

    # --------------------------------------------------------
    # Phase ranges
    # --------------------------------------------------------

    for phase, (
        start_frame,
        end_frame
    ) in phase_ranges.items():

        if (
            start_frame is None
            or end_frame is None
        ):
            continue

        x1 = (
            margin_x
            + int(
                (
                    (start_frame - 1)
                    / max(
                        total_frames - 1,
                        1
                    )
                )
                * timeline_width
            )
        )

        x2 = (
            margin_x
            + int(
                (
                    (end_frame - 1)
                    / max(
                        total_frames - 1,
                        1
                    )
                )
                * timeline_width
            )
        )

        color = PHASE_COLORS.get(
            phase,
            (150, 150, 150)
        )

        cv2.rectangle(
            frame,
            (
                x1,
                timeline_y - 8
            ),
            (
                max(
                    x2,
                    x1 + 1
                ),
                timeline_y + 8
            ),
            color,
            -1
        )

    # --------------------------------------------------------
    # Current frame cursor
    # --------------------------------------------------------

    current_x = (
        margin_x
        + int(
            (
                (current_frame - 1)
                / max(
                    total_frames - 1,
                    1
                )
            )
            * timeline_width
        )
    )

    cv2.line(
        frame,
        (
            current_x,
            timeline_y - 15
        ),
        (
            current_x,
            timeline_y + 15
        ),
        (255, 255, 255),
        3
    )

    cv2.circle(
        frame,
        (
            current_x,
            timeline_y
        ),
        6,
        (255, 255, 255),
        -1
    )


# ============================================================
# PHASE RANGES FROM CSV
# ============================================================

def extract_phase_ranges(df):

    phase_ranges = {}

    for phase in [
        "SETUP",
        "LOAD",
        "SWING",
        "PEAK_SWING",
        "FOLLOW_THROUGH"
    ]:

        subset = df[
            df["swing_phase"]
            == phase
        ]

        if len(subset) == 0:

            phase_ranges[
                phase
            ] = (
                None,
                None
            )

        else:

            phase_ranges[
                phase
            ] = (
                int(
                    subset["frame"]
                    .min()
                ),
                int(
                    subset["frame"]
                    .max()
                )
            )

    return phase_ranges


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    video_path = Path(
        args.video
    ).resolve()

    pose_csv_path = Path(
        args.pose_csv
    ).resolve()

    analysis_csv_path = Path(
        args.analysis_csv
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    for path in [
        video_path,
        pose_csv_path,
        analysis_csv_path
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_video = (
        output_dir
        / (
            f"{video_path.stem}"
            f"_swing_visualization_v2.mp4"
        )
    )

    # ========================================================
    # LOAD CSVs
    # ========================================================

    pose_df = pd.read_csv(
        pose_csv_path
    )

    analysis_df = pd.read_csv(
        analysis_csv_path
    )

    if "frame" not in pose_df.columns:

        raise RuntimeError(
            "Pose CSV has no 'frame' column."
        )

    if "frame" not in analysis_df.columns:

        raise RuntimeError(
            "Analysis CSV has no 'frame' column."
        )

    # Index by frame number
    pose_map = {
        int(row["frame"]): row
        for _, row in pose_df.iterrows()
    }

    analysis_map = {
        int(row["frame"]): row
        for _, row in analysis_df.iterrows()
    }

    # ========================================================
    # PHASE RANGES
    # ========================================================

    phase_ranges = extract_phase_ranges(
        analysis_df
    )

    # ========================================================
    # PEAK FRAME
    # ========================================================

    peak_rows = analysis_df[
        analysis_df[
            "peak_swing_candidate"
        ] == 1
    ]

    if len(peak_rows) > 0:

        peak_frame = int(
            peak_rows.iloc[0][
                "frame"
            ]
        )

    else:

        peak_frame = None

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

    total_video_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    total_frames = min(
        total_video_frames,
        len(analysis_df)
    )

    # --------------------------------------------------------
    # Writer
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_video),
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
    # PRINT INFO
    # ========================================================

    print()
    print("=" * 76)
    print("BaseVisionAI - Swing Visualization V2")
    print("=" * 76)

    print(
        f"Video       : {video_path}"
    )

    print(
        f"Pose CSV    : {pose_csv_path}"
    )

    print(
        f"Analysis CSV: {analysis_csv_path}"
    )

    print(
        f"Frames      : {total_video_frames}"
    )

    print(
        f"FPS         : {fps:.3f}"
    )

    print(
        f"Resolution  : {width}x{height}"
    )

    print(
        f"Peak frame  : {peak_frame}"
    )

    print("=" * 76)

    # ========================================================
    # FRAME LOOP
    # ========================================================

    frame_number = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # ----------------------------------------------------
        # Get rows
        # ----------------------------------------------------

        pose_row = pose_map.get(
            frame_number
        )

        analysis_row = analysis_map.get(
            frame_number
        )

        # ----------------------------------------------------
        # Draw pose
        # ----------------------------------------------------

        if pose_row is not None:

            draw_pose(
                frame,
                pose_row,
                args.keypoint_conf
            )

        # ----------------------------------------------------
        # Analysis overlay
        # ----------------------------------------------------

        if analysis_row is not None:

            phase = str(
                analysis_row.get(
                    "swing_phase",
                    "UNKNOWN"
                )
            )

            phase_label = PHASE_LABELS.get(
                phase,
                phase
            )

            phase_color = PHASE_COLORS.get(
                phase,
                (255, 255, 255)
            )

            swing_energy = safe_float(
                analysis_row.get(
                    "swing_motion_energy",
                    np.nan
                )
            )

            wrist_motion = safe_float(
                analysis_row.get(
                    "wrist_motion_norm",
                    np.nan
                )
            )

            pose_quality = safe_float(
                analysis_row.get(
                    "pose_quality",
                    np.nan
                )
            )

            # ------------------------------------------------
            # Top-left phase
            # ------------------------------------------------

            draw_text_with_background(
                frame,
                f"PHASE: {phase_label}",
                (
                    30,
                    45
                ),
                font_scale=0.85,
                thickness=2,
                background_color=phase_color
            )

            # ------------------------------------------------
            # Frame number / time
            # ------------------------------------------------

            time_sec = (
                (frame_number - 1)
                / fps
                if fps > 0
                else 0.0
            )

            draw_text_with_background(
                frame,
                (
                    f"Frame: {frame_number}  "
                    f"Time: {time_sec:.3f}s"
                ),
                (
                    30,
                    85
                ),
                font_scale=0.55,
                thickness=1
            )

            # ------------------------------------------------
            # Scores
            # ------------------------------------------------

            draw_text_with_background(
                frame,
                (
                    f"Swing score: "
                    f"{swing_energy:.3f}"
                ),
                (
                    30,
                    125
                ),
                font_scale=0.55,
                thickness=1
            )

            draw_text_with_background(
                frame,
                (
                    f"Wrist motion: "
                    f"{wrist_motion:.3f}"
                ),
                (
                    30,
                    160
                ),
                font_scale=0.55,
                thickness=1
            )

            draw_text_with_background(
                frame,
                (
                    f"Pose quality: "
                    f"{pose_quality:.3f}"
                ),
                (
                    30,
                    195
                ),
                font_scale=0.55,
                thickness=1
            )

        # ----------------------------------------------------
        # PEAK WARNING
        # ----------------------------------------------------

        if (
            peak_frame is not None
            and frame_number == peak_frame
        ):

            box_width = min(
                width - 80,
                700
            )

            x1 = (
                width // 2
                - box_width // 2
            )

            x2 = (
                width // 2
                + box_width // 2
            )

            y1 = 40
            y2 = 125

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
                (0, 0, 255),
                -1
            )

            text = (
                "PEAK SWING CANDIDATE"
            )

            warning = (
                "NOT CONFIRMED BAT-BALL CONTACT"
            )

            cv2.putText(
                frame,
                text,
                (
                    x1 + 25,
                    y1 + 35
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                warning,
                (
                    x1 + 25,
                    y1 + 70
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # Timeline
        # ----------------------------------------------------

        draw_timeline(
            frame,
            frame_number,
            total_frames,
            phase_ranges
        )

        # ----------------------------------------------------
        # Disclaimer
        # ----------------------------------------------------

        disclaimer = (
            "2D image-space motion analysis | "
            "not calibrated biomechanics"
        )

        cv2.putText(
            frame,
            disclaimer,
            (
                30,
                height - 75
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Write
        # ----------------------------------------------------

        writer.write(
            frame
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            frame_number % 30 == 0
            or frame_number
            == total_video_frames
        ):

            print(
                f"Processed frame "
                f"{frame_number}/"
                f"{total_video_frames}"
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    writer.release()

    print()
    print("=" * 76)
    print("SWING VISUALIZATION V2 COMPLETE")
    print("=" * 76)

    print(
        f"Frames processed : "
        f"{frame_number}"
    )

    print(
        f"Peak swing frame : "
        f"{peak_frame}"
    )

    print()

    print(
        "Phase ranges:"
    )

    for phase, (
        start_frame,
        end_frame
    ) in phase_ranges.items():

        print(
            f"  {phase:<18}: "
            f"{start_frame} -> {end_frame}"
        )

    print()

    print(
        f"Output video     : "
        f"{output_video}"
    )

    print("=" * 76)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()