from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd


# ============================================================
# BASEVISIONAI - FULL ANALYSIS VISUALIZATION V2.1
# ============================================================

POSE_CONNECTIONS = [
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

POSE_KEYPOINTS = [
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
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Full Analysis Visualization V2.1"
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Original input video"
    )

    parser.add_argument(
        "--pose-csv",
        required=True,
        help="Smoothed pose CSV"
    )

    parser.add_argument(
        "--swing-csv",
        required=True,
        help="Swing Analysis V2.2 frame CSV"
    )

    parser.add_argument(
        "--ball-csv",
        required=True,
        help="Ball Tracking V3.2 CSV"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output MP4 path"
    )

    return parser.parse_args()


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_float(value):

    try:

        if pd.isna(value):
            return None

        return float(value)

    except Exception:
        return None


def bool_value(value):

    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    text = str(value).strip().lower()

    return text in [
        "true",
        "1",
        "yes",
        "y",
    ]


def put_text(
    image,
    text,
    xy,
    scale=0.6,
    thickness=2,
    color=(255, 255, 255),
):

    cv2.putText(
        image,
        str(text),
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_panel(
    frame,
    x1,
    y1,
    x2,
    y2,
    alpha=0.58,
):

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
        frame
    )


# ============================================================
# POSE HELPERS
# ============================================================

def get_pose_point(
    row,
    keypoint,
):

    x_candidates = [
        f"{keypoint}_x_smooth",
        f"{keypoint}_x_interp",
        f"{keypoint}_x_filtered",
        f"{keypoint}_x",
    ]

    y_candidates = [
        f"{keypoint}_y_smooth",
        f"{keypoint}_y_interp",
        f"{keypoint}_y_filtered",
        f"{keypoint}_y",
    ]

    x = None
    y = None

    for column in x_candidates:

        if column in row.index:

            value = safe_float(
                row[column]
            )

            if value is not None:
                x = value
                break

    for column in y_candidates:

        if column in row.index:

            value = safe_float(
                row[column]
            )

            if value is not None:
                y = value
                break

    if x is None or y is None:
        return None

    if (
        not np.isfinite(x)
        or not np.isfinite(y)
    ):
        return None

    return (
        int(round(x)),
        int(round(y))
    )


def draw_pose(
    frame,
    row,
):

    points = {}

    for keypoint in POSE_KEYPOINTS:

        point = get_pose_point(
            row,
            keypoint
        )

        if point is not None:
            points[keypoint] = point

    # --------------------------------------------------------
    # BODY CONNECTIONS
    # --------------------------------------------------------

    for a, b in POSE_CONNECTIONS:

        if (
            a in points
            and b in points
        ):

            cv2.line(
                frame,
                points[a],
                points[b],
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

    # --------------------------------------------------------
    # KEYPOINTS
    # --------------------------------------------------------

    for point in points.values():

        cv2.circle(
            frame,
            point,
            4,
            (255, 255, 255),
            -1,
            cv2.LINE_AA
        )


# ============================================================
# BALL HELPERS
# ============================================================

def get_ball_point(
    row,
):

    if row is None:
        return None, None

    state = str(
        row.get(
            "state",
            ""
        )
    ).upper()

    source = str(
        row.get(
            "detection_source",
            ""
        )
    ).upper()

    yolo_detected = False

    if "yolo_detected" in row.index:

        yolo_detected = bool_value(
            row["yolo_detected"]
        )

    # --------------------------------------------------------
    # NORMAL YOLO OBSERVATION
    # --------------------------------------------------------

    if yolo_detected:

        x = safe_float(
            row.get(
                "observed_x"
            )
        )

        y = safe_float(
            row.get(
                "observed_y"
            )
        )

        if (
            x is not None
            and y is not None
        ):

            return (
                int(round(x)),
                int(round(y))
            ), "YOLO"

    # --------------------------------------------------------
    # LOCAL RECOVERY
    # --------------------------------------------------------

    if (
        state == "RECOVERED"
        or source == "RECOVERED"
    ):

        x = safe_float(
            row.get(
                "observed_x"
            )
        )

        y = safe_float(
            row.get(
                "observed_y"
            )
        )

        if (
            x is not None
            and y is not None
        ):

            return (
                int(round(x)),
                int(round(y))
            ), "RECOVERED"

    # --------------------------------------------------------
    # KALMAN PREDICTION
    # --------------------------------------------------------

    if state == "KALMAN":

        x = safe_float(
            row.get(
                "filtered_x"
            )
        )

        y = safe_float(
            row.get(
                "filtered_y"
            )
        )

        if (
            x is not None
            and y is not None
        ):

            return (
                int(round(x)),
                int(round(y))
            ), "KALMAN"

    # --------------------------------------------------------
    # FALLBACK YOLO SOURCE
    # --------------------------------------------------------

    if source == "YOLO":

        x = safe_float(
            row.get(
                "observed_x"
            )
        )

        y = safe_float(
            row.get(
                "observed_y"
            )
        )

        if (
            x is not None
            and y is not None
        ):

            return (
                int(round(x)),
                int(round(y))
            ), "YOLO"

    return None, None


# ============================================================
# TIMELINE
# ============================================================

def draw_timeline(
    frame,
    frame_number,
    total_frames,
    phase,
):

    height, width = frame.shape[:2]

    margin = 55

    x1 = margin
    x2 = width - margin

    y = height - 42

    # --------------------------------------------------------
    # BACKGROUND LINE
    # --------------------------------------------------------

    cv2.line(
        frame,
        (x1, y),
        (x2, y),
        (210, 210, 210),
        3,
        cv2.LINE_AA
    )

    if total_frames <= 1:
        return

    # --------------------------------------------------------
    # CURRENT POSITION
    # --------------------------------------------------------

    ratio = (
        (frame_number - 1)
        / (total_frames - 1)
    )

    ratio = float(
        np.clip(
            ratio,
            0,
            1
        )
    )

    current_x = int(
        x1
        + ratio * (x2 - x1)
    )

    cv2.circle(
        frame,
        (current_x, y),
        7,
        (255, 255, 255),
        -1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # CURRENT PHASE
    # --------------------------------------------------------

    put_text(
        frame,
        phase,
        (
            max(
                x1,
                current_x - 75
            ),
            y - 14
        ),
        scale=0.52,
        thickness=2,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # PATHS
    # --------------------------------------------------------

    video_path = Path(
        args.video
    )

    pose_path = Path(
        args.pose_csv
    )

    swing_path = Path(
        args.swing_csv
    )

    ball_path = Path(
        args.ball_csv
    )

    output_path = Path(
        args.output
    )

    # --------------------------------------------------------
    # INPUT CHECKS
    # --------------------------------------------------------

    for path in [
        video_path,
        pose_path,
        swing_path,
        ball_path,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing input:\n{path}"
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD CSV FILES
    # --------------------------------------------------------

    pose_df = pd.read_csv(
        pose_path
    )

    swing_df = pd.read_csv(
        swing_path
    )

    ball_df = pd.read_csv(
        ball_path
    )

    # --------------------------------------------------------
    # NORMALIZE FRAME COLUMNS
    # --------------------------------------------------------

    for df_name, df in [
        ("pose", pose_df),
        ("swing", swing_df),
        ("ball", ball_df),
    ]:

        if "frame" not in df.columns:

            raise ValueError(
                f"{df_name} CSV does not contain a "
                f"'frame' column."
            )

        df["frame"] = pd.to_numeric(
            df["frame"],
            errors="coerce"
        )

        df.dropna(
            subset=["frame"],
            inplace=True
        )

        df["frame"] = (
            df["frame"]
            .astype(int)
        )

    # --------------------------------------------------------
    # LOOKUPS
    # --------------------------------------------------------

    pose_lookup = (
        pose_df
        .set_index("frame")
    )

    swing_lookup = (
        swing_df
        .set_index("frame")
    )

    ball_lookup = (
        ball_df
        .set_index("frame")
    )

    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
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

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # --------------------------------------------------------
    # VIDEO WRITER
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
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
            f"{output_path}"
        )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print()
    print("=" * 76)
    print(
        "BaseVisionAI - Full Analysis Visualization V2.1"
    )
    print("=" * 76)

    print(
        f"Video       : {video_path}"
    )

    print(
        f"Frames      : {frame_count}"
    )

    print(
        f"FPS         : {fps:.2f}"
    )

    print(
        f"Resolution  : {width}x{height}"
    )

    print(
        f"Pose CSV    : {pose_path}"
    )

    print(
        f"Swing CSV   : {swing_path}"
    )

    print(
        f"Ball CSV    : {ball_path}"
    )

    print(
        f"Output      : {output_path}"
    )

    print("=" * 76)

    # ========================================================
    # BALL TRAIL
    # ========================================================

    # IMPORTANT:
    #
    # Only REAL detector observations are added here.
    #
    # YOLO       -> real
    # RECOVERED  -> real
    # KALMAN     -> prediction only, never added
    # LOST       -> trail is cleared

    ball_trail = []

    max_trail = 15

    # --------------------------------------------------------
    # COUNTERS FOR FINAL SUMMARY
    # --------------------------------------------------------

    yolo_frames = 0
    recovered_frames = 0
    kalman_frames = 0
    lost_frames = 0

    pose_frames = 0

    frame_number = 0

    # ========================================================
    # VIDEO LOOP
    # ========================================================

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        frame_number += 1

        # ====================================================
        # POSE
        # ====================================================

        pose_quality = None

        if frame_number in pose_lookup.index:

            pose_row = pose_lookup.loc[
                frame_number
            ]

            if isinstance(
                pose_row,
                pd.DataFrame
            ):

                pose_row = (
                    pose_row.iloc[0]
                )

            draw_pose(
                frame,
                pose_row
            )

            pose_frames += 1

            pose_quality = safe_float(
                pose_row.get(
                    "pose_quality"
                )
            )

        # ====================================================
        # SWING ANALYSIS
        # ====================================================

        phase = "UNKNOWN"

        swing_score = None

        if frame_number in swing_lookup.index:

            swing_row = swing_lookup.loc[
                frame_number
            ]

            if isinstance(
                swing_row,
                pd.DataFrame
            ):

                swing_row = (
                    swing_row.iloc[0]
                )

            # ------------------------------------------------
            # PHASE
            # ------------------------------------------------

            for phase_column in [
                "phase",
                "swing_phase",
                "phase_name",
            ]:

                if phase_column in swing_row.index:

                    value = swing_row[
                        phase_column
                    ]

                    if not pd.isna(value):

                        phase = str(
                            value
                        )

                        break

            # ------------------------------------------------
            # MOTION SCORE
            # ------------------------------------------------

            for score_column in [
                "swing_score",
                "combined_score",
                "motion_score",
                "smoothed_swing_score",
            ]:

                if score_column in swing_row.index:

                    value = safe_float(
                        swing_row[
                            score_column
                        ]
                    )

                    if value is not None:

                        swing_score = value

                        break

        # ====================================================
        # BALL
        # ====================================================

        ball_state = "WAITING"

        ball_source = "NONE"

        point = None

        if frame_number in ball_lookup.index:

            ball_row = ball_lookup.loc[
                frame_number
            ]

            if isinstance(
                ball_row,
                pd.DataFrame
            ):

                ball_row = (
                    ball_row.iloc[0]
                )

            ball_state = str(
                ball_row.get(
                    "state",
                    "WAITING"
                )
            ).upper()

            point, ball_source = get_ball_point(
                ball_row
            )

            # ------------------------------------------------
            # TRACK LOST
            # ------------------------------------------------

            if ball_state == "LOST":

                ball_trail.clear()

                lost_frames += 1

            # ------------------------------------------------
            # REAL YOLO OBSERVATION
            # ------------------------------------------------

            if (
                point is not None
                and ball_source == "YOLO"
            ):

                yolo_frames += 1

                ball_trail.append(
                    (
                        point,
                        ball_source
                    )
                )

            # ------------------------------------------------
            # REAL LOCAL RECOVERY OBSERVATION
            # ------------------------------------------------

            elif (
                point is not None
                and ball_source == "RECOVERED"
            ):

                recovered_frames += 1

                ball_trail.append(
                    (
                        point,
                        ball_source
                    )
                )

            # ------------------------------------------------
            # KALMAN ONLY
            # ------------------------------------------------

            elif (
                point is not None
                and ball_source == "KALMAN"
            ):

                kalman_frames += 1

            # ------------------------------------------------
            # TRAIL LIMIT
            # ------------------------------------------------

            if len(ball_trail) > max_trail:

                ball_trail = (
                    ball_trail[-max_trail:]
                )

        # ====================================================
        # DRAW REAL BALL TRAIL
        # ====================================================

        if len(ball_trail) >= 2:

            for i in range(
                1,
                len(ball_trail)
            ):

                p1 = ball_trail[
                    i - 1
                ][0]

                p2 = ball_trail[
                    i
                ][0]

                cv2.line(
                    frame,
                    p1,
                    p2,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        # ====================================================
        # DRAW CURRENT BALL
        # ====================================================

        if point is not None:

            # ------------------------------------------------
            # YOLO
            # ------------------------------------------------

            if ball_source == "YOLO":

                cv2.circle(
                    frame,
                    point,
                    12,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA
                )

                cv2.circle(
                    frame,
                    point,
                    4,
                    (0, 255, 0),
                    -1,
                    cv2.LINE_AA
                )

                put_text(
                    frame,
                    "BALL YOLO",
                    (
                        point[0] + 16,
                        point[1] - 12
                    ),
                    scale=0.50,
                    thickness=2,
                    color=(0, 255, 0),
                )

            # ------------------------------------------------
            # RECOVERED
            # ------------------------------------------------

            elif ball_source == "RECOVERED":

                cv2.circle(
                    frame,
                    point,
                    12,
                    (255, 255, 0),
                    3,
                    cv2.LINE_AA
                )

                cv2.circle(
                    frame,
                    point,
                    4,
                    (255, 255, 0),
                    -1,
                    cv2.LINE_AA
                )

                put_text(
                    frame,
                    "BALL RECOVERED",
                    (
                        point[0] + 16,
                        point[1] - 12
                    ),
                    scale=0.50,
                    thickness=2,
                    color=(255, 255, 0),
                )

            # ------------------------------------------------
            # KALMAN PREDICTION ONLY
            # ------------------------------------------------

            elif ball_source == "KALMAN":

                x, y = point

                cv2.line(
                    frame,
                    (
                        x - 10,
                        y - 10
                    ),
                    (
                        x + 10,
                        y + 10
                    ),
                    (0, 255, 255),
                    3,
                    cv2.LINE_AA
                )

                cv2.line(
                    frame,
                    (
                        x + 10,
                        y - 10
                    ),
                    (
                        x - 10,
                        y + 10
                    ),
                    (0, 255, 255),
                    3,
                    cv2.LINE_AA
                )

                put_text(
                    frame,
                    "KALMAN PRED",
                    (
                        x + 16,
                        y - 12
                    ),
                    scale=0.48,
                    thickness=2,
                    color=(0, 255, 255),
                )

        # ====================================================
        # TOP INFORMATION PANEL
        # ====================================================

        panel_width = min(
            650,
            width - 30
        )

        draw_panel(
            frame,
            15,
            15,
            panel_width,
            225
        )

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        put_text(
            frame,
            "BASEVISION AI",
            (
                30,
                52
            ),
            scale=1.0,
            thickness=2,
        )

        # ----------------------------------------------------
        # FRAME
        # ----------------------------------------------------

        put_text(
            frame,
            f"Frame: {frame_number}/{frame_count}",
            (
                30,
                88
            ),
            scale=0.65,
            thickness=2,
        )

        # ----------------------------------------------------
        # PHASE
        # ----------------------------------------------------

        put_text(
            frame,
            f"Phase: {phase}",
            (
                30,
                125
            ),
            scale=0.72,
            thickness=2,
        )

        # ----------------------------------------------------
        # POSE QUALITY
        # ----------------------------------------------------

        if pose_quality is not None:

            put_text(
                frame,
                f"Pose quality: {pose_quality:.3f}",
                (
                    30,
                    162
                ),
                scale=0.65,
                thickness=2,
            )

        else:

            put_text(
                frame,
                "Pose quality: N/A",
                (
                    30,
                    162
                ),
                scale=0.65,
                thickness=2,
            )

        # ----------------------------------------------------
        # BALL LABEL
        # ----------------------------------------------------

        if ball_source == "YOLO":

            ball_label = (
                "REAL YOLO OBSERVATION"
            )

        elif ball_source == "RECOVERED":

            ball_label = (
                "REAL RECOVERED OBSERVATION"
            )

        elif ball_source == "KALMAN":

            ball_label = (
                "KALMAN PREDICTION ONLY"
            )

        elif ball_state == "LOST":

            ball_label = (
                "NO ACTIVE BALL TRACK"
            )

        else:

            ball_label = (
                "NO BALL OBSERVATION"
            )

        put_text(
            frame,
            f"Ball: {ball_label}",
            (
                30,
                199
            ),
            scale=0.62,
            thickness=2,
        )

        # ====================================================
        # MOTION SCORE
        # ====================================================

        if swing_score is not None:

            put_text(
                frame,
                f"Motion score: {swing_score:.3f}",
                (
                    390,
                    88
                ),
                scale=0.56,
                thickness=2,
            )

        # ====================================================
        # DISCLAIMER
        # ====================================================

        disclaimer = (
            "2D vision analysis - no strike/ball/hit classification"
        )

        text_size = cv2.getTextSize(
            disclaimer,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            1
        )[0]

        draw_panel(
            frame,
            15,
            height - 105,
            min(
                width - 15,
                40 + text_size[0]
            ),
            height - 68,
            alpha=0.68,
        )

        put_text(
            frame,
            disclaimer,
            (
                27,
                height - 80
            ),
            scale=0.48,
            thickness=1,
        )

        # ====================================================
        # TIMELINE
        # ====================================================

        draw_timeline(
            frame,
            frame_number,
            frame_count,
            phase
        )

        # ====================================================
        # WRITE FRAME
        # ====================================================

        writer.write(
            frame
        )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            frame_number % 30 == 0
            or frame_number == frame_count
        ):

            print(
                f"Processing "
                f"{frame_number}/{frame_count}"
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    writer.release()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print("=" * 76)

    print(
        "FULL ANALYSIS VISUALIZATION V2.1 COMPLETE"
    )

    print("=" * 76)

    print(
        f"Frames processed          : "
        f"{frame_number}"
    )

    print(
        f"Pose frames               : "
        f"{pose_frames}"
    )

    print(
        f"Real YOLO ball frames     : "
        f"{yolo_frames}"
    )

    print(
        f"Recovered ball frames     : "
        f"{recovered_frames}"
    )

    print(
        f"Kalman prediction frames  : "
        f"{kalman_frames}"
    )

    print(
        f"LOST frames               : "
        f"{lost_frames}"
    )

    print()

    print(
        "Visualization rules:"
    )

    print(
        "  YOLO      = real detector observation"
    )

    print(
        "  RECOVERED = real localized detector observation"
    )

    print(
        "  KALMAN    = prediction only"
    )

    print(
        "  LOST      = clears the real ball trail"
    )

    print()

    print(
        f"Output : {output_path.resolve()}"
    )

    print("=" * 76)


if __name__ == "__main__":
    main()