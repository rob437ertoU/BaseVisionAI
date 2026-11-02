from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd


# ============================================================
# VISUAL CONFIG
# ============================================================

BALL_RADIUS = 7
TRAIL_LENGTH = 15

# BGR colors
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
ORANGE = (0, 165, 255)
GRAY = (180, 180, 180)


def parse_args():
    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Pitch Crossing Visualization V1"
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Original input video"
    )

    parser.add_argument(
        "--zone-csv",
        required=True,
        help="Strike Zone V1.1 CSV"
    )

    parser.add_argument(
        "--ball-csv",
        required=True,
        help="Ball Tracking V3.2 CSV"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output visualization video"
    )

    return parser.parse_args()


def safe_float(value):
    try:
        value = float(value)

        if np.isnan(value):
            return None

        return value

    except (TypeError, ValueError):
        return None


def bool_from_value(value):
    if isinstance(value, str):
        return value.strip().lower() in [
            "true",
            "1",
            "yes"
        ]

    try:
        return bool(value)
    except Exception:
        return False


def main():

    args = parse_args()

    video_path = Path(args.video)
    zone_path = Path(args.zone_csv)
    ball_path = Path(args.ball_csv)
    output_path = Path(args.output)

    # ========================================================
    # CHECK FILES
    # ========================================================

    for path in [
        video_path,
        zone_path,
        ball_path
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # LOAD CSVs
    # ========================================================

    zone_df = pd.read_csv(
        zone_path
    )

    ball_df = pd.read_csv(
        ball_path
    )

    print()
    print("=" * 78)
    print("BaseVisionAI - Pitch Crossing Visualization V1")
    print("=" * 78)
    print(f"Video     : {video_path}")
    print(f"Zone CSV  : {zone_path}")
    print(f"Ball CSV  : {ball_path}")
    print(f"Output    : {output_path}")

    # ========================================================
    # EXACT STRIKE ZONE V1.1 COLUMNS
    # ========================================================

    zone_frame_col = "frame"
    plate_x_col = "plate_x_proxy"

    x1_col = "zone_left"
    x2_col = "zone_right"

    y1_col = "zone_top"
    y2_col = "zone_bottom"

    zone_available_col = "zone_available"

    required_zone_columns = [
        zone_frame_col,
        plate_x_col,
        x1_col,
        x2_col,
        y1_col,
        y2_col,
    ]

    missing_zone_columns = [
        col
        for col in required_zone_columns
        if col not in zone_df.columns
    ]

    if missing_zone_columns:

        print()
        print("Available strike-zone columns:")
        print(zone_df.columns.tolist())

        raise ValueError(
            "Missing required strike-zone columns: "
            + ", ".join(
                missing_zone_columns
            )
        )

    print()
    print("Strike-zone columns detected:")
    print(
        f"Frame       : {zone_frame_col}"
    )
    print(
        f"Plate X     : {plate_x_col}"
    )
    print(
        f"Zone left   : {x1_col}"
    )
    print(
        f"Zone right  : {x2_col}"
    )
    print(
        f"Zone top    : {y1_col}"
    )
    print(
        f"Zone bottom : {y2_col}"
    )

    # ========================================================
    # EXACT BALL TRACKING V3.2 COLUMNS
    # ========================================================

    ball_frame_col = "frame"
    yolo_col = "yolo_detected"

    observed_x_col = "observed_x"
    observed_y_col = "observed_y"

    filtered_x_col = "filtered_x"
    filtered_y_col = "filtered_y"

    state_col = "state"

    source_col = (
        "detection_source"
        if "detection_source" in ball_df.columns
        else None
    )

    required_ball_columns = [
        ball_frame_col,
        yolo_col,
        observed_x_col,
        observed_y_col,
        filtered_x_col,
        filtered_y_col,
        state_col,
    ]

    missing_ball_columns = [
        col
        for col in required_ball_columns
        if col not in ball_df.columns
    ]

    if missing_ball_columns:

        print()
        print("Available ball-tracking columns:")
        print(ball_df.columns.tolist())

        raise ValueError(
            "Missing required ball-tracking columns: "
            + ", ".join(
                missing_ball_columns
            )
        )

    print()
    print("Ball-tracking columns detected:")
    print(
        f"Frame       : {ball_frame_col}"
    )
    print(
        f"YOLO flag   : {yolo_col}"
    )
    print(
        f"Observed X  : {observed_x_col}"
    )
    print(
        f"Observed Y  : {observed_y_col}"
    )
    print(
        f"Filtered X  : {filtered_x_col}"
    )
    print(
        f"Filtered Y  : {filtered_y_col}"
    )
    print(
        f"State       : {state_col}"
    )

    if source_col is not None:
        print(
            f"Source      : {source_col}"
        )

    # ========================================================
    # INDEX DATA BY FRAME
    # ========================================================

    zone_by_frame = {}

    for _, row in zone_df.iterrows():

        frame_value = safe_float(
            row[zone_frame_col]
        )

        if frame_value is None:
            continue

        frame_number = int(
            frame_value
        )

        zone_by_frame[
            frame_number
        ] = row

    ball_by_frame = {}

    for _, row in ball_df.iterrows():

        frame_value = safe_float(
            row[ball_frame_col]
        )

        if frame_value is None:
            continue

        frame_number = int(
            frame_value
        )

        ball_by_frame[
            frame_number
        ] = row

    # ========================================================
    # COLLECT REAL YOLO OBSERVATIONS
    # ========================================================

    real_points = []

    for _, row in ball_df.iterrows():

        frame_value = safe_float(
            row[ball_frame_col]
        )

        if frame_value is None:
            continue

        frame_number = int(
            frame_value
        )

        is_yolo = bool_from_value(
            row[yolo_col]
        )

        if not is_yolo:
            continue

        observed_x = safe_float(
            row[observed_x_col]
        )

        observed_y = safe_float(
            row[observed_y_col]
        )

        if (
            observed_x is None
            or observed_y is None
        ):
            continue

        real_points.append(
            (
                frame_number,
                observed_x,
                observed_y
            )
        )

    if len(real_points) < 4:
        raise ValueError(
            "At least 4 real YOLO observations are required."
        )

    real_points = sorted(
        real_points,
        key=lambda p: p[0]
    )

    # ========================================================
    # USE SAME LOCAL FIT AS PITCH CROSSING V1
    # Latest 4 real YOLO measurements
    # ========================================================

    fit_points = real_points[-4:]

    frames_fit = np.array(
        [
            point[0]
            for point in fit_points
        ],
        dtype=float
    )

    xs_fit = np.array(
        [
            point[1]
            for point in fit_points
        ],
        dtype=float
    )

    ys_fit = np.array(
        [
            point[2]
            for point in fit_points
        ],
        dtype=float
    )

    x_coeff = np.polyfit(
        frames_fit,
        xs_fit,
        1
    )

    y_coeff = np.polyfit(
        frames_fit,
        ys_fit,
        1
    )

    ax, bx = x_coeff
    ay, by = y_coeff

    predicted_x_fit = (
        ax * frames_fit
        + bx
    )

    predicted_y_fit = (
        ay * frames_fit
        + by
    )

    residuals = np.sqrt(
        (
            xs_fit
            - predicted_x_fit
        ) ** 2
        +
        (
            ys_fit
            - predicted_y_fit
        ) ** 2
    )

    mean_residual = float(
        np.mean(
            residuals
        )
    )

    # ========================================================
    # PLATE X PROXY
    # ========================================================

    valid_plate_values = []

    for value in zone_df[
        plate_x_col
    ].tolist():

        parsed = safe_float(
            value
        )

        if parsed is not None:
            valid_plate_values.append(
                parsed
            )

    if len(valid_plate_values) == 0:
        raise ValueError(
            "Could not find a valid plate_x_proxy value."
        )

    plate_x = float(
        np.median(
            valid_plate_values
        )
    )

    # ========================================================
    # CROSSING ESTIMATE
    # ========================================================

    if abs(ax) < 1e-8:
        raise ValueError(
            "Trajectory X slope is too small to estimate crossing."
        )

    crossing_frame = (
        plate_x
        - bx
    ) / ax

    crossing_y = (
        ay * crossing_frame
        + by
    )

    last_real_frame = float(
        frames_fit[-1]
    )

    extrapolation_frames = (
        crossing_frame
        - last_real_frame
    )

    # ========================================================
    # FIND NEAREST AVAILABLE ZONE
    # ========================================================

    candidate_zone_frames = []

    for _, row in zone_df.iterrows():

        frame_value = safe_float(
            row[zone_frame_col]
        )

        if frame_value is None:
            continue

        if zone_available_col in zone_df.columns:

            zone_available = bool_from_value(
                row[zone_available_col]
            )

            if not zone_available:
                continue

        left = safe_float(
            row[x1_col]
        )

        right = safe_float(
            row[x2_col]
        )

        top = safe_float(
            row[y1_col]
        )

        bottom = safe_float(
            row[y2_col]
        )

        if None in [
            left,
            right,
            top,
            bottom
        ]:
            continue

        candidate_zone_frames.append(
            int(
                frame_value
            )
        )

    if len(candidate_zone_frames) == 0:
        raise ValueError(
            "No valid strike-zone frames were found."
        )

    candidate_zone_frames = np.array(
        candidate_zone_frames,
        dtype=int
    )

    crossing_reference_frame = int(
        round(
            crossing_frame
        )
    )

    nearest_zone_frame = int(
        candidate_zone_frames[
            np.argmin(
                np.abs(
                    candidate_zone_frames
                    - crossing_reference_frame
                )
            )
        ]
    )

    crossing_zone = zone_by_frame[
        nearest_zone_frame
    ]

    zx1 = safe_float(
        crossing_zone[x1_col]
    )

    zx2 = safe_float(
        crossing_zone[x2_col]
    )

    zy1 = safe_float(
        crossing_zone[y1_col]
    )

    zy2 = safe_float(
        crossing_zone[y2_col]
    )

    if None in [
        zx1,
        zx2,
        zy1,
        zy2
    ]:
        raise ValueError(
            "Strike zone unavailable at crossing reference frame."
        )

    horizontal_inside = (
        zx1
        <= plate_x
        <= zx2
    )

    vertical_inside = (
        zy1
        <= crossing_y
        <= zy2
    )

    if (
        horizontal_inside
        and vertical_inside
    ):
        classification = (
            "STRIKE_CANDIDATE"
        )
    else:
        classification = (
            "BALL_CANDIDATE"
        )

    evidence_quality = (
        "LIMITED"
    )

    # ========================================================
    # PRINT NUMERICAL RESULT
    # ========================================================

    print()
    print("=" * 78)
    print("PITCH CROSSING ESTIMATE")
    print("=" * 78)

    print(
        f"Real YOLO observations : {len(real_points)}"
    )

    print(
        f"Fit frames             : "
        f"{int(frames_fit[0])} -> "
        f"{int(frames_fit[-1])}"
    )

    print(
        f"Mean fit residual      : "
        f"{mean_residual:.2f} px"
    )

    print(
        f"Plate X proxy          : "
        f"{plate_x:.1f} px"
    )

    print(
        f"Crossing frame         : "
        f"{crossing_frame:.2f}"
    )

    print(
        f"Crossing Y             : "
        f"{crossing_y:.1f} px"
    )

    print(
        f"Extrapolation          : "
        f"{extrapolation_frames:.2f} frames"
    )

    print(
        f"Zone reference frame   : "
        f"{nearest_zone_frame}"
    )

    print(
        f"Horizontal zone        : "
        f"{zx1:.1f} -> {zx2:.1f}"
    )

    print(
        f"Vertical zone          : "
        f"{zy1:.1f} -> {zy2:.1f}"
    )

    print(
        f"Horizontal inside      : "
        f"{horizontal_inside}"
    )

    print(
        f"Vertical inside        : "
        f"{vertical_inside}"
    )

    print(
        f"Classification         : "
        f"{classification}"
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

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(
            output_path
        ),
        fourcc,
        fps,
        (
            width,
            height
        )
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create output video:\n{output_path}"
        )

    # ========================================================
    # VIDEO LOOP
    # ========================================================

    frame_number = 0

    real_trail = []

    crossing_point = (
        int(
            round(
                plate_x
            )
        ),
        int(
            round(
                crossing_y
            )
        )
    )

    trajectory_start_frame = int(
        frames_fit[0]
    )

    trajectory_end_frame = int(
        np.ceil(
            crossing_frame
        )
    )

    trajectory_points = []

    for trajectory_frame in range(
        trajectory_start_frame,
        trajectory_end_frame + 1
    ):

        px = (
            ax
            * trajectory_frame
            + bx
        )

        py = (
            ay
            * trajectory_frame
            + by
        )

        trajectory_points.append(
            (
                int(
                    round(
                        px
                    )
                ),
                int(
                    round(
                        py
                    )
                )
            )
        )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # ====================================================
        # CURRENT STRIKE ZONE
        # ====================================================

        zone_row = zone_by_frame.get(
            frame_number
        )

        if zone_row is not None:

            zone_available = True

            if (
                zone_available_col
                in zone_df.columns
            ):
                zone_available = bool_from_value(
                    zone_row[
                        zone_available_col
                    ]
                )

            if zone_available:

                x1 = safe_float(
                    zone_row[x1_col]
                )

                x2 = safe_float(
                    zone_row[x2_col]
                )

                y1 = safe_float(
                    zone_row[y1_col]
                )

                y2 = safe_float(
                    zone_row[y2_col]
                )

                if None not in [
                    x1,
                    x2,
                    y1,
                    y2
                ]:

                    cv2.rectangle(
                        frame,
                        (
                            int(
                                round(
                                    x1
                                )
                            ),
                            int(
                                round(
                                    y1
                                )
                            )
                        ),
                        (
                            int(
                                round(
                                    x2
                                )
                            ),
                            int(
                                round(
                                    y2
                                )
                            )
                        ),
                        CYAN,
                        2
                    )

                    cv2.putText(
                        frame,
                        "ESTIMATED 2D STRIKE ZONE",
                        (
                            int(
                                round(
                                    x1
                                )
                            ),
                            max(
                                30,
                                int(
                                    round(
                                        y1
                                    )
                                )
                                - 10
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        CYAN,
                        2,
                        cv2.LINE_AA
                    )

        # ====================================================
        # PLATE X PROXY
        # ====================================================

        cv2.line(
            frame,
            (
                int(
                    round(
                        plate_x
                    )
                ),
                0
            ),
            (
                int(
                    round(
                        plate_x
                    )
                ),
                height
            ),
            GRAY,
            1
        )

        cv2.putText(
            frame,
            "PLATE X PROXY",
            (
                min(
                    width - 180,
                    int(
                        round(
                            plate_x
                        )
                    ) + 8
                ),
                30
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            GRAY,
            1,
            cv2.LINE_AA
        )

        # ====================================================
        # BALL TRACKING
        # ====================================================

        ball_row = ball_by_frame.get(
            frame_number
        )

        if ball_row is not None:

            is_yolo = bool_from_value(
                ball_row[yolo_col]
            )

            observed_x = safe_float(
                ball_row[
                    observed_x_col
                ]
            )

            observed_y = safe_float(
                ball_row[
                    observed_y_col
                ]
            )

            filtered_x = safe_float(
                ball_row[
                    filtered_x_col
                ]
            )

            filtered_y = safe_float(
                ball_row[
                    filtered_y_col
                ]
            )

            state = str(
                ball_row[
                    state_col
                ]
            ).upper()

            # -----------------------------------------------
            # REAL YOLO BALL
            # -----------------------------------------------

            if (
                is_yolo
                and observed_x is not None
                and observed_y is not None
            ):

                point = (
                    int(
                        round(
                            observed_x
                        )
                    ),
                    int(
                        round(
                            observed_y
                        )
                    )
                )

                real_trail.append(
                    point
                )

                real_trail = (
                    real_trail[
                        -TRAIL_LENGTH:
                    ]
                )

                cv2.circle(
                    frame,
                    point,
                    BALL_RADIUS,
                    GREEN,
                    -1
                )

                cv2.circle(
                    frame,
                    point,
                    BALL_RADIUS + 3,
                    GREEN,
                    1
                )

                cv2.putText(
                    frame,
                    "YOLO BALL",
                    (
                        point[0] + 10,
                        point[1] - 10
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    GREEN,
                    2,
                    cv2.LINE_AA
                )

            # -----------------------------------------------
            # KALMAN ONLY
            # -----------------------------------------------

            elif (
                state == "KALMAN"
                and filtered_x is not None
                and filtered_y is not None
            ):

                point = (
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

                cv2.drawMarker(
                    frame,
                    point,
                    YELLOW,
                    cv2.MARKER_TILTED_CROSS,
                    14,
                    2
                )

                cv2.putText(
                    frame,
                    "KALMAN ONLY",
                    (
                        point[0] + 10,
                        point[1] - 10
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    YELLOW,
                    2,
                    cv2.LINE_AA
                )

        # ====================================================
        # REAL YOLO TRAIL
        # ====================================================

        if len(
            real_trail
        ) >= 2:

            for i in range(
                1,
                len(
                    real_trail
                )
            ):

                cv2.line(
                    frame,
                    real_trail[i - 1],
                    real_trail[i],
                    GREEN,
                    2
                )

        # ====================================================
        # FIT / EXTRAPOLATED TRAJECTORY
        # ====================================================

        if (
            frame_number
            >= trajectory_start_frame - 2
        ):

            for i in range(
                1,
                len(
                    trajectory_points
                )
            ):

                cv2.line(
                    frame,
                    trajectory_points[
                        i - 1
                    ],
                    trajectory_points[i],
                    ORANGE,
                    2
                )

            cv2.putText(
                frame,
                "LOCAL TRAJECTORY FIT",
                (
                    max(
                        20,
                        trajectory_points[0][0]
                        - 40
                    ),
                    max(
                        30,
                        trajectory_points[0][1]
                        - 20
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                ORANGE,
                1,
                cv2.LINE_AA
            )

        # ====================================================
        # CROSSING POINT
        # ====================================================

        if (
            frame_number
            >= int(
                np.floor(
                    crossing_frame
                )
            )
            - 2
        ):

            cv2.drawMarker(
                frame,
                crossing_point,
                RED,
                cv2.MARKER_CROSS,
                24,
                3
            )

            cv2.putText(
                frame,
                "ESTIMATED PLATE CROSSING",
                (
                    max(
                        20,
                        crossing_point[0]
                        - 260
                    ),
                    max(
                        30,
                        crossing_point[1]
                        - 20
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                RED,
                2,
                cv2.LINE_AA
            )

        # ====================================================
        # INFO PANEL
        # ====================================================

        panel = frame.copy()

        cv2.rectangle(
            panel,
            (20, 20),
            (650, 245),
            (0, 0, 0),
            -1
        )

        cv2.addWeighted(
            panel,
            0.60,
            frame,
            0.40,
            0,
            frame
        )

        cv2.putText(
            frame,
            "BaseVisionAI - Pitch Crossing V1",
            (40, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            WHITE,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Frame: {frame_number}/{total_frames}",
            (40, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            WHITE,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Fit frames: "
            f"{int(frames_fit[0])}"
            f" -> "
            f"{int(frames_fit[-1])}",
            (40, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            WHITE,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Mean residual: "
            f"{mean_residual:.2f}px",
            (40, 134),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            WHITE,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Crossing frame: "
            f"{crossing_frame:.2f}",
            (40, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            WHITE,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Crossing Y: "
            f"{crossing_y:.1f}px",
            (40, 186),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            WHITE,
            1,
            cv2.LINE_AA
        )

        result_color = (
            GREEN
            if classification
            == "STRIKE_CANDIDATE"
            else ORANGE
        )

        cv2.putText(
            frame,
            classification,
            (40, 216),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            result_color,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            (
                "EVIDENCE: "
                f"{evidence_quality}"
                " | 2D ESTIMATE"
                " - NOT UMPIRE GRADE"
            ),
            (290, 216),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            WHITE,
            1,
            cv2.LINE_AA
        )

        # ====================================================
        # WRITE FRAME
        # ====================================================

        writer.write(
            frame
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    writer.release()

    print()
    print("=" * 78)
    print("VISUALIZATION COMPLETE")
    print("=" * 78)

    print(
        f"Real YOLO observations : {len(real_points)}"
    )

    print(
        f"Fit frames             : "
        f"{int(frames_fit[0])} -> "
        f"{int(frames_fit[-1])}"
    )

    print(
        f"Mean fit residual      : "
        f"{mean_residual:.2f} px"
    )

    print(
        f"Plate X                : "
        f"{plate_x:.1f} px"
    )

    print(
        f"Crossing frame         : "
        f"{crossing_frame:.2f}"
    )

    print(
        f"Crossing Y             : "
        f"{crossing_y:.1f} px"
    )

    print(
        f"Extrapolation          : "
        f"{extrapolation_frames:.2f} frames"
    )

    print(
        f"Zone reference frame   : "
        f"{nearest_zone_frame}"
    )

    print(
        f"Horizontal inside      : "
        f"{horizontal_inside}"
    )

    print(
        f"Vertical inside        : "
        f"{vertical_inside}"
    )

    print(
        f"Classification         : "
        f"{classification}"
    )

    print()
    print(
        f"Output video:\n{output_path}"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()