from pathlib import Path
import pandas as pd
import numpy as np
import math


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "ball_tracking_v3_1.csv"
)

OUTPUT_SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "pitch_analysis_summary.csv"
)

OUTPUT_POINTS_CSV = (
    PROJECT_ROOT
    / "results"
    / "pitch_analysis_points.csv"
)


# ============================================================
# VIDEO FPS
# ============================================================

# IMPORTANT:
# Change this if your video uses a different FPS.
#
# You can later read FPS directly from the source video,
# but for now we keep it explicit.
FPS = 30.0


# ============================================================
# FUNCTIONS
# ============================================================

def euclidean_distance(x1, y1, x2, y2):
    """
    Calculate Euclidean distance between two image points.
    """

    return math.sqrt(
        (x2 - x1) ** 2
        +
        (y2 - y1) ** 2
    )


def calculate_angle(dx, dy):
    """
    Calculate movement angle in degrees.

    Image coordinates:
        +X = right
        +Y = down

    atan2 returns angle relative to positive X axis.
    """

    return math.degrees(
        math.atan2(dy, dx)
    )


def direction_label(dx, dy):
    """
    Simple direction label based on image coordinates.
    """

    threshold = 3

    if abs(dx) < threshold and abs(dy) < threshold:
        return "STATIONARY"

    horizontal = ""
    vertical = ""

    if dx > threshold:
        horizontal = "RIGHT"
    elif dx < -threshold:
        horizontal = "LEFT"

    if dy > threshold:
        vertical = "DOWN"
    elif dy < -threshold:
        vertical = "UP"

    if horizontal and vertical:
        return f"{vertical}_{horizontal}"

    if horizontal:
        return horizontal

    if vertical:
        return vertical

    return "UNKNOWN"


def clean_track_points(track_df):
    """
    Keep only points with a valid filtered position.

    This includes:
        TRACKING points
        KALMAN points

    LOST/RESET rows without position are removed.
    """

    clean_df = track_df.copy()

    clean_df = clean_df.dropna(
        subset=[
            "filtered_x",
            "filtered_y"
        ]
    )

    clean_df = clean_df.sort_values(
        "frame"
    )

    clean_df = clean_df.reset_index(
        drop=True
    )

    return clean_df


def analyze_track(track_id, track_df):
    """
    Analyze one individual trajectory.
    """

    clean_df = clean_track_points(
        track_df
    )

    if len(clean_df) < 2:
        return None, clean_df

    # ========================================================
    # BASIC TRACK INFORMATION
    # ========================================================

    start_frame = int(
        clean_df.iloc[0]["frame"]
    )

    end_frame = int(
        clean_df.iloc[-1]["frame"]
    )

    frame_span = (
        end_frame
        -
        start_frame
        +
        1
    )

    duration_seconds = (
        (end_frame - start_frame)
        /
        FPS
    )

    # ========================================================
    # REAL YOLO VS KALMAN POINTS
    # ========================================================

    yolo_points = int(
        (
            clean_df["yolo_detected"]
            == True
        ).sum()
    )

    kalman_points = int(
        (
            clean_df["state"]
            == "KALMAN"
        ).sum()
    )

    total_valid_points = len(
        clean_df
    )

    # ========================================================
    # START / END
    # ========================================================

    start_x = float(
        clean_df.iloc[0]["filtered_x"]
    )

    start_y = float(
        clean_df.iloc[0]["filtered_y"]
    )

    end_x = float(
        clean_df.iloc[-1]["filtered_x"]
    )

    end_y = float(
        clean_df.iloc[-1]["filtered_y"]
    )

    displacement_x = (
        end_x - start_x
    )

    displacement_y = (
        end_y - start_y
    )

    direct_displacement_px = (
        euclidean_distance(
            start_x,
            start_y,
            end_x,
            end_y
        )
    )

    # ========================================================
    # POINT-BY-POINT MOVEMENT
    # ========================================================

    step_distances = []
    dx_values = []
    dy_values = []
    speeds_px_frame = []
    speeds_px_second = []
    angles = []
    directions = []

    previous_row = None

    for index, row in clean_df.iterrows():

        if previous_row is None:

            step_distances.append(
                np.nan
            )

            dx_values.append(
                np.nan
            )

            dy_values.append(
                np.nan
            )

            speeds_px_frame.append(
                np.nan
            )

            speeds_px_second.append(
                np.nan
            )

            angles.append(
                np.nan
            )

            directions.append(
                "START"
            )

            previous_row = row
            continue

        frame_delta = (
            row["frame"]
            -
            previous_row["frame"]
        )

        if frame_delta <= 0:
            frame_delta = 1

        dx = (
            row["filtered_x"]
            -
            previous_row["filtered_x"]
        )

        dy = (
            row["filtered_y"]
            -
            previous_row["filtered_y"]
        )

        distance = (
            euclidean_distance(
                previous_row["filtered_x"],
                previous_row["filtered_y"],
                row["filtered_x"],
                row["filtered_y"]
            )
        )

        speed_px_frame = (
            distance
            /
            frame_delta
        )

        speed_px_second = (
            speed_px_frame
            *
            FPS
        )

        angle = calculate_angle(
            dx,
            dy
        )

        direction = direction_label(
            dx,
            dy
        )

        step_distances.append(
            distance
        )

        dx_values.append(
            dx
        )

        dy_values.append(
            dy
        )

        speeds_px_frame.append(
            speed_px_frame
        )

        speeds_px_second.append(
            speed_px_second
        )

        angles.append(
            angle
        )

        directions.append(
            direction
        )

        previous_row = row

    # Add point metrics to dataframe
    clean_df = clean_df.copy()

    clean_df["dx"] = dx_values
    clean_df["dy"] = dy_values
    clean_df["step_distance_px"] = (
        step_distances
    )

    clean_df["speed_px_per_frame"] = (
        speeds_px_frame
    )

    clean_df["speed_px_per_second"] = (
        speeds_px_second
    )

    clean_df["movement_angle_deg"] = (
        angles
    )

    clean_df["direction"] = (
        directions
    )

    # ========================================================
    # TOTAL DISTANCE
    # ========================================================

    total_distance_px = float(
        np.nansum(
            step_distances
        )
    )

    # ========================================================
    # SPEED STATISTICS
    # ========================================================

    valid_speeds_frame = np.array(
        [
            value
            for value
            in speeds_px_frame
            if not pd.isna(value)
        ],
        dtype=float
    )

    valid_speeds_second = np.array(
        [
            value
            for value
            in speeds_px_second
            if not pd.isna(value)
        ],
        dtype=float
    )

    if len(valid_speeds_frame) > 0:

        avg_speed_px_frame = float(
            np.mean(
                valid_speeds_frame
            )
        )

        max_speed_px_frame = float(
            np.max(
                valid_speeds_frame
            )
        )

        min_speed_px_frame = float(
            np.min(
                valid_speeds_frame
            )
        )

    else:

        avg_speed_px_frame = np.nan
        max_speed_px_frame = np.nan
        min_speed_px_frame = np.nan

    if len(valid_speeds_second) > 0:

        avg_speed_px_second = float(
            np.mean(
                valid_speeds_second
            )
        )

        max_speed_px_second = float(
            np.max(
                valid_speeds_second
            )
        )

    else:

        avg_speed_px_second = np.nan
        max_speed_px_second = np.nan

    # ========================================================
    # OVERALL MOVEMENT ANGLE
    # ========================================================

    overall_angle_deg = (
        calculate_angle(
            displacement_x,
            displacement_y
        )
    )

    overall_direction = (
        direction_label(
            displacement_x,
            displacement_y
        )
    )

    # ========================================================
    # DIRECTION CHANGES
    # ========================================================

    valid_angles = [
        angle
        for angle
        in angles
        if not pd.isna(angle)
    ]

    angle_changes = []

    for i in range(
        1,
        len(valid_angles)
    ):

        diff = abs(
            valid_angles[i]
            -
            valid_angles[i - 1]
        )

        # Normalize to 0-180 degrees
        if diff > 180:
            diff = 360 - diff

        angle_changes.append(
            diff
        )

    if angle_changes:

        avg_angle_change_deg = float(
            np.mean(
                angle_changes
            )
        )

        max_angle_change_deg = float(
            np.max(
                angle_changes
            )
        )

    else:

        avg_angle_change_deg = np.nan
        max_angle_change_deg = np.nan

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "track_id": int(track_id),

        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_span": frame_span,
        "duration_seconds": duration_seconds,

        "valid_points": total_valid_points,
        "yolo_points": yolo_points,
        "kalman_points": kalman_points,

        "start_x": start_x,
        "start_y": start_y,

        "end_x": end_x,
        "end_y": end_y,

        "displacement_x_px":
            displacement_x,

        "displacement_y_px":
            displacement_y,

        "direct_displacement_px":
            direct_displacement_px,

        "total_path_distance_px":
            total_distance_px,

        "avg_speed_px_per_frame":
            avg_speed_px_frame,

        "max_speed_px_per_frame":
            max_speed_px_frame,

        "min_speed_px_per_frame":
            min_speed_px_frame,

        "avg_speed_px_per_second":
            avg_speed_px_second,

        "max_speed_px_per_second":
            max_speed_px_second,

        "overall_angle_deg":
            overall_angle_deg,

        "overall_direction":
            overall_direction,

        "avg_angle_change_deg":
            avg_angle_change_deg,

        "max_angle_change_deg":
            max_angle_change_deg,
    }

    return summary, clean_df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BaseVisionAI - Pitch Analytics V1")
    print("=" * 70)

    print(f"\nInput CSV:")
    print(INPUT_CSV)

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Tracking CSV not found:\n"
            f"{INPUT_CSV}"
        )

    # ========================================================
    # LOAD TRACKING CSV
    # ========================================================

    df = pd.read_csv(
        INPUT_CSV
    )

    print(
        f"\nRows loaded: "
        f"{len(df)}"
    )

    # ========================================================
    # KEEP REAL TRACK IDS
    # ========================================================

    valid_tracks = df[
        df["track_id"] > 0
    ].copy()

    track_ids = sorted(
        valid_tracks[
            "track_id"
        ].unique()
    )

    print(
        f"Tracks found: "
        f"{track_ids}"
    )

    # ========================================================
    # ANALYZE EACH TRACK
    # ========================================================

    summaries = []

    point_dataframes = []

    for track_id in track_ids:

        print(
            "\n"
            +
            "-" * 70
        )

        print(
            f"Analyzing Track "
            f"{track_id}"
        )

        track_df = valid_tracks[
            valid_tracks["track_id"]
            ==
            track_id
        ].copy()

        summary, point_df = (
            analyze_track(
                track_id,
                track_df
            )
        )

        if summary is None:

            print(
                "Not enough valid points."
            )

            continue

        summaries.append(
            summary
        )

        point_df = point_df.copy()

        point_df.insert(
            0,
            "analysis_track_id",
            track_id
        )

        point_dataframes.append(
            point_df
        )

        print(
            f"Frames: "
            f"{summary['start_frame']} "
            f"-> "
            f"{summary['end_frame']}"
        )

        print(
            f"YOLO points: "
            f"{summary['yolo_points']}"
        )

        print(
            f"Kalman points: "
            f"{summary['kalman_points']}"
        )

        print(
            f"Displacement X: "
            f"{summary['displacement_x_px']:.2f} px"
        )

        print(
            f"Displacement Y: "
            f"{summary['displacement_y_px']:.2f} px"
        )

        print(
            f"Direct displacement: "
            f"{summary['direct_displacement_px']:.2f} px"
        )

        print(
            f"Total path distance: "
            f"{summary['total_path_distance_px']:.2f} px"
        )

        print(
            f"Average speed: "
            f"{summary['avg_speed_px_per_frame']:.2f} "
            f"px/frame"
        )

        print(
            f"Average speed: "
            f"{summary['avg_speed_px_per_second']:.2f} "
            f"px/s"
        )

        print(
            f"Direction: "
            f"{summary['overall_direction']}"
        )

        print(
            f"Movement angle: "
            f"{summary['overall_angle_deg']:.2f} deg"
        )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    OUTPUT_SUMMARY_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_df = pd.DataFrame(
        summaries
    )

    summary_df.to_csv(
        OUTPUT_SUMMARY_CSV,
        index=False
    )

    # ========================================================
    # SAVE POINT-BY-POINT ANALYSIS
    # ========================================================

    if point_dataframes:

        points_df = pd.concat(
            point_dataframes,
            ignore_index=True
        )

        points_df.to_csv(
            OUTPUT_POINTS_CSV,
            index=False
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "PITCH ANALYTICS COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        "\nSummary:"
    )

    print(
        OUTPUT_SUMMARY_CSV
    )

    print(
        "\nPoint-by-point:"
    )

    print(
        OUTPUT_POINTS_CSV
    )


if __name__ == "__main__":
    main()