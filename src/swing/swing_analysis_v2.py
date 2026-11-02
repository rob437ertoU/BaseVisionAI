from pathlib import Path
import argparse
import math

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "results"
    / "batter_pose_v2"
    / "baseball_train10_batter_pose_v2_smoothed.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "swing_analysis_v2"
)


# ============================================================
# CONFIGURATION
# ============================================================

MIN_POSE_QUALITY = 0.65

# Small rolling window for derived motion metrics.
# We already smoothed the pose, so keep this conservative.
MOTION_SMOOTHING_WINDOW = 3

# Number of frames around the peak classified as PEAK_SWING.
PEAK_WINDOW = 2

# These thresholds are relative to the maximum swing motion
# found in THIS clip. They are only temporal heuristics.
LOAD_THRESHOLD = 0.22
SWING_THRESHOLD = 0.40


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Swing Analysis V2"
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Smoothed batter pose CSV"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory"
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help=(
            "Video FPS. If omitted, estimated from "
            "time_sec in the CSV."
        )
    )

    return parser.parse_args()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_numeric(df, column):
    """
    Return numeric pandas Series.
    Missing/non-numeric values become NaN.
    """

    if column not in df.columns:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float
        )

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


def point(df, name):
    """
    Get smoothed x/y for one keypoint.
    """

    x = safe_numeric(
        df,
        f"{name}_x_smooth"
    )

    y = safe_numeric(
        df,
        f"{name}_y_smooth"
    )

    return x, y


def midpoint(x1, y1, x2, y2):

    x = (x1 + x2) / 2.0
    y = (y1 + y2) / 2.0

    return x, y


def distance(x1, y1, x2, y2):

    return np.sqrt(
        (x2 - x1) ** 2
        + (y2 - y1) ** 2
    )


def velocity(x, y):
    """
    Frame-to-frame 2D displacement.

    Units:
        pixels/frame

    This is NOT physical velocity.
    """

    dx = x.diff()
    dy = y.diff()

    speed = np.sqrt(
        dx ** 2
        + dy ** 2
    )

    return speed


def body_line_angle(x1, y1, x2, y2):
    """
    Orientation of a body segment in image space.

    Result mapped to [0, 180) because a shoulder/hip axis
    has no meaningful directional arrow.
    """

    dx = x2 - x1
    dy = y2 - y1

    angle = np.degrees(
        np.arctan2(
            dy,
            dx
        )
    )

    angle = np.mod(
        angle,
        180.0
    )

    return pd.Series(
        angle,
        index=x1.index
    )


def axial_angle_difference(angle_a, angle_b):
    """
    Smallest difference between two axial orientations.

    Since an axis at 0 degrees is equivalent to one at
    180 degrees, the result is constrained to [0, 90].
    """

    diff = np.abs(
        angle_a - angle_b
    )

    diff = np.minimum(
        diff,
        180.0 - diff
    )

    return diff


def rolling_median(series, window):

    return (
        series
        .rolling(
            window=window,
            center=True,
            min_periods=1
        )
        .median()
    )


def normalize_series(series):

    valid = series.dropna()

    if len(valid) == 0:
        return pd.Series(
            0.0,
            index=series.index
        )

    max_value = valid.max()

    if (
        not np.isfinite(max_value)
        or max_value <= 0
    ):

        return pd.Series(
            0.0,
            index=series.index
        )

    return (
        series / max_value
    ).clip(
        lower=0.0,
        upper=1.0
    )


def valid_argmax(series):

    valid = series.dropna()

    if len(valid) == 0:
        return None

    return int(
        valid.idxmax()
    )


# ============================================================
# FPS
# ============================================================

def estimate_fps(df):

    if "time_sec" not in df.columns:
        return 30.0

    time = safe_numeric(
        df,
        "time_sec"
    )

    dt = (
        time.diff()
        .dropna()
    )

    dt = dt[
        dt > 0
    ]

    if len(dt) == 0:
        return 30.0

    median_dt = float(
        dt.median()
    )

    if median_dt <= 0:
        return 30.0

    return 1.0 / median_dt


# ============================================================
# PHASE DETECTION
# ============================================================

def detect_phases(
    df,
    peak_index,
):
    """
    Temporal heuristic.

    These labels describe motion regions in the clip.
    They are NOT biomechanical ground truth and
    PEAK_SWING is NOT confirmed ball contact.
    """

    n = len(df)

    phases = np.array(
        ["SETUP"] * n,
        dtype=object
    )

    if peak_index is None:
        return phases

    energy = (
        df["swing_motion_energy"]
        .fillna(0.0)
        .to_numpy()
    )

    peak_value = energy[
        peak_index
    ]

    if peak_value <= 0:
        return phases

    normalized = (
        energy / peak_value
    )

    # --------------------------------------------------------
    # Find swing start
    # --------------------------------------------------------

    before_peak = np.arange(
        0,
        peak_index + 1
    )

    swing_candidates = before_peak[
        normalized[
            before_peak
        ] >= SWING_THRESHOLD
    ]

    if len(swing_candidates) > 0:

        swing_start = int(
            swing_candidates[0]
        )

    else:

        swing_start = max(
            0,
            peak_index - 5
        )

    # --------------------------------------------------------
    # Find approximate load start
    # --------------------------------------------------------

    load_search = np.arange(
        0,
        swing_start
    )

    load_candidates = load_search[
        normalized[
            load_search
        ] >= LOAD_THRESHOLD
    ]

    if len(load_candidates) > 0:

        load_start = int(
            load_candidates[0]
        )

    else:

        load_start = max(
            0,
            swing_start - 8
        )

    # --------------------------------------------------------
    # Peak window
    # --------------------------------------------------------

    peak_start = max(
        0,
        peak_index - PEAK_WINDOW
    )

    peak_end = min(
        n - 1,
        peak_index + PEAK_WINDOW
    )

    # --------------------------------------------------------
    # Assign phases
    # --------------------------------------------------------

    phases[:load_start] = "SETUP"

    phases[
        load_start:swing_start
    ] = "LOAD"

    phases[
        swing_start:peak_start
    ] = "SWING"

    phases[
        peak_start:peak_end + 1
    ] = "PEAK_SWING"

    if peak_end + 1 < n:

        phases[
            peak_end + 1:
        ] = "FOLLOW_THROUGH"

    return phases


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    input_path = Path(
        args.input
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    if not input_path.exists():

        raise FileNotFoundError(
            f"Input CSV not found:\n"
            f"{input_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_frame_csv = (
        output_dir
        / f"{input_path.stem}_swing_analysis.csv"
    )

    output_summary_csv = (
        output_dir
        / f"{input_path.stem}_swing_summary.csv"
    )

    # ========================================================
    # LOAD
    # ========================================================

    df = pd.read_csv(
        input_path
    )

    if len(df) == 0:

        raise RuntimeError(
            "Input CSV contains no frames."
        )

    if args.fps is not None:

        fps = args.fps

    else:

        fps = estimate_fps(df)

    print()
    print("=" * 76)
    print("BaseVisionAI - Swing Analysis V2")
    print("=" * 76)

    print(
        f"Input       : {input_path}"
    )

    print(
        f"Frames      : {len(df)}"
    )

    print(
        f"FPS         : {fps:.3f}"
    )

    print(
        f"Pose quality: >= {MIN_POSE_QUALITY:.2f}"
    )

    print("=" * 76)

    # ========================================================
    # POSE QUALITY FILTER
    # ========================================================

    pose_quality = safe_numeric(
        df,
        "pose_quality"
    )

    df["analysis_valid"] = (
        pose_quality
        >= MIN_POSE_QUALITY
    ).astype(int)

    # ========================================================
    # LOAD KEYPOINTS
    # ========================================================

    lsx, lsy = point(
        df,
        "left_shoulder"
    )

    rsx, rsy = point(
        df,
        "right_shoulder"
    )

    lex, ley = point(
        df,
        "left_elbow"
    )

    rex, rey = point(
        df,
        "right_elbow"
    )

    lwx, lwy = point(
        df,
        "left_wrist"
    )

    rwx, rwy = point(
        df,
        "right_wrist"
    )

    lhx, lhy = point(
        df,
        "left_hip"
    )

    rhx, rhy = point(
        df,
        "right_hip"
    )

    lkx, lky = point(
        df,
        "left_knee"
    )

    rkx, rky = point(
        df,
        "right_knee"
    )

    lax, lay = point(
        df,
        "left_ankle"
    )

    rax, ray = point(
        df,
        "right_ankle"
    )

    # ========================================================
    # BODY CENTERS
    # ========================================================

    shoulder_cx, shoulder_cy = midpoint(
        lsx,
        lsy,
        rsx,
        rsy
    )

    hip_cx, hip_cy = midpoint(
        lhx,
        lhy,
        rhx,
        rhy
    )

    df["shoulder_center_x"] = shoulder_cx
    df["shoulder_center_y"] = shoulder_cy

    df["hip_center_x"] = hip_cx
    df["hip_center_y"] = hip_cy

    # ========================================================
    # BODY SCALE
    # ========================================================

    torso_length = distance(
        shoulder_cx,
        shoulder_cy,
        hip_cx,
        hip_cy
    )

    # Prevent unstable normalization if torso length gets tiny.
    torso_length = torso_length.where(
        torso_length >= 10.0,
        np.nan
    )

    df["torso_length_px"] = (
        torso_length
    )

    median_torso = float(
        torso_length.median()
    )

    # ========================================================
    # WRIST VELOCITIES
    # ========================================================

    left_wrist_speed = velocity(
        lwx,
        lwy
    )

    right_wrist_speed = velocity(
        rwx,
        rwy
    )

    left_wrist_speed = rolling_median(
        left_wrist_speed,
        MOTION_SMOOTHING_WINDOW
    )

    right_wrist_speed = rolling_median(
        right_wrist_speed,
        MOTION_SMOOTHING_WINDOW
    )

    df["left_wrist_speed_px_frame"] = (
        left_wrist_speed
    )

    df["right_wrist_speed_px_frame"] = (
        right_wrist_speed
    )

    df["left_wrist_speed_px_sec"] = (
        left_wrist_speed
        * fps
    )

    df["right_wrist_speed_px_sec"] = (
        right_wrist_speed
        * fps
    )

    # --------------------------------------------------------
    # Normalized velocities
    #
    # torso lengths per frame
    # --------------------------------------------------------

    df["left_wrist_speed_norm"] = (
        left_wrist_speed
        / torso_length
    )

    df["right_wrist_speed_norm"] = (
        right_wrist_speed
        / torso_length
    )

    # ========================================================
    # HIP / SHOULDER CENTER MOTION
    # ========================================================

    shoulder_speed = velocity(
        shoulder_cx,
        shoulder_cy
    )

    hip_speed = velocity(
        hip_cx,
        hip_cy
    )

    shoulder_speed = rolling_median(
        shoulder_speed,
        MOTION_SMOOTHING_WINDOW
    )

    hip_speed = rolling_median(
        hip_speed,
        MOTION_SMOOTHING_WINDOW
    )

    df["shoulder_center_speed_px_frame"] = (
        shoulder_speed
    )

    df["hip_center_speed_px_frame"] = (
        hip_speed
    )

    df["shoulder_center_speed_norm"] = (
        shoulder_speed
        / torso_length
    )

    df["hip_center_speed_norm"] = (
        hip_speed
        / torso_length
    )

    # ========================================================
    # SHOULDER / HIP ORIENTATION
    # ========================================================

    shoulder_angle = body_line_angle(
        lsx,
        lsy,
        rsx,
        rsy
    )

    hip_angle = body_line_angle(
        lhx,
        lhy,
        rhx,
        rhy
    )

    shoulder_hip_separation = (
        axial_angle_difference(
            shoulder_angle,
            hip_angle
        )
    )

    df["shoulder_axis_deg"] = (
        shoulder_angle
    )

    df["hip_axis_deg"] = (
        hip_angle
    )

    df["shoulder_hip_separation_deg"] = (
        shoulder_hip_separation
    )

    # ========================================================
    # KNEE / ANKLE MOTION
    # ========================================================

    left_knee_speed = velocity(
        lkx,
        lky
    )

    right_knee_speed = velocity(
        rkx,
        rky
    )

    left_ankle_speed = velocity(
        lax,
        lay
    )

    right_ankle_speed = velocity(
        rax,
        ray
    )

    df["left_knee_speed_norm"] = (
        rolling_median(
            left_knee_speed,
            MOTION_SMOOTHING_WINDOW
        )
        / torso_length
    )

    df["right_knee_speed_norm"] = (
        rolling_median(
            right_knee_speed,
            MOTION_SMOOTHING_WINDOW
        )
        / torso_length
    )

    df["left_ankle_speed_norm"] = (
        rolling_median(
            left_ankle_speed,
            MOTION_SMOOTHING_WINDOW
        )
        / torso_length
    )

    df["right_ankle_speed_norm"] = (
        rolling_median(
            right_ankle_speed,
            MOTION_SMOOTHING_WINDOW
        )
        / torso_length
    )

    # ========================================================
    # COMBINED WRIST MOTION
    # ========================================================

    wrist_motion = pd.concat(
        [
            df["left_wrist_speed_norm"],
            df["right_wrist_speed_norm"]
        ],
        axis=1
    ).max(
        axis=1
    )

    df["wrist_motion_norm"] = (
        wrist_motion
    )

    # ========================================================
    # SWING MOTION ENERGY
    # ========================================================
    #
    # This is an evidence score for temporal segmentation.
    #
    # It is NOT:
    # - physical kinetic energy
    # - contact probability
    # - biomechanical power
    #
    # Wrist motion has the largest weight because hand motion
    # is particularly useful for identifying the swing burst.
    # ========================================================

    wrist_component = normalize_series(
        df["wrist_motion_norm"]
    )

    hip_component = normalize_series(
        df["hip_center_speed_norm"]
    )

    shoulder_component = normalize_series(
        df["shoulder_center_speed_norm"]
    )

    df["swing_motion_energy"] = (
        0.60 * wrist_component
        + 0.20 * hip_component
        + 0.20 * shoulder_component
    )

    # --------------------------------------------------------
    # Reject frames where pose quality is insufficient
    # --------------------------------------------------------

    df.loc[
        df["analysis_valid"] == 0,
        "swing_motion_energy"
    ] = np.nan

    # ========================================================
    # FIND PEAK
    # ========================================================

    peak_index = valid_argmax(
        df["swing_motion_energy"]
    )

    if peak_index is not None:

        peak_frame = int(
            df.loc[
                peak_index,
                "frame"
            ]
        )

        peak_time = float(
            df.loc[
                peak_index,
                "time_sec"
            ]
        )

    else:

        peak_frame = None
        peak_time = None

    # ========================================================
    # PHASES
    # ========================================================

    phases = detect_phases(
        df,
        peak_index
    )

    df["swing_phase"] = (
        phases
    )

    df["peak_swing_candidate"] = 0

    if peak_index is not None:

        df.loc[
            peak_index,
            "peak_swing_candidate"
        ] = 1

    # ========================================================
    # PEAK METRICS
    # ========================================================

    left_wrist_peak_index = (
        valid_argmax(
            df[
                "left_wrist_speed_norm"
            ]
        )
    )

    right_wrist_peak_index = (
        valid_argmax(
            df[
                "right_wrist_speed_norm"
            ]
        )
    )

    hip_peak_index = (
        valid_argmax(
            df[
                "hip_center_speed_norm"
            ]
        )
    )

    shoulder_peak_index = (
        valid_argmax(
            df[
                "shoulder_center_speed_norm"
            ]
        )
    )

    separation_peak_index = (
        valid_argmax(
            df[
                "shoulder_hip_separation_deg"
            ]
        )
    )

    # ========================================================
    # HELPER FOR SUMMARY
    # ========================================================

    def frame_from_index(index):

        if index is None:
            return np.nan

        return int(
            df.loc[
                index,
                "frame"
            ]
        )

    def value_from_index(
        index,
        column
    ):

        if index is None:
            return np.nan

        return float(
            df.loc[
                index,
                column
            ]
        )

    # ========================================================
    # PHASE FRAME RANGES
    # ========================================================

    phase_ranges = {}

    for phase in [
        "SETUP",
        "LOAD",
        "SWING",
        "PEAK_SWING",
        "FOLLOW_THROUGH"
    ]:

        selected = df[
            df["swing_phase"]
            == phase
        ]

        if len(selected) > 0:

            phase_ranges[
                f"{phase.lower()}_start_frame"
            ] = int(
                selected["frame"].min()
            )

            phase_ranges[
                f"{phase.lower()}_end_frame"
            ] = int(
                selected["frame"].max()
            )

        else:

            phase_ranges[
                f"{phase.lower()}_start_frame"
            ] = np.nan

            phase_ranges[
                f"{phase.lower()}_end_frame"
            ] = np.nan

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "video_frames": len(df),

        "fps": fps,

        "median_torso_length_px":
            median_torso,

        "average_pose_quality":
            float(
                pose_quality.mean()
            ),

        "peak_swing_candidate_frame":
            peak_frame,

        "peak_swing_candidate_time_sec":
            peak_time,

        "peak_swing_energy":
            (
                value_from_index(
                    peak_index,
                    "swing_motion_energy"
                )
                if peak_index is not None
                else np.nan
            ),

        "left_wrist_peak_frame":
            frame_from_index(
                left_wrist_peak_index
            ),

        "left_wrist_peak_norm":
            value_from_index(
                left_wrist_peak_index,
                "left_wrist_speed_norm"
            ),

        "right_wrist_peak_frame":
            frame_from_index(
                right_wrist_peak_index
            ),

        "right_wrist_peak_norm":
            value_from_index(
                right_wrist_peak_index,
                "right_wrist_speed_norm"
            ),

        "hip_motion_peak_frame":
            frame_from_index(
                hip_peak_index
            ),

        "hip_motion_peak_norm":
            value_from_index(
                hip_peak_index,
                "hip_center_speed_norm"
            ),

        "shoulder_motion_peak_frame":
            frame_from_index(
                shoulder_peak_index
            ),

        "shoulder_motion_peak_norm":
            value_from_index(
                shoulder_peak_index,
                "shoulder_center_speed_norm"
            ),

        "max_shoulder_hip_separation_frame":
            frame_from_index(
                separation_peak_index
            ),

        "max_shoulder_hip_separation_deg":
            value_from_index(
                separation_peak_index,
                "shoulder_hip_separation_deg"
            ),
    }

    summary.update(
        phase_ranges
    )

    summary_df = pd.DataFrame(
        [summary]
    )

    # ========================================================
    # SAVE
    # ========================================================

    df.to_csv(
        output_frame_csv,
        index=False
    )

    summary_df.to_csv(
        output_summary_csv,
        index=False
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 76)
    print("SWING ANALYSIS V2 SUMMARY")
    print("=" * 76)

    print(
        f"Frames analyzed             : "
        f"{len(df)}"
    )

    print(
        f"Average pose quality        : "
        f"{pose_quality.mean():.3f}"
    )

    print(
        f"Median torso length         : "
        f"{median_torso:.2f} px"
    )

    print()

    print(
        f"Peak swing candidate        : "
        f"frame {peak_frame}"
    )

    if peak_time is not None:

        print(
            f"Peak swing time             : "
            f"{peak_time:.3f} s"
        )

    print()

    print(
        f"Left wrist peak             : "
        f"frame "
        f"{frame_from_index(left_wrist_peak_index)}"
    )

    print(
        f"Right wrist peak            : "
        f"frame "
        f"{frame_from_index(right_wrist_peak_index)}"
    )

    print(
        f"Hip motion peak             : "
        f"frame "
        f"{frame_from_index(hip_peak_index)}"
    )

    print(
        f"Shoulder motion peak        : "
        f"frame "
        f"{frame_from_index(shoulder_peak_index)}"
    )

    print()

    print(
        f"Max shoulder/hip separation : "
        f"{value_from_index(separation_peak_index, 'shoulder_hip_separation_deg'):.2f} deg "
        f"@ frame "
        f"{frame_from_index(separation_peak_index)}"
    )

    print()

    print("Detected temporal phases:")

    for phase in [
        "SETUP",
        "LOAD",
        "SWING",
        "PEAK_SWING",
        "FOLLOW_THROUGH"
    ]:

        start_key = (
            f"{phase.lower()}_start_frame"
        )

        end_key = (
            f"{phase.lower()}_end_frame"
        )

        print(
            f"  {phase:<18}: "
            f"{summary[start_key]} "
            f"-> "
            f"{summary[end_key]}"
        )

    print()
    print(
        "IMPORTANT: PEAK_SWING is a motion candidate, "
        "not confirmed bat-ball contact."
    )

    print()

    print(
        f"Frame analysis CSV : "
        f"{output_frame_csv}"
    )

    print(
        f"Summary CSV        : "
        f"{output_summary_csv}"
    )

    print("=" * 76)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()