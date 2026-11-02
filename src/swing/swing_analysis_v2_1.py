from pathlib import Path
import argparse

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

# More smoothing for derived motion signals
MOTION_SMOOTHING_WINDOW = 5

# Frames around the selected peak
PEAK_WINDOW = 2

# Relative temporal thresholds
LOAD_THRESHOLD = 0.18
SWING_THRESHOLD = 0.35


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Swing Analysis V2.1"
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
            "Video FPS. If omitted, estimated "
            "from time_sec in the CSV."
        )
    )

    return parser.parse_args()


# ============================================================
# UTILITIES
# ============================================================

def safe_numeric(df, column):

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

    x = safe_numeric(
        df,
        f"{name}_x_smooth"
    )

    y = safe_numeric(
        df,
        f"{name}_y_smooth"
    )

    return x, y


def midpoint(
    x1,
    y1,
    x2,
    y2
):

    x = (
        x1 + x2
    ) / 2.0

    y = (
        y1 + y2
    ) / 2.0

    return x, y


def distance(
    x1,
    y1,
    x2,
    y2
):

    return np.sqrt(
        (x2 - x1) ** 2
        + (y2 - y1) ** 2
    )


def velocity(
    x,
    y
):

    dx = x.diff()
    dy = y.diff()

    speed = np.sqrt(
        dx ** 2
        + dy ** 2
    )

    return speed


def body_line_angle(
    x1,
    y1,
    x2,
    y2
):

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


def axial_angle_difference(
    angle_a,
    angle_b
):

    diff = np.abs(
        angle_a - angle_b
    )

    diff = np.minimum(
        diff,
        180.0 - diff
    )

    return diff


def rolling_median(
    series,
    window
):

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

    valid = (
        series
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    if len(valid) == 0:

        return pd.Series(
            0.0,
            index=series.index
        )

    max_value = float(
        valid.max()
    )

    if (
        not np.isfinite(max_value)
        or max_value <= 0
    ):

        return pd.Series(
            0.0,
            index=series.index
        )

    normalized = (
        series
        / max_value
    )

    return normalized.clip(
        lower=0.0,
        upper=1.0
    )


def valid_argmax(series):

    clean = (
        series
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    if len(clean) == 0:
        return None

    return int(
        clean.idxmax()
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
        time
        .diff()
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

    return (
        1.0
        / median_dt
    )


# ============================================================
# PHASE DETECTION
# ============================================================

def detect_phases(
    df,
    peak_index
):

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

    peak_value = float(
        energy[
            peak_index
        ]
    )

    if peak_value <= 0:
        return phases

    normalized = (
        energy
        / peak_value
    )

    # --------------------------------------------------------
    # Find start of swing
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
    # Find load start
    # --------------------------------------------------------

    load_search = np.arange(
        0,
        swing_start
    )

    if len(load_search) > 0:

        load_candidates = load_search[
            normalized[
                load_search
            ] >= LOAD_THRESHOLD
        ]

    else:

        load_candidates = np.array(
            [],
            dtype=int
        )

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
    # Peak region
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
    # Assign labels
    # --------------------------------------------------------

    phases[
        :load_start
    ] = "SETUP"

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

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

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
        / (
            f"{input_path.stem}"
            f"_swing_analysis_v2_1.csv"
        )
    )

    output_summary_csv = (
        output_dir
        / (
            f"{input_path.stem}"
            f"_swing_summary_v2_1.csv"
        )
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(
        input_path
    )

    if len(df) == 0:

        raise RuntimeError(
            "Input CSV contains no frames."
        )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    if args.fps is not None:
        fps = args.fps

    else:
        fps = estimate_fps(df)

    # --------------------------------------------------------
    # Print header
    # --------------------------------------------------------

    print()
    print("=" * 76)
    print("BaseVisionAI - Swing Analysis V2.1")
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
    # POSE QUALITY
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
    # KEYPOINTS
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

    df["shoulder_center_x"] = (
        shoulder_cx
    )

    df["shoulder_center_y"] = (
        shoulder_cy
    )

    df["hip_center_x"] = (
        hip_cx
    )

    df["hip_center_y"] = (
        hip_cy
    )

    # ========================================================
    # TORSO SCALE
    # ========================================================

    torso_length = distance(
        shoulder_cx,
        shoulder_cy,
        hip_cx,
        hip_cy
    )

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
    # WRIST MOTION
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

    df[
        "left_wrist_speed_px_frame"
    ] = left_wrist_speed

    df[
        "right_wrist_speed_px_frame"
    ] = right_wrist_speed

    df[
        "left_wrist_speed_px_sec"
    ] = (
        left_wrist_speed
        * fps
    )

    df[
        "right_wrist_speed_px_sec"
    ] = (
        right_wrist_speed
        * fps
    )

    df[
        "left_wrist_speed_norm"
    ] = (
        left_wrist_speed
        / torso_length
    )

    df[
        "right_wrist_speed_norm"
    ] = (
        right_wrist_speed
        / torso_length
    )

    # ========================================================
    # ELBOW MOTION
    # ========================================================

    left_elbow_speed = velocity(
        lex,
        ley
    )

    right_elbow_speed = velocity(
        rex,
        rey
    )

    left_elbow_speed = rolling_median(
        left_elbow_speed,
        MOTION_SMOOTHING_WINDOW
    )

    right_elbow_speed = rolling_median(
        right_elbow_speed,
        MOTION_SMOOTHING_WINDOW
    )

    df[
        "left_elbow_speed_px_frame"
    ] = left_elbow_speed

    df[
        "right_elbow_speed_px_frame"
    ] = right_elbow_speed

    df[
        "left_elbow_speed_norm"
    ] = (
        left_elbow_speed
        / torso_length
    )

    df[
        "right_elbow_speed_norm"
    ] = (
        right_elbow_speed
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

    df[
        "shoulder_center_speed_px_frame"
    ] = shoulder_speed

    df[
        "hip_center_speed_px_frame"
    ] = hip_speed

    df[
        "shoulder_center_speed_norm"
    ] = (
        shoulder_speed
        / torso_length
    )

    df[
        "hip_center_speed_norm"
    ] = (
        hip_speed
        / torso_length
    )

    # ========================================================
    # BODY AXES
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

    df[
        "shoulder_axis_deg"
    ] = shoulder_angle

    df[
        "hip_axis_deg"
    ] = hip_angle

    df[
        "shoulder_hip_separation_deg"
    ] = shoulder_hip_separation

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

    df[
        "left_knee_speed_norm"
    ] = (
        rolling_median(
            left_knee_speed,
            MOTION_SMOOTHING_WINDOW
        )
        / torso_length
    )

    df[
        "right_knee_speed_norm"
    ] = (
        rolling_median(
            right_knee_speed,
            MOTION_SMOOTHING_WINDOW
        )
        / torso_length
    )

    df[
        "left_ankle_speed_norm"
    ] = (
        rolling_median(
            left_ankle_speed,
            MOTION_SMOOTHING_WINDOW
        )
        / torso_length
    )

    df[
        "right_ankle_speed_norm"
    ] = (
        rolling_median(
            right_ankle_speed,
            MOTION_SMOOTHING_WINDOW
        )
        / torso_length
    )

    # ========================================================
    # ARM MOTION
    # ========================================================

    wrist_motion = pd.concat(
        [
            df[
                "left_wrist_speed_norm"
            ],
            df[
                "right_wrist_speed_norm"
            ]
        ],
        axis=1
    ).max(
        axis=1
    )

    elbow_motion = pd.concat(
        [
            df[
                "left_elbow_speed_norm"
            ],
            df[
                "right_elbow_speed_norm"
            ]
        ],
        axis=1
    ).max(
        axis=1
    )

    df[
        "wrist_motion_norm"
    ] = wrist_motion

    df[
        "elbow_motion_norm"
    ] = elbow_motion

    # ========================================================
    # SWING MOTION SCORE V2.1
    # ========================================================
    #
    # IMPORTANT:
    #
    # This is a temporal motion evidence score.
    #
    # It is NOT:
    # - physical energy
    # - bat speed
    # - power
    # - contact probability
    #
    # Hands dominate the score.
    # Elbows support the estimate.
    # Torso contributes only weakly.
    # ========================================================

    wrist_component = normalize_series(
        df[
            "wrist_motion_norm"
        ]
    )

    elbow_component = normalize_series(
        df[
            "elbow_motion_norm"
        ]
    )

    hip_component = normalize_series(
        df[
            "hip_center_speed_norm"
        ]
    )

    shoulder_component = normalize_series(
        df[
            "shoulder_center_speed_norm"
        ]
    )

    torso_component = (
        0.5 * hip_component
        + 0.5 * shoulder_component
    )

    df[
        "swing_motion_energy"
    ] = (
        0.70 * wrist_component
        + 0.20 * elbow_component
        + 0.10 * torso_component
    )

    # Stabilize derived score
    df[
        "swing_motion_energy"
    ] = rolling_median(
        df[
            "swing_motion_energy"
        ],
        5
    )

    # --------------------------------------------------------
    # Remove invalid pose frames
    # --------------------------------------------------------

    df.loc[
        df[
            "analysis_valid"
        ] == 0,
        "swing_motion_energy"
    ] = np.nan

    # ========================================================
    # PEAK SWING CANDIDATE V2.1
    # ========================================================

    wrist_peak_value = float(
        df[
            "wrist_motion_norm"
        ].max()
    )

    if (
        not np.isfinite(
            wrist_peak_value
        )
        or wrist_peak_value <= 0
    ):

        peak_index = None

    else:

        strong_hand_mask = (
            df[
                "wrist_motion_norm"
            ]
            >= (
                0.55
                * wrist_peak_value
            )
        )

        candidate_score = (
            df[
                "swing_motion_energy"
            ]
            .where(
                strong_hand_mask
            )
        )

        peak_index = valid_argmax(
            candidate_score
        )

    # ========================================================
    # PEAK FRAME / TIME
    # ========================================================

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

    df[
        "swing_phase"
    ] = phases

    df[
        "peak_swing_candidate"
    ] = 0

    if peak_index is not None:

        df.loc[
            peak_index,
            "peak_swing_candidate"
        ] = 1

    # ========================================================
    # INDIVIDUAL PEAKS
    # ========================================================

    left_wrist_peak_index = valid_argmax(
        df[
            "left_wrist_speed_norm"
        ]
    )

    right_wrist_peak_index = valid_argmax(
        df[
            "right_wrist_speed_norm"
        ]
    )

    left_elbow_peak_index = valid_argmax(
        df[
            "left_elbow_speed_norm"
        ]
    )

    right_elbow_peak_index = valid_argmax(
        df[
            "right_elbow_speed_norm"
        ]
    )

    hip_peak_index = valid_argmax(
        df[
            "hip_center_speed_norm"
        ]
    )

    shoulder_peak_index = valid_argmax(
        df[
            "shoulder_center_speed_norm"
        ]
    )

    separation_peak_index = valid_argmax(
        df[
            "shoulder_hip_separation_deg"
        ]
    )

    # ========================================================
    # SUMMARY HELPERS
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

        value = df.loc[
            index,
            column
        ]

        if pd.isna(value):
            return np.nan

        return float(
            value
        )

    # ========================================================
    # PHASE RANGES
    # ========================================================

    phase_ranges = {}

    phase_names = [
        "SETUP",
        "LOAD",
        "SWING",
        "PEAK_SWING",
        "FOLLOW_THROUGH"
    ]

    for phase in phase_names:

        selected = df[
            df[
                "swing_phase"
            ] == phase
        ]

        if len(selected) > 0:

            phase_ranges[
                f"{phase.lower()}_start_frame"
            ] = int(
                selected[
                    "frame"
                ].min()
            )

            phase_ranges[
                f"{phase.lower()}_end_frame"
            ] = int(
                selected[
                    "frame"
                ].max()
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

        "video_frames":
            len(df),

        "fps":
            fps,

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
            value_from_index(
                peak_index,
                "swing_motion_energy"
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

        "left_elbow_peak_frame":
            frame_from_index(
                left_elbow_peak_index
            ),

        "left_elbow_peak_norm":
            value_from_index(
                left_elbow_peak_index,
                "left_elbow_speed_norm"
            ),

        "right_elbow_peak_frame":
            frame_from_index(
                right_elbow_peak_index
            ),

        "right_elbow_peak_norm":
            value_from_index(
                right_elbow_peak_index,
                "right_elbow_speed_norm"
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
    # SAVE RESULTS
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
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 76)
    print("SWING ANALYSIS V2.1 SUMMARY")
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
        f"Left elbow peak             : "
        f"frame "
        f"{frame_from_index(left_elbow_peak_index)}"
    )

    print(
        f"Right elbow peak            : "
        f"frame "
        f"{frame_from_index(right_elbow_peak_index)}"
    )

    print()

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

    separation_value = (
        value_from_index(
            separation_peak_index,
            "shoulder_hip_separation_deg"
        )
    )

    separation_frame = (
        frame_from_index(
            separation_peak_index
        )
    )

    print(
        f"Max shoulder/hip separation : "
        f"{separation_value:.2f} deg "
        f"@ frame "
        f"{separation_frame}"
    )

    print()
    print(
        "Detected temporal phases:"
    )

    for phase in phase_names:

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

    print(
        "2D normalized motion metrics are relative image-space "
        "features, not calibrated biomechanics."
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