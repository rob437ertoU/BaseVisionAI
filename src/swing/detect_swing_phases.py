from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "swing_analysis_v1_3.csv"
)

OUTPUT_PHASES_CSV = (
    PROJECT_ROOT
    / "results"
    / "swing_phases_v1.csv"
)

OUTPUT_SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "swing_phases_summary_v1.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

FPS = 30.0

MIN_POSE_QUALITY = 0.65

SMOOTHING_WINDOW = 5

# Approximate activation thresholds after normalization
LOAD_THRESHOLD = 0.30
SWING_THRESHOLD = 0.25

# How many consecutive frames we want before accepting
# the beginning of a phase.
MIN_CONSECUTIVE_FRAMES = 2

# Peak swing window around detected peak
PEAK_WINDOW_BEFORE = 2
PEAK_WINDOW_AFTER = 2


# ============================================================
# HELPERS
# ============================================================

def rolling_median(series, window=5):
    """
    Smooth noisy signals without strongly amplifying spikes.
    """

    return (
        series
        .rolling(
            window=window,
            center=True,
            min_periods=1
        )
        .median()
    )


def robust_normalize(series):
    """
    Normalize a signal approximately to 0-1 using its
    95th percentile rather than absolute maximum.

    This reduces sensitivity to one extreme frame.
    """

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    values = values.clip(lower=0)

    valid = values.dropna()

    if valid.empty:
        return pd.Series(
            0.0,
            index=series.index
        )

    scale = np.nanpercentile(
        valid,
        95
    )

    if (
        not np.isfinite(scale)
        or scale <= 1e-6
    ):
        return pd.Series(
            0.0,
            index=series.index
        )

    normalized = values / scale

    return normalized.clip(
        lower=0.0,
        upper=1.0
    )


def find_first_sustained(
    df,
    column,
    threshold,
    start_frame=None,
    end_frame=None,
    consecutive=2
):
    """
    Find the first frame where the selected signal remains
    above threshold for N consecutive frames.
    """

    work = df.copy()

    if start_frame is not None:
        work = work[
            work["frame"] >= start_frame
        ]

    if end_frame is not None:
        work = work[
            work["frame"] <= end_frame
        ]

    if work.empty:
        return None

    frames = work["frame"].to_numpy()
    values = work[column].to_numpy()

    for i in range(
        0,
        len(work) - consecutive + 1
    ):

        window = values[
            i:i + consecutive
        ]

        if np.all(
            np.isfinite(window)
            &
            (window >= threshold)
        ):
            return int(
                frames[i]
            )

    return None


def safe_frame_of_max(
    df,
    column,
    start_frame=None,
    end_frame=None
):
    """
    Return the actual video frame associated with the
    maximum value of a signal.
    """

    work = df.copy()

    if start_frame is not None:
        work = work[
            work["frame"] >= start_frame
        ]

    if end_frame is not None:
        work = work[
            work["frame"] <= end_frame
        ]

    work = work.dropna(
        subset=[column]
    )

    if work.empty:
        return None

    idx = work[column].idxmax()

    return int(
        work.loc[
            idx,
            "frame"
        ]
    )


def backtrack_phase_start(
    df,
    peak_frame,
    column,
    threshold
):
    """
    Starting at a peak, move backward until the activity
    falls below a lower threshold.

    This gives us an approximate beginning of LOAD rather
    than labeling only the peak frame itself.
    """

    before = df[
        df["frame"] <= peak_frame
    ].copy()

    if before.empty:
        return peak_frame

    before = before.sort_values(
        "frame",
        ascending=False
    )

    candidate = peak_frame

    for _, row in before.iterrows():

        value = row[column]

        if not np.isfinite(value):
            continue

        if value >= threshold:
            candidate = int(
                row["frame"]
            )
        else:
            break

    return candidate


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("BaseVisionAI - Swing Phase Detection V1")
    print("=" * 80)

    print(
        f"\nInput:\n{INPUT_CSV}"
    )

    # ========================================================
    # LOAD
    # ========================================================

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{INPUT_CSV}\n\n"
            "Run analyze_swing.py first."
        )

    df = pd.read_csv(
        INPUT_CSV
    )

    df = (
        df
        .sort_values("frame")
        .reset_index(drop=True)
    )

    print(
        f"\nFrames loaded: {len(df)}"
    )

    # ========================================================
    # VERIFY REQUIRED COLUMNS
    # ========================================================

    required_columns = [
        "frame",
        "swing_pose_quality",
        "left_wrist_relative_speed_px_frame",
        "right_wrist_relative_speed_px_frame",
        "hip_center_speed_px_frame",
        "shoulder_center_speed_px_frame",
        "ankle_change_from_baseline_px",
        "knee_change_from_baseline_px"
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    # ========================================================
    # POSE VALIDITY
    # ========================================================

    df["pose_valid"] = (
        df["swing_pose_quality"]
        >= MIN_POSE_QUALITY
    )

    # ========================================================
    # SMOOTH RAW MOTION SIGNALS
    # ========================================================

    df[
        "left_wrist_speed_smooth"
    ] = rolling_median(
        df[
            "left_wrist_relative_speed_px_frame"
        ],
        SMOOTHING_WINDOW
    )

    df[
        "right_wrist_speed_smooth"
    ] = rolling_median(
        df[
            "right_wrist_relative_speed_px_frame"
        ],
        SMOOTHING_WINDOW
    )

    df[
        "hip_speed_smooth"
    ] = rolling_median(
        df[
            "hip_center_speed_px_frame"
        ],
        SMOOTHING_WINDOW
    )

    df[
        "shoulder_speed_smooth"
    ] = rolling_median(
        df[
            "shoulder_center_speed_px_frame"
        ],
        SMOOTHING_WINDOW
    )

    # ========================================================
    # LEG ACTIVITY
    # ========================================================

    ankle_activity = (
        df[
            "ankle_change_from_baseline_px"
        ]
        .abs()
    )

    knee_activity = (
        df[
            "knee_change_from_baseline_px"
        ]
        .abs()
    )

    ankle_activity = rolling_median(
        ankle_activity,
        SMOOTHING_WINDOW
    )

    knee_activity = rolling_median(
        knee_activity,
        SMOOTHING_WINDOW
    )

    # Normalize each leg signal independently first
    ankle_norm = robust_normalize(
        ankle_activity
    )

    knee_norm = robust_normalize(
        knee_activity
    )

    # Ankle slightly more important for leg-kick detection
    df["leg_activity_score"] = (
        0.60 * ankle_norm
        +
        0.40 * knee_norm
    )

    # ========================================================
    # NORMALIZE MOTION SIGNALS
    # ========================================================

    df["left_wrist_score"] = (
        robust_normalize(
            df[
                "left_wrist_speed_smooth"
            ]
        )
    )

    df["right_wrist_score"] = (
        robust_normalize(
            df[
                "right_wrist_speed_smooth"
            ]
        )
    )

    df["hip_motion_score"] = (
        robust_normalize(
            df[
                "hip_speed_smooth"
            ]
        )
    )

    df["shoulder_motion_score"] = (
        robust_normalize(
            df[
                "shoulder_speed_smooth"
            ]
        )
    )

    # ========================================================
    # SWING ACTIVITY SCORE
    # ========================================================

    # Left wrist receives the highest weight because it had
    # much better raw keypoint reliability in our current clip.
    #
    # Right wrist remains useful, but secondary.
    #
    # Hip and shoulder motion provide supporting context.

    df["swing_activity_score"] = (
        0.50
        *
        df["left_wrist_score"]
        +
        0.15
        *
        df["right_wrist_score"]
        +
        0.20
        *
        df["hip_motion_score"]
        +
        0.15
        *
        df["shoulder_motion_score"]
    )

    # ========================================================
    # PEAK SWING SCORE
    # ========================================================

    # Peak detection emphasizes hand speed even more.
    df["peak_swing_score"] = (
        0.60
        *
        df["left_wrist_score"]
        +
        0.15
        *
        df["right_wrist_score"]
        +
        0.15
        *
        df["hip_motion_score"]
        +
        0.10
        *
        df["shoulder_motion_score"]
    )

    # Invalid pose frames should not create phase events.
    df.loc[
        ~df["pose_valid"],
        [
            "swing_activity_score",
            "peak_swing_score"
        ]
    ] = np.nan

    # ========================================================
    # DETECT PEAK SWING
    # ========================================================

    peak_swing_frame = safe_frame_of_max(
        df,
        "peak_swing_score"
    )

    if peak_swing_frame is None:

        raise ValueError(
            "Could not detect a peak swing frame."
        )

    # ========================================================
    # DETECT LEG-KICK PEAK
    # ========================================================

    leg_peak_frame = safe_frame_of_max(
        df,
        "leg_activity_score",
        end_frame=peak_swing_frame
    )

    if leg_peak_frame is None:

        leg_peak_frame = max(
            1,
            peak_swing_frame - 20
        )

    # ========================================================
    # ESTIMATE LOAD START
    # ========================================================

    load_start = backtrack_phase_start(
        df,
        leg_peak_frame,
        "leg_activity_score",
        threshold=LOAD_THRESHOLD
    )

    # Prevent baseline noise from starting the phase too early
    minimum_reasonable_load_frame = max(
        int(df["frame"].min()) + 5,
        1
    )

    load_start = max(
        load_start,
        minimum_reasonable_load_frame
    )

    # ========================================================
    # DETECT SWING START
    # ========================================================

    swing_start = find_first_sustained(
        df,
        "swing_activity_score",
        threshold=SWING_THRESHOLD,
        start_frame=load_start,
        end_frame=peak_swing_frame,
        consecutive=MIN_CONSECUTIVE_FRAMES
    )

    if swing_start is None:

        swing_start = max(
            load_start + 1,
            peak_swing_frame - 10
        )

    # Swing must logically occur after load.
    if swing_start <= load_start:

        swing_start = load_start + 1

    # ========================================================
    # PEAK WINDOW
    # ========================================================

    peak_start = max(
        swing_start,
        peak_swing_frame
        -
        PEAK_WINDOW_BEFORE
    )

    peak_end = min(
        int(df["frame"].max()),
        peak_swing_frame
        +
        PEAK_WINDOW_AFTER
    )

    # ========================================================
    # ASSIGN PHASES
    # ========================================================

    phases = []

    for _, row in df.iterrows():

        frame = int(
            row["frame"]
        )

        if frame < load_start:

            phase = "SETUP"

        elif frame < swing_start:

            phase = "LOAD"

        elif frame < peak_start:

            phase = "SWING"

        elif frame <= peak_end:

            phase = "PEAK_SWING"

        else:

            phase = "FOLLOW_THROUGH"

        phases.append(
            phase
        )

    df["swing_phase"] = phases

    # ========================================================
    # CONTACT CANDIDATE
    # ========================================================

    # IMPORTANT:
    #
    # This is NOT confirmed contact.
    #
    # It only identifies the region with the strongest swing
    # activity. Ball trajectory analysis will later be fused
    # with this information.

    df[
        "contact_candidate"
    ] = (
        (
            df["frame"]
            >= peak_start
        )
        &
        (
            df["frame"]
            <= peak_end
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    first_frame = int(
        df["frame"].min()
    )

    last_frame = int(
        df["frame"].max()
    )

    setup_end = load_start - 1
    load_end = swing_start - 1
    swing_end = peak_start - 1
    follow_start = peak_end + 1

    summary = {
        "first_frame":
            first_frame,

        "last_frame":
            last_frame,

        "setup_start_frame":
            first_frame,

        "setup_end_frame":
            setup_end,

        "load_start_frame":
            load_start,

        "load_peak_frame":
            leg_peak_frame,

        "load_end_frame":
            load_end,

        "swing_start_frame":
            swing_start,

        "swing_end_frame":
            swing_end,

        "peak_swing_start_frame":
            peak_start,

        "peak_swing_frame":
            peak_swing_frame,

        "peak_swing_end_frame":
            peak_end,

        "follow_through_start_frame":
            follow_start,

        "follow_through_end_frame":
            last_frame,

        "contact_candidate_start_frame":
            peak_start,

        "contact_candidate_center_frame":
            peak_swing_frame,

        "contact_candidate_end_frame":
            peak_end,

        "peak_swing_time_seconds":
            (
                peak_swing_frame - 1
            )
            /
            FPS,

        "load_peak_time_seconds":
            (
                leg_peak_frame - 1
            )
            /
            FPS
    }

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_PHASES_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PHASES_CSV,
        index=False
    )

    pd.DataFrame(
        [summary]
    ).to_csv(
        OUTPUT_SUMMARY_CSV,
        index=False
    )

    # ========================================================
    # TERMINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 80)
    print("SWING PHASE SUMMARY V1")
    print("=" * 80)

    print("\nSETUP")
    print(
        f"Frames: "
        f"{first_frame} -> {setup_end}"
    )

    print("\nLOAD")
    print(
        f"Frames: "
        f"{load_start} -> {load_end}"
    )
    print(
        f"Leg activity peak: "
        f"frame {leg_peak_frame}"
    )

    print("\nSWING")
    print(
        f"Frames: "
        f"{swing_start} -> {swing_end}"
    )

    print("\nPEAK SWING")
    print(
        f"Frames: "
        f"{peak_start} -> {peak_end}"
    )
    print(
        f"Peak swing frame: "
        f"{peak_swing_frame}"
    )

    print("\nFOLLOW THROUGH")
    print(
        f"Frames: "
        f"{follow_start} -> {last_frame}"
    )

    print("\nCONTACT CANDIDATE")
    print(
        f"Frames: "
        f"{peak_start} -> {peak_end}"
    )
    print(
        "Status: CANDIDATE ONLY - "
        "not confirmed ball/bat contact"
    )

    print("\n" + "=" * 80)
    print("SWING PHASE DETECTION V1 COMPLETED")
    print("=" * 80)

    print(
        f"\nFrame analysis:\n"
        f"{OUTPUT_PHASES_CSV}"
    )

    print(
        f"\nSummary:\n"
        f"{OUTPUT_SUMMARY_CSV}"
    )


if __name__ == "__main__":
    main()