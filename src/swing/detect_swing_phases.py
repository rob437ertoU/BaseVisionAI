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
    / "swing_phases_v1_1.csv"
)

OUTPUT_SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "swing_phases_summary_v1_1.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

FPS = 30.0

MIN_POSE_QUALITY = 0.65

SMOOTHING_WINDOW = 5

# Number of frames before leg peak allowed for LOAD search
MAX_LOAD_LOOKBACK = 18

# Number of frames before peak swing allowed for SWING search
MAX_SWING_LOOKBACK = 30

# Activation thresholds
LOAD_ACTIVITY_THRESHOLD = 0.45
SWING_ACTIVITY_THRESHOLD = 0.30

# Peak swing visualization window
PEAK_WINDOW_BEFORE = 2
PEAK_WINDOW_AFTER = 2


# ============================================================
# HELPERS
# ============================================================

def rolling_median(series, window=5):

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

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    values = values.clip(
        lower=0
    )

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

    normalized = (
        values
        /
        scale
    )

    return normalized.clip(
        0.0,
        1.0
    )


def frame_of_max(
    df,
    column,
    start_frame=None,
    end_frame=None
):

    work = df.copy()

    if start_frame is not None:

        work = work[
            work["frame"]
            >= start_frame
        ]

    if end_frame is not None:

        work = work[
            work["frame"]
            <= end_frame
        ]

    work = work.dropna(
        subset=[column]
    )

    if work.empty:
        return None

    idx = work[
        column
    ].idxmax()

    return int(
        work.loc[
            idx,
            "frame"
        ]
    )


def find_onset_before_peak(
    df,
    column,
    peak_frame,
    threshold,
    max_lookback
):
    """
    Search backward from a known peak.

    We find the earliest meaningful activation close to that
    peak instead of allowing weak activity much earlier in
    the clip to start the phase.
    """

    search_start = max(
        int(df["frame"].min()),
        peak_frame - max_lookback
    )

    work = df[
        (
            df["frame"]
            >= search_start
        )
        &
        (
            df["frame"]
            <= peak_frame
        )
    ].copy()

    if work.empty:
        return peak_frame

    active = (
        work[column]
        >= threshold
    )

    active_frames = work.loc[
        active,
        "frame"
    ]

    if active_frames.empty:

        return search_start

    first_active = int(
        active_frames.iloc[0]
    )

    return first_active


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("BaseVisionAI - Swing Phase Detection V1.1")
    print("=" * 80)

    print(
        f"\nInput:\n{INPUT_CSV}"
    )

    # ========================================================
    # LOAD CSV
    # ========================================================

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"File not found:\n"
            f"{INPUT_CSV}"
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
    # VALIDATE COLUMNS
    # ========================================================

    required = [
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
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns:\n"
            +
            "\n".join(missing)
        )

    # ========================================================
    # VALID POSE
    # ========================================================

    df["pose_valid"] = (
        df["swing_pose_quality"]
        >= MIN_POSE_QUALITY
    )

    # ========================================================
    # SMOOTH MOTION SIGNALS
    # ========================================================

    df["left_wrist_smooth"] = (
        rolling_median(
            df[
                "left_wrist_relative_speed_px_frame"
            ],
            SMOOTHING_WINDOW
        )
    )

    df["right_wrist_smooth"] = (
        rolling_median(
            df[
                "right_wrist_relative_speed_px_frame"
            ],
            SMOOTHING_WINDOW
        )
    )

    df["hip_smooth"] = (
        rolling_median(
            df[
                "hip_center_speed_px_frame"
            ],
            SMOOTHING_WINDOW
        )
    )

    df["shoulder_smooth"] = (
        rolling_median(
            df[
                "shoulder_center_speed_px_frame"
            ],
            SMOOTHING_WINDOW
        )
    )

    ankle_activity = rolling_median(
        df[
            "ankle_change_from_baseline_px"
        ].abs(),
        SMOOTHING_WINDOW
    )

    knee_activity = rolling_median(
        df[
            "knee_change_from_baseline_px"
        ].abs(),
        SMOOTHING_WINDOW
    )

    # ========================================================
    # NORMALIZED SCORES
    # ========================================================

    df["left_wrist_score"] = (
        robust_normalize(
            df[
                "left_wrist_smooth"
            ]
        )
    )

    df["right_wrist_score"] = (
        robust_normalize(
            df[
                "right_wrist_smooth"
            ]
        )
    )

    df["hip_score"] = (
        robust_normalize(
            df[
                "hip_smooth"
            ]
        )
    )

    df["shoulder_score"] = (
        robust_normalize(
            df[
                "shoulder_smooth"
            ]
        )
    )

    ankle_score = robust_normalize(
        ankle_activity
    )

    knee_score = robust_normalize(
        knee_activity
    )

    # ========================================================
    # LEG ACTIVITY
    # ========================================================

    df["leg_activity_score"] = (
        0.60
        *
        ankle_score
        +
        0.40
        *
        knee_score
    )

    # ========================================================
    # SWING ACTIVITY
    # ========================================================

    df["swing_activity_score"] = (
        0.55
        *
        df["left_wrist_score"]
        +
        0.15
        *
        df["right_wrist_score"]
        +
        0.20
        *
        df["hip_score"]
        +
        0.10
        *
        df["shoulder_score"]
    )

    # ========================================================
    # PEAK SCORE
    # ========================================================

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
        df["hip_score"]
        +
        0.10
        *
        df["shoulder_score"]
    )

    # Invalid pose should not trigger events
    df.loc[
        ~df["pose_valid"],
        [
            "leg_activity_score",
            "swing_activity_score",
            "peak_swing_score"
        ]
    ] = np.nan

    # ========================================================
    # PEAK SWING
    # ========================================================

    peak_swing_frame = frame_of_max(
        df,
        "peak_swing_score"
    )

    if peak_swing_frame is None:

        raise ValueError(
            "Peak swing could not be detected."
        )

    # ========================================================
    # LEG PEAK
    # ========================================================

    leg_peak_frame = frame_of_max(
        df,
        "leg_activity_score",
        end_frame=peak_swing_frame
    )

    if leg_peak_frame is None:

        raise ValueError(
            "Leg activity peak could not be detected."
        )

    # ========================================================
    # LOAD START
    # ========================================================

    load_start = find_onset_before_peak(
        df,
        "leg_activity_score",
        peak_frame=leg_peak_frame,
        threshold=LOAD_ACTIVITY_THRESHOLD,
        max_lookback=MAX_LOAD_LOOKBACK
    )

    # ========================================================
    # SWING START
    # ========================================================

    swing_search_start = max(
        leg_peak_frame - 3,
        peak_swing_frame - MAX_SWING_LOOKBACK
    )

    swing_window = df[
        (
            df["frame"]
            >= swing_search_start
        )
        &
        (
            df["frame"]
            <= peak_swing_frame
        )
    ].copy()

    active_swing = swing_window[
        swing_window[
            "swing_activity_score"
        ]
        >=
        SWING_ACTIVITY_THRESHOLD
    ]

    if active_swing.empty:

        swing_start = max(
            leg_peak_frame,
            peak_swing_frame - 12
        )

    else:

        swing_start = int(
            active_swing[
                "frame"
            ].iloc[0]
        )

    # Swing should not begin before LOAD begins
    swing_start = max(
        swing_start,
        load_start + 1
    )

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
    # PHASE ASSIGNMENT
    # ========================================================

    phases = []

    for frame in df["frame"]:

        frame = int(frame)

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

    df["contact_candidate"] = (
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

    setup_end = (
        load_start - 1
    )

    load_end = (
        swing_start - 1
    )

    swing_end = (
        peak_start - 1
    )

    follow_start = (
        peak_end + 1
    )

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

        "load_peak_time_seconds":
            (
                leg_peak_frame - 1
            )
            /
            FPS,

        "peak_swing_time_seconds":
            (
                peak_swing_frame - 1
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
    # TERMINAL
    # ========================================================

    print(
        "\n"
        +
        "=" * 80
    )

    print(
        "SWING PHASE SUMMARY V1.1"
    )

    print(
        "=" * 80
    )

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

    print(
        "\n"
        +
        "=" * 80
    )

    print(
        "SWING PHASE DETECTION V1.1 COMPLETED"
    )

    print(
        "=" * 80
    )

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