from pathlib import Path
import argparse
import math

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SWING_CSV = (
    PROJECT_ROOT
    / "results"
    / "swing_analysis_v2"
    / "baseball_train10_batter_pose_v2_smoothed_swing_analysis_v2_2.csv"
)

DEFAULT_BALL_CSV = (
    PROJECT_ROOT
    / "results"
    / "ball_tracking_v3_1"
    / "baseball_train10"
    / "ball_tracking_v3_1.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "contact_fusion_v2"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Temporal relationship between ball events and swing peak
MAX_TEMPORAL_GAP_FRAMES = 6

# Minimum YOLO-observed points required to estimate a
# meaningful trajectory segment
MIN_TRAJECTORY_POINTS = 3

# Local points before/after candidate
TRAJECTORY_WINDOW = 5

# Angle thresholds for evidence only
MODERATE_DIRECTION_CHANGE_DEG = 25.0
STRONG_DIRECTION_CHANGE_DEG = 45.0

# Evidence threshold.
# This is NOT a probability threshold.
CONTACT_CANDIDATE_THRESHOLD = 0.60

# If there is too little ball evidence, we do not force
# a contact conclusion even when temporal evidence is strong.
MIN_BALL_EVIDENCE_FOR_CONTACT = 0.35


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Contact Fusion V2"
    )

    parser.add_argument(
        "--swing-csv",
        type=str,
        default=str(DEFAULT_SWING_CSV),
        help="Swing Analysis V2.2 per-frame CSV"
    )

    parser.add_argument(
        "--ball-csv",
        type=str,
        default=str(DEFAULT_BALL_CSV),
        help="Ball Tracking V3.1 CSV"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for Contact Fusion V2 results"
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video FPS"
    )

    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=np.nan):

    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


def normalize_bool(value):

    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    value = str(value).strip().lower()

    return value in {
        "true",
        "1",
        "yes",
        "y"
    }


def clamp01(value):

    return max(
        0.0,
        min(
            1.0,
            float(value)
        )
    )


def angle_difference_deg(
    angle_a,
    angle_b
):

    diff = abs(
        angle_a - angle_b
    )

    while diff > 180.0:
        diff = abs(
            diff - 360.0
        )

    return diff


# ============================================================
# TRAJECTORY
# ============================================================

def calculate_segment_motion(points):
    """
    points:
        DataFrame containing:
        frame, observed_x, observed_y

    Returns:
        direction angle
        average px/frame
        dx
        dy

    Uses only actual YOLO observations.
    """

    if len(points) < 2:

        return {
            "valid": False,
            "angle_deg": np.nan,
            "avg_speed_px_frame": np.nan,
            "dx": np.nan,
            "dy": np.nan,
        }

    points = (
        points
        .sort_values("frame")
        .copy()
    )

    first = points.iloc[0]
    last = points.iloc[-1]

    x1 = safe_float(
        first["observed_x"]
    )

    y1 = safe_float(
        first["observed_y"]
    )

    x2 = safe_float(
        last["observed_x"]
    )

    y2 = safe_float(
        last["observed_y"]
    )

    f1 = int(
        first["frame"]
    )

    f2 = int(
        last["frame"]
    )

    frame_delta = (
        f2 - f1
    )

    if (
        not np.isfinite(x1)
        or not np.isfinite(y1)
        or not np.isfinite(x2)
        or not np.isfinite(y2)
        or frame_delta <= 0
    ):

        return {
            "valid": False,
            "angle_deg": np.nan,
            "avg_speed_px_frame": np.nan,
            "dx": np.nan,
            "dy": np.nan,
        }

    dx = (
        x2 - x1
    )

    dy = (
        y2 - y1
    )

    angle_deg = math.degrees(
        math.atan2(
            dy,
            dx
        )
    )

    distance = math.sqrt(
        dx ** 2
        + dy ** 2
    )

    avg_speed = (
        distance
        / frame_delta
    )

    return {
        "valid": True,
        "angle_deg": angle_deg,
        "avg_speed_px_frame": avg_speed,
        "dx": dx,
        "dy": dy,
    }


# ============================================================
# SWING INFORMATION
# ============================================================

def get_swing_info(swing_df):

    if "frame" not in swing_df.columns:
        raise RuntimeError(
            "Swing CSV is missing 'frame'."
        )

    if "swing_phase" not in swing_df.columns:
        raise RuntimeError(
            "Swing CSV is missing 'swing_phase'."
        )

    peak_rows = swing_df[
        swing_df.get(
            "peak_swing_candidate",
            0
        ) == 1
    ]

    if len(peak_rows) == 0:
        raise RuntimeError(
            "No peak_swing_candidate found in Swing CSV."
        )

    peak_frame = int(
        peak_rows.iloc[0]["frame"]
    )

    swing_phase_rows = swing_df[
        swing_df["swing_phase"].isin(
            [
                "SWING",
                "PEAK_SWING"
            ]
        )
    ]

    peak_phase_rows = swing_df[
        swing_df["swing_phase"]
        == "PEAK_SWING"
    ]

    if len(swing_phase_rows) > 0:

        swing_window_start = int(
            swing_phase_rows[
                "frame"
            ].min()
        )

        swing_window_end = int(
            swing_phase_rows[
                "frame"
            ].max()
        )

    else:

        swing_window_start = peak_frame
        swing_window_end = peak_frame

    if len(peak_phase_rows) > 0:

        peak_window_start = int(
            peak_phase_rows[
                "frame"
            ].min()
        )

        peak_window_end = int(
            peak_phase_rows[
                "frame"
            ].max()
        )

    else:

        peak_window_start = peak_frame
        peak_window_end = peak_frame

    return {
        "peak_frame":
            peak_frame,

        "swing_window_start":
            swing_window_start,

        "swing_window_end":
            swing_window_end,

        "peak_window_start":
            peak_window_start,

        "peak_window_end":
            peak_window_end,
    }


# ============================================================
# BALL INFORMATION
# ============================================================

def prepare_ball_df(ball_df):

    required = [
        "frame",
        "track_id",
        "state",
        "yolo_detected",
        "observed_x",
        "observed_y",
        "filtered_x",
        "filtered_y",
        "missing_frames",
    ]

    missing = [
        column
        for column in required
        if column not in ball_df.columns
    ]

    if missing:

        raise RuntimeError(
            "Ball CSV is missing columns: "
            + ", ".join(
                missing
            )
        )

    ball_df = ball_df.copy()

    ball_df[
        "yolo_detected"
    ] = (
        ball_df[
            "yolo_detected"
        ]
        .apply(
            normalize_bool
        )
    )

    for column in [
        "observed_x",
        "observed_y",
        "filtered_x",
        "filtered_y",
        "confidence",
        "distance_to_prediction",
        "candidate_score",
    ]:

        if column in ball_df.columns:

            ball_df[column] = pd.to_numeric(
                ball_df[column],
                errors="coerce"
            )

    ball_df[
        "frame"
    ] = pd.to_numeric(
        ball_df["frame"],
        errors="coerce"
    )

    ball_df = ball_df.dropna(
        subset=[
            "frame"
        ]
    )

    ball_df[
        "frame"
    ] = (
        ball_df[
            "frame"
        ]
        .astype(int)
    )

    return ball_df


# ============================================================
# CONTACT EVIDENCE
# ============================================================

def analyze_contact(
    swing_df,
    ball_df,
    fps
):

    swing = get_swing_info(
        swing_df
    )

    ball_df = prepare_ball_df(
        ball_df
    )

    peak_frame = (
        swing[
            "peak_frame"
        ]
    )

    swing_start = (
        swing[
            "swing_window_start"
        ]
    )

    swing_end = (
        swing[
            "swing_window_end"
        ]
    )

    peak_start = (
        swing[
            "peak_window_start"
        ]
    )

    peak_end = (
        swing[
            "peak_window_end"
        ]
    )

    # ========================================================
    # YOLO OBSERVATIONS
    # ========================================================

    yolo_points = ball_df[
        (
            ball_df[
                "yolo_detected"
            ]
        )
        &
        (
            ball_df[
                "observed_x"
            ].notna()
        )
        &
        (
            ball_df[
                "observed_y"
            ].notna()
        )
    ].copy()

    if len(yolo_points) > 0:

        first_yolo_frame = int(
            yolo_points[
                "frame"
            ].min()
        )

        last_yolo_frame = int(
            yolo_points[
                "frame"
            ].max()
        )

    else:

        first_yolo_frame = None
        last_yolo_frame = None

    # ========================================================
    # NEAREST YOLO OBSERVATION TO PEAK
    # ========================================================

    if len(yolo_points) > 0:

        temp = yolo_points.copy()

        temp[
            "peak_distance"
        ] = (
            temp[
                "frame"
            ]
            - peak_frame
        ).abs()

        nearest_row = (
            temp
            .sort_values(
                [
                    "peak_distance",
                    "frame"
                ]
            )
            .iloc[0]
        )

        nearest_yolo_frame = int(
            nearest_row[
                "frame"
            ]
        )

        nearest_yolo_gap = int(
            abs(
                nearest_yolo_frame
                - peak_frame
            )
        )

    else:

        nearest_yolo_frame = None
        nearest_yolo_gap = None

    # ========================================================
    # BALL STATES AROUND SWING
    # ========================================================

    relevant_ball = ball_df[
        (
            ball_df["frame"]
            >= swing_start
        )
        &
        (
            ball_df["frame"]
            <= (
                swing_end
                + MAX_TEMPORAL_GAP_FRAMES
            )
        )
    ].copy()

    state_counts = (
        relevant_ball[
            "state"
        ]
        .value_counts()
        .to_dict()
    )

    tracking_frames = int(
        state_counts.get(
            "TRACKING",
            0
        )
    )

    kalman_frames = int(
        state_counts.get(
            "KALMAN",
            0
        )
    )

    lost_frames = int(
        state_counts.get(
            "LOST",
            0
        )
    )

    # ========================================================
    # TRACK LOSS AFTER LAST YOLO
    # ========================================================

    track_loss_after_detection = False
    first_lost_frame = None

    if last_yolo_frame is not None:

        after_last_yolo = ball_df[
            ball_df[
                "frame"
            ] > last_yolo_frame
        ]

        lost_rows = after_last_yolo[
            after_last_yolo[
                "state"
            ] == "LOST"
        ]

        if len(lost_rows) > 0:

            first_lost_frame = int(
                lost_rows[
                    "frame"
                ].min()
            )

            if (
                first_lost_frame
                - last_yolo_frame
                <= MAX_TEMPORAL_GAP_FRAMES
            ):

                track_loss_after_detection = True

    # ========================================================
    # TEMPORAL EVIDENCE
    # ========================================================

    if nearest_yolo_gap is None:

        temporal_evidence = 0.0

    else:

        temporal_evidence = clamp01(
            1.0
            - (
                nearest_yolo_gap
                / MAX_TEMPORAL_GAP_FRAMES
            )
        )

    # Extra boost if observed ball overlaps PEAK_SWING
    yolo_in_peak = yolo_points[
        (
            yolo_points["frame"]
            >= peak_start
        )
        &
        (
            yolo_points["frame"]
            <= peak_end
        )
    ]

    if len(yolo_in_peak) > 0:

        temporal_evidence = max(
            temporal_evidence,
            0.90
        )

    # ========================================================
    # TRACK LOSS EVIDENCE
    # ========================================================

    if (
        last_yolo_frame is not None
        and first_lost_frame is not None
    ):

        loss_gap = (
            first_lost_frame
            - last_yolo_frame
        )

        if 0 <= loss_gap <= 2:

            loss_evidence = 1.0

        elif loss_gap <= 4:

            loss_evidence = 0.75

        elif loss_gap <= 6:

            loss_evidence = 0.50

        else:

            loss_evidence = 0.0

    else:

        loss_gap = None
        loss_evidence = 0.0

    # ========================================================
    # TRAJECTORY BEFORE / AFTER PEAK
    # ========================================================

    pre_points = yolo_points[
        (
            yolo_points[
                "frame"
            ] < peak_frame
        )
        &
        (
            yolo_points[
                "frame"
            ] >= (
                peak_frame
                - TRAJECTORY_WINDOW
            )
        )
    ].copy()

    post_points = yolo_points[
        (
            yolo_points[
                "frame"
            ] > peak_frame
        )
        &
        (
            yolo_points[
                "frame"
            ] <= (
                peak_frame
                + TRAJECTORY_WINDOW
            )
        )
    ].copy()

    pre_motion = (
        calculate_segment_motion(
            pre_points
        )
    )

    post_motion = (
        calculate_segment_motion(
            post_points
        )
    )

    trajectory_available = (
        len(pre_points)
        >= MIN_TRAJECTORY_POINTS
        and len(post_points)
        >= MIN_TRAJECTORY_POINTS
        and pre_motion[
            "valid"
        ]
        and post_motion[
            "valid"
        ]
    )

    if trajectory_available:

        direction_change_deg = (
            angle_difference_deg(
                pre_motion[
                    "angle_deg"
                ],
                post_motion[
                    "angle_deg"
                ]
            )
        )

        pre_speed = (
            pre_motion[
                "avg_speed_px_frame"
            ]
        )

        post_speed = (
            post_motion[
                "avg_speed_px_frame"
            ]
        )

        if (
            pre_speed > 0
            and np.isfinite(
                post_speed
            )
        ):

            speed_ratio = (
                post_speed
                / pre_speed
            )

        else:

            speed_ratio = np.nan

        if (
            direction_change_deg
            >= STRONG_DIRECTION_CHANGE_DEG
        ):

            trajectory_evidence = 1.0

        elif (
            direction_change_deg
            >= MODERATE_DIRECTION_CHANGE_DEG
        ):

            trajectory_evidence = 0.65

        else:

            trajectory_evidence = (
                direction_change_deg
                / MODERATE_DIRECTION_CHANGE_DEG
                * 0.40
            )

    else:

        direction_change_deg = np.nan
        speed_ratio = np.nan
        trajectory_evidence = 0.0

    # ========================================================
    # BALL EVIDENCE QUALITY
    # ========================================================

    yolo_in_swing = yolo_points[
        (
            yolo_points[
                "frame"
            ] >= swing_start
        )
        &
        (
            yolo_points[
                "frame"
            ] <= swing_end
        )
    ]

    yolo_count_in_swing = len(
        yolo_in_swing
    )

    # 5 YOLO observations within swing = full local coverage
    detection_coverage = clamp01(
        yolo_count_in_swing
        / 5.0
    )

    ball_evidence_quality = (
        0.60 * detection_coverage
        + 0.40 * (
            1.0
            if trajectory_available
            else 0.0
        )
    )

    # ========================================================
    # SWING EVIDENCE
    # ========================================================

    # Since swing V2.2 produced a valid peak and SWING /
    # PEAK_SWING windows, this module treats swing evidence
    # as strong but not perfect.
    swing_evidence = 0.90

    # ========================================================
    # FUSION SCORE
    # ========================================================
    #
    # Evidence score only.
    # NOT probability.
    #
    # ========================================================

    fusion_score = (
        0.35 * temporal_evidence
        + 0.25 * loss_evidence
        + 0.25 * trajectory_evidence
        + 0.15 * swing_evidence
    )

    fusion_score = clamp01(
        fusion_score
    )

    # ========================================================
    # FINAL CLASSIFICATION
    # ========================================================

    if (
        ball_evidence_quality
        < MIN_BALL_EVIDENCE_FOR_CONTACT
    ):

        if (
            temporal_evidence >= 0.65
            and loss_evidence >= 0.50
        ):

            final_label = (
                "CONTACT_CANDIDATE_"
                "INSUFFICIENT_BALL_EVIDENCE"
            )

        else:

            final_label = (
                "INSUFFICIENT_BALL_EVIDENCE"
            )

    elif (
        fusion_score
        >= CONTACT_CANDIDATE_THRESHOLD
    ):

        final_label = (
            "CONTACT_CANDIDATE"
        )

    else:

        final_label = (
            "NO_CONTACT_EVIDENCE"
        )

    # ========================================================
    # CANDIDATE EVENT FRAME
    # ========================================================

    # Do not blindly use swing peak.
    # Prefer last real YOLO observation when it occurs
    # near the peak window.
    if (
        last_yolo_frame is not None
        and abs(
            last_yolo_frame
            - peak_frame
        )
        <= MAX_TEMPORAL_GAP_FRAMES
    ):

        event_frame = int(
            round(
                (
                    last_yolo_frame
                    + peak_frame
                )
                / 2.0
            )
        )

    else:

        event_frame = (
            peak_frame
        )

    event_time_sec = (
        (event_frame - 1)
        / fps
    )

    return {
        "swing_peak_frame":
            peak_frame,

        "swing_window_start":
            swing_start,

        "swing_window_end":
            swing_end,

        "peak_window_start":
            peak_start,

        "peak_window_end":
            peak_end,

        "first_yolo_frame":
            first_yolo_frame,

        "last_yolo_frame":
            last_yolo_frame,

        "nearest_yolo_frame_to_peak":
            nearest_yolo_frame,

        "nearest_yolo_gap_frames":
            nearest_yolo_gap,

        "tracking_frames_near_swing":
            tracking_frames,

        "kalman_frames_near_swing":
            kalman_frames,

        "lost_frames_near_swing":
            lost_frames,

        "first_lost_frame":
            first_lost_frame,

        "loss_gap_frames":
            loss_gap,

        "yolo_points_pre_peak":
            len(pre_points),

        "yolo_points_post_peak":
            len(post_points),

        "trajectory_available":
            trajectory_available,

        "pre_angle_deg":
            pre_motion[
                "angle_deg"
            ],

        "post_angle_deg":
            post_motion[
                "angle_deg"
            ],

        "direction_change_deg":
            direction_change_deg,

        "pre_speed_px_frame":
            pre_motion[
                "avg_speed_px_frame"
            ],

        "post_speed_px_frame":
            post_motion[
                "avg_speed_px_frame"
            ],

        "speed_ratio":
            speed_ratio,

        "yolo_detections_in_swing":
            yolo_count_in_swing,

        "detection_coverage_score":
            detection_coverage,

        "ball_evidence_quality":
            ball_evidence_quality,

        "temporal_evidence":
            temporal_evidence,

        "track_loss_evidence":
            loss_evidence,

        "trajectory_evidence":
            trajectory_evidence,

        "swing_evidence":
            swing_evidence,

        "fusion_score":
            fusion_score,

        "event_frame":
            event_frame,

        "event_time_sec":
            event_time_sec,

        "classification":
            final_label,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    swing_path = Path(
        args.swing_csv
    ).resolve()

    ball_path = Path(
        args.ball_csv
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    if not swing_path.exists():

        raise FileNotFoundError(
            f"Swing CSV not found:\n"
            f"{swing_path}"
        )

    if not ball_path.exists():

        raise FileNotFoundError(
            f"Ball CSV not found:\n"
            f"{ball_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_csv = (
        output_dir
        / "contact_fusion_v2.csv"
    )

    # ========================================================
    # LOAD
    # ========================================================

    swing_df = pd.read_csv(
        swing_path
    )

    ball_df = pd.read_csv(
        ball_path
    )

    # ========================================================
    # ANALYZE
    # ========================================================

    result = analyze_contact(
        swing_df,
        ball_df,
        args.fps
    )

    result_df = pd.DataFrame(
        [result]
    )

    result_df.to_csv(
        output_csv,
        index=False
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 76)
    print("BaseVisionAI - Contact Fusion V2")
    print("=" * 76)

    print(
        f"Swing peak candidate      : "
        f"frame {result['swing_peak_frame']}"
    )

    print(
        f"Swing window              : "
        f"{result['swing_window_start']} "
        f"-> "
        f"{result['swing_window_end']}"
    )

    print(
        f"Peak swing window         : "
        f"{result['peak_window_start']} "
        f"-> "
        f"{result['peak_window_end']}"
    )

    print()

    print(
        f"First ball YOLO           : "
        f"{result['first_yolo_frame']}"
    )

    print(
        f"Last ball YOLO            : "
        f"{result['last_yolo_frame']}"
    )

    print(
        f"Nearest YOLO to peak      : "
        f"{result['nearest_yolo_frame_to_peak']}"
    )

    print(
        f"Temporal gap              : "
        f"{result['nearest_yolo_gap_frames']} frames"
    )

    print()

    print(
        f"TRACKING frames           : "
        f"{result['tracking_frames_near_swing']}"
    )

    print(
        f"KALMAN frames             : "
        f"{result['kalman_frames_near_swing']}"
    )

    print(
        f"LOST frames               : "
        f"{result['lost_frames_near_swing']}"
    )

    print(
        f"First LOST frame          : "
        f"{result['first_lost_frame']}"
    )

    print()

    print(
        f"YOLO points before peak   : "
        f"{result['yolo_points_pre_peak']}"
    )

    print(
        f"YOLO points after peak    : "
        f"{result['yolo_points_post_peak']}"
    )

    print(
        f"Trajectory available      : "
        f"{result['trajectory_available']}"
    )

    if result[
        "trajectory_available"
    ]:

        print(
            f"Direction change          : "
            f"{result['direction_change_deg']:.2f} deg"
        )

        print(
            f"Speed ratio               : "
            f"{result['speed_ratio']:.3f}"
        )

    else:

        print(
            "Direction change          : "
            "INSUFFICIENT EVIDENCE"
        )

        print(
            "Speed ratio               : "
            "INSUFFICIENT EVIDENCE"
        )

    print()
    print("Evidence scores:")

    print(
        f"  Temporal evidence       : "
        f"{result['temporal_evidence']:.3f}"
    )

    print(
        f"  Track-loss evidence     : "
        f"{result['track_loss_evidence']:.3f}"
    )

    print(
        f"  Trajectory evidence     : "
        f"{result['trajectory_evidence']:.3f}"
    )

    print(
        f"  Swing evidence          : "
        f"{result['swing_evidence']:.3f}"
    )

    print(
        f"  Ball evidence quality   : "
        f"{result['ball_evidence_quality']:.3f}"
    )

    print()

    print(
        f"Fusion evidence score     : "
        f"{result['fusion_score']:.3f}"
    )

    print(
        f"Candidate event frame     : "
        f"{result['event_frame']}"
    )

    print(
        f"Candidate event time      : "
        f"{result['event_time_sec']:.3f} s"
    )

    print()

    print(
        f"FINAL CLASSIFICATION      : "
        f"{result['classification']}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Fusion score is an evidence score, "
        "not a probability."
    )

    print(
        "Ball disappearance or track loss does not "
        "by itself prove bat-ball contact."
    )

    print(
        "A CONTACT_CANDIDATE is not automatically "
        "a confirmed HIT."
    )

    print()

    print(
        f"Output CSV                : "
        f"{output_csv}"
    )

    print("=" * 76)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()