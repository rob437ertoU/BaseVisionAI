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
    / "contact_fusion_v2_1"
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_TEMPORAL_GAP_FRAMES = 6

MIN_TRAJECTORY_POINTS = 3
TRAJECTORY_WINDOW = 5

MODERATE_DIRECTION_CHANGE_DEG = 25.0
STRONG_DIRECTION_CHANGE_DEG = 45.0

# Evidence score threshold only.
# NOT probability.
CONTACT_CANDIDATE_THRESHOLD = 0.60


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Contact Fusion V2.1"
    )

    parser.add_argument(
        "--swing-csv",
        type=str,
        default=str(DEFAULT_SWING_CSV),
        help="Swing Analysis V2.2 CSV"
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
        help="Output directory"
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


def angle_difference_deg(angle_a, angle_b):

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

    frame_delta = f2 - f1

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

    dx = x2 - x1
    dy = y2 - y1

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
# SWING INFO
# ============================================================

def get_swing_info(swing_df):

    required = [
        "frame",
        "swing_phase",
        "peak_swing_candidate",
    ]

    missing = [
        col
        for col in required
        if col not in swing_df.columns
    ]

    if missing:

        raise RuntimeError(
            "Swing CSV missing columns: "
            + ", ".join(missing)
        )

    peak_rows = swing_df[
        swing_df[
            "peak_swing_candidate"
        ] == 1
    ]

    if len(peak_rows) == 0:

        raise RuntimeError(
            "No peak_swing_candidate found."
        )

    peak_frame = int(
        peak_rows.iloc[0]["frame"]
    )

    swing_rows = swing_df[
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

    if len(swing_rows) > 0:

        swing_start = int(
            swing_rows[
                "frame"
            ].min()
        )

        swing_end = int(
            swing_rows[
                "frame"
            ].max()
        )

    else:

        swing_start = peak_frame
        swing_end = peak_frame

    if len(peak_phase_rows) > 0:

        peak_start = int(
            peak_phase_rows[
                "frame"
            ].min()
        )

        peak_end = int(
            peak_phase_rows[
                "frame"
            ].max()
        )

    else:

        peak_start = peak_frame
        peak_end = peak_frame

    return {
        "peak_frame": peak_frame,
        "swing_window_start": swing_start,
        "swing_window_end": swing_end,
        "peak_window_start": peak_start,
        "peak_window_end": peak_end,
    }


# ============================================================
# BALL PREPARATION
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
        col
        for col in required
        if col not in ball_df.columns
    ]

    if missing:

        raise RuntimeError(
            "Ball CSV missing columns: "
            + ", ".join(missing)
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

    numeric_columns = [
        "frame",
        "track_id",
        "confidence",
        "observed_x",
        "observed_y",
        "kalman_predicted_x",
        "kalman_predicted_y",
        "filtered_x",
        "filtered_y",
        "distance_to_prediction",
        "candidate_score",
        "missing_frames",
    ]

    for col in numeric_columns:

        if col in ball_df.columns:

            ball_df[col] = pd.to_numeric(
                ball_df[col],
                errors="coerce"
            )

    ball_df = ball_df.dropna(
        subset=["frame"]
    )

    ball_df["frame"] = (
        ball_df["frame"]
        .astype(int)
    )

    return ball_df


# ============================================================
# EVIDENCE QUALITY
# ============================================================

def classify_evidence_quality(
    pre_yolo_count,
    post_yolo_count,
    trajectory_available,
    temporal_evidence,
    loss_evidence
):

    """
    Evidence quality is independent from event classification.

    IMPORTANT:
    This is not probability or confidence in the statistical sense.
    """

    # Strong:
    # observed ball before and after event
    # and trajectory can be estimated.
    if (
        pre_yolo_count >= MIN_TRAJECTORY_POINTS
        and post_yolo_count >= MIN_TRAJECTORY_POINTS
        and trajectory_available
        and temporal_evidence >= 0.65
    ):

        return "STRONG"

    # Moderate:
    # some actual observations exist on both sides,
    # but trajectory evidence is incomplete.
    if (
        pre_yolo_count >= 2
        and post_yolo_count >= 2
        and temporal_evidence >= 0.50
    ):

        return "MODERATE"

    # Limited:
    # temporal relationship exists,
    # but only one side of trajectory is really observed,
    # or disappearance dominates the evidence.
    if (
        pre_yolo_count >= 2
        and post_yolo_count < 2
        and temporal_evidence >= 0.50
    ):

        return "LIMITED"

    if (
        temporal_evidence >= 0.65
        and loss_evidence >= 0.50
    ):

        return "LIMITED"

    return "INSUFFICIENT"


# ============================================================
# TRAJECTORY STATUS
# ============================================================

def classify_trajectory_status(
    pre_count,
    post_count,
    trajectory_available
):

    if trajectory_available:
        return "AVAILABLE"

    if (
        pre_count >= MIN_TRAJECTORY_POINTS
        and post_count < MIN_TRAJECTORY_POINTS
    ):
        return "INSUFFICIENT_POST_EVENT_OBSERVATIONS"

    if (
        pre_count < MIN_TRAJECTORY_POINTS
        and post_count >= MIN_TRAJECTORY_POINTS
    ):
        return "INSUFFICIENT_PRE_EVENT_OBSERVATIONS"

    return "INSUFFICIENT_PRE_AND_POST_EVENT_OBSERVATIONS"


# ============================================================
# CONTACT FUSION
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

    peak_frame = swing[
        "peak_frame"
    ]

    swing_start = swing[
        "swing_window_start"
    ]

    swing_end = swing[
        "swing_window_end"
    ]

    peak_start = swing[
        "peak_window_start"
    ]

    peak_end = swing[
        "peak_window_end"
    ]

    # ========================================================
    # REAL YOLO BALL OBSERVATIONS
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
    # NEAREST YOLO TO SWING PEAK
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
            nearest_row["frame"]
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
    ]

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
    # TRACK LOSS
    # ========================================================

    first_lost_frame = None

    if last_yolo_frame is not None:

        lost_rows = ball_df[
            (
                ball_df["frame"]
                > last_yolo_frame
            )
            &
            (
                ball_df["state"]
                == "LOST"
            )
        ]

        if len(lost_rows) > 0:

            first_lost_frame = int(
                lost_rows[
                    "frame"
                ].min()
            )

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
    # LOCAL TRAJECTORY
    # ========================================================

    pre_points = yolo_points[
        (
            yolo_points["frame"]
            < peak_frame
        )
        &
        (
            yolo_points["frame"]
            >= (
                peak_frame
                - TRAJECTORY_WINDOW
            )
        )
    ].copy()

    post_points = yolo_points[
        (
            yolo_points["frame"]
            > peak_frame
        )
        &
        (
            yolo_points["frame"]
            <= (
                peak_frame
                + TRAJECTORY_WINDOW
            )
        )
    ].copy()

    pre_motion = calculate_segment_motion(
        pre_points
    )

    post_motion = calculate_segment_motion(
        post_points
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

        pre_speed = pre_motion[
            "avg_speed_px_frame"
        ]

        post_speed = post_motion[
            "avg_speed_px_frame"
        ]

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
    # SWING EVIDENCE
    # ========================================================

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
    # EVENT CLASSIFICATION
    # ========================================================

    if (
        fusion_score
        >= CONTACT_CANDIDATE_THRESHOLD
    ):

        event_classification = (
            "CONTACT_CANDIDATE"
        )

    else:

        event_classification = (
            "NO_CONTACT_EVIDENCE"
        )

    # ========================================================
    # EVIDENCE QUALITY
    # ========================================================

    evidence_quality = (
        classify_evidence_quality(
            pre_yolo_count=len(
                pre_points
            ),
            post_yolo_count=len(
                post_points
            ),
            trajectory_available=(
                trajectory_available
            ),
            temporal_evidence=(
                temporal_evidence
            ),
            loss_evidence=(
                loss_evidence
            ),
        )
    )

    # ========================================================
    # TRAJECTORY STATUS
    # ========================================================

    trajectory_status = (
        classify_trajectory_status(
            pre_count=len(
                pre_points
            ),
            post_count=len(
                post_points
            ),
            trajectory_available=(
                trajectory_available
            )
        )
    )

    # ========================================================
    # CANDIDATE EVENT FRAME
    # ========================================================

    if (
        last_yolo_frame is not None
        and abs(
            last_yolo_frame
            - peak_frame
        )
        <= MAX_TEMPORAL_GAP_FRAMES
    ):

        candidate_event_frame = int(
            round(
                (
                    last_yolo_frame
                    + peak_frame
                )
                / 2.0
            )
        )

        candidate_frame_method = (
            "MIDPOINT_LAST_YOLO_AND_SWING_PEAK"
        )

    else:

        candidate_event_frame = (
            peak_frame
        )

        candidate_frame_method = (
            "SWING_PEAK_FALLBACK"
        )

    candidate_event_time_sec = (
        (candidate_event_frame - 1)
        / fps
    )

    # ========================================================
    # HUMAN-READABLE SUMMARY
    # ========================================================

    if (
        event_classification
        == "CONTACT_CANDIDATE"
    ):

        if evidence_quality == "STRONG":

            interpretation = (
                "Contact-like event with strong "
                "temporal and ball-trajectory evidence."
            )

        elif evidence_quality == "MODERATE":

            interpretation = (
                "Contact-like event with moderate "
                "ball evidence."
            )

        elif evidence_quality == "LIMITED":

            interpretation = (
                "Contact-like temporal event, but "
                "ball evidence is limited."
            )

        else:

            interpretation = (
                "Contact-like temporal event, but "
                "ball evidence is insufficient."
            )

    else:

        interpretation = (
            "Available evidence does not support "
            "a contact candidate."
        )

    return {
        "event_classification":
            event_classification,

        "evidence_quality":
            evidence_quality,

        "interpretation":
            interpretation,

        "fusion_evidence_score":
            fusion_score,

        "candidate_event_frame":
            candidate_event_frame,

        "candidate_event_time_sec":
            candidate_event_time_sec,

        "candidate_frame_method":
            candidate_frame_method,

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

        "first_lost_frame":
            first_lost_frame,

        "loss_gap_frames":
            loss_gap,

        "tracking_frames_near_swing":
            tracking_frames,

        "kalman_frames_near_swing":
            kalman_frames,

        "lost_frames_near_swing":
            lost_frames,

        "pre_event_yolo_points":
            len(pre_points),

        "post_event_yolo_points":
            len(post_points),

        "trajectory_status":
            trajectory_status,

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

        "temporal_evidence":
            temporal_evidence,

        "track_loss_evidence":
            loss_evidence,

        "trajectory_evidence":
            trajectory_evidence,

        "swing_evidence":
            swing_evidence,
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
        / "contact_fusion_v2_1.csv"
    )

    swing_df = pd.read_csv(
        swing_path
    )

    ball_df = pd.read_csv(
        ball_path
    )

    result = analyze_contact(
        swing_df,
        ball_df,
        args.fps
    )

    pd.DataFrame(
        [result]
    ).to_csv(
        output_csv,
        index=False
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 78)
    print("BaseVisionAI - Contact Fusion V2.1")
    print("=" * 78)

    print()
    print("EVENT")
    print("-" * 78)

    print(
        f"Classification            : "
        f"{result['event_classification']}"
    )

    print(
        f"Evidence quality          : "
        f"{result['evidence_quality']}"
    )

    print(
        f"Fusion evidence score     : "
        f"{result['fusion_evidence_score']:.3f}"
    )

    print(
        f"Candidate event frame     : "
        f"{result['candidate_event_frame']}"
    )

    print(
        f"Candidate event time      : "
        f"{result['candidate_event_time_sec']:.3f} s"
    )

    print(
        f"Candidate frame method    : "
        f"{result['candidate_frame_method']}"
    )

    print()

    print(
        f"Interpretation            : "
        f"{result['interpretation']}"
    )

    print()
    print("SWING")
    print("-" * 78)

    print(
        f"Swing peak candidate      : "
        f"{result['swing_peak_frame']}"
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
    print("BALL")
    print("-" * 78)

    print(
        f"First YOLO observation    : "
        f"{result['first_yolo_frame']}"
    )

    print(
        f"Last YOLO observation     : "
        f"{result['last_yolo_frame']}"
    )

    print(
        f"Nearest YOLO to peak      : "
        f"{result['nearest_yolo_frame_to_peak']}"
    )

    print(
        f"Gap to swing peak         : "
        f"{result['nearest_yolo_gap_frames']} frames"
    )

    print(
        f"First LOST frame          : "
        f"{result['first_lost_frame']}"
    )

    print(
        f"YOLO pre-event points     : "
        f"{result['pre_event_yolo_points']}"
    )

    print(
        f"YOLO post-event points    : "
        f"{result['post_event_yolo_points']}"
    )

    print()
    print("TRAJECTORY")
    print("-" * 78)

    print(
        f"Status                    : "
        f"{result['trajectory_status']}"
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
            "UNKNOWN"
        )

        print(
            "Speed ratio               : "
            "UNKNOWN"
        )

    print()
    print("EVIDENCE COMPONENTS")
    print("-" * 78)

    print(
        f"Temporal evidence         : "
        f"{result['temporal_evidence']:.3f}"
    )

    print(
        f"Track-loss evidence       : "
        f"{result['track_loss_evidence']:.3f}"
    )

    print(
        f"Trajectory evidence       : "
        f"{result['trajectory_evidence']:.3f}"
    )

    print(
        f"Swing evidence            : "
        f"{result['swing_evidence']:.3f}"
    )

    print()
    print("IMPORTANT")
    print("-" * 78)

    print(
        "Fusion evidence score is NOT a probability."
    )

    print(
        "CONTACT_CANDIDATE does NOT mean confirmed hit."
    )

    print(
        "Candidate event frame is a heuristic temporal estimate."
    )

    print(
        "Trajectory claims require real YOLO observations "
        "before and after the candidate."
    )

    print()
    print(
        f"Output CSV                : "
        f"{output_csv}"
    )

    print("=" * 78)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()