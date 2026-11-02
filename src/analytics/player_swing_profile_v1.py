from pathlib import Path
import argparse

import numpy as np
import pandas as pd


# ============================================================
# BASEVISIONAI - PLAYER SWING PROFILE V1
# ============================================================
#
# Purpose:
#   Convert Swing Metrics V3.3 summary output into a compact,
#   consistent swing-level profile.
#
# Input:
#   swing_metrics_v3_3_summary.csv
#
# Output:
#   player_swing_profile_v1.csv
#
# IMPORTANT:
# - This script DOES NOT rerun models.
# - This script DOES NOT modify thresholds.
# - This script DOES NOT infer strike/ball/hit.
# - All motion / rotation / stability values remain
#   comparative 2D vision metrics.
#
# ============================================================


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Player Swing Profile V1"
    )

    parser.add_argument(
        "--metrics-summary",
        required=True,
        help="Swing Metrics V3.3 summary CSV"
    )

    parser.add_argument(
        "--video-id",
        required=True,
        help="Video or swing identifier"
    )

    parser.add_argument(
        "--player-id",
        default="unknown_player",
        help="Player identifier"
    )

    parser.add_argument(
        "--session-id",
        default="unknown_session",
        help="Session identifier"
    )

    parser.add_argument(
        "--source-video",
        default="",
        help="Original video path or filename"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output profile CSV"
    )

    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def safe_float(
    row,
    column,
    default=np.nan
):

    if column not in row.index:
        return default

    try:

        value = float(
            row[column]
        )

        if np.isfinite(
            value
        ):
            return value

    except Exception:
        pass

    return default


def safe_int(
    row,
    column,
    default=None
):

    value = safe_float(
        row,
        column
    )

    if np.isfinite(
        value
    ):
        return int(
            round(value)
        )

    return default


def safe_text(
    value,
    default=""
):

    if value is None:
        return default

    if pd.isna(
        value
    ):
        return default

    value = str(
        value
    ).strip()

    return value if value else default


def ratio_safe(
    a,
    b
):

    if not (
        np.isfinite(a)
        and np.isfinite(b)
    ):
        return np.nan

    if abs(b) < 1e-9:
        return np.nan

    return float(
        a / b
    )


def difference_safe(
    a,
    b
):

    if not (
        np.isfinite(a)
        and np.isfinite(b)
    ):
        return np.nan

    return float(
        a - b
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    metrics_path = Path(
        args.metrics_summary
    )

    output_path = Path(
        args.output
    )

    if not metrics_path.exists():

        raise FileNotFoundError(
            f"Metrics summary not found:\n"
            f"{metrics_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics_df = pd.read_csv(
        metrics_path
    )

    if len(
        metrics_df
    ) == 0:

        raise ValueError(
            "Metrics summary CSV is empty."
        )

    row = metrics_df.iloc[
        0
    ]

    # ========================================================
    # RAW METRICS
    # ========================================================

    fps = safe_float(
        row,
        "fps"
    )

    duration_sec = safe_float(
        row,
        "duration_sec"
    )

    frames_analyzed = safe_int(
        row,
        "frames_analyzed"
    )

    torso_scale = safe_float(
        row,
        "body_scale_px"
    )

    mean_pose_quality = safe_float(
        row,
        "mean_pose_quality"
    )

    # --------------------------------------------------------
    # Motion peaks
    # --------------------------------------------------------

    left_wrist_peak = safe_float(
        row,
        "left_wrist_peak_swing_px_frame"
    )

    left_wrist_peak_frame = safe_int(
        row,
        "left_wrist_peak_swing_frame"
    )

    right_wrist_peak = safe_float(
        row,
        "right_wrist_peak_swing_px_frame"
    )

    right_wrist_peak_frame = safe_int(
        row,
        "right_wrist_peak_swing_frame"
    )

    hip_peak = safe_float(
        row,
        "hip_center_peak_load_to_peak_px_frame"
    )

    hip_peak_frame = safe_int(
        row,
        "hip_center_peak_load_to_peak_frame"
    )

    shoulder_peak = safe_float(
        row,
        "shoulder_center_peak_load_to_peak_px_frame"
    )

    shoulder_peak_frame = safe_int(
        row,
        "shoulder_center_peak_load_to_peak_frame"
    )

    left_wrist_norm = safe_float(
        row,
        "left_wrist_peak_torsoscale_frame"
    )

    right_wrist_norm = safe_float(
        row,
        "right_wrist_peak_torsoscale_frame"
    )

    # --------------------------------------------------------
    # Rotation
    # --------------------------------------------------------

    shoulder_rotation = safe_float(
        row,
        "max_abs_shoulder_rotation_from_load_deg"
    )

    hip_rotation = safe_float(
        row,
        "max_abs_hip_rotation_from_load_deg"
    )

    max_separation = safe_float(
        row,
        "max_hip_shoulder_separation_deg"
    )

    max_separation_frame = safe_int(
        row,
        "max_separation_frame"
    )

    # --------------------------------------------------------
    # Stability
    # --------------------------------------------------------

    head_prepeak = safe_float(
        row,
        "prepeak_head_max_torsoscale"
    )

    left_ankle_prepeak = safe_float(
        row,
        "prepeak_left_ankle_max_torsoscale"
    )

    right_ankle_prepeak = safe_float(
        row,
        "prepeak_right_ankle_max_torsoscale"
    )

    left_knee_prepeak = safe_float(
        row,
        "prepeak_left_knee_max_torsoscale"
    )

    right_knee_prepeak = safe_float(
        row,
        "prepeak_right_knee_max_torsoscale"
    )

    follow_head = safe_float(
        row,
        "follow_head_max_torsoscale"
    )

    follow_left_ankle = safe_float(
        row,
        "follow_left_ankle_max_torsoscale"
    )

    follow_right_ankle = safe_float(
        row,
        "follow_right_ankle_max_torsoscale"
    )

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    setup_duration = safe_float(
        row,
        "setup_duration_sec"
    )

    load_duration = safe_float(
        row,
        "load_duration_sec"
    )

    swing_duration = safe_float(
        row,
        "swing_duration_sec"
    )

    peak_duration = safe_float(
        row,
        "peak_swing_duration_sec"
    )

    follow_duration = safe_float(
        row,
        "follow_through_duration_sec"
    )

    load_to_peak = safe_float(
        row,
        "load_to_peak_sec"
    )

    swing_to_peak = safe_float(
        row,
        "swing_to_peak_sec"
    )

    setup_start_frame = safe_int(
        row,
        "setup_start_frame"
    )

    setup_end_frame = safe_int(
        row,
        "setup_end_frame"
    )

    load_start_frame = safe_int(
        row,
        "load_start_frame"
    )

    load_end_frame = safe_int(
        row,
        "load_end_frame"
    )

    swing_start_frame = safe_int(
        row,
        "swing_start_frame"
    )

    swing_end_frame = safe_int(
        row,
        "swing_end_frame"
    )

    peak_start_frame = safe_int(
        row,
        "peak_swing_start_frame"
    )

    peak_end_frame = safe_int(
        row,
        "peak_swing_end_frame"
    )

    follow_start_frame = safe_int(
        row,
        "follow_through_start_frame"
    )

    follow_end_frame = safe_int(
        row,
        "follow_through_end_frame"
    )

    # ========================================================
    # DERIVED COMPARATIVE FEATURES
    # ========================================================

    wrist_peak_max = np.nanmax(
        [
            left_wrist_peak,
            right_wrist_peak,
        ]
    )

    wrist_peak_min = np.nanmin(
        [
            left_wrist_peak,
            right_wrist_peak,
        ]
    )

    wrist_speed_difference = difference_safe(
        right_wrist_peak,
        left_wrist_peak
    )

    wrist_speed_ratio = ratio_safe(
        right_wrist_peak,
        left_wrist_peak
    )

    shoulder_hip_rotation_difference = difference_safe(
        shoulder_rotation,
        hip_rotation
    )

    shoulder_hip_rotation_ratio = ratio_safe(
        shoulder_rotation,
        hip_rotation
    )

    peak_sequence_gap_frames = np.nan

    if (
        left_wrist_peak_frame is not None
        and
        right_wrist_peak_frame is not None
    ):

        peak_sequence_gap_frames = int(
            right_wrist_peak_frame
            -
            left_wrist_peak_frame
        )

    peak_sequence_gap_sec = np.nan

    if (
        np.isfinite(
            peak_sequence_gap_frames
        )
        and
        np.isfinite(
            fps
        )
        and
        fps > 0
    ):

        peak_sequence_gap_sec = (
            peak_sequence_gap_frames
            /
            fps
        )

    # ========================================================
    # PROFILE
    # ========================================================

    profile = {
        # ----------------------------------------------------
        # Identity / source
        # ----------------------------------------------------

        "profile_version":
            "Player Swing Profile V1",

        "video_id":
            safe_text(
                args.video_id
            ),

        "player_id":
            safe_text(
                args.player_id,
                default="unknown_player"
            ),

        "session_id":
            safe_text(
                args.session_id,
                default="unknown_session"
            ),

        "source_video":
            safe_text(
                args.source_video
            ),

        # ----------------------------------------------------
        # General
        # ----------------------------------------------------

        "fps":
            fps,

        "frames_analyzed":
            frames_analyzed,

        "duration_sec":
            duration_sec,

        "torso_scale_px":
            torso_scale,

        "mean_pose_quality":
            mean_pose_quality,

        # ----------------------------------------------------
        # Wrist motion
        # ----------------------------------------------------

        "left_wrist_peak_px_frame":
            left_wrist_peak,

        "left_wrist_peak_frame":
            left_wrist_peak_frame,

        "right_wrist_peak_px_frame":
            right_wrist_peak,

        "right_wrist_peak_frame":
            right_wrist_peak_frame,

        "left_wrist_peak_torsoscale_frame":
            left_wrist_norm,

        "right_wrist_peak_torsoscale_frame":
            right_wrist_norm,

        "max_wrist_peak_px_frame":
            wrist_peak_max,

        "min_wrist_peak_px_frame":
            wrist_peak_min,

        "right_minus_left_wrist_peak":
            wrist_speed_difference,

        "right_left_wrist_peak_ratio":
            wrist_speed_ratio,

        "wrist_peak_frame_gap":
            peak_sequence_gap_frames,

        "wrist_peak_time_gap_sec":
            peak_sequence_gap_sec,

        # ----------------------------------------------------
        # Body center motion
        # ----------------------------------------------------

        "hip_center_peak_px_frame":
            hip_peak,

        "hip_center_peak_frame":
            hip_peak_frame,

        "shoulder_center_peak_px_frame":
            shoulder_peak,

        "shoulder_center_peak_frame":
            shoulder_peak_frame,

        # ----------------------------------------------------
        # Rotation
        # ----------------------------------------------------

        "shoulder_rotation_projected_deg":
            shoulder_rotation,

        "hip_rotation_projected_deg":
            hip_rotation,

        "hip_shoulder_separation_projected_deg":
            max_separation,

        "hip_shoulder_separation_peak_frame":
            max_separation_frame,

        "shoulder_minus_hip_rotation_deg":
            shoulder_hip_rotation_difference,

        "shoulder_hip_rotation_ratio":
            shoulder_hip_rotation_ratio,

        # ----------------------------------------------------
        # Pre-peak stability
        # ----------------------------------------------------

        "head_prepeak_torso":
            head_prepeak,

        "left_ankle_prepeak_torso":
            left_ankle_prepeak,

        "right_ankle_prepeak_torso":
            right_ankle_prepeak,

        "left_knee_prepeak_torso":
            left_knee_prepeak,

        "right_knee_prepeak_torso":
            right_knee_prepeak,

        # ----------------------------------------------------
        # Follow-through
        # ----------------------------------------------------

        "head_follow_torso":
            follow_head,

        "left_ankle_follow_torso":
            follow_left_ankle,

        "right_ankle_follow_torso":
            follow_right_ankle,

        # ----------------------------------------------------
        # Timing
        # ----------------------------------------------------

        "setup_duration_sec":
            setup_duration,

        "load_duration_sec":
            load_duration,

        "swing_duration_sec":
            swing_duration,

        "peak_duration_sec":
            peak_duration,

        "follow_duration_sec":
            follow_duration,

        "load_to_peak_sec":
            load_to_peak,

        "swing_to_peak_sec":
            swing_to_peak,

        # ----------------------------------------------------
        # Phase frames
        # ----------------------------------------------------

        "setup_start_frame":
            setup_start_frame,

        "setup_end_frame":
            setup_end_frame,

        "load_start_frame":
            load_start_frame,

        "load_end_frame":
            load_end_frame,

        "swing_start_frame":
            swing_start_frame,

        "swing_end_frame":
            swing_end_frame,

        "peak_start_frame":
            peak_start_frame,

        "peak_end_frame":
            peak_end_frame,

        "follow_start_frame":
            follow_start_frame,

        "follow_end_frame":
            follow_end_frame,

        # ----------------------------------------------------
        # Methodology metadata
        # ----------------------------------------------------

        "metrics_source":
            "Swing Metrics V3.3",

        "motion_units":
            "px/frame and torso-length/frame",

        "rotation_units":
            "2D projected degrees",

        "stability_units":
            "torso lengths",

        "calibrated_biomechanics":
            False,
    }

    # ========================================================
    # SAVE
    # ========================================================

    profile_df = pd.DataFrame(
        [
            profile
        ]
    )

    profile_df.to_csv(
        output_path,
        index=False
    )

    # ========================================================
    # CONSOLE REPORT
    # ========================================================

    print()
    print("=" * 80)
    print(
        "PLAYER SWING PROFILE V1 COMPLETE"
    )
    print("=" * 80)

    print()
    print("IDENTITY")
    print("-" * 80)

    print(
        f"Video ID                  : {args.video_id}"
    )

    print(
        f"Player ID                 : {args.player_id}"
    )

    print(
        f"Session ID                : {args.session_id}"
    )

    print()
    print("SWING MOTION")
    print("-" * 80)

    if np.isfinite(
        left_wrist_peak
    ):

        print(
            f"Left wrist peak           : "
            f"{left_wrist_peak:.2f} px/frame "
            f"@ frame {left_wrist_peak_frame}"
        )

    if np.isfinite(
        right_wrist_peak
    ):

        print(
            f"Right wrist peak          : "
            f"{right_wrist_peak:.2f} px/frame "
            f"@ frame {right_wrist_peak_frame}"
        )

    if np.isfinite(
        hip_peak
    ):

        print(
            f"Hip center peak           : "
            f"{hip_peak:.2f} px/frame "
            f"@ frame {hip_peak_frame}"
        )

    if np.isfinite(
        shoulder_peak
    ):

        print(
            f"Shoulder center peak      : "
            f"{shoulder_peak:.2f} px/frame "
            f"@ frame {shoulder_peak_frame}"
        )

    print()
    print("PROJECTED ROTATION")
    print("-" * 80)

    if np.isfinite(
        shoulder_rotation
    ):

        print(
            f"Shoulder deviation        : "
            f"{shoulder_rotation:.2f} deg"
        )

    if np.isfinite(
        hip_rotation
    ):

        print(
            f"Hip deviation             : "
            f"{hip_rotation:.2f} deg"
        )

    if np.isfinite(
        max_separation
    ):

        print(
            f"Hip-shoulder separation   : "
            f"{max_separation:.2f} deg "
            f"@ frame {max_separation_frame}"
        )

    print()
    print("PRE-PEAK STABILITY")
    print("-" * 80)

    if np.isfinite(
        head_prepeak
    ):

        print(
            f"Head movement             : "
            f"{head_prepeak:.3f} torso lengths"
        )

    if np.isfinite(
        left_ankle_prepeak
    ):

        print(
            f"Left ankle movement       : "
            f"{left_ankle_prepeak:.3f} torso lengths"
        )

    if np.isfinite(
        right_ankle_prepeak
    ):

        print(
            f"Right ankle movement      : "
            f"{right_ankle_prepeak:.3f} torso lengths"
        )

    print()
    print("TIMING")
    print("-" * 80)

    if np.isfinite(
        load_duration
    ):

        print(
            f"Load duration             : "
            f"{load_duration:.3f} s"
        )

    if np.isfinite(
        swing_duration
    ):

        print(
            f"Swing duration            : "
            f"{swing_duration:.3f} s"
        )

    if np.isfinite(
        load_to_peak
    ):

        print(
            f"Load -> Peak              : "
            f"{load_to_peak:.3f} s"
        )

    if np.isfinite(
        swing_to_peak
    ):

        print(
            f"Swing -> Peak             : "
            f"{swing_to_peak:.3f} s"
        )

    print()
    print("QUALITY")
    print("-" * 80)

    if np.isfinite(
        mean_pose_quality
    ):

        print(
            f"Mean pose quality         : "
            f"{mean_pose_quality:.3f}"
        )

    print()
    print("OUTPUT")
    print("-" * 80)

    print(
        output_path.resolve()
    )

    print()
    print("IMPORTANT")
    print("-" * 80)

    print(
        "Profile contains comparative 2D computer-vision metrics."
    )

    print(
        "It is intended for repeated-swing comparison, dashboards and ML features."
    )

    print(
        "It does not represent calibrated physical biomechanics."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()