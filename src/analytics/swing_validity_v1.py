from pathlib import Path
import argparse

import numpy as np
import pandas as pd


# ============================================================
# BASEVISIONAI - SWING VALIDITY V1
# ============================================================
#
# Purpose:
#   Evaluate whether each swing profile is structurally usable
#   for swing-to-swing comparison.
#
# Input:
#   Player Swing Database V1
#
# Output statuses:
#
#   VALID_SWING
#       Core phase/timing/motion evidence is available and
#       structurally coherent enough for comparison.
#
#   LOW_CONFIDENCE_SWING
#       Most information exists, but timing is extremely short
#       or important evidence is incomplete.
#
#   NO_CLEAR_SWING
#       Required swing/peak evidence is missing.
#
# IMPORTANT:
#   - This is NOT hit / miss / strike / ball classification.
#   - It is NOT a swing quality score.
#   - It does NOT use cross-video outliers to reject clips.
#   - No model inference is performed.
#
# ============================================================


def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Swing Validity V1"
    )

    parser.add_argument(
        "--database",
        required=True,
        help="Player Swing Database V1 CSV"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output swing validity CSV"
    )

    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def safe_float(row, column):

    if column not in row.index:
        return np.nan

    try:
        value = float(row[column])

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return np.nan


def safe_int(row, column):

    value = safe_float(
        row,
        column
    )

    if np.isfinite(value):
        return int(round(value))

    return None


def finite(value):

    try:
        return np.isfinite(float(value))
    except Exception:
        return False


# ============================================================
# VALIDITY
# ============================================================

def evaluate_row(row):

    fps = safe_float(
        row,
        "fps"
    )

    if not finite(fps) or fps <= 0:
        fps = 30.0

    # --------------------------------------------------------
    # Core motion evidence
    # --------------------------------------------------------

    left_wrist = safe_float(
        row,
        "left_wrist_peak_px_frame"
    )

    right_wrist = safe_float(
        row,
        "right_wrist_peak_px_frame"
    )

    hip_peak = safe_float(
        row,
        "hip_center_peak_px_frame"
    )

    shoulder_peak = safe_float(
        row,
        "shoulder_center_peak_px_frame"
    )

    pose_quality = safe_float(
        row,
        "mean_pose_quality"
    )

    # --------------------------------------------------------
    # Phase timing
    # --------------------------------------------------------

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
        "peak_duration_sec"
    )

    load_to_peak = safe_float(
        row,
        "load_to_peak_sec"
    )

    swing_to_peak = safe_float(
        row,
        "swing_to_peak_sec"
    )

    # --------------------------------------------------------
    # Phase frames
    # --------------------------------------------------------

    swing_start = safe_int(
        row,
        "swing_start_frame"
    )

    swing_end = safe_int(
        row,
        "swing_end_frame"
    )

    peak_start = safe_int(
        row,
        "peak_start_frame"
    )

    peak_end = safe_int(
        row,
        "peak_end_frame"
    )

    # ========================================================
    # STRUCTURAL FLAGS
    # ========================================================

    has_swing_phase = (
        swing_start is not None
        and swing_end is not None
        and swing_end >= swing_start
        and finite(swing_duration)
        and swing_duration > 0
    )

    has_peak_phase = (
        peak_start is not None
        and peak_end is not None
        and peak_end >= peak_start
        and finite(peak_duration)
        and peak_duration > 0
    )

    has_wrist_motion = (
        finite(left_wrist)
        and finite(right_wrist)
    )

    has_body_motion = (
        finite(hip_peak)
        and finite(shoulder_peak)
    )

    has_timing = (
        finite(load_to_peak)
        and finite(swing_to_peak)
    )

    has_pose_quality = finite(
        pose_quality
    )

    # ========================================================
    # FRAME-BASED TIMING CHECK
    # ========================================================
    #
    # This is not tuned against eval1-eval6.
    #
    # We only identify segments that are extremely short in
    # temporal resolution: fewer than 3 video frames.
    #
    # At 30 FPS:
    #     3 frames = 0.100 sec
    #
    # This is a structural warning, not a baseball-performance
    # threshold.
    # ========================================================

    minimum_temporal_support_sec = (
        3.0 / fps
    )

    very_short_swing = (
        finite(swing_duration)
        and
        swing_duration < minimum_temporal_support_sec
    )

    very_short_load_to_peak = (
        finite(load_to_peak)
        and
        load_to_peak < minimum_temporal_support_sec
    )

    very_short_swing_to_peak = (
        finite(swing_to_peak)
        and
        swing_to_peak < minimum_temporal_support_sec
    )

    # ========================================================
    # STATUS
    # ========================================================

    reasons = []

    # --------------------------------------------------------
    # NO CLEAR SWING
    # --------------------------------------------------------

    if not has_swing_phase:

        reasons.append(
            "missing_or_invalid_swing_phase"
        )

    if not has_peak_phase:

        reasons.append(
            "missing_or_invalid_peak_phase"
        )

    if not has_wrist_motion:

        reasons.append(
            "missing_wrist_motion"
        )

    if (
        not has_swing_phase
        or
        not has_peak_phase
        or
        not has_wrist_motion
    ):

        status = "NO_CLEAR_SWING"
        evidence_quality = "INSUFFICIENT"

    else:

        # ----------------------------------------------------
        # LOW CONFIDENCE
        # ----------------------------------------------------

        low_confidence_reasons = []

        if not has_body_motion:

            low_confidence_reasons.append(
                "missing_body_motion"
            )

        if not has_timing:

            low_confidence_reasons.append(
                "missing_phase_timing"
            )

        if not has_pose_quality:

            low_confidence_reasons.append(
                "missing_pose_quality"
            )

        if very_short_swing:

            low_confidence_reasons.append(
                "swing_phase_under_3_frames"
            )

        if very_short_load_to_peak:

            low_confidence_reasons.append(
                "load_to_peak_under_3_frames"
            )

        if very_short_swing_to_peak:

            low_confidence_reasons.append(
                "swing_to_peak_under_3_frames"
            )

        if len(low_confidence_reasons) > 0:

            status = "LOW_CONFIDENCE_SWING"
            evidence_quality = "LIMITED"

            reasons.extend(
                low_confidence_reasons
            )

        else:

            status = "VALID_SWING"
            evidence_quality = "SUFFICIENT"

            reasons.append(
                "structural_swing_evidence_available"
            )

    # ========================================================
    # OUTPUT
    # ========================================================

    return {

        "video_id":
            row.get(
                "video_id",
                ""
            ),

        "player_id":
            row.get(
                "player_id",
                ""
            ),

        "session_id":
            row.get(
                "session_id",
                ""
            ),

        "validity_status":
            status,

        "evidence_quality":
            evidence_quality,

        "has_swing_phase":
            has_swing_phase,

        "has_peak_phase":
            has_peak_phase,

        "has_wrist_motion":
            has_wrist_motion,

        "has_body_motion":
            has_body_motion,

        "has_timing":
            has_timing,

        "has_pose_quality":
            has_pose_quality,

        "fps":
            fps,

        "minimum_temporal_support_sec":
            minimum_temporal_support_sec,

        "load_duration_sec":
            load_duration,

        "swing_duration_sec":
            swing_duration,

        "peak_duration_sec":
            peak_duration,

        "load_to_peak_sec":
            load_to_peak,

        "swing_to_peak_sec":
            swing_to_peak,

        "left_wrist_peak_px_frame":
            left_wrist,

        "right_wrist_peak_px_frame":
            right_wrist,

        "hip_center_peak_px_frame":
            hip_peak,

        "shoulder_center_peak_px_frame":
            shoulder_peak,

        "mean_pose_quality":
            pose_quality,

        "very_short_swing":
            very_short_swing,

        "very_short_load_to_peak":
            very_short_load_to_peak,

        "very_short_swing_to_peak":
            very_short_swing_to_peak,

        "reason":
            ";".join(reasons),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    database_path = Path(
        args.database
    )

    output_path = Path(
        args.output
    )

    if not database_path.exists():

        raise FileNotFoundError(
            f"Database not found:\n{database_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    database_df = pd.read_csv(
        database_path
    )

    if len(database_df) == 0:

        raise ValueError(
            "Swing database is empty."
        )

    rows = []

    for _, row in database_df.iterrows():

        rows.append(
            evaluate_row(
                row
            )
        )

    validity_df = pd.DataFrame(
        rows
    )

    validity_df.to_csv(
        output_path,
        index=False
    )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 80)
    print(
        "SWING VALIDITY V1 COMPLETE"
    )
    print("=" * 80)

    print()
    print("SUMMARY")
    print("-" * 80)

    print(
        f"Swings evaluated          : {len(validity_df)}"
    )

    for status in [
        "VALID_SWING",
        "LOW_CONFIDENCE_SWING",
        "NO_CLEAR_SWING",
    ]:

        count = int(
            (
                validity_df[
                    "validity_status"
                ]
                ==
                status
            ).sum()
        )

        print(
            f"{status:<26}: {count}"
        )

    print()
    print("SWINGS")
    print("-" * 80)

    for _, row in validity_df.iterrows():

        print(
            f"{str(row['video_id']):<20} | "
            f"{str(row['validity_status']):<22} | "
            f"{str(row['evidence_quality']):<12} | "
            f"{row['reason']}"
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
        "VALID_SWING means structurally usable for comparison."
    )

    print(
        "It does not prove that a true baseball swing occurred."
    )

    print(
        "The temporal warning uses only a minimum of 3 video frames."
    )

    print(
        "No cross-video metric thresholds or outlier rejection are used."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()