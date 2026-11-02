"""
BaseVisionAI - Swing Validity V1.1
==================================

Adds temporal-confidence validation on top of Swing Validity V1.

PURPOSE
-------
Swing Validity V1 verifies that enough structural computer-vision
evidence exists to analyze a candidate swing.

V1.1 adds a second layer:
    TEMPORAL SUPPORT

This prevents extremely short detected phases from contributing to
session baselines with the same confidence as temporally supported
detections.

IMPORTANT
---------
Temporal thresholds are algorithmic support thresholds.

They are NOT:
- biomechanical thresholds
- coaching recommendations
- proof that a swing is good or bad
- calibrated baseball-performance criteria

Input:
    Swing Validity V1 CSV

Output:
    Swing Validity V1.1 CSV
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


# =============================================================================
# VERSION
# =============================================================================

VERSION = "Swing Validity V1.1"


# =============================================================================
# DEFAULT TEMPORAL SUPPORT
# =============================================================================

DEFAULT_MIN_LOAD_FRAMES = 3
DEFAULT_MIN_SWING_FRAMES = 4
DEFAULT_MIN_LOAD_TO_PEAK_FRAMES = 6


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Swing Validity V1.1"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Swing Validity V1 CSV",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Swing Validity V1.1 output CSV",
    )

    parser.add_argument(
        "--min-load-frames",
        type=int,
        default=DEFAULT_MIN_LOAD_FRAMES,
        help=(
            "Minimum LOAD temporal support in frames "
            f"(default: {DEFAULT_MIN_LOAD_FRAMES})"
        ),
    )

    parser.add_argument(
        "--min-swing-frames",
        type=int,
        default=DEFAULT_MIN_SWING_FRAMES,
        help=(
            "Minimum SWING temporal support in frames "
            f"(default: {DEFAULT_MIN_SWING_FRAMES})"
        ),
    )

    parser.add_argument(
        "--min-load-to-peak-frames",
        type=int,
        default=DEFAULT_MIN_LOAD_TO_PEAK_FRAMES,
        help=(
            "Minimum LOAD->PEAK temporal support in frames "
            f"(default: {DEFAULT_MIN_LOAD_TO_PEAK_FRAMES})"
        ),
    )

    return parser.parse_args()


# =============================================================================
# HELPERS
# =============================================================================

def separator(char="-", width=92):
    print(char * width)


def safe_float(value):

    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except (TypeError, ValueError):
        pass

    return np.nan


def duration_to_frames(duration_sec, fps):

    duration_sec = safe_float(duration_sec)
    fps = safe_float(fps)

    if not np.isfinite(duration_sec):
        return np.nan

    if not np.isfinite(fps):
        return np.nan

    if fps <= 0:
        return np.nan

    # Durations in the existing pipeline are frame-derived.
    # round() prevents floating-point values such as
    # 0.099999999999 from becoming the wrong frame count.
    return int(round(duration_sec * fps))


def bool_text(value):

    if bool(value):
        return "True"

    return "False"


# =============================================================================
# LOAD
# =============================================================================

def load_validity_v1(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Swing Validity V1 file not found: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "video_id",
        "player_id",
        "session_id",
        "validity_status",
        "evidence_quality",
        "fps",
        "load_duration_sec",
        "swing_duration_sec",
        "load_to_peak_sec",
    }

    missing = required - set(df.columns)

    if missing:

        raise ValueError(
            "Input validity database is missing required columns: "
            + ", ".join(sorted(missing))
        )

    return df


# =============================================================================
# TEMPORAL VALIDATION
# =============================================================================

def analyze_temporal_support(
    row,
    min_load_frames,
    min_swing_frames,
    min_load_to_peak_frames,
):

    fps = safe_float(
        row.get("fps", np.nan)
    )

    load_frames = duration_to_frames(
        row.get("load_duration_sec", np.nan),
        fps,
    )

    swing_frames = duration_to_frames(
        row.get("swing_duration_sec", np.nan),
        fps,
    )

    load_to_peak_frames = duration_to_frames(
        row.get("load_to_peak_sec", np.nan),
        fps,
    )

    # -------------------------------------------------------------------------
    # Missing temporal evidence
    # -------------------------------------------------------------------------

    temporal_data_available = all(
        np.isfinite(x)
        for x in [
            load_frames,
            swing_frames,
            load_to_peak_frames,
        ]
    )

    if temporal_data_available:

        load_too_short = (
            int(load_frames) < min_load_frames
        )

        swing_too_short = (
            int(swing_frames) < min_swing_frames
        )

        load_to_peak_too_short = (
            int(load_to_peak_frames)
            < min_load_to_peak_frames
        )

    else:

        load_too_short = False
        swing_too_short = False
        load_to_peak_too_short = False

    temporal_flags = int(
        load_too_short
        + swing_too_short
        + load_to_peak_too_short
    )

    # -------------------------------------------------------------------------
    # Temporal confidence
    # -------------------------------------------------------------------------

    if not temporal_data_available:

        temporal_confidence = "INSUFFICIENT"

    elif temporal_flags == 0:

        temporal_confidence = "SUPPORTED"

    else:

        temporal_confidence = "LOW_CONFIDENCE"

    return {
        "load_frames": load_frames,
        "swing_frames": swing_frames,
        "load_to_peak_frames": load_to_peak_frames,

        "min_load_frames": min_load_frames,
        "min_swing_frames": min_swing_frames,
        "min_load_to_peak_frames":
            min_load_to_peak_frames,

        "load_too_short_v1_1":
            load_too_short,

        "swing_too_short_v1_1":
            swing_too_short,

        "load_to_peak_too_short_v1_1":
            load_to_peak_too_short,

        "temporal_flags":
            temporal_flags,

        "temporal_data_available":
            temporal_data_available,

        "temporal_confidence":
            temporal_confidence,
    }


# =============================================================================
# FINAL CLASSIFICATION
# =============================================================================

def classify_v1_1(
    original_status,
    original_evidence,
    temporal_result,
):

    original_status = str(
        original_status
    ).strip().upper()

    original_evidence = str(
        original_evidence
    ).strip().upper()

    temporal_confidence = (
        temporal_result["temporal_confidence"]
    )

    temporal_flags = (
        temporal_result["temporal_flags"]
    )

    # -------------------------------------------------------------------------
    # Preserve existing low-confidence / invalid classifications
    # -------------------------------------------------------------------------

    if original_status != "VALID_SWING":

        return {
            "validity_status_v1_1":
                original_status,

            "evidence_quality_v1_1":
                original_evidence,

            "baseline_eligible":
                False,

            "validity_reason_v1_1":
                "original_v1_status_preserved",
        }

    # -------------------------------------------------------------------------
    # Original swing was structurally valid, but temporal data is unavailable
    # -------------------------------------------------------------------------

    if temporal_confidence == "INSUFFICIENT":

        return {
            "validity_status_v1_1":
                "TEMPORAL_LOW_CONFIDENCE",

            "evidence_quality_v1_1":
                "LIMITED",

            "baseline_eligible":
                False,

            "validity_reason_v1_1":
                "temporal_support_unavailable",
        }

    # -------------------------------------------------------------------------
    # Original swing was structurally valid but temporal support is too short
    # -------------------------------------------------------------------------

    if temporal_flags > 0:

        reasons = []

        if temporal_result["load_too_short_v1_1"]:
            reasons.append(
                "load_support_too_short"
            )

        if temporal_result["swing_too_short_v1_1"]:
            reasons.append(
                "swing_support_too_short"
            )

        if temporal_result[
            "load_to_peak_too_short_v1_1"
        ]:
            reasons.append(
                "load_to_peak_support_too_short"
            )

        return {
            "validity_status_v1_1":
                "TEMPORAL_LOW_CONFIDENCE",

            "evidence_quality_v1_1":
                "LIMITED",

            "baseline_eligible":
                False,

            "validity_reason_v1_1":
                ";".join(reasons),
        }

    # -------------------------------------------------------------------------
    # Structurally valid + temporally supported
    # -------------------------------------------------------------------------

    return {
        "validity_status_v1_1":
            "VALID_SWING",

        "evidence_quality_v1_1":
            original_evidence,

        "baseline_eligible":
            True,

        "validity_reason_v1_1":
            "structural_and_temporal_support_available",
    }


# =============================================================================
# PROCESS
# =============================================================================

def process_database(
    df,
    min_load_frames,
    min_swing_frames,
    min_load_to_peak_frames,
):

    rows = []

    for _, row in df.iterrows():

        temporal = analyze_temporal_support(
            row=row,
            min_load_frames=min_load_frames,
            min_swing_frames=min_swing_frames,
            min_load_to_peak_frames=
                min_load_to_peak_frames,
        )

        final = classify_v1_1(
            original_status=row["validity_status"],
            original_evidence=row[
                "evidence_quality"
            ],
            temporal_result=temporal,
        )

        output_row = row.to_dict()

        # ---------------------------------------------------------------------
        # Preserve V1 values explicitly
        # ---------------------------------------------------------------------

        output_row["validity_version"] = VERSION

        output_row["validity_status_v1"] = (
            row["validity_status"]
        )

        output_row["evidence_quality_v1"] = (
            row["evidence_quality"]
        )

        # ---------------------------------------------------------------------
        # Temporal support
        # ---------------------------------------------------------------------

        output_row.update(temporal)

        # ---------------------------------------------------------------------
        # V1.1 classification
        # ---------------------------------------------------------------------

        output_row.update(final)

        rows.append(output_row)

    return pd.DataFrame(rows)


# =============================================================================
# REPORT
# =============================================================================

def print_report(
    result,
    input_path,
    output_path,
    min_load_frames,
    min_swing_frames,
    min_load_to_peak_frames,
):

    print()
    separator("=")

    print(
        "BASEVISIONAI - SWING VALIDITY V1.1 COMPLETE"
    )

    separator("=")

    print()
    print("INPUT")
    separator()

    print(
        f"Source validity           : {input_path}"
    )

    print(
        f"Profiles analyzed         : {len(result)}"
    )

    print()
    print("TEMPORAL SUPPORT THRESHOLDS")
    separator()

    print(
        f"Minimum LOAD              : "
        f"{min_load_frames} frames"
    )

    print(
        f"Minimum SWING             : "
        f"{min_swing_frames} frames"
    )

    print(
        f"Minimum LOAD -> PEAK      : "
        f"{min_load_to_peak_frames} frames"
    )

    print()
    print("V1 -> V1.1 CLASSIFICATION")
    separator()

    for _, row in result.iterrows():

        video_id = str(
            row["video_id"]
        )

        old_status = str(
            row["validity_status_v1"]
        )

        new_status = str(
            row["validity_status_v1_1"]
        )

        evidence = str(
            row["evidence_quality_v1_1"]
        )

        baseline = bool(
            row["baseline_eligible"]
        )

        print()
        print(video_id)

        print(
            f"  V1 status               : "
            f"{old_status}"
        )

        print(
            f"  V1.1 status             : "
            f"{new_status}"
        )

        print(
            f"  Evidence quality        : "
            f"{evidence}"
        )

        print(
            f"  LOAD support            : "
            f"{row['load_frames']} frames"
        )

        print(
            f"  SWING support           : "
            f"{row['swing_frames']} frames"
        )

        print(
            f"  LOAD -> PEAK support    : "
            f"{row['load_to_peak_frames']} frames"
        )

        print(
            f"  Temporal confidence     : "
            f"{row['temporal_confidence']}"
        )

        print(
            f"  Baseline eligible       : "
            f"{bool_text(baseline)}"
        )

        print(
            f"  Reason                  : "
            f"{row['validity_reason_v1_1']}"
        )

    print()
    print("SUMMARY")
    separator()

    counts = (
        result[
            "validity_status_v1_1"
        ]
        .value_counts()
        .to_dict()
    )

    valid_count = int(
        counts.get(
            "VALID_SWING",
            0,
        )
    )

    temporal_low_count = int(
        counts.get(
            "TEMPORAL_LOW_CONFIDENCE",
            0,
        )
    )

    low_confidence_count = int(
        counts.get(
            "LOW_CONFIDENCE_SWING",
            0,
        )
    )

    baseline_count = int(
        result["baseline_eligible"]
        .astype(bool)
        .sum()
    )

    print(
        f"VALID_SWING               : "
        f"{valid_count}"
    )

    print(
        f"TEMPORAL_LOW_CONFIDENCE   : "
        f"{temporal_low_count}"
    )

    print(
        f"LOW_CONFIDENCE_SWING      : "
        f"{low_confidence_count}"
    )

    print(
        f"Baseline eligible         : "
        f"{baseline_count}"
    )

    print()
    print("OUTPUT")
    separator()

    print(
        f"{output_path}"
    )

    print()
    print("IMPORTANT")
    separator()

    print(
        "V1.1 adds temporal-support confidence to the "
        "existing structural validity analysis."
    )

    print(
        "Temporal thresholds are algorithmic support "
        "requirements, not biomechanical thresholds."
    )

    print(
        "TEMPORAL_LOW_CONFIDENCE does not mean the "
        "baseball swing itself was poor or invalid."
    )

    print(
        "It means the detected phase timing does not "
        "have enough temporal support for the session baseline."
    )

    print(
        "Original V1 classifications are preserved in "
        "the output for traceability."
    )

    separator("=")


# =============================================================================
# MAIN
# =============================================================================

def main():

    args = parse_args()

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    try:

        if args.min_load_frames < 1:
            raise ValueError(
                "--min-load-frames must be >= 1"
            )

        if args.min_swing_frames < 1:
            raise ValueError(
                "--min-swing-frames must be >= 1"
            )

        if args.min_load_to_peak_frames < 1:
            raise ValueError(
                "--min-load-to-peak-frames must be >= 1"
            )

        df = load_validity_v1(
            input_path
        )

        result = process_database(
            df=df,
            min_load_frames=
                args.min_load_frames,
            min_swing_frames=
                args.min_swing_frames,
            min_load_to_peak_frames=
                args.min_load_to_peak_frames,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        result.to_csv(
            output_path,
            index=False,
        )

        print_report(
            result=result,
            input_path=input_path,
            output_path=output_path,
            min_load_frames=
                args.min_load_frames,
            min_swing_frames=
                args.min_swing_frames,
            min_load_to_peak_frames=
                args.min_load_to_peak_frames,
        )

    except Exception as exc:

        print()
        separator("=")

        print(
            "BASEVISIONAI - SWING VALIDITY V1.1 FAILED"
        )

        separator("=")

        print()
        print(f"ERROR: {exc}")
        print()

        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())