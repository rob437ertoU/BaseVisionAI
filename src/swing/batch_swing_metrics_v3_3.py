from pathlib import Path
import argparse
import subprocess
import sys

import pandas as pd


# ============================================================
# BASEVISIONAI - BATCH SWING METRICS V3.3
# ============================================================
#
# Purpose:
#   Run Swing Metrics V3.3 across multiple already-processed
#   evaluation videos.
#
# This script DOES NOT rerun:
#   - YOLO
#   - batter tracking
#   - pose estimation
#   - pose smoothing
#   - swing phase detection
#
# It only consumes existing CSV outputs.
# ============================================================


def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Batch Swing Metrics V3.3"
    )

    parser.add_argument(
        "--evaluation-root",
        required=True,
        help="Root containing per-video evaluation result folders"
    )

    parser.add_argument(
        "--output-root",
        required=True,
        help="Root where Swing Metrics V3.3 outputs will be written"
    )

    parser.add_argument(
        "--metrics-script",
        default="src/swing/swing_metrics_v3_3.py",
        help="Path to Swing Metrics V3.3 script"
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="FPS used by Swing Metrics V3.3"
    )

    parser.add_argument(
        "--setup-baseline-frames",
        type=int,
        default=10,
        help="Late SETUP frames used as stability baseline"
    )

    return parser.parse_args()


def find_pose_csv(video_dir):

    pose_dir = (
        video_dir
        /
        "pose_smoothing"
    )

    if not pose_dir.exists():
        return None

    patterns = [
        "*_batter_pose_v2_smoothed.csv",
        "*pose*smoothed*.csv",
    ]

    for pattern in patterns:

        matches = sorted(
            pose_dir.glob(pattern)
        )

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            return matches[0]

    return None


def find_swing_csv(video_dir):

    swing_dir = (
        video_dir
        /
        "swing_analysis"
    )

    if not swing_dir.exists():
        return None

    patterns = [
        "*_batter_pose_v2_smoothed_swing_analysis_v2_2.csv",
        "*swing_analysis_v2_2.csv",
    ]

    for pattern in patterns:

        matches = sorted(
            swing_dir.glob(pattern)
        )

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            return matches[0]

    return None


def run_command(command):

    print()
    print("-" * 80)
    print("RUNNING:")
    print(" ".join(str(x) for x in command))
    print("-" * 80)

    result = subprocess.run(
        command
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"Command failed with exit code {result.returncode}"
        )


def main():

    args = parse_args()

    evaluation_root = Path(
        args.evaluation_root
    )

    output_root = Path(
        args.output_root
    )

    metrics_script = Path(
        args.metrics_script
    )

    if not evaluation_root.exists():

        raise FileNotFoundError(
            f"Evaluation root not found:\n{evaluation_root}"
        )

    if not metrics_script.exists():

        raise FileNotFoundError(
            f"Swing Metrics V3.3 script not found:\n{metrics_script}"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    video_dirs = sorted(
        [
            p
            for p in evaluation_root.iterdir()
            if p.is_dir()
        ]
    )

    if len(video_dirs) == 0:

        raise ValueError(
            f"No video result folders found in:\n{evaluation_root}"
        )

    print()
    print("=" * 80)
    print("BATCH SWING METRICS V3.3")
    print("=" * 80)

    print(
        f"Evaluation root           : {evaluation_root.resolve()}"
    )

    print(
        f"Output root               : {output_root.resolve()}"
    )

    print(
        f"Folders discovered        : {len(video_dirs)}"
    )

    print()

    rows = []

    success_count = 0
    skipped_count = 0
    failed_count = 0

    for index, video_dir in enumerate(
        video_dirs,
        start=1
    ):

        video_id = video_dir.name

        print()
        print("=" * 80)
        print(
            f"[{index}/{len(video_dirs)}] {video_id}"
        )
        print("=" * 80)

        pose_csv = find_pose_csv(
            video_dir
        )

        swing_csv = find_swing_csv(
            video_dir
        )

        if pose_csv is None:

            print(
                "SKIPPED: pose smoothing CSV not found."
            )

            skipped_count += 1

            rows.append(
                {
                    "video_id": video_id,
                    "status": "SKIPPED",
                    "reason": "pose_csv_not_found",
                    "pose_csv": "",
                    "swing_csv": str(swing_csv) if swing_csv else "",
                    "output_dir": "",
                }
            )

            continue

        if swing_csv is None:

            print(
                "SKIPPED: swing analysis V2.2 CSV not found."
            )

            skipped_count += 1

            rows.append(
                {
                    "video_id": video_id,
                    "status": "SKIPPED",
                    "reason": "swing_csv_not_found",
                    "pose_csv": str(pose_csv),
                    "swing_csv": "",
                    "output_dir": "",
                }
            )

            continue

        output_dir = (
            output_root
            /
            video_id
        )

        command = [
            sys.executable,
            str(metrics_script),
            "--pose-csv",
            str(pose_csv),
            "--swing-csv",
            str(swing_csv),
            "--fps",
            str(args.fps),
            "--setup-baseline-frames",
            str(args.setup_baseline_frames),
            "--output-dir",
            str(output_dir),
        ]

        try:

            run_command(
                command
            )

            success_count += 1

            rows.append(
                {
                    "video_id": video_id,
                    "status": "SUCCESS",
                    "reason": "",
                    "pose_csv": str(pose_csv),
                    "swing_csv": str(swing_csv),
                    "output_dir": str(output_dir),
                }
            )

        except Exception as exc:

            print(
                f"FAILED: {exc}"
            )

            failed_count += 1

            rows.append(
                {
                    "video_id": video_id,
                    "status": "FAILED",
                    "reason": str(exc),
                    "pose_csv": str(pose_csv),
                    "swing_csv": str(swing_csv),
                    "output_dir": str(output_dir),
                }
            )

    summary_df = pd.DataFrame(
        rows
    )

    summary_path = (
        output_root
        /
        "batch_swing_metrics_v3_3_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print()
    print("=" * 80)
    print("BATCH SWING METRICS V3.3 COMPLETE")
    print("=" * 80)

    print()
    print("RESULTS")
    print("-" * 80)

    print(
        f"Folders discovered        : {len(video_dirs)}"
    )

    print(
        f"Successful                : {success_count}"
    )

    print(
        f"Skipped                   : {skipped_count}"
    )

    print(
        f"Failed                    : {failed_count}"
    )

    print()
    print("OUTPUT")
    print("-" * 80)

    print(
        f"Metrics root              : {output_root.resolve()}"
    )

    print(
        f"Batch summary             : {summary_path.resolve()}"
    )

    print()
    print("IMPORTANT")
    print("-" * 80)

    print(
        "This batch only computes Swing Metrics V3.3 from existing pose and swing CSVs."
    )

    print(
        "It does not rerun detection, tracking, pose estimation or phase detection."
    )

    print(
        "All metrics remain comparative 2D vision measurements."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()