from pathlib import Path
import argparse
import subprocess
import sys

import pandas as pd


# ============================================================
# BASEVISIONAI - BATCH FULL ANALYSIS V3
# ============================================================
#
# Purpose:
#   Generate Full Analysis V3 videos for all already-processed
#   evaluation clips.
#
# Uses existing:
#   - Pose smoothing CSV
#   - Swing analysis V2.2 CSV
#   - Ball tracking V3.2 CSV
#   - Swing metrics V3.3 frame-level CSV
#   - Swing metrics V3.3 summary CSV
#
# Does NOT rerun:
#   - YOLO
#   - batter tracking
#   - pose estimation
#   - pose smoothing
#   - swing phase detection
#   - ball tracking
#
# It only creates visualization videos.
# ============================================================


def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Batch Full Analysis V3"
    )

    parser.add_argument(
        "--evaluation-root",
        required=True,
        help="Root directory containing per-video pipeline outputs"
    )

    parser.add_argument(
        "--metrics-root",
        required=True,
        help="Root directory containing Swing Metrics V3.3 outputs"
    )

    parser.add_argument(
        "--video-root",
        required=True,
        help="Directory containing original videos"
    )

    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where Full Analysis V3 videos will be written"
    )

    parser.add_argument(
        "--visualizer-script",
        default="src/analytics/visualize_full_analysis_v3.py",
        help="Path to Full Analysis V3 visualizer"
    )

    parser.add_argument(
        "--trail-length",
        type=int,
        default=None,
        help="Optional maximum real-observation ball trail points"
    )

    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def find_single(folder, patterns):

    if not folder.exists():
        return None

    for pattern in patterns:

        matches = sorted(
            folder.glob(pattern)
        )

        if len(matches) > 0:
            return matches[0]

    return None


def find_video(video_root, video_id):

    candidates = [
        video_root / f"{video_id}.mp4",
        video_root / f"{video_id}.avi",
        video_root / f"{video_id}.mov",
        video_root / f"{video_id}.mkv",
    ]

    for candidate in candidates:

        if candidate.exists():
            return candidate

    matches = []

    for pattern in [
        "*.mp4",
        "*.avi",
        "*.mov",
        "*.mkv",
    ]:

        matches.extend(
            video_root.glob(pattern)
        )

    exact_matches = [
        path
        for path in matches
        if path.stem == video_id
    ]

    if len(exact_matches) > 0:
        return exact_matches[0]

    return None


def run_command(command):

    print()
    print("-" * 100)
    print("RUNNING:")
    print(" ".join(str(x) for x in command))
    print("-" * 100)

    result = subprocess.run(
        command
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"Command failed with exit code {result.returncode}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    evaluation_root = Path(
        args.evaluation_root
    )

    metrics_root = Path(
        args.metrics_root
    )

    video_root = Path(
        args.video_root
    )

    output_root = Path(
        args.output_root
    )

    visualizer_script = Path(
        args.visualizer_script
    )

    # ========================================================
    # VALIDATE ROOT PATHS
    # ========================================================

    if not evaluation_root.exists():

        raise FileNotFoundError(
            f"Evaluation root not found:\n{evaluation_root}"
        )

    if not metrics_root.exists():

        raise FileNotFoundError(
            f"Metrics root not found:\n{metrics_root}"
        )

    if not video_root.exists():

        raise FileNotFoundError(
            f"Video root not found:\n{video_root}"
        )

    if not visualizer_script.exists():

        raise FileNotFoundError(
            f"Visualizer script not found:\n{visualizer_script}"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # DISCOVER VIDEO RESULT FOLDERS
    # ========================================================

    video_dirs = sorted(
        [
            path
            for path in evaluation_root.iterdir()
            if path.is_dir()
        ]
    )

    if len(video_dirs) == 0:

        raise ValueError(
            f"No video result folders found in:\n{evaluation_root}"
        )

    # ========================================================
    # START REPORT
    # ========================================================

    print()
    print("=" * 100)
    print("BATCH FULL ANALYSIS V3")
    print("=" * 100)

    print(
        f"Evaluation root           : {evaluation_root.resolve()}"
    )

    print(
        f"Metrics root              : {metrics_root.resolve()}"
    )

    print(
        f"Video root                : {video_root.resolve()}"
    )

    print(
        f"Output root               : {output_root.resolve()}"
    )

    print(
        f"Visualizer                : {visualizer_script.resolve()}"
    )

    print(
        f"Folders discovered        : {len(video_dirs)}"
    )

    rows = []

    success_count = 0
    skipped_count = 0
    failed_count = 0

    # ========================================================
    # PROCESS EACH VIDEO
    # ========================================================

    for index, video_dir in enumerate(
        video_dirs,
        start=1
    ):

        video_id = video_dir.name

        print()
        print("=" * 100)
        print(
            f"[{index}/{len(video_dirs)}] {video_id}"
        )
        print("=" * 100)

        # ----------------------------------------------------
        # ORIGINAL VIDEO
        # ----------------------------------------------------

        video_path = find_video(
            video_root,
            video_id
        )

        # ----------------------------------------------------
        # POSE SMOOTHING
        # ----------------------------------------------------

        pose_csv = find_single(
            video_dir / "pose_smoothing",
            [
                f"{video_id}_batter_pose_v2_smoothed.csv",
                "*_batter_pose_v2_smoothed.csv",
                "*pose*smoothed*.csv",
            ]
        )

        # ----------------------------------------------------
        # SWING ANALYSIS V2.2
        # ----------------------------------------------------

        swing_csv = find_single(
            video_dir / "swing_analysis",
            [
                (
                    f"{video_id}_batter_pose_v2_smoothed_"
                    f"swing_analysis_v2_2.csv"
                ),
                "*_swing_analysis_v2_2.csv",
                "*swing_analysis_v2_2.csv",
            ]
        )

        # ----------------------------------------------------
        # BALL TRACKING V3.2
        # ----------------------------------------------------

        ball_csv = find_single(
            video_dir / "ball_tracking",
            [
                "ball_tracking_v3_2.csv",
                "*ball_tracking_v3_2.csv",
            ]
        )

        # ----------------------------------------------------
        # SWING METRICS V3.3
        # ----------------------------------------------------

        metrics_dir = (
            metrics_root
            /
            video_id
        )

        metrics_frames_csv = find_single(
            metrics_dir,
            [
                "swing_metrics_v3_3_frames.csv",
                "*swing_metrics_v3_3_frames.csv",
            ]
        )

        metrics_summary_csv = find_single(
            metrics_dir,
            [
                "swing_metrics_v3_3_summary.csv",
                "*swing_metrics_v3_3_summary.csv",
            ]
        )

        # ====================================================
        # VALIDATE REQUIRED INPUTS
        # ====================================================

        missing = []

        if video_path is None:
            missing.append("video")

        if pose_csv is None:
            missing.append("pose_csv")

        if swing_csv is None:
            missing.append("swing_csv")

        if ball_csv is None:
            missing.append("ball_csv")

        if metrics_frames_csv is None:
            missing.append("metrics_frames_csv")

        if metrics_summary_csv is None:
            missing.append("metrics_summary_csv")

        if len(missing) > 0:

            print()
            print("SKIPPED")
            print("-" * 100)

            print(
                "Missing inputs           : "
                +
                ", ".join(missing)
            )

            print()
            print("PATHS FOUND")
            print("-" * 100)

            print(
                f"Video                     : {video_path}"
            )

            print(
                f"Pose CSV                  : {pose_csv}"
            )

            print(
                f"Swing CSV                 : {swing_csv}"
            )

            print(
                f"Ball CSV                  : {ball_csv}"
            )

            print(
                f"Metrics frames CSV        : {metrics_frames_csv}"
            )

            print(
                f"Metrics summary CSV       : {metrics_summary_csv}"
            )

            skipped_count += 1

            rows.append(
                {
                    "video_id":
                        video_id,

                    "status":
                        "SKIPPED",

                    "reason":
                        ",".join(missing),

                    "video":
                        str(video_path)
                        if video_path
                        else "",

                    "pose_csv":
                        str(pose_csv)
                        if pose_csv
                        else "",

                    "swing_csv":
                        str(swing_csv)
                        if swing_csv
                        else "",

                    "ball_csv":
                        str(ball_csv)
                        if ball_csv
                        else "",

                    "metrics_frames_csv":
                        str(metrics_frames_csv)
                        if metrics_frames_csv
                        else "",

                    "metrics_summary_csv":
                        str(metrics_summary_csv)
                        if metrics_summary_csv
                        else "",

                    "output":
                        "",
                }
            )

            continue

        # ====================================================
        # OUTPUT
        # ====================================================

        output_path = (
            output_root
            /
            f"{video_id}_full_analysis_v3.mp4"
        )

        # ====================================================
        # COMMAND
        # ====================================================

        command = [
            sys.executable,
            str(visualizer_script),

            "--video",
            str(video_path),

            "--pose-csv",
            str(pose_csv),

            "--swing-csv",
            str(swing_csv),

            "--ball-csv",
            str(ball_csv),

            "--metrics-frames-csv",
            str(metrics_frames_csv),

            "--metrics-summary-csv",
            str(metrics_summary_csv),

            "--output",
            str(output_path),
        ]

        if args.trail_length is not None:

            command.extend(
                [
                    "--trail-length",
                    str(args.trail_length),
                ]
            )

        # ====================================================
        # RUN VISUALIZER
        # ====================================================

        try:

            run_command(
                command
            )

            if not output_path.exists():

                raise RuntimeError(
                    "Visualizer finished but output video was not created."
                )

            size_mb = (
                output_path.stat().st_size
                /
                (1024 * 1024)
            )

            print()
            print("SUCCESS")
            print("-" * 100)

            print(
                f"Output                    : {output_path.resolve()}"
            )

            print(
                f"File size                 : {size_mb:.2f} MB"
            )

            success_count += 1

            rows.append(
                {
                    "video_id":
                        video_id,

                    "status":
                        "SUCCESS",

                    "reason":
                        "",

                    "video":
                        str(video_path),

                    "pose_csv":
                        str(pose_csv),

                    "swing_csv":
                        str(swing_csv),

                    "ball_csv":
                        str(ball_csv),

                    "metrics_frames_csv":
                        str(metrics_frames_csv),

                    "metrics_summary_csv":
                        str(metrics_summary_csv),

                    "output":
                        str(output_path),
                }
            )

        except Exception as exc:

            print()
            print("FAILED")
            print("-" * 100)

            print(
                str(exc)
            )

            failed_count += 1

            rows.append(
                {
                    "video_id":
                        video_id,

                    "status":
                        "FAILED",

                    "reason":
                        str(exc),

                    "video":
                        str(video_path),

                    "pose_csv":
                        str(pose_csv),

                    "swing_csv":
                        str(swing_csv),

                    "ball_csv":
                        str(ball_csv),

                    "metrics_frames_csv":
                        str(metrics_frames_csv),

                    "metrics_summary_csv":
                        str(metrics_summary_csv),

                    "output":
                        str(output_path),
                }
            )

    # ========================================================
    # BATCH SUMMARY
    # ========================================================

    summary_df = pd.DataFrame(
        rows
    )

    summary_path = (
        output_root
        /
        "batch_full_analysis_v3_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 100)
    print("BATCH FULL ANALYSIS V3 COMPLETE")
    print("=" * 100)

    print()
    print("RESULTS")
    print("-" * 100)

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
    print("-" * 100)

    print(
        f"Videos directory          : {output_root.resolve()}"
    )

    print(
        f"Batch summary             : {summary_path.resolve()}"
    )

    print()
    print("IMPORTANT")
    print("-" * 100)

    print(
        "This batch only generates visualization videos from existing pipeline outputs."
    )

    print(
        "It does not rerun detection, tracking, pose estimation or swing analysis."
    )

    print("=" * 100)


if __name__ == "__main__":
    main()