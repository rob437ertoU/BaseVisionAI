from pathlib import Path
import argparse
import json
import re
import subprocess
import sys
import time

import pandas as pd


# ============================================================
# BASEVISIONAI - INDEPENDENT EVALUATION PIPELINE V1
# ============================================================

DEFAULT_MANIFEST = Path(
    "data/evaluation/evaluation_manifest.csv"
)

DEFAULT_RESULTS = Path(
    "results/evaluation_v1"
)

BASEBALL_MODEL = Path(
    "runs/baseball/baseball_v2_yolo11n/weights/best.pt"
)

BATTER_MODEL = Path(
    "runs/player_roles/player_roles_v2_yolo26n/weights/best.pt"
)

ROLE_MODEL = Path(
    "runs/player_roles/player_roles_v2_yolo11n/weights/best.pt"
)

POSE_MODEL = Path(
    "yolo11n-pose.pt"
)


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "BaseVisionAI - Independent Evaluation "
            "Pipeline V1"
        )
    )

    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Evaluation manifest CSV"
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS),
        help="Root evaluation output directory"
    )

    parser.add_argument(
        "--video-id",
        default=None,
        help=(
            "Run only one video, for example "
            "baseball_eval1. "
            "If omitted, all videos are processed."
        )
    )

    return parser.parse_args()


# ============================================================
# UTILITIES
# ============================================================

def banner(text):

    print()
    print("=" * 80)
    print(text)
    print("=" * 80)


def run_command(
    stage_name,
    command,
    log_path,
):
    """
    Runs one BaseVisionAI stage using the same Python
    interpreter as the active virtual environment.

    stdout/stderr are captured and also printed.
    """

    print()
    print("-" * 80)
    print(f"STAGE: {stage_name}")
    print("-" * 80)

    print(
        "COMMAND:",
        subprocess.list2cmdline(command)
    )

    start = time.time()

    process = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    elapsed = time.time() - start

    output = process.stdout or ""

    print(output)

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    log_path.write_text(
        output,
        encoding="utf-8",
        errors="replace"
    )

    if process.returncode != 0:

        raise RuntimeError(
            f"{stage_name} failed "
            f"with exit code {process.returncode}"
        )

    print(
        f"[OK] {stage_name} "
        f"({elapsed:.1f} sec)"
    )

    return output, elapsed


def find_csv(
    directory,
    keywords=None,
):
    """
    Search recursively for a CSV.

    If keywords are provided, filenames containing
    those keywords are preferred.
    """

    directory = Path(directory)

    csvs = sorted(
        directory.rglob("*.csv")
    )

    if not csvs:

        raise FileNotFoundError(
            f"No CSV generated in:\n{directory}"
        )

    if keywords:

        keywords = [
            keyword.lower()
            for keyword in keywords
        ]

        for csv_path in csvs:

            filename = (
                csv_path.name.lower()
            )

            if all(
                keyword in filename
                for keyword in keywords
            ):
                return csv_path

    if len(csvs) == 1:
        return csvs[0]

    print()
    print(
        f"[WARNING] Multiple CSV files found in "
        f"{directory}"
    )

    for path in csvs:
        print(f"    {path}")

    print(
        f"Using newest CSV: "
        f"{max(csvs, key=lambda p: p.stat().st_mtime)}"
    )

    return max(
        csvs,
        key=lambda p: p.stat().st_mtime
    )


def extract_float(
    text,
    patterns,
):
    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            try:
                return float(
                    match.group(1)
                )
            except Exception:
                pass

    return None


def extract_int(
    text,
    patterns,
):
    value = extract_float(
        text,
        patterns
    )

    if value is None:
        return None

    return int(
        round(value)
    )


def extract_bool(
    text,
    patterns,
):
    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            value = (
                match.group(1)
                .strip()
                .lower()
            )

            if value == "true":
                return True

            if value == "false":
                return False

    return None


def extract_text(
    text,
    patterns,
):
    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            return (
                match.group(1)
                .strip()
            )

    return None


def safe_manifest_value(
    row,
    column,
):
    value = row.get(
        column,
        ""
    )

    if pd.isna(value):
        return ""

    return value


# ============================================================
# RESULT PARSERS
# ============================================================

def parse_ball_output(
    text,
    ball_csv,
):

    real_yolo = extract_int(
        text,
        [
            r"Normal YOLO observations\s*[:=]\s*([0-9]+)",
            r"Real YOLO observations\s*[:=]\s*([0-9]+)",
        ]
    )

    recovered = extract_int(
        text,
        [
            r"Recovered observations\s*[:=]\s*([0-9]+)"
        ]
    )

    kalman = extract_int(
        text,
        [
            r"Kalman-only\s*[:=]\s*([0-9]+)"
        ]
    )

    lost = extract_int(
        text,
        [
            r"LOST\s*[:=]\s*([0-9]+)"
        ]
    )

    # Fallback: compute real YOLO count directly
    # from CSV if stdout wording changes.
    try:

        df = pd.read_csv(
            ball_csv
        )

        if (
            real_yolo is None
            and "yolo_detected"
            in df.columns
        ):

            values = (
                df["yolo_detected"]
                .astype(str)
                .str.lower()
            )

            real_yolo = int(
                values.isin(
                    ["true", "1", "yes"]
                ).sum()
            )

    except Exception:
        pass

    return {
        "real_yolo_observations": real_yolo,
        "recovered_observations": recovered,
        "kalman_only_frames": kalman,
        "lost_events": lost,
    }


def parse_pitch_crossing_output(
    text,
):

    classification = extract_text(
        text,
        [
            r"FINAL CLASSIFICATION\s*[:=]\s*([A-Z_]+)",
            r"Classification\s*[:=]\s*([A-Z_]+)",
        ]
    )

    crossing_frame = extract_float(
        text,
        [
            r"Crossing frame estimate\s*[:=]\s*([-0-9.]+)",
            r"Crossing frame\s*[:=]\s*([-0-9.]+)",
        ]
    )

    crossing_y = extract_float(
        text,
        [
            r"Crossing Y estimate\s*[:=]\s*([-0-9.]+)",
            r"Crossing Y\s*[:=]\s*([-0-9.]+)",
        ]
    )

    extrapolation = extract_float(
        text,
        [
            r"Extrapolation\s*[:=]\s*([-0-9.]+)"
        ]
    )

    residual = extract_float(
        text,
        [
            r"Mean fit residual\s*[:=]\s*([-0-9.]+)"
        ]
    )

    horizontal_inside = extract_bool(
        text,
        [
            r"Horizontal inside\s*[:=]\s*(True|False)"
        ]
    )

    vertical_inside = extract_bool(
        text,
        [
            r"Vertical inside\s*[:=]\s*(True|False)"
        ]
    )

    evidence = extract_text(
        text,
        [
            r"Evidence quality\s*[:=]\s*([A-Z_]+)"
        ]
    )

    return {
        "classification": classification,
        "crossing_frame": crossing_frame,
        "crossing_y": crossing_y,
        "extrapolation_frames": extrapolation,
        "mean_fit_residual_px": residual,
        "horizontal_inside": horizontal_inside,
        "vertical_inside": vertical_inside,
        "evidence_quality": evidence,
    }


def parse_contact_output(
    text,
):

    classification = extract_text(
        text,
        [
            r"Classification\s*[:=]\s*([A-Z_]+)",
            r"FINAL CLASSIFICATION\s*[:=]\s*([A-Z_]+)",
        ]
    )

    candidate_frame = extract_float(
        text,
        [
            r"Candidate event frame\s*[:=]\s*([-0-9.]+)",
            r"Candidate frame\s*[:=]\s*([-0-9.]+)",
            r"Pred(?:icted)? contact\s*[:=]\s*([-0-9.]+)",
        ]
    )

    score = extract_float(
        text,
        [
            r"Fusion evidence score\s*[:=]\s*([-0-9.]+)",
            r"Evidence score\s*[:=]\s*([-0-9.]+)",
            r"Score\s*[:=]\s*([-0-9.]+)",
        ]
    )

    evidence = extract_text(
        text,
        [
            r"Evidence quality\s*[:=]\s*([A-Z_]+)"
        ]
    )

    return {
        "classification": classification,
        "candidate_frame": candidate_frame,
        "evidence_score": score,
        "evidence_quality": evidence,
    }


# ============================================================
# ONE VIDEO
# ============================================================

def process_video(
    row,
    output_root,
):

    video_id = str(
        row["video_id"]
    )

    video_path = Path(
        str(
            row["video_path"]
        )
    )

    fps = float(
        row["fps"]
    )

    banner(
        f"EVALUATING: {video_id}"
    )

    print(
        f"Video : {video_path}"
    )

    print(
        f"FPS   : {fps}"
    )

    # --------------------------------------------------------
    # FROZEN MANUAL GROUND TRUTH
    # --------------------------------------------------------

    gt = {
        "pitch_result": safe_manifest_value(
            row,
            "gt_pitch_result"
        ),
        "contact_type": safe_manifest_value(
            row,
            "gt_contact_type"
        ),
        "contact_frame": safe_manifest_value(
            row,
            "gt_contact_frame"
        ),
        "notes": safe_manifest_value(
            row,
            "gt_notes"
        ),
    }

    print()
    print("FROZEN GROUND TRUTH")
    print(
        f"Pitch result : {gt['pitch_result']}"
    )
    print(
        f"Contact type : {gt['contact_type']}"
    )
    print(
        f"Contact frame: {gt['contact_frame']}"
    )
    print(
        f"Notes        : {gt['notes']}"
    )

    # --------------------------------------------------------
    # FOLDERS
    # --------------------------------------------------------

    video_root = (
        output_root
        / video_id
    )

    ball_dir = (
        video_root
        / "ball_tracking"
    )

    batter_dir = (
        video_root
        / "batter_tracking"
    )

    pose_dir = (
        video_root
        / "batter_pose"
    )

    smooth_dir = (
        video_root
        / "pose_smoothing"
    )

    swing_dir = (
        video_root
        / "swing_analysis"
    )

    zone_dir = (
        video_root
        / "strike_zone"
    )

    crossing_dir = (
        video_root
        / "pitch_crossing"
    )

    contact_dir = (
        video_root
        / "contact_fusion"
    )

    logs_dir = (
        video_root
        / "logs"
    )

    for directory in [
        ball_dir,
        batter_dir,
        pose_dir,
        smooth_dir,
        swing_dir,
        zone_dir,
        crossing_dir,
        contact_dir,
        logs_dir,
    ]:

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    summary = {
        "video_id": video_id,
        "video_path": str(video_path),
        "fps": fps,
        "ground_truth": gt,
        "status": "RUNNING",
        "failure_stage": None,
        "stages": {},
    }

    summary_path = (
        video_root
        / "summary.json"
    )

    try:

        # ====================================================
        # 1. BALL TRACKING V3.2 + BASEBALL DETECTOR V2
        # ====================================================

        ball_stdout, ball_time = run_command(
            "Ball Tracking V3.2",
            [
                sys.executable,
                "src/ball/track_ball_v3_2.py",
                "--video",
                str(video_path),
                "--model",
                str(BASEBALL_MODEL),
                "--output-dir",
                str(ball_dir),
            ],
            logs_dir
            / "01_ball_tracking.log",
        )

        ball_csv = find_csv(
            ball_dir,
            ["ball", "tracking"]
        )

        summary["stages"][
            "ball_tracking"
        ] = {
            "status": "OK",
            "csv": str(ball_csv),
            "runtime_sec": ball_time,
            **parse_ball_output(
                ball_stdout,
                ball_csv
            ),
        }

        # ====================================================
        # 2. BATTER TRACKING V2
        # ====================================================

        batter_stdout, batter_time = run_command(
            "Batter Tracking V2",
            [
                sys.executable,
                "src/detection/track_batter_v2.py",
                "--video",
                str(video_path),
                "--model",
                str(BATTER_MODEL),
                "--output-dir",
                str(batter_dir),
            ],
            logs_dir
            / "02_batter_tracking.log",
        )

        batter_csv = find_csv(
            batter_dir,
            ["batter"]
        )

        summary["stages"][
            "batter_tracking"
        ] = {
            "status": "OK",
            "csv": str(batter_csv),
            "runtime_sec": batter_time,
        }

        # ====================================================
        # 3. BATTER POSE V2
        # ====================================================

        pose_stdout, pose_time = run_command(
            "Batter Pose V2",
            [
                sys.executable,
                "src/swing/batter_pose_v2.py",
                "--video",
                str(video_path),
                "--tracking-csv",
                str(batter_csv),
                "--model",
                str(POSE_MODEL),
                "--output-dir",
                str(pose_dir),
            ],
            logs_dir
            / "03_batter_pose.log",
        )

        pose_csv = find_csv(
            pose_dir,
            ["pose"]
        )

        summary["stages"][
            "batter_pose"
        ] = {
            "status": "OK",
            "csv": str(pose_csv),
            "runtime_sec": pose_time,
        }

        # ====================================================
        # 4. POSE SMOOTHING V2
        # Frozen config:
        # confidence = 0.30
        # max gap = 3
        # EMA alpha = 0.60
        # ====================================================

        smooth_stdout, smooth_time = run_command(
            "Pose Smoothing V2",
            [
                sys.executable,
                "src/swing/smooth_pose_v2.py",
                "--input",
                str(pose_csv),
                "--output-dir",
                str(smooth_dir),
                "--conf-threshold",
                "0.30",
                "--max-gap",
                "3",
                "--ema-alpha",
                "0.60",
            ],
            logs_dir
            / "04_pose_smoothing.log",
        )

        smooth_csv = find_csv(
            smooth_dir
        )

        summary["stages"][
            "pose_smoothing"
        ] = {
            "status": "OK",
            "csv": str(smooth_csv),
            "runtime_sec": smooth_time,
        }

        # ====================================================
        # 5. SWING ANALYSIS V2.2
        # ====================================================

        swing_stdout, swing_time = run_command(
            "Swing Analysis V2.2",
            [
                sys.executable,
                "src/swing/swing_analysis_v2_2.py",
                "--input",
                str(smooth_csv),
                "--output-dir",
                str(swing_dir),
                "--fps",
                str(fps),
            ],
            logs_dir
            / "05_swing_analysis.log",
        )

        swing_csv = find_csv(
            swing_dir,
            ["swing"]
        )

        summary["stages"][
            "swing_analysis"
        ] = {
            "status": "OK",
            "csv": str(swing_csv),
            "runtime_sec": swing_time,
        }

        # ====================================================
        # 6. STRIKE ZONE V1.1
        # Uses YOLO11 role model as frozen general-role model.
        # ====================================================

        zone_stdout, zone_time = run_command(
            "Strike Zone V1.1",
            [
                sys.executable,
                "src/pitch/strike_zone_v1_1.py",
                "--video",
                str(video_path),
                "--pose-csv",
                str(smooth_csv),
                "--ball-csv",
                str(ball_csv),
                "--role-model",
                str(ROLE_MODEL),
                "--output-dir",
                str(zone_dir),
            ],
            logs_dir
            / "06_strike_zone.log",
        )

        zone_csv = find_csv(
            zone_dir,
            ["strike", "zone"]
        )

        summary["stages"][
            "strike_zone"
        ] = {
            "status": "OK",
            "csv": str(zone_csv),
            "runtime_sec": zone_time,
        }

        # ====================================================
        # 7. PITCH CROSSING V1
        # ====================================================

        crossing_stdout, crossing_time = run_command(
            "Pitch Crossing V1",
            [
                sys.executable,
                "src/pitch/pitch_crossing_v1.py",
                "--zone-csv",
                str(zone_csv),
                "--ball-csv",
                str(ball_csv),
                "--output-dir",
                str(crossing_dir),
            ],
            logs_dir
            / "07_pitch_crossing.log",
        )

        crossing_result = (
            parse_pitch_crossing_output(
                crossing_stdout
            )
        )

        summary["stages"][
            "pitch_crossing"
        ] = {
            "status": "OK",
            "runtime_sec": crossing_time,
            **crossing_result,
        }

        # ====================================================
        # 8. CONTACT FUSION V2.1
        # ====================================================

        contact_stdout, contact_time = run_command(
            "Contact Fusion V2.1",
            [
                sys.executable,
                "src/analytics/contact_fusion_v2_1.py",
                "--swing-csv",
                str(swing_csv),
                "--ball-csv",
                str(ball_csv),
                "--output-dir",
                str(contact_dir),
                "--fps",
                str(fps),
            ],
            logs_dir
            / "08_contact_fusion.log",
        )

        contact_result = (
            parse_contact_output(
                contact_stdout
            )
        )

        summary["stages"][
            "contact_fusion"
        ] = {
            "status": "OK",
            "runtime_sec": contact_time,
            **contact_result,
        }

        # ====================================================
        # COMPLETE
        # ====================================================

        summary[
            "status"
        ] = "COMPLETE"

    except Exception as error:

        print()
        print("!" * 80)
        print(
            f"PIPELINE FAILURE: {video_id}"
        )
        print("!" * 80)
        print(error)

        summary[
            "status"
        ] = "FAILED"

        summary[
            "failure_stage"
        ] = str(error)

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    summary_path.write_text(
        json.dumps(
            summary,
            indent=4,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print()
    print(
        f"Summary saved:"
    )
    print(
        summary_path
    )

    return summary


# ============================================================
# GLOBAL SUMMARY
# ============================================================

def create_global_summary(
    summaries,
    output_root,
):

    rows = []

    for summary in summaries:

        video_id = summary[
            "video_id"
        ]

        gt = summary.get(
            "ground_truth",
            {}
        )

        stages = summary.get(
            "stages",
            {}
        )

        ball = stages.get(
            "ball_tracking",
            {}
        )

        pitch = stages.get(
            "pitch_crossing",
            {}
        )

        contact = stages.get(
            "contact_fusion",
            {}
        )

        rows.append(
            {
                "video_id": video_id,

                "status": summary.get(
                    "status"
                ),

                # Ground truth
                "gt_pitch_result": gt.get(
                    "pitch_result"
                ),

                "gt_contact_type": gt.get(
                    "contact_type"
                ),

                "gt_contact_frame": gt.get(
                    "contact_frame"
                ),

                # Ball
                "real_yolo_observations": (
                    ball.get(
                        "real_yolo_observations"
                    )
                ),

                # Pitch crossing
                "pred_pitch_result": (
                    pitch.get(
                        "classification"
                    )
                ),

                "crossing_frame": (
                    pitch.get(
                        "crossing_frame"
                    )
                ),

                "crossing_y": (
                    pitch.get(
                        "crossing_y"
                    )
                ),

                "extrapolation_frames": (
                    pitch.get(
                        "extrapolation_frames"
                    )
                ),

                "mean_fit_residual_px": (
                    pitch.get(
                        "mean_fit_residual_px"
                    )
                ),

                "horizontal_inside": (
                    pitch.get(
                        "horizontal_inside"
                    )
                ),

                "vertical_inside": (
                    pitch.get(
                        "vertical_inside"
                    )
                ),

                "pitch_evidence": (
                    pitch.get(
                        "evidence_quality"
                    )
                ),

                # Contact
                "pred_contact": (
                    contact.get(
                        "classification"
                    )
                ),

                "contact_candidate_frame": (
                    contact.get(
                        "candidate_frame"
                    )
                ),

                "contact_evidence_score": (
                    contact.get(
                        "evidence_score"
                    )
                ),

                "contact_evidence_quality": (
                    contact.get(
                        "evidence_quality"
                    )
                ),

                "failure_stage": summary.get(
                    "failure_stage"
                ),
            }
        )

    dataframe = pd.DataFrame(
        rows
    )

    csv_path = (
        output_root
        / "evaluation_summary_v1.csv"
    )

    dataframe.to_csv(
        csv_path,
        index=False
    )

    return (
        dataframe,
        csv_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    manifest_path = Path(
        args.manifest
    )

    output_root = Path(
        args.output_dir
    )

    # --------------------------------------------------------
    # CHECK INPUTS
    # --------------------------------------------------------

    if not manifest_path.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n"
            f"{manifest_path}"
        )

    for model_path in [
        BASEBALL_MODEL,
        BATTER_MODEL,
        ROLE_MODEL,
    ]:

        if not model_path.exists():

            raise FileNotFoundError(
                f"Required model not found:\n"
                f"{model_path}"
            )

    manifest = pd.read_csv(
        manifest_path
    )

    required_columns = [
        "video_id",
        "video_path",
        "fps",
        "gt_pitch_result",
        "gt_contact_type",
        "gt_contact_frame",
        "gt_notes",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in manifest.columns
    ]

    if missing_columns:

        raise ValueError(
            "Manifest missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    # --------------------------------------------------------
    # SELECT VIDEO
    # --------------------------------------------------------

    if args.video_id:

        manifest = manifest[
            manifest["video_id"]
            == args.video_id
        ].copy()

        if len(manifest) == 0:

            raise ValueError(
                f"video_id not found in manifest: "
                f"{args.video_id}"
            )

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    banner(
        "BASEVISIONAI - INDEPENDENT EVALUATION V1"
    )

    print(
        f"Manifest : {manifest_path}"
    )

    print(
        f"Videos   : {len(manifest)}"
    )

    print(
        f"Output   : {output_root}"
    )

    print()
    print("FROZEN MODELS")
    print(
        f"Baseball : {BASEBALL_MODEL}"
    )

    print(
        f"Batter   : {BATTER_MODEL}"
    )

    print(
        f"Roles    : {ROLE_MODEL}"
    )

    print()
    print(
        "IMPORTANT: No thresholds or model weights "
        "are changed during evaluation."
    )

    summaries = []

    total_start = time.time()

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    for index, row in manifest.iterrows():

        print()
        print(
            "#" * 80
        )

        print(
            f"VIDEO "
            f"{len(summaries) + 1}"
            f"/{len(manifest)}"
        )

        print(
            "#" * 80
        )

        summary = process_video(
            row,
            output_root
        )

        summaries.append(
            summary
        )

    total_time = (
        time.time()
        - total_start
    )

    # --------------------------------------------------------
    # GLOBAL SUMMARY
    # --------------------------------------------------------

    dataframe, csv_path = (
        create_global_summary(
            summaries,
            output_root
        )
    )

    banner(
        "BASEVISIONAI - EVALUATION V1 COMPLETE"
    )

    complete_count = sum(
        summary["status"]
        == "COMPLETE"
        for summary in summaries
    )

    failed_count = (
        len(summaries)
        - complete_count
    )

    print(
        f"Videos processed : {len(summaries)}"
    )

    print(
        f"Complete         : {complete_count}"
    )

    print(
        f"Failed           : {failed_count}"
    )

    print(
        f"Total runtime    : "
        f"{total_time / 60:.2f} min"
    )

    print()
    print(
        f"Global summary:"
    )

    print(
        csv_path
    )

    print()
    print(
        dataframe.to_string(
            index=False
        )
    )

    print()
    print(
        "Do NOT modify thresholds based on these "
        "six evaluation videos."
    )

    print(
        "Any improvements discovered here should "
        "be implemented in a future model/version "
        "and evaluated on a new independent set."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()