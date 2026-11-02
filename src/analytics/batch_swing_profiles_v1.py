from pathlib import Path
import argparse
import subprocess
import sys

import pandas as pd


# ============================================================
# BASEVISIONAI - BATCH SWING PROFILES V1
# ============================================================
#
# Purpose:
#   Batch-generate Player Swing Profile V1 files from existing
#   Swing Metrics V3.3 summaries and append them to the
#   Player Swing Database V1.
#
# IMPORTANT:
#   - No model inference.
#   - No pose rerun.
#   - No ball rerun.
#   - No threshold changes.
#
# Expected existing structure:
#
# results/swing_metrics_v3_3/
#   baseball_eval1/
#       swing_metrics_v3_3_summary.csv
#   baseball_eval2/
#       swing_metrics_v3_3_summary.csv
#   ...
#
# ============================================================


def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Batch Swing Profiles V1"
    )

    parser.add_argument(
        "--metrics-root",
        required=True,
        help="Root directory containing per-video Swing Metrics V3.3 folders"
    )

    parser.add_argument(
        "--profiles-dir",
        required=True,
        help="Directory where Player Swing Profile CSVs will be written"
    )

    parser.add_argument(
        "--database",
        required=True,
        help="Player Swing Database V1 CSV"
    )

    parser.add_argument(
        "--player-id",
        required=True,
        help="Player ID assigned to all videos in this batch"
    )

    parser.add_argument(
        "--session-id",
        required=True,
        help="Session ID assigned to all videos in this batch"
    )

    parser.add_argument(
        "--video-root",
        default="data/evaluation",
        help="Directory containing original videos"
    )

    parser.add_argument(
        "--profile-script",
        default="src/analytics/player_swing_profile_v1.py",
        help="Path to Player Swing Profile V1 script"
    )

    parser.add_argument(
        "--database-script",
        default="src/analytics/player_swing_database_v1.py",
        help="Path to Player Swing Database V1 script"
    )

    parser.add_argument(
        "--video-pattern",
        default="*.mp4",
        help="Video filename pattern"
    )

    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace database rows that already exist"
    )

    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

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


def find_summary_file(folder):

    candidates = [
        folder / "swing_metrics_v3_3_summary.csv",
        folder / f"{folder.name}_swing_metrics_v3_3_summary.csv",
    ]

    for candidate in candidates:

        if candidate.exists():
            return candidate

    matches = list(
        folder.glob("*swing_metrics_v3_3_summary.csv")
    )

    if len(matches) == 1:
        return matches[0]

    return None


def find_video(video_root, video_id, pattern):

    direct = video_root / f"{video_id}.mp4"

    if direct.exists():
        return direct

    matches = list(
        video_root.glob(pattern)
    )

    exact_stem = [
        p
        for p in matches
        if p.stem == video_id
    ]

    if len(exact_stem) == 1:
        return exact_stem[0]

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    metrics_root = Path(
        args.metrics_root
    )

    profiles_dir = Path(
        args.profiles_dir
    )

    database_path = Path(
        args.database
    )

    video_root = Path(
        args.video_root
    )

    profile_script = Path(
        args.profile_script
    )

    database_script = Path(
        args.database_script
    )

    if not metrics_root.exists():

        raise FileNotFoundError(
            f"Metrics root not found:\n{metrics_root}"
        )

    if not profile_script.exists():

        raise FileNotFoundError(
            f"Profile script not found:\n{profile_script}"
        )

    if not database_script.exists():

        raise FileNotFoundError(
            f"Database script not found:\n{database_script}"
        )

    profiles_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # DISCOVER VIDEO FOLDERS
    # ========================================================

    folders = sorted(
        [
            p
            for p in metrics_root.iterdir()
            if p.is_dir()
        ]
    )

    if len(folders) == 0:

        raise ValueError(
            f"No subfolders found in:\n{metrics_root}"
        )

    print()
    print("=" * 80)
    print("BATCH SWING PROFILES V1")
    print("=" * 80)

    print(
        f"Metrics root              : {metrics_root.resolve()}"
    )

    print(
        f"Folders discovered        : {len(folders)}"
    )

    print(
        f"Player ID                 : {args.player_id}"
    )

    print(
        f"Session ID                : {args.session_id}"
    )

    print()

    # ========================================================
    # PROCESS
    # ========================================================

    rows = []

    success_count = 0
    skipped_count = 0
    failed_count = 0

    for index, folder in enumerate(
        folders,
        start=1
    ):

        video_id = folder.name

        print()
        print("=" * 80)
        print(
            f"[{index}/{len(folders)}] {video_id}"
        )
        print("=" * 80)

        summary_path = find_summary_file(
            folder
        )

        if summary_path is None:

            print(
                "SKIPPED: Swing Metrics V3.3 summary not found."
            )

            skipped_count += 1

            rows.append(
                {
                    "video_id": video_id,
                    "status": "SKIPPED",
                    "reason": "metrics_summary_not_found",
                    "profile_path": "",
                }
            )

            continue

        video_path = find_video(
            video_root,
            video_id,
            args.video_pattern
        )

        if video_path is None:

            print(
                "WARNING: source video not found."
            )

            source_video = ""

        else:

            source_video = str(
                video_path
            )

        profile_output = (
            profiles_dir
            /
            f"{video_id}_player_swing_profile_v1.csv"
        )

        # ----------------------------------------------------
        # GENERATE PROFILE
        # ----------------------------------------------------

        profile_command = [
            sys.executable,
            str(profile_script),
            "--metrics-summary",
            str(summary_path),
            "--video-id",
            video_id,
            "--player-id",
            args.player_id,
            "--session-id",
            args.session_id,
            "--source-video",
            source_video,
            "--output",
            str(profile_output),
        ]

        try:

            run_command(
                profile_command
            )

        except Exception as exc:

            print(
                f"FAILED while generating profile: {exc}"
            )

            failed_count += 1

            rows.append(
                {
                    "video_id": video_id,
                    "status": "FAILED",
                    "reason": "profile_generation_failed",
                    "profile_path": str(profile_output),
                }
            )

            continue

        # ----------------------------------------------------
        # APPEND TO DATABASE
        # ----------------------------------------------------

        database_command = [
            sys.executable,
            str(database_script),
            "--profile",
            str(profile_output),
            "--database",
            str(database_path),
        ]

        if args.replace_existing:

            database_command.append(
                "--replace-existing"
            )

        try:

            run_command(
                database_command
            )

        except Exception as exc:

            print(
                f"FAILED while updating database: {exc}"
            )

            failed_count += 1

            rows.append(
                {
                    "video_id": video_id,
                    "status": "FAILED",
                    "reason": "database_update_failed",
                    "profile_path": str(profile_output),
                }
            )

            continue

        success_count += 1

        rows.append(
            {
                "video_id": video_id,
                "status": "SUCCESS",
                "reason": "",
                "profile_path": str(profile_output),
            }
        )

    # ========================================================
    # BATCH SUMMARY
    # ========================================================

    summary_df = pd.DataFrame(
        rows
    )

    batch_summary_path = (
        profiles_dir
        /
        "batch_swing_profiles_v1_summary.csv"
    )

    summary_df.to_csv(
        batch_summary_path,
        index=False
    )

    # ========================================================
    # DATABASE STATS
    # ========================================================

    total_database_rows = 0
    player_rows = 0
    session_rows = 0

    if database_path.exists():

        database_df = pd.read_csv(
            database_path
        )

        total_database_rows = len(
            database_df
        )

        if (
            "player_id" in database_df.columns
        ):

            player_rows = int(
                (
                    database_df[
                        "player_id"
                    ].astype(str)
                    ==
                    str(args.player_id)
                ).sum()
            )

        if (
            "player_id" in database_df.columns
            and
            "session_id" in database_df.columns
        ):

            session_rows = int(
                (
                    (
                        database_df[
                            "player_id"
                        ].astype(str)
                        ==
                        str(args.player_id)
                    )
                    &
                    (
                        database_df[
                            "session_id"
                        ].astype(str)
                        ==
                        str(args.session_id)
                    )
                ).sum()
            )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 80)
    print("BATCH SWING PROFILES V1 COMPLETE")
    print("=" * 80)

    print()
    print("BATCH RESULTS")
    print("-" * 80)

    print(
        f"Folders discovered        : {len(folders)}"
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
    print("DATABASE")
    print("-" * 80)

    print(
        f"Total database rows       : {total_database_rows}"
    )

    print(
        f"Rows for player           : {player_rows}"
    )

    print(
        f"Rows for session          : {session_rows}"
    )

    print()
    print("OUTPUTS")
    print("-" * 80)

    print(
        f"Profiles directory        : {profiles_dir.resolve()}"
    )

    print(
        f"Database                  : {database_path.resolve()}"
    )

    print(
        f"Batch summary             : {batch_summary_path.resolve()}"
    )

    print()
    print("IMPORTANT")
    print("-" * 80)

    print(
        "This batch only organizes already-computed Swing Metrics V3.3."
    )

    print(
        "It does not rerun detection, tracking, pose or swing analysis."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()