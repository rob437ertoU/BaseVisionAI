"""
BaseVisionAI - Batch Swing Feedback V1

Runs Swing Feedback V1.1 for every swing/profile belonging to a selected
player and session.

Each target swing is evaluated independently.

VALID_SWING profiles are used to construct session baselines.
LOW_CONFIDENCE_SWING profiles can still receive feedback but are never
used as baseline reference swings.

Outputs:
- Individual feedback CSVs
- Individual summary CSVs
- Combined feedback database
- Combined summary database
- Batch execution summary
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd


# =============================================================================
# HELPERS
# =============================================================================

def run_command(command: list[str]) -> tuple[bool, str]:

    print()
    print("-" * 88)
    print("RUNNING:")
    print(" ".join(command))
    print("-" * 88)
    print()

    result = subprocess.run(
        command,
        text=True,
    )

    if result.returncode != 0:
        return (
            False,
            f"Command failed with exit code {result.returncode}",
        )

    return True, "OK"


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Batch Swing Feedback V1"
    )

    parser.add_argument(
        "--database",
        required=True,
        help="Player Swing Database V1 CSV",
    )

    parser.add_argument(
        "--validity",
        required=True,
        help="Swing Validity V1 CSV",
    )

    parser.add_argument(
        "--player-id",
        required=True,
        help="Player identifier",
    )

    parser.add_argument(
        "--session-id",
        required=True,
        help="Session identifier",
    )

    parser.add_argument(
        "--output-dir",
        default="results/swing_feedback",
        help="Batch output directory",
    )

    parser.add_argument(
        "--feedback-script",
        default="src/analytics/swing_feedback_v1.py",
        help="Swing Feedback V1.1 script",
    )

    args = parser.parse_args()

    # =========================================================================
    # PATHS
    # =========================================================================

    database_path = Path(args.database)
    validity_path = Path(args.validity)
    output_dir = Path(args.output_dir)
    feedback_script = Path(args.feedback_script)

    if not database_path.exists():
        print(f"ERROR: Database not found: {database_path}")
        sys.exit(1)

    if not validity_path.exists():
        print(f"ERROR: Validity file not found: {validity_path}")
        sys.exit(1)

    if not feedback_script.exists():
        print(f"ERROR: Feedback script not found: {feedback_script}")
        sys.exit(1)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # LOAD DATABASE
    # =========================================================================

    database = pd.read_csv(
        database_path
    )

    required_columns = [
        "video_id",
        "player_id",
        "session_id",
    ]

    for column in required_columns:

        if column not in database.columns:
            print(
                f"ERROR: Required database column missing: {column}"
            )
            sys.exit(1)

    # =========================================================================
    # SELECT PLAYER / SESSION
    # =========================================================================

    selected = database[
        (
            database["player_id"].astype(str)
            == str(args.player_id)
        )
        &
        (
            database["session_id"].astype(str)
            == str(args.session_id)
        )
    ].copy()

    if selected.empty:

        print(
            "ERROR: No swings found for "
            f"player={args.player_id}, "
            f"session={args.session_id}"
        )

        sys.exit(1)

    selected = (
        selected
        .drop_duplicates(
            subset=["video_id"]
        )
        .reset_index(drop=True)
    )

    # =========================================================================
    # HEADER
    # =========================================================================

    print()
    print("=" * 88)
    print("BASEVISIONAI - BATCH SWING FEEDBACK V1")
    print("=" * 88)

    print()
    print("SELECTION")
    print("-" * 88)

    print(
        f"Player ID                 : "
        f"{args.player_id}"
    )

    print(
        f"Session ID                : "
        f"{args.session_id}"
    )

    print(
        f"Swings discovered         : "
        f"{len(selected)}"
    )

    print(
        f"Output directory          : "
        f"{output_dir.resolve()}"
    )

    # =========================================================================
    # PROCESS EACH SWING
    # =========================================================================

    batch_rows = []

    successful = 0
    failed = 0

    for index, row in selected.iterrows():

        video_id = str(
            row["video_id"]
        )

        print()
        print("=" * 88)
        print(
            f"[{index + 1}/{len(selected)}] "
            f"{video_id}"
        )
        print("=" * 88)

        command = [
            sys.executable,
            str(feedback_script),

            "--database",
            str(database_path),

            "--validity",
            str(validity_path),

            "--video-id",
            video_id,

            "--player-id",
            str(args.player_id),

            "--session-id",
            str(args.session_id),

            "--output-dir",
            str(output_dir),
        ]

        success, reason = run_command(
            command
        )

        if success:
            successful += 1
            status = "COMPLETE"
        else:
            failed += 1
            status = "FAILED"

        batch_rows.append(
            {
                "video_id": video_id,
                "status": status,
                "reason": reason,
            }
        )

    # =========================================================================
    # COMBINE OUTPUT FILES
    # =========================================================================

    detailed_frames = []
    summary_frames = []

    for video_id in selected["video_id"].astype(str):

        detailed_file = (
            output_dir
            / f"{video_id}_swing_feedback_v1_1.csv"
        )

        summary_file = (
            output_dir
            / f"{video_id}_swing_feedback_v1_1_summary.csv"
        )

        if detailed_file.exists():

            try:
                df = pd.read_csv(
                    detailed_file
                )

                detailed_frames.append(
                    df
                )

            except Exception as exc:

                print(
                    f"WARNING: Could not read "
                    f"{detailed_file}: {exc}"
                )

        if summary_file.exists():

            try:
                df = pd.read_csv(
                    summary_file
                )

                summary_frames.append(
                    df
                )

            except Exception as exc:

                print(
                    f"WARNING: Could not read "
                    f"{summary_file}: {exc}"
                )

    # =========================================================================
    # COMBINED FEEDBACK DATABASE
    # =========================================================================

    combined_feedback_path = (
        output_dir
        / "swing_feedback_database_v1.csv"
    )

    if detailed_frames:

        combined_feedback = pd.concat(
            detailed_frames,
            ignore_index=True,
        )

        combined_feedback.to_csv(
            combined_feedback_path,
            index=False,
        )

    else:

        combined_feedback = pd.DataFrame()

    # =========================================================================
    # COMBINED SUMMARY DATABASE
    # =========================================================================

    combined_summary_path = (
        output_dir
        / "swing_feedback_summary_database_v1.csv"
    )

    if summary_frames:

        combined_summary = pd.concat(
            summary_frames,
            ignore_index=True,
        )

        combined_summary.to_csv(
            combined_summary_path,
            index=False,
        )

    else:

        combined_summary = pd.DataFrame()

    # =========================================================================
    # BATCH SUMMARY
    # =========================================================================

    batch_summary = pd.DataFrame(
        batch_rows
    )

    batch_summary_path = (
        output_dir
        / "batch_swing_feedback_v1_summary.csv"
    )

    batch_summary.to_csv(
        batch_summary_path,
        index=False,
    )

    # =========================================================================
    # FINAL REPORT
    # =========================================================================

    print()
    print("=" * 88)
    print(
        "BASEVISIONAI - "
        "BATCH SWING FEEDBACK V1 COMPLETE"
    )
    print("=" * 88)

    print()
    print("BATCH RESULTS")
    print("-" * 88)

    print(
        f"Swings discovered         : "
        f"{len(selected)}"
    )

    print(
        f"Successful                : "
        f"{successful}"
    )

    print(
        f"Failed                    : "
        f"{failed}"
    )

    print()
    print("COMBINED DATABASE")
    print("-" * 88)

    print(
        f"Feedback rows             : "
        f"{len(combined_feedback)}"
    )

    print(
        f"Summary rows              : "
        f"{len(combined_summary)}"
    )

    print()
    print("OUTPUT FILES")
    print("-" * 88)

    print(
        f"Feedback database         : "
        f"{combined_feedback_path.resolve()}"
    )

    print(
        f"Summary database          : "
        f"{combined_summary_path.resolve()}"
    )

    print(
        f"Batch summary             : "
        f"{batch_summary_path.resolve()}"
    )

    print()
    print("IMPORTANT")
    print("-" * 88)

    print(
        "Every swing receives an independent "
        "session-relative comparison."
    )

    print(
        "VALID_SWING profiles define the baseline."
    )

    print(
        "LOW_CONFIDENCE_SWING profiles may be analyzed "
        "but do not contribute to the baseline."
    )

    print(
        "Feedback remains comparative 2D analysis "
        "and is not a biomechanical performance score."
    )

    print("=" * 88)


if __name__ == "__main__":
    main()