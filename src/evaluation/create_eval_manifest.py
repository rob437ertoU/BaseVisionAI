from pathlib import Path
import argparse

import cv2
import pandas as pd


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Create Independent Evaluation Manifest"
    )

    parser.add_argument(
        "--input-dir",
        default="data/evaluation",
        help="Folder containing independent evaluation videos",
    )

    parser.add_argument(
        "--output",
        default="data/evaluation/evaluation_manifest.csv",
        help="Output CSV manifest",
    )

    return parser.parse_args()


def inspect_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {
            "valid_video": False,
            "fps": None,
            "frames": None,
            "duration_sec": None,
            "width": None,
            "height": None,
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    duration = (
        frames / fps
        if fps is not None and fps > 0
        else None
    )

    return {
        "valid_video": True,
        "fps": fps,
        "frames": frames,
        "duration_sec": duration,
        "width": width,
        "height": height,
    }


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input folder does not exist: {input_dir}"
        )

    videos = sorted(
        [
            path
            for path in input_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
        ]
    )

    if len(videos) == 0:
        raise RuntimeError(
            f"No evaluation videos found in {input_dir}"
        )

    rows = []

    print()
    print("=" * 80)
    print("BaseVisionAI - Independent Evaluation Dataset")
    print("=" * 80)

    for index, video in enumerate(videos, start=1):
        info = inspect_video(video)

        print()
        print(f"[{index}/{len(videos)}] {video.name}")

        if info["valid_video"]:
            print(f"    Frames     : {info['frames']}")
            print(f"    FPS        : {info['fps']:.3f}")
            print(f"    Duration   : {info['duration_sec']:.3f} s")
            print(
                f"    Resolution : "
                f"{info['width']}x{info['height']}"
            )
        else:
            print("    ERROR: video could not be opened")

        rows.append(
            {
                # ==========================================
                # VIDEO INFORMATION
                # ==========================================
                "video_id": video.stem,
                "video_path": str(video.as_posix()),
                "valid_video": info["valid_video"],
                "fps": info["fps"],
                "frames": info["frames"],
                "duration_sec": info["duration_sec"],
                "width": info["width"],
                "height": info["height"],

                # ==========================================
                # MANUAL GROUND TRUTH
                # Fill BEFORE running BaseVisionAI
                # ==========================================
                "gt_pitch_result": "",
                "gt_contact_type": "",
                "gt_contact_frame": "",
                "gt_notes": "",

                # ==========================================
                # BASEVISIONAI RESULTS
                # Populated later
                # ==========================================
                "ball_real_yolo_points": "",
                "ball_track_available": "",
                "strike_zone_available": "",
                "crossing_available": "",
                "crossing_frame": "",
                "crossing_y": "",
                "crossing_extrapolation_frames": "",
                "crossing_residual_px": "",
                "horizontal_inside": "",
                "vertical_inside": "",
                "pred_pitch_result": "",
                "evidence_quality": "",

                # ==========================================
                # CONTACT FUSION RESULTS
                # ==========================================
                "contact_candidate_frame": "",
                "contact_evidence_score": "",
                "contact_classification": "",
                "contact_evidence_quality": "",

                # ==========================================
                # FINAL EVALUATION
                # ==========================================
                "pitch_result_correct": "",
                "contact_result_correct": "",
                "failure_stage": "",
            }
        )

    df = pd.DataFrame(rows)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("MANIFEST CREATED")
    print("=" * 80)

    print(f"Videos detected : {len(df)}")
    print(f"Output          : {output_path}")

    print()
    print("Evaluation videos:")

    for video_id in df["video_id"]:
        print(f"  - {video_id}")

    print()
    print("IMPORTANT:")
    print(
        "Fill the manual ground-truth columns BEFORE "
        "running BaseVisionAI on these videos."
    )

    print()
    print("Ground-truth columns:")
    print("  gt_pitch_result")
    print("  gt_contact_type")
    print("  gt_contact_frame")
    print("  gt_notes")

    print("=" * 80)


if __name__ == "__main__":
    main()