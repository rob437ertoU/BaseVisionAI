from pathlib import Path
import cv2


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_VIDEO_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "player_roles_dataset"
)


# ============================================================
# DATASET SPLIT
# ============================================================

DATASET_SPLITS = {

    "train": [
        "baseball_test1.mp4",
        "baseball_test2.mp4",
        "baseball_test3.mp4",
    ],

    "valid": [
        "baseball_test4.mp4",
    ],

    "test": [
        "baseball_test5.mp4",
    ],
}


# ============================================================
# EXTRACTION SETTINGS
# ============================================================

# Extract 1 frame every N original frames.
#
# Videos are ~30 FPS.
#
# FRAME_STEP = 3
# -> approximately 10 extracted images per second.
FRAME_STEP = 3


# JPEG quality
JPEG_QUALITY = 95


# ============================================================
# EXTRACT ONE VIDEO
# ============================================================

def extract_video_frames(
    video_path,
    output_dir,
    frame_step=3
):
    """
    Extract frames from one video.

    Frames are sampled every `frame_step` frames.

    Example:
        frame_step = 3
        original video = 30 FPS
        output ~= 10 FPS
    """

    print("=" * 70)
    print(f"Processing: {video_path.name}")
    print("=" * 70)

    if not video_path.exists():

        print(
            f"ERROR: Video not found:\n"
            f"{video_path}"
        )

        return 0

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            f"ERROR: Could not open video:\n"
            f"{video_path}"
        )

        return 0

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print(
        f"FPS: {fps:.2f}"
    )

    print(
        f"Resolution: "
        f"{width}x{height}"
    )

    print(
        f"Total frames: "
        f"{total_frames}"
    )

    frame_number = 0
    saved_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # ====================================================
        # SAMPLE FRAME
        # ====================================================

        if (
            (frame_number - 1)
            %
            frame_step
            !=
            0
        ):
            continue

        # ====================================================
        # OUTPUT NAME
        # ====================================================

        filename = (
            f"{video_path.stem}"
            f"_frame_{frame_number:04d}.jpg"
        )

        output_path = (
            output_dir
            /
            filename
        )

        success = cv2.imwrite(
            str(output_path),
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                JPEG_QUALITY
            ]
        )

        if success:

            saved_count += 1

            print(
                f"Saved: "
                f"{filename}"
            )

        else:

            print(
                f"WARNING: Could not save "
                f"frame {frame_number}"
            )

    cap.release()

    print()

    print(
        f"Finished {video_path.name}"
    )

    print(
        f"Extracted images: "
        f"{saved_count}"
    )

    print()

    return saved_count


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BaseVisionAI - Player Roles Dataset Extraction")
    print("=" * 70)

    print(
        f"Input directory:\n"
        f"{RAW_VIDEO_DIR}"
    )

    print()

    print(
        f"Output directory:\n"
        f"{OUTPUT_ROOT}"
    )

    print()

    print(
        f"Frame step: "
        f"{FRAME_STEP}"
    )

    print()

    total_images = 0

    split_counts = {}

    # ========================================================
    # PROCESS SPLITS
    # ========================================================

    for (
        split_name,
        video_names
    ) in DATASET_SPLITS.items():

        print()
        print("#" * 70)
        print(
            f"SPLIT: "
            f"{split_name.upper()}"
        )
        print("#" * 70)
        print()

        split_total = 0

        for video_name in video_names:

            video_path = (
                RAW_VIDEO_DIR
                /
                video_name
            )

            # ------------------------------------------------
            # Keep each video in its own folder.
            # ------------------------------------------------

            video_output_dir = (
                OUTPUT_ROOT
                /
                split_name
                /
                Path(video_name).stem
            )

            extracted = (
                extract_video_frames(
                    video_path=
                        video_path,

                    output_dir=
                        video_output_dir,

                    frame_step=
                        FRAME_STEP
                )
            )

            split_total += (
                extracted
            )

            total_images += (
                extracted
            )

        split_counts[
            split_name
        ] = split_total

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("DATASET EXTRACTION COMPLETED")
    print("=" * 70)

    print()

    for split_name in [
        "train",
        "valid",
        "test"
    ]:

        count = (
            split_counts.get(
                split_name,
                0
            )
        )

        print(
            f"{split_name.upper():<6}: "
            f"{count} images"
        )

    print()

    print(
        f"TOTAL: "
        f"{total_images} images"
    )

    print()

    print(
        "Dataset saved in:"
    )

    print(
        OUTPUT_ROOT
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()