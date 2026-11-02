from pathlib import Path
import cv2
import shutil


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "player_roles_dataset_v2"
)


# ============================================================
# DATASET SPLIT
# ============================================================
#
# IMPORTANT:
#
# Training and validation are separated BY VIDEO.
#
# baseball_test5.mp4 is intentionally NOT included.
# It remains the final held-out test.
# ============================================================


TRAIN_VIDEOS = [

    # Previous training clips
    "baseball_test1.mp4",
    "baseball_test2.mp4",
    "baseball_test3.mp4",

    # New independent training clips
    "baseball_train1.mp4",
    "baseball_train2.mp4",
    "baseball_train3.mp4",
    "baseball_train4.mp4",
    "baseball_train5.mp4",
    "baseball_train6.mp4",
    "baseball_train7.mp4",
    "baseball_train8.mp4",
]


VALID_VIDEOS = [

    # Previous validation clip
    "baseball_test4.mp4",

    # New independent validation clips
    "baseball_train9.mp4",
    "baseball_train10.mp4",
]


# ============================================================
# EXTRACTION SETTINGS
# ============================================================

# Extract approximately 3 frames per second.
#
# Example:
# 30 FPS source -> every 10 frames
# 60 FPS source -> every 20 frames

TARGET_FPS = 3.0

JPEG_QUALITY = 95


# ============================================================
# EXTRACT ONE VIDEO
# ============================================================

def extract_video(
    video_path: Path,
    output_folder: Path,
    target_fps: float
):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        print()
        print("[ERROR] Could not open video:")
        print(video_path)

        return 0

    # --------------------------------------------------------
    # Video information
    # --------------------------------------------------------

    source_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if source_fps <= 0:

        print()
        print(
            f"[ERROR] Invalid FPS for "
            f"{video_path.name}"
        )

        cap.release()

        return 0

    duration_seconds = (
        total_frames
        /
        source_fps
    )

    # --------------------------------------------------------
    # Determine sampling interval
    # --------------------------------------------------------

    frame_interval = max(
        1,
        round(
            source_fps
            /
            target_fps
        )
    )

    # --------------------------------------------------------
    # Create folder
    # --------------------------------------------------------

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    print()
    print("-" * 70)

    print(
        f"Video          : "
        f"{video_path.name}"
    )

    print(
        f"Resolution     : "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}"
        f"x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    )

    print(
        f"Source FPS     : "
        f"{source_fps:.2f}"
    )

    print(
        f"Total frames   : "
        f"{total_frames}"
    )

    print(
        f"Duration       : "
        f"{duration_seconds:.2f} s"
    )

    print(
        f"Target FPS     : "
        f"{target_fps:.2f}"
    )

    print(
        f"Frame interval : "
        f"{frame_interval}"
    )

    print(
        f"Output folder  : "
        f"{output_folder}"
    )

    # ========================================================
    # FRAME EXTRACTION
    # ========================================================

    frame_index = 0

    saved_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Human-readable frame number.
        current_frame = frame_index + 1

        # ----------------------------------------------------
        # Save selected frame
        # ----------------------------------------------------

        if frame_index % frame_interval == 0:

            filename = (
                f"{video_path.stem}"
                f"_frame_"
                f"{current_frame:05d}"
                f".jpg"
            )

            output_path = (
                output_folder
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

            else:

                print(
                    f"[WARNING] Could not save: "
                    f"{output_path}"
                )

        frame_index += 1

    cap.release()

    print(
        f"Saved images   : "
        f"{saved_count}"
    )

    return saved_count


# ============================================================
# PROCESS DATASET SPLIT
# ============================================================

def process_split(
    split_name: str,
    video_names: list[str]
):

    split_dir = (
        OUTPUT_DIR
        /
        split_name
    )

    total_saved = 0

    successful_videos = 0

    missing_videos = 0

    print()
    print("=" * 70)

    print(
        f"PROCESSING SPLIT: "
        f"{split_name.upper()}"
    )

    print("=" * 70)

    for video_name in video_names:

        video_path = (
            RAW_DIR
            /
            video_name
        )

        # ----------------------------------------------------
        # Check video exists
        # ----------------------------------------------------

        if not video_path.exists():

            print()
            print(
                "[WARNING] Missing video:"
            )

            print(
                video_path
            )

            print(
                "Skipping..."
            )

            missing_videos += 1

            continue

        # ----------------------------------------------------
        # Folder for this specific video
        # ----------------------------------------------------

        video_output_folder = (
            split_dir
            /
            video_path.stem
        )

        saved = extract_video(
            video_path=video_path,
            output_folder=video_output_folder,
            target_fps=TARGET_FPS
        )

        if saved > 0:
            successful_videos += 1

        total_saved += saved

    return {
        "images": total_saved,
        "videos": successful_videos,
        "missing": missing_videos
    }


# ============================================================
# VERIFY SPLIT
# ============================================================

def verify_split():

    train_set = set(
        TRAIN_VIDEOS
    )

    valid_set = set(
        VALID_VIDEOS
    )

    # --------------------------------------------------------
    # Check duplicates between train and validation
    # --------------------------------------------------------

    overlap = (
        train_set
        &
        valid_set
    )

    if overlap:

        raise RuntimeError(
            "\nERROR:\n"
            "The same video appears in TRAIN "
            "and VALID:\n"
            f"{sorted(overlap)}"
        )

    # --------------------------------------------------------
    # Protect final test video
    # --------------------------------------------------------

    forbidden_video = (
        "baseball_test5.mp4"
    )

    if (
        forbidden_video in train_set
        or
        forbidden_video in valid_set
    ):

        raise RuntimeError(
            "\nERROR:\n"
            "baseball_test5.mp4 is the final "
            "held-out test and cannot appear "
            "in TRAIN or VALID."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "BaseVisionAI"
    )

    print(
        "Player Roles Dataset V2"
    )

    print("=" * 70)

    print()
    print(
        f"Project root:"
    )

    print(
        PROJECT_ROOT
    )

    print()
    print(
        f"Raw video folder:"
    )

    print(
        RAW_DIR
    )

    print()
    print(
        f"Dataset output:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    # ========================================================
    # VERIFY DATASET DESIGN
    # ========================================================

    verify_split()

    print(
        f"TRAIN videos configured : "
        f"{len(TRAIN_VIDEOS)}"
    )

    print(
        f"VALID videos configured : "
        f"{len(VALID_VIDEOS)}"
    )

    print()

    print(
        "Final held-out test:"
    )

    print(
        "baseball_test5.mp4"
    )

    # ========================================================
    # CLEAN PREVIOUS V2 EXTRACTION
    # ========================================================

    if OUTPUT_DIR.exists():

        print()
        print(
            "Existing V2 extraction found."
        )

        print(
            "Removing previous extracted frames..."
        )

        shutil.rmtree(
            OUTPUT_DIR
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # TRAIN
    # ========================================================

    train_stats = process_split(
        split_name="train",
        video_names=TRAIN_VIDEOS
    )

    # ========================================================
    # VALID
    # ========================================================

    valid_stats = process_split(
        split_name="valid",
        video_names=VALID_VIDEOS
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    total_images = (
        train_stats["images"]
        +
        valid_stats["images"]
    )

    total_videos = (
        train_stats["videos"]
        +
        valid_stats["videos"]
    )

    print()
    print("=" * 70)

    print(
        "DATASET V2 EXTRACTION COMPLETE"
    )

    print("=" * 70)

    print()

    print("TRAIN")

    print(
        f"  Videos : "
        f"{train_stats['videos']}"
    )

    print(
        f"  Images : "
        f"{train_stats['images']}"
    )

    if train_stats["missing"]:

        print(
            f"  Missing: "
            f"{train_stats['missing']}"
        )

    print()

    print("VALIDATION")

    print(
        f"  Videos : "
        f"{valid_stats['videos']}"
    )

    print(
        f"  Images : "
        f"{valid_stats['images']}"
    )

    if valid_stats["missing"]:

        print(
            f"  Missing: "
            f"{valid_stats['missing']}"
        )

    print()

    print(
        f"TOTAL VIDEOS : "
        f"{total_videos}"
    )

    print(
        f"TOTAL IMAGES : "
        f"{total_images}"
    )

    print()

    print("-" * 70)

    print(
        "FINAL TEST REMAINS HELD OUT:"
    )

    print(
        "baseball_test5.mp4"
    )

    print("-" * 70)

    print()

    print(
        "Next step:"
    )

    print(
        "Inspect the extracted images before "
        "uploading them to Roboflow."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()