from pathlib import Path
import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "baseball_dataset_v2_source"
)

VIDEOS = [
    f"baseball_train{i}.mp4"
    for i in range(1, 11)
]

GENERAL_SAMPLE_EVERY = 10

# Dense sampling in final portion of each video.
DENSE_SAMPLE_EVERY = 3

DENSE_START_RATIO = 0.55


def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    total_saved = 0

    print()
    print("=" * 70)
    print("BaseVisionAI - Baseball Dataset V2 Extractor")
    print("=" * 70)

    for video_name in VIDEOS:

        video_path = RAW_DIR / video_name

        if not video_path.exists():

            print(
                f"[SKIP] Missing: {video_path}"
            )
            continue

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():

            print(
                f"[ERROR] Could not open: {video_name}"
            )
            continue

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        fps = float(
            cap.get(
                cv2.CAP_PROP_FPS
            )
        )

        dense_start = int(
            total_frames
            * DENSE_START_RATIO
        )

        output_dir = (
            OUTPUT_ROOT
            / Path(video_name).stem
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        frame_number = 0
        saved_video = 0

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            frame_number += 1

            if frame_number >= dense_start:

                should_save = (
                    frame_number
                    % DENSE_SAMPLE_EVERY
                    == 0
                )

            else:

                should_save = (
                    frame_number
                    % GENERAL_SAMPLE_EVERY
                    == 0
                )

            if not should_save:
                continue

            output_name = (
                f"{Path(video_name).stem}"
                f"_frame_{frame_number:04d}.jpg"
            )

            output_path = (
                output_dir
                / output_name
            )

            cv2.imwrite(
                str(output_path),
                frame
            )

            saved_video += 1
            total_saved += 1

        cap.release()

        print(
            f"{video_name:25s} | "
            f"{total_frames:4d} frames | "
            f"{fps:5.1f} fps | "
            f"saved {saved_video:3d}"
        )

    print()
    print("=" * 70)
    print(
        f"TOTAL IMAGES SAVED: {total_saved}"
    )

    print(
        f"Output: {OUTPUT_ROOT}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()