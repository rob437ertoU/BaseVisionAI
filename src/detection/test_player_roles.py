from pathlib import Path
import argparse

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_MODEL = (
    PROJECT_ROOT
    / "runs"
    / "player_roles"
    / "player_roles_v1_1"
    / "weights"
    / "best.pt"
)

DEFAULT_VIDEO = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "baseball_test4.mp4"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "player_roles_v1_1_validation.mp4"
)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Visualize Player Role Detection"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help="Path to trained YOLO model"
    )

    parser.add_argument(
        "--video",
        type=str,
        default=str(DEFAULT_VIDEO),
        help="Input video"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output annotated video"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Detection confidence threshold"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Inference image size"
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    model_path = Path(args.model)
    video_path = Path(args.video)
    output_path = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found:\n{video_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("BaseVisionAI - Player Role Validation")
    print("=" * 70)

    print(f"Model  : {model_path}")
    print(f"Video  : {video_path}")
    print(f"Output : {output_path}")
    print(f"Conf   : {args.conf}")
    print(f"Image  : {args.imgsz}")

    print("=" * 70)

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = YOLO(str(model_path))

    print()
    print("Classes:")
    print(model.names)
    print()

    # ========================================================
    # VIDEO
    # ========================================================

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    frame_number = 0

    # ========================================================
    # INFERENCE LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        results = model.predict(
            source=frame,
            conf=args.conf,
            imgsz=args.imgsz,
            device=0,
            verbose=False
        )

        result = results[0]

        annotated_frame = result.plot()

        # ----------------------------------------------------
        # Frame information
        # ----------------------------------------------------

        cv2.putText(
            annotated_frame,
            f"Frame: {frame_number}/{total_frames}",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        writer.write(
            annotated_frame
        )

        if frame_number % 30 == 0:

            print(
                f"Processed "
                f"{frame_number}/{total_frames}"
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    writer.release()

    print()
    print("=" * 70)
    print("Finished")
    print("=" * 70)

    print(
        f"Saved to:\n{output_path}"
    )


if __name__ == "__main__":
    main()