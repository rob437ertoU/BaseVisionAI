from pathlib import Path

import cv2


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "baseball_test5.mp4"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("BaseVisionAI - Manual Contact Labeler")
    print("=" * 80)

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found:\n{VIDEO_PATH}"
        )

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open video."
        )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    current_frame = 1

    print(
        f"\nVideo: {VIDEO_PATH.name}"
    )

    print(
        f"Frames: {total_frames}"
    )

    print(
        f"FPS: {fps:.2f}"
    )

    print(
        "\nCONTROLS"
    )

    print(
        "D / Right Arrow  -> next frame"
    )

    print(
        "A / Left Arrow   -> previous frame"
    )

    print(
        "J                -> jump -5 frames"
    )

    print(
        "L                -> jump +5 frames"
    )

    print(
        "Q                -> quit"
    )

    while True:

        # ----------------------------------------------------
        # Move video to requested frame
        # OpenCV uses zero-based indexing
        # ----------------------------------------------------

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            current_frame - 1
        )

        ret, frame = cap.read()

        if not ret:
            break

        display = frame.copy()

        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        cv2.rectangle(
            display,
            (20, 20),
            (470, 130),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            display,
            f"Frame: {current_frame}/{total_frames}",
            (40, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        time_seconds = (
            current_frame - 1
        ) / fps

        cv2.putText(
            display,
            f"Time: {time_seconds:.3f} s",
            (40, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            "BaseVisionAI - Manual Contact Labeler",
            display
        )

        key = cv2.waitKey(0)

        # ====================================================
        # QUIT
        # ====================================================

        if key in [
            ord("q"),
            ord("Q"),
            27
        ]:
            break

        # ====================================================
        # NEXT FRAME
        # ====================================================

        elif key in [
            ord("d"),
            ord("D"),
            83
        ]:

            current_frame += 1

        # ====================================================
        # PREVIOUS FRAME
        # ====================================================

        elif key in [
            ord("a"),
            ord("A"),
            81
        ]:

            current_frame -= 1

        # ====================================================
        # +5 FRAMES
        # ====================================================

        elif key in [
            ord("l"),
            ord("L")
        ]:

            current_frame += 5

        # ====================================================
        # -5 FRAMES
        # ====================================================

        elif key in [
            ord("j"),
            ord("J")
        ]:

            current_frame -= 5

        # ----------------------------------------------------
        # Clamp
        # ----------------------------------------------------

        current_frame = max(
            1,
            min(
                total_frames,
                current_frame
            )
        )

    cap.release()
    cv2.destroyAllWindows()

    print(
        "\nLabeler closed."
    )


if __name__ == "__main__":
    main()