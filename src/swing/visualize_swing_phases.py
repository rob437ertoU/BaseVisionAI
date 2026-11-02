from pathlib import Path

import cv2
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = PROJECT_ROOT / "data" / "raw" / "baseball_test.mp4"

PHASES_CSV = (
    PROJECT_ROOT
    / "results"
    / "swing_phases_v1_1.csv"
)

OUTPUT_VIDEO = (
    PROJECT_ROOT
    / "results"
    / "swing_phases_visualization_v1.mp4"
)


def put_text(
    frame,
    text,
    position,
    scale=0.7,
    thickness=2
):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


def main():

    print("=" * 80)
    print("BaseVisionAI - Swing Phase Visualization V1")
    print("=" * 80)

    print(f"\nVideo:\n{VIDEO_PATH}")
    print(f"\nPhases:\n{PHASES_CSV}")

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found:\n{VIDEO_PATH}"
        )

    if not PHASES_CSV.exists():
        raise FileNotFoundError(
            f"Phase CSV not found:\n{PHASES_CSV}"
        )

    df = pd.read_csv(PHASES_CSV)

    df = (
        df
        .sort_values("frame")
        .reset_index(drop=True)
    )

    phase_lookup = {}

    for _, row in df.iterrows():

        frame_id = int(row["frame"])

        phase_lookup[frame_id] = {
            "phase": row["swing_phase"],

            "swing_activity":
                row.get(
                    "swing_activity_score",
                    0.0
                ),

            "leg_activity":
                row.get(
                    "leg_activity_score",
                    0.0
                ),

            "peak_score":
                row.get(
                    "peak_swing_score",
                    0.0
                ),

            "contact_candidate":
                bool(
                    row.get(
                        "contact_candidate",
                        False
                    )
                )
        }

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open input video."
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
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

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    OUTPUT_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height)
    )

    print(
        f"\nVideo info:"
        f"\nFPS: {fps:.2f}"
        f"\nResolution: {width}x{height}"
        f"\nFrames: {total_frames}"
    )

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        info = phase_lookup.get(
            frame_number,
            None
        )

        overlay = frame.copy()

        panel_width = 430
        panel_height = 220

        cv2.rectangle(
            overlay,
            (20, 20),
            (
                20 + panel_width,
                20 + panel_height
            ),
            (0, 0, 0),
            -1
        )

        alpha = 0.65

        frame = cv2.addWeighted(
            overlay,
            alpha,
            frame,
            1 - alpha,
            0
        )

        put_text(
            frame,
            "BaseVision AI",
            (40, 55),
            scale=0.9
        )

        put_text(
            frame,
            f"Frame: {frame_number}",
            (40, 90)
        )

        if info is not None:

            phase = str(
                info["phase"]
            )

            swing_score = info[
                "swing_activity"
            ]

            leg_score = info[
                "leg_activity"
            ]

            peak_score = info[
                "peak_score"
            ]

            contact_candidate = info[
                "contact_candidate"
            ]

            put_text(
                frame,
                f"Phase: {phase}",
                (40, 125),
                scale=0.8
            )

            put_text(
                frame,
                f"Swing activity: {swing_score:.2f}",
                (40, 160),
                scale=0.65
            )

            put_text(
                frame,
                f"Leg activity: {leg_score:.2f}",
                (40, 190),
                scale=0.65
            )

            put_text(
                frame,
                f"Peak score: {peak_score:.2f}",
                (40, 220),
                scale=0.65
            )

            if contact_candidate:

                cv2.rectangle(
                    frame,
                    (
                        width - 420,
                        35
                    ),
                    (
                        width - 35,
                        120
                    ),
                    (0, 0, 255),
                    -1
                )

                put_text(
                    frame,
                    "CONTACT CANDIDATE",
                    (
                        width - 390,
                        88
                    ),
                    scale=0.9,
                    thickness=3
                )

        writer.write(frame)

        if (
            frame_number % 20
            == 0
        ):
            print(
                f"Processed "
                f"{frame_number}/"
                f"{total_frames}"
            )

    cap.release()
    writer.release()

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETED")
    print("=" * 80)

    print(
        f"\nOutput:\n{OUTPUT_VIDEO}"
    )


if __name__ == "__main__":
    main()