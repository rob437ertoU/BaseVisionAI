from ultralytics import YOLO
from pathlib import Path
import cv2
import csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "baseball"
    / "baseball_v1-3"
    / "weights"
    / "best.pt"
)

VIDEO_PATH = PROJECT_ROOT / "data" / "raw" / "baseball_test.mp4"

OUTPUT_VIDEO = PROJECT_ROOT / "results" / "ball_tracking_v1.mp4"
OUTPUT_CSV = PROJECT_ROOT / "results" / "ball_tracking_v1.csv"


def main():

    print("=" * 60)
    print("BaseVisionAI - Ball Tracker V1")
    print("=" * 60)

    print(f"Model: {MODEL_PATH}")
    print(f"Video: {VIDEO_PATH}")

    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height)
    )

    trajectory = []

    csv_file = open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    )

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(
        [
            "frame",
            "x_center",
            "y_center",
            "confidence"
        ]
    )

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        results = model.predict(
            frame,
            imgsz=1280,
            conf=0.15,
            device=0,
            verbose=False
        )

        boxes = results[0].boxes

        best_detection = None
        best_confidence = 0

        if boxes is not None:

            for box in boxes:

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                if confidence > best_confidence:

                    best_confidence = confidence

                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    best_detection = (
                        x1,
                        y1,
                        x2,
                        y2,
                        center_x,
                        center_y,
                        confidence
                    )

        if best_detection is not None:

            (
                x1,
                y1,
                x2,
                y2,
                center_x,
                center_y,
                confidence
            ) = best_detection

            trajectory.append(
                (center_x, center_y)
            )

            csv_writer.writerow(
                [
                    frame_number,
                    center_x,
                    center_y,
                    confidence
                ]
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (255, 255, 255),
                -1
            )

            cv2.putText(
                frame,
                f"Baseball {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # Dibujar trayectoria
        if len(trajectory) > 1:

            for i in range(1, len(trajectory)):

                cv2.line(
                    frame,
                    trajectory[i - 1],
                    trajectory[i],
                    (255, 255, 255),
                    2
                )

        writer.write(frame)

        print(
            f"Frame {frame_number} | "
            f"Detected: {best_detection is not None} | "
            f"Confidence: {best_confidence:.2f}"
        )

    cap.release()
    writer.release()
    csv_file.close()

    print("\n" + "=" * 60)
    print("TRACKING COMPLETED")
    print("=" * 60)

    print(f"Video saved to: {OUTPUT_VIDEO}")
    print(f"CSV saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()