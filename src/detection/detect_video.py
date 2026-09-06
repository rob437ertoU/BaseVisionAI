from ultralytics import YOLO
import cv2
from pathlib import Path

# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_VIDEO = PROJECT_ROOT / "data" / "raw" / "baseball_test.mp4"
OUTPUT_DIR = PROJECT_ROOT / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Load YOLO
# =========================
print("Loading YOLO model...")

model = YOLO("yolo26n.pt")

# =========================
# Open video
# =========================
cap = cv2.VideoCapture(str(INPUT_VIDEO))

if not cap.isOpened():
    raise FileNotFoundError(
        f"Could not open video: {INPUT_VIDEO}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {INPUT_VIDEO.name}")
print(f"Resolution: {width}x{height}")
print(f"FPS: {fps}")
print(f"Frames: {total_frames}")

# =========================
# Output video
# =========================
output_path = OUTPUT_DIR / "baseball_detection.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(output_path),
    fourcc,
    fps,
    (width, height)
)

# =========================
# Process frames
# =========================
frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    results = model.predict(
        frame,
        device=0,
        conf=0.55,
        verbose=False
    )

    annotated_frame = results[0].plot()

    writer.write(annotated_frame)

    print(
        f"\rProcessing frame "
        f"{frame_number}/{total_frames}",
        end=""
    )

cap.release()
writer.release()

print("\n")
print("Detection completed.")
print(f"Output saved to: {output_path}")