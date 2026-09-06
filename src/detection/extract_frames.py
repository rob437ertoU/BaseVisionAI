import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = PROJECT_ROOT / "data" / "raw" / "baseball_test.mp4"
OUTPUT_DIR = PROJECT_ROOT / "data" / "dataset" / "images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise FileNotFoundError(f"No se pudo abrir: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS: {fps}")
print(f"Frames totales: {total_frames}")

frame_id = 0
saved = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Guardamos todos los frames por ahora
    filename = OUTPUT_DIR / f"frame_{frame_id:04d}.jpg"
    cv2.imwrite(str(filename), frame)

    saved += 1
    frame_id += 1

cap.release()

print(f"\nFrames guardados: {saved}")
print(f"Carpeta: {OUTPUT_DIR}")