from ultralytics import YOLO
from pathlib import Path

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

print("=" * 60)
print("BaseVisionAI - Baseball Detection Test")
print("=" * 60)

print(f"Model: {MODEL_PATH}")
print(f"Video: {VIDEO_PATH}")

model = YOLO(str(MODEL_PATH))

results = model.predict(
    source=str(VIDEO_PATH),
    imgsz=1280,
    conf=0.15,
    device=0,
    save=True,
    project=str(PROJECT_ROOT / "results"),
    name="baseball_v1_test",
    verbose=True
)

print("\nDetection completed.")
print(
    "Result saved in:",
    PROJECT_ROOT / "results" / "baseball_v1_test"
)