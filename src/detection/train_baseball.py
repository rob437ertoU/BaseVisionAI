from ultralytics import YOLO
from pathlib import Path
import torch
from multiprocessing import freeze_support


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = PROJECT_ROOT / "data" / "baseball_v1" / "data.yaml"


def main():

    print("=" * 60)
    print("BaseVisionAI - Baseball Detector Training")
    print("=" * 60)

    print(f"Dataset: {DATA_YAML}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print("\nLoading pretrained YOLO model...")

    model = YOLO("yolo26n.pt")

    print("Starting training...\n")

    results = model.train(
        data=str(DATA_YAML),

        # Training
        epochs=100,
        patience=20,

        # Small-object detection
        imgsz=1280,

        # RTX 4060 8GB
        batch=4,
        device=0,

        # Windows DataLoader
        workers=4,

        # Output
        project=str(PROJECT_ROOT / "runs" / "baseball"),
        name="baseball_v1",

        # Performance
        amp=True,
        cache=False,

        # Reproducibility
        seed=42,
        deterministic=True,

        # Logging
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(
        "Results:",
        PROJECT_ROOT / "runs" / "baseball" / "baseball_v1"
    )


if __name__ == "__main__":
    freeze_support()
    main()