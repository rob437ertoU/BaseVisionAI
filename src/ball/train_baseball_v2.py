from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_YAML = (
    PROJECT_ROOT
    / "data"
    / "baseball_v2"
    / "data.yaml"
)

RUNS_DIR = (
    PROJECT_ROOT
    / "runs"
    / "baseball"
)


def main():

    print()
    print("=" * 78)
    print("BaseVisionAI - Baseball Detector V2 Training")
    print("=" * 78)
    print(f"Dataset : {DATA_YAML}")
    print("Model   : yolo11n.pt")
    print("Image   : 1280")
    print("=" * 78)

    model = YOLO(
        "yolo11n.pt"
    )

    model.train(
        data=str(DATA_YAML),

        epochs=120,
        patience=25,

        imgsz=1280,
        batch=4,

        device=0,
        workers=4,

        project=str(RUNS_DIR),
        name="baseball_v2_yolo11n",

        optimizer="AdamW",

        lr0=0.0005,
        lrf=0.01,

        weight_decay=0.0005,

        seed=42,
        deterministic=True,

        amp=True,

        # ====================================================
        # MODERATE ONLINE AUGMENTATION
        # ====================================================

        hsv_h=0.015,
        hsv_s=0.40,
        hsv_v=0.30,

        degrees=2.0,
        translate=0.04,
        scale=0.12,

        shear=0.0,
        perspective=0.0,

        flipud=0.0,
        fliplr=0.0,

        mosaic=0.15,
        mixup=0.0,

        close_mosaic=10,

        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()