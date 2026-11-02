from pathlib import Path
import argparse

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA = (
    PROJECT_ROOT
    / "data"
    / "player_roles_v2"
    / "data.yaml"
)

DEFAULT_PROJECT = (
    PROJECT_ROOT
    / "runs"
    / "player_roles"
)


def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Train Player Roles V2 with YOLO26"
    )

    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA),
        help="Path to data.yaml"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n.pt",
        help="Ultralytics pretrained model"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=2
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=25
    )

    parser.add_argument(
        "--name",
        type=str,
        default="player_roles_v2_yolo26n"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    data_path = Path(args.data)

    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    data_path = data_path.resolve()

    if not data_path.exists():
        raise FileNotFoundError(
            f"\ndata.yaml not found:\n{data_path}\n"
        )

    print("=" * 70)
    print("BaseVisionAI - Player Roles V2")
    print("YOLO26n Training")
    print("=" * 70)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data         : {data_path}")
    print(f"Model        : {args.model}")
    print(f"Epochs       : {args.epochs}")
    print(f"Image size   : {args.imgsz}")
    print(f"Batch        : {args.batch}")
    print(f"Device       : {args.device}")
    print(f"Workers      : {args.workers}")
    print(f"Patience     : {args.patience}")
    print(f"Run name     : {args.name}")

    print("=" * 70)
    print()

    model = YOLO(args.model)

    model.train(
        data=str(data_path),

        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,

        project=str(DEFAULT_PROJECT),
        name=args.name,
        exist_ok=False,

        save=True,
        plots=True,
        verbose=True,

        pretrained=True,

        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        weight_decay=0.0005,

        seed=42,
        deterministic=True,

        amp=True,

        hsv_h=0.005,
        hsv_s=0.20,
        hsv_v=0.15,

        degrees=2.0,
        translate=0.05,
        scale=0.15,

        shear=0.0,
        perspective=0.0,

        flipud=0.0,
        fliplr=0.0,

        mosaic=0.20,

        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,
    )


if __name__ == "__main__":
    main()