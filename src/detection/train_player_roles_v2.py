from pathlib import Path
import argparse

from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

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


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Train Player Roles V2 with YOLO11"
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
        default="yolo11n.pt",
        help="Ultralytics pretrained model"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of epochs"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Training image size"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=2,
        help="Batch size"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataloader workers"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Early stopping patience"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="player_roles_v2_yolo11n",
        help="Training run name"
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Resolve dataset path
    # --------------------------------------------------------

    data_path = Path(args.data)

    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    data_path = data_path.resolve()

    if not data_path.exists():
        raise FileNotFoundError(
            f"\ndata.yaml not found:\n{data_path}\n"
        )

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    print("=" * 70)
    print("BaseVisionAI - Player Roles V2")
    print("YOLO11n Training")
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

    # ========================================================
    # LOAD PRETRAINED MODEL
    # ========================================================

    model = YOLO(args.model)

    # ========================================================
    # TRAIN
    # ========================================================

    model.train(

        # ----------------------------------------------------
        # DATASET
        # ----------------------------------------------------

        data=str(data_path),

        # ----------------------------------------------------
        # TRAINING PARAMETERS
        # ----------------------------------------------------

        epochs=args.epochs,

        imgsz=args.imgsz,

        batch=args.batch,

        device=args.device,

        workers=args.workers,

        patience=args.patience,

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        project=str(DEFAULT_PROJECT),

        name=args.name,

        exist_ok=False,

        save=True,

        plots=True,

        verbose=True,

        # ----------------------------------------------------
        # TRANSFER LEARNING
        # ----------------------------------------------------

        pretrained=True,

        # ----------------------------------------------------
        # OPTIMIZER
        # ----------------------------------------------------

        optimizer="AdamW",

        lr0=0.0005,

        lrf=0.01,

        weight_decay=0.0005,

        # ----------------------------------------------------
        # REPRODUCIBILITY
        # ----------------------------------------------------

        seed=42,

        deterministic=True,

        # ----------------------------------------------------
        # GPU
        # ----------------------------------------------------

        amp=True,

        # ====================================================
        # MODERATE ONLINE AUGMENTATION
        #
        # Dataset V2 has NO Roboflow augmentation baked in.
        # Therefore we use moderate augmentation here.
        # ====================================================

        # ----------------------------------------------------
        # COLOR
        # ----------------------------------------------------

        hsv_h=0.005,

        hsv_s=0.20,

        hsv_v=0.15,

        # ----------------------------------------------------
        # GEOMETRIC
        # ----------------------------------------------------

        degrees=2.0,

        translate=0.05,

        scale=0.15,

        shear=0.0,

        perspective=0.0,

        # ----------------------------------------------------
        # FLIPS
        # ----------------------------------------------------

        flipud=0.0,

        fliplr=0.0,

        # ----------------------------------------------------
        # COMPOSITE AUGMENTATION
        # ----------------------------------------------------

        mosaic=0.20,

        mixup=0.0,

        copy_paste=0.0,

        # ----------------------------------------------------
        # ERASING
        # ----------------------------------------------------

        erasing=0.0,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()