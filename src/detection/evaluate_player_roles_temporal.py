from pathlib import Path
import argparse
import csv

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_YOLO11 = (
    PROJECT_ROOT
    / "runs"
    / "player_roles"
    / "player_roles_v2_yolo11n"
    / "weights"
    / "best.pt"
)

DEFAULT_YOLO26 = (
    PROJECT_ROOT
    / "runs"
    / "player_roles"
    / "player_roles_v2_yolo26n"
    / "weights"
    / "best.pt"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "player_roles_temporal_comparison"
)


# ============================================================
# VALIDATION VIDEOS
# ============================================================

VALIDATION_VIDEOS = [
    PROJECT_ROOT / "data" / "raw" / "baseball_test4.mp4",
    PROJECT_ROOT / "data" / "raw" / "baseball_train9.mp4",
    PROJECT_ROOT / "data" / "raw" / "baseball_train10.mp4",
]


# ============================================================
# CLASS CONFIGURATION
# ============================================================

CLASS_NAMES = {
    0: "batter",
    1: "catcher",
    2: "pitcher",
    3: "umpire",
}

ROLE_NAMES = [
    "batter",
    "catcher",
    "pitcher",
    "umpire",
]


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "BaseVisionAI - Temporal comparison "
            "YOLO11n vs YOLO26n player roles"
        )
    )

    parser.add_argument(
        "--yolo11",
        type=str,
        default=str(DEFAULT_YOLO11),
        help="YOLO11n best.pt"
    )

    parser.add_argument(
        "--yolo26",
        type=str,
        default=str(DEFAULT_YOLO26),
        help="YOLO26n best.pt"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Inference image size"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output directory"
    )

    return parser.parse_args()


# ============================================================
# UTILITY
# ============================================================

def longest_false_streak(values):
    """
    Receives a list of booleans:
        True  = role detected
        False = role missing

    Returns the longest consecutive missing streak.
    """

    longest = 0
    current = 0

    for detected in values:

        if not detected:
            current += 1
            longest = max(longest, current)

        else:
            current = 0

    return longest


# ============================================================
# PROCESS VIDEO
# ============================================================

def evaluate_video(
    model,
    model_name,
    video_path,
    conf,
    imgsz,
    device,
):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    total_frames_reported = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print()
    print("=" * 80)
    print(f"Model : {model_name}")
    print(f"Video : {video_path.name}")
    print(f"FPS   : {fps:.2f}")
    print(f"Frames: {total_frames_reported}")
    print("=" * 80)

    # --------------------------------------------------------
    # Temporal storage
    # --------------------------------------------------------

    role_detected = {
        role: []
        for role in ROLE_NAMES
    }

    role_confidences = {
        role: []
        for role in ROLE_NAMES
    }

    role_duplicate_frames = {
        role: 0
        for role in ROLE_NAMES
    }

    frame_rows = []

    complete_role_frames = 0

    processed_frames = 0

    frame_number = 0

    # ========================================================
    # FRAME LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1
        processed_frames += 1

        # ----------------------------------------------------
        # YOLO inference
        # ----------------------------------------------------

        results = model.predict(
            source=frame,
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )

        result = results[0]

        detections_by_role = {
            role: []
            for role in ROLE_NAMES
        }

        # ----------------------------------------------------
        # Read detections
        # ----------------------------------------------------

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                if class_id not in CLASS_NAMES:
                    continue

                role = CLASS_NAMES[class_id]

                xyxy = box.xyxy[0].cpu().numpy()

                x1, y1, x2, y2 = xyxy.tolist()

                detections_by_role[role].append({
                    "confidence": confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                })

        # ----------------------------------------------------
        # Process each role
        # ----------------------------------------------------

        row = {
            "model": model_name,
            "video": video_path.name,
            "frame": frame_number,
            "time_sec": (frame_number - 1) / fps,
        }

        all_roles_present = True

        for role in ROLE_NAMES:

            detections = detections_by_role[role]

            count = len(detections)

            detected = count > 0

            role_detected[role].append(detected)

            # ------------------------------------------------
            # Duplicate detection
            # ------------------------------------------------

            if count > 1:
                role_duplicate_frames[role] += 1

            # ------------------------------------------------
            # Highest confidence detection
            # ------------------------------------------------

            if detected:

                best_detection = max(
                    detections,
                    key=lambda x: x["confidence"]
                )

                best_conf = best_detection["confidence"]

                role_confidences[role].append(best_conf)

                row[f"{role}_detected"] = 1
                row[f"{role}_count"] = count
                row[f"{role}_conf"] = best_conf

                row[f"{role}_x1"] = best_detection["x1"]
                row[f"{role}_y1"] = best_detection["y1"]
                row[f"{role}_x2"] = best_detection["x2"]
                row[f"{role}_y2"] = best_detection["y2"]

            else:

                all_roles_present = False

                row[f"{role}_detected"] = 0
                row[f"{role}_count"] = 0
                row[f"{role}_conf"] = ""

                row[f"{role}_x1"] = ""
                row[f"{role}_y1"] = ""
                row[f"{role}_x2"] = ""
                row[f"{role}_y2"] = ""

        if all_roles_present:
            complete_role_frames += 1

        frame_rows.append(row)

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if frame_number % 30 == 0:

            print(
                f"\rProcessing frame "
                f"{frame_number}/{total_frames_reported}",
                end="",
                flush=True
            )

    cap.release()

    print()

    # ========================================================
    # VIDEO SUMMARY
    # ========================================================

    summary_rows = []

    for role in ROLE_NAMES:

        detected_values = role_detected[role]

        detected_frames = sum(detected_values)

        coverage = (
            detected_frames
            / processed_frames
            * 100.0
        )

        confidences = role_confidences[role]

        if confidences:

            avg_conf = float(np.mean(confidences))
            median_conf = float(np.median(confidences))
            min_conf = float(np.min(confidences))
            max_conf = float(np.max(confidences))

        else:

            avg_conf = 0.0
            median_conf = 0.0
            min_conf = 0.0
            max_conf = 0.0

        longest_missing = longest_false_streak(
            detected_values
        )

        longest_missing_seconds = (
            longest_missing / fps
        )

        duplicate_frames = (
            role_duplicate_frames[role]
        )

        duplicate_percent = (
            duplicate_frames
            / processed_frames
            * 100.0
        )

        summary_rows.append({
            "model": model_name,
            "video": video_path.name,
            "role": role,

            "total_frames": processed_frames,

            "detected_frames": detected_frames,

            "coverage_percent": coverage,

            "avg_confidence": avg_conf,

            "median_confidence": median_conf,

            "min_confidence": min_conf,

            "max_confidence": max_conf,

            "longest_missing_frames":
                longest_missing,

            "longest_missing_seconds":
                longest_missing_seconds,

            "duplicate_frames":
                duplicate_frames,

            "duplicate_percent":
                duplicate_percent,
        })

    complete_percent = (
        complete_role_frames
        / processed_frames
        * 100.0
    )

    video_summary = {
        "model": model_name,
        "video": video_path.name,
        "fps": fps,
        "total_frames": processed_frames,
        "all_4_roles_frames": complete_role_frames,
        "all_4_roles_percent": complete_percent,
    }

    return summary_rows, frame_rows, video_summary


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(path, rows):

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# AGGREGATE SUMMARY
# ============================================================

def create_aggregate_summary(summary_rows):

    aggregate_rows = []

    models = sorted(
        set(row["model"] for row in summary_rows)
    )

    for model_name in models:

        for role in ROLE_NAMES:

            selected = [
                row
                for row in summary_rows
                if row["model"] == model_name
                and row["role"] == role
            ]

            if not selected:
                continue

            total_frames = sum(
                row["total_frames"]
                for row in selected
            )

            detected_frames = sum(
                row["detected_frames"]
                for row in selected
            )

            coverage = (
                detected_frames
                / total_frames
                * 100.0
            )

            # Weighted confidence by number of detected frames
            confidence_numerator = sum(
                row["avg_confidence"]
                * row["detected_frames"]
                for row in selected
            )

            if detected_frames > 0:

                avg_confidence = (
                    confidence_numerator
                    / detected_frames
                )

            else:
                avg_confidence = 0.0

            duplicate_frames = sum(
                row["duplicate_frames"]
                for row in selected
            )

            duplicate_percent = (
                duplicate_frames
                / total_frames
                * 100.0
            )

            longest_missing_frames = max(
                row["longest_missing_frames"]
                for row in selected
            )

            longest_missing_seconds = max(
                row["longest_missing_seconds"]
                for row in selected
            )

            aggregate_rows.append({
                "model": model_name,
                "role": role,

                "total_frames":
                    total_frames,

                "detected_frames":
                    detected_frames,

                "coverage_percent":
                    coverage,

                "avg_confidence":
                    avg_confidence,

                "duplicate_frames":
                    duplicate_frames,

                "duplicate_percent":
                    duplicate_percent,

                "worst_missing_frames":
                    longest_missing_frames,

                "worst_missing_seconds":
                    longest_missing_seconds,
            })

    return aggregate_rows


# ============================================================
# PRINT COMPARISON
# ============================================================

def print_aggregate_comparison(rows):

    print()
    print("=" * 110)
    print("TEMPORAL AGGREGATE COMPARISON")
    print("=" * 110)

    header = (
        f"{'MODEL':<10}"
        f"{'ROLE':<12}"
        f"{'COVERAGE':>12}"
        f"{'AVG CONF':>12}"
        f"{'DUPLICATES':>14}"
        f"{'MAX MISSING':>16}"
    )

    print(header)
    print("-" * 110)

    for row in rows:

        print(
            f"{row['model']:<10}"
            f"{row['role']:<12}"
            f"{row['coverage_percent']:>11.2f}%"
            f"{row['avg_confidence']:>12.3f}"
            f"{row['duplicate_percent']:>13.2f}%"
            f"{row['worst_missing_seconds']:>15.3f}s"
        )

    print("=" * 110)


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    yolo11_path = Path(args.yolo11).resolve()
    yolo26_path = Path(args.yolo26).resolve()

    output_dir = Path(args.output).resolve()

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not yolo11_path.exists():
        raise FileNotFoundError(
            f"YOLO11 model not found:\n{yolo11_path}"
        )

    if not yolo26_path.exists():
        raise FileNotFoundError(
            f"YOLO26 model not found:\n{yolo26_path}"
        )

    for video in VALIDATION_VIDEOS:

        if not video.exists():
            raise FileNotFoundError(
                f"Validation video not found:\n{video}"
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    print()
    print("Loading YOLO11n...")
    model11 = YOLO(str(yolo11_path))

    print("Loading YOLO26n...")
    model26 = YOLO(str(yolo26_path))

    models = [
        ("YOLO11n", model11),
        ("YOLO26n", model26),
    ]

    # --------------------------------------------------------
    # Global storage
    # --------------------------------------------------------

    all_summary_rows = []
    all_frame_rows = []
    all_video_summary_rows = []

    # ========================================================
    # RUN EXPERIMENT
    # ========================================================

    for model_name, model in models:

        for video_path in VALIDATION_VIDEOS:

            summary_rows, frame_rows, video_summary = (
                evaluate_video(
                    model=model,
                    model_name=model_name,
                    video_path=video_path,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    device=args.device,
                )
            )

            all_summary_rows.extend(summary_rows)
            all_frame_rows.extend(frame_rows)
            all_video_summary_rows.append(
                video_summary
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    save_csv(
        output_dir
        / "temporal_per_video.csv",
        all_summary_rows
    )

    save_csv(
        output_dir
        / "temporal_per_frame.csv",
        all_frame_rows
    )

    save_csv(
        output_dir
        / "complete_roles_per_video.csv",
        all_video_summary_rows
    )

    # --------------------------------------------------------
    # Aggregate comparison
    # --------------------------------------------------------

    aggregate_rows = create_aggregate_summary(
        all_summary_rows
    )

    save_csv(
        output_dir
        / "temporal_aggregate.csv",
        aggregate_rows
    )

    print_aggregate_comparison(
        aggregate_rows
    )

    print()
    print("Results saved to:")
    print(output_dir)

    print()
    print("Files:")
    print("  temporal_per_video.csv")
    print("  temporal_per_frame.csv")
    print("  complete_roles_per_video.csv")
    print("  temporal_aggregate.csv")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()