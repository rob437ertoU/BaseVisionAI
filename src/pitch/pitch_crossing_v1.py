from pathlib import Path
import argparse

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ZONE_CSV = (
    PROJECT_ROOT
    / "results"
    / "strike_zone_v1_1"
    / "baseball_train10_strike_zone_v1_1.csv"
)

DEFAULT_BALL_CSV = (
    PROJECT_ROOT
    / "results"
    / "ball_tracking_v3_1"
    / "baseball_train10"
    / "ball_tracking_v3_1.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "pitch_crossing_v1"
)


# ============================================================
# CONFIG
# ============================================================

MIN_YOLO_POINTS = 3

# How many latest real YOLO observations are allowed
# to fit incoming trajectory.
FIT_WINDOW = 4

# Crossing must not be extrapolated too far from
# the last observed ball frame.
MAX_EXTRAPOLATION_FRAMES = 6

# Residual threshold in pixels for a simple linear fit.
# This is an engineering quality check, not a physical constant.
MAX_MEAN_RESIDUAL_PX = 25.0


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Pitch Crossing Estimation V1"
    )

    parser.add_argument(
        "--zone-csv",
        type=str,
        default=str(DEFAULT_ZONE_CSV)
    )

    parser.add_argument(
        "--ball-csv",
        type=str,
        default=str(DEFAULT_BALL_CSV)
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR)
    )

    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def normalize_bool(value):

    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y"
    }


def safe_float(value, default=np.nan):

    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


# ============================================================
# LOAD / PREPARE
# ============================================================

def prepare_ball(ball_df):

    ball_df = ball_df.copy()

    ball_df["yolo_detected"] = (
        ball_df["yolo_detected"]
        .apply(normalize_bool)
    )

    for col in [
        "frame",
        "observed_x",
        "observed_y"
    ]:

        ball_df[col] = pd.to_numeric(
            ball_df[col],
            errors="coerce"
        )

    ball_df = ball_df[
        (
            ball_df["yolo_detected"]
        )
        &
        (
            ball_df["observed_x"].notna()
        )
        &
        (
            ball_df["observed_y"].notna()
        )
    ].copy()

    ball_df["frame"] = (
        ball_df["frame"]
        .astype(int)
    )

    return ball_df


def prepare_zone(zone_df):

    zone_df = zone_df.copy()

    for col in [
        "frame",
        "plate_x_proxy",
        "zone_left",
        "zone_right",
        "zone_top",
        "zone_bottom"
    ]:

        zone_df[col] = pd.to_numeric(
            zone_df[col],
            errors="coerce"
        )

    zone_df["zone_available"] = (
        zone_df["zone_available"]
        .apply(normalize_bool)
    )

    zone_df = zone_df[
        zone_df["zone_available"]
    ].copy()

    return zone_df


# ============================================================
# TRAJECTORY FIT
# ============================================================

def fit_linear_trajectory(points):

    if len(points) < MIN_YOLO_POINTS:

        return None

    points = (
        points
        .sort_values("frame")
        .tail(FIT_WINDOW)
        .copy()
    )

    frames = (
        points["frame"]
        .to_numpy(dtype=float)
    )

    xs = (
        points["observed_x"]
        .to_numpy(dtype=float)
    )

    ys = (
        points["observed_y"]
        .to_numpy(dtype=float)
    )

    # x(frame)
    coef_x = np.polyfit(
        frames,
        xs,
        1
    )

    # y(frame)
    coef_y = np.polyfit(
        frames,
        ys,
        1
    )

    pred_x = np.polyval(
        coef_x,
        frames
    )

    pred_y = np.polyval(
        coef_y,
        frames
    )

    residuals = np.sqrt(
        (pred_x - xs) ** 2
        +
        (pred_y - ys) ** 2
    )

    mean_residual = float(
        np.mean(residuals)
    )

    return {
        "points_used": len(points),
        "first_frame": int(frames[0]),
        "last_frame": int(frames[-1]),

        "x_slope": float(coef_x[0]),
        "x_intercept": float(coef_x[1]),

        "y_slope": float(coef_y[0]),
        "y_intercept": float(coef_y[1]),

        "mean_residual_px": mean_residual,
    }


# ============================================================
# CROSSING
# ============================================================

def estimate_crossing(
    trajectory,
    plate_x
):

    vx = trajectory["x_slope"]

    x0 = trajectory["x_intercept"]

    if abs(vx) < 1e-6:
        return None

    crossing_frame = (
        plate_x - x0
    ) / vx

    crossing_y = (
        trajectory["y_slope"]
        * crossing_frame
        +
        trajectory["y_intercept"]
    )

    return {
        "crossing_frame_float":
            float(crossing_frame),

        "crossing_y":
            float(crossing_y)
    }


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_crossing(
    crossing_x,
    crossing_y,
    zone_left,
    zone_right,
    zone_top,
    zone_bottom
):

    horizontal_inside = (
        zone_left
        <= crossing_x
        <= zone_right
    )

    vertical_inside = (
        zone_top
        <= crossing_y
        <= zone_bottom
    )

    if (
        horizontal_inside
        and vertical_inside
    ):

        classification = (
            "STRIKE_CANDIDATE"
        )

    else:

        classification = (
            "BALL_CANDIDATE"
        )

    return {
        "classification":
            classification,

        "horizontal_inside":
            horizontal_inside,

        "vertical_inside":
            vertical_inside
    }


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    zone_path = Path(
        args.zone_csv
    ).resolve()

    ball_path = Path(
        args.ball_csv
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    for path in [
        zone_path,
        ball_path
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_csv = (
        output_dir
        / "baseball_train10_pitch_crossing_v1.csv"
    )

    zone_df = pd.read_csv(
        zone_path
    )

    ball_df = pd.read_csv(
        ball_path
    )

    zone_df = prepare_zone(
        zone_df
    )

    ball_df = prepare_ball(
        ball_df
    )

    print()
    print("=" * 78)
    print(
        "BaseVisionAI - Pitch Crossing Estimation V1"
    )
    print("=" * 78)

    print(
        f"Real YOLO ball points     : "
        f"{len(ball_df)}"
    )

    if len(ball_df) < MIN_YOLO_POINTS:

        result = {
            "classification":
                "INSUFFICIENT_CROSSING_EVIDENCE",

            "reason":
                "NOT_ENOUGH_REAL_YOLO_POINTS"
        }

        pd.DataFrame(
            [result]
        ).to_csv(
            output_csv,
            index=False
        )

        print()
        print(
            "FINAL CLASSIFICATION      : "
            "INSUFFICIENT_CROSSING_EVIDENCE"
        )

        print(
            "Reason                    : "
            "NOT_ENOUGH_REAL_YOLO_POINTS"
        )

        print()
        print(
            f"Output CSV                : "
            f"{output_csv}"
        )

        return

    # ========================================================
    # FIT TRAJECTORY
    # ========================================================

    trajectory = fit_linear_trajectory(
        ball_df
    )

    if trajectory is None:

        raise RuntimeError(
            "Trajectory fitting failed."
        )

    # ========================================================
    # PLATE REFERENCE
    # ========================================================

    plate_x = float(
        zone_df[
            "plate_x_proxy"
        ].median()
    )

    crossing = estimate_crossing(
        trajectory,
        plate_x
    )

    if crossing is None:

        result = {
            "classification":
                "INSUFFICIENT_CROSSING_EVIDENCE",

            "reason":
                "TRAJECTORY_PARALLEL_TO_PLATE_X"
        }

        pd.DataFrame(
            [result]
        ).to_csv(
            output_csv,
            index=False
        )

        print()
        print(
            "FINAL CLASSIFICATION      : "
            "INSUFFICIENT_CROSSING_EVIDENCE"
        )

        print(
            "Reason                    : "
            "TRAJECTORY_PARALLEL_TO_PLATE_X"
        )

        return

    crossing_frame = (
        crossing[
            "crossing_frame_float"
        ]
    )

    crossing_y = (
        crossing[
            "crossing_y"
        ]
    )

    last_yolo_frame = (
        trajectory[
            "last_frame"
        ]
    )

    extrapolation_frames = (
        crossing_frame
        - last_yolo_frame
    )

    # ========================================================
    # TRAJECTORY QUALITY
    # ========================================================

    residual_ok = (
        trajectory[
            "mean_residual_px"
        ]
        <= MAX_MEAN_RESIDUAL_PX
    )

    extrapolation_ok = (
        0.0
        <= extrapolation_frames
        <= MAX_EXTRAPOLATION_FRAMES
    )

    # ========================================================
    # ZONE AT CROSSING
    # ========================================================

    nearest_zone_row = (
        zone_df
        .assign(
            frame_distance=(
                zone_df["frame"]
                - crossing_frame
            ).abs()
        )
        .sort_values(
            "frame_distance"
        )
        .iloc[0]
    )

    zone_left = safe_float(
        nearest_zone_row[
            "zone_left"
        ]
    )

    zone_right = safe_float(
        nearest_zone_row[
            "zone_right"
        ]
    )

    zone_top = safe_float(
        nearest_zone_row[
            "zone_top"
        ]
    )

    zone_bottom = safe_float(
        nearest_zone_row[
            "zone_bottom"
        ]
    )

    # ========================================================
    # DECISION
    # ========================================================

    if (
        not residual_ok
        or not extrapolation_ok
    ):

        if not residual_ok:
            reason = (
                "POOR_LINEAR_TRAJECTORY_FIT"
            )

        else:
            reason = (
                "CROSSING_EXTRAPOLATION_TOO_LARGE"
            )

        classification = (
            "INSUFFICIENT_CROSSING_EVIDENCE"
        )

        horizontal_inside = False
        vertical_inside = False

    else:

        decision = classify_crossing(
            crossing_x=plate_x,
            crossing_y=crossing_y,
            zone_left=zone_left,
            zone_right=zone_right,
            zone_top=zone_top,
            zone_bottom=zone_bottom
        )

        classification = (
            decision[
                "classification"
            ]
        )

        horizontal_inside = (
            decision[
                "horizontal_inside"
            ]
        )

        vertical_inside = (
            decision[
                "vertical_inside"
            ]
        )

        reason = (
            "ESTIMATED_2D_PLATE_CROSSING"
        )

    # ========================================================
    # QUALITY LABEL
    # ========================================================

    if classification == (
        "INSUFFICIENT_CROSSING_EVIDENCE"
    ):

        evidence_quality = (
            "INSUFFICIENT"
        )

    else:

        # Since this is extrapolated from only a few
        # 2D YOLO observations, V1 is never STRONG.
        if extrapolation_frames <= 2:

            evidence_quality = (
                "MODERATE"
            )

        else:

            evidence_quality = (
                "LIMITED"
            )

    # ========================================================
    # RESULT
    # ========================================================

    result = {
        "classification":
            classification,

        "evidence_quality":
            evidence_quality,

        "reason":
            reason,

        "real_yolo_points":
            len(ball_df),

        "trajectory_points_used":
            trajectory[
                "points_used"
            ],

        "trajectory_first_frame":
            trajectory[
                "first_frame"
            ],

        "trajectory_last_frame":
            trajectory[
                "last_frame"
            ],

        "trajectory_mean_residual_px":
            trajectory[
                "mean_residual_px"
            ],

        "plate_x_proxy":
            plate_x,

        "crossing_frame_float":
            crossing_frame,

        "crossing_frame_estimate":
            int(
                round(
                    crossing_frame
                )
            ),

        "crossing_y_px":
            crossing_y,

        "extrapolation_frames":
            extrapolation_frames,

        "zone_reference_frame":
            int(
                nearest_zone_row[
                    "frame"
                ]
            ),

        "zone_left":
            zone_left,

        "zone_right":
            zone_right,

        "zone_top":
            zone_top,

        "zone_bottom":
            zone_bottom,

        "horizontal_inside":
            horizontal_inside,

        "vertical_inside":
            vertical_inside,
    }

    pd.DataFrame(
        [result]
    ).to_csv(
        output_csv,
        index=False
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("TRAJECTORY")
    print("-" * 78)

    print(
        f"Points used               : "
        f"{trajectory['points_used']}"
    )

    print(
        f"YOLO frame range          : "
        f"{trajectory['first_frame']} "
        f"-> "
        f"{trajectory['last_frame']}"
    )

    print(
        f"Mean fit residual         : "
        f"{trajectory['mean_residual_px']:.2f} px"
    )

    print()
    print("CROSSING")
    print("-" * 78)

    print(
        f"Plate X proxy             : "
        f"{plate_x:.1f} px"
    )

    print(
        f"Crossing frame estimate   : "
        f"{crossing_frame:.2f}"
    )

    print(
        f"Crossing Y estimate       : "
        f"{crossing_y:.1f} px"
    )

    print(
        f"Extrapolation             : "
        f"{extrapolation_frames:.2f} frames"
    )

    print()
    print("STRIKE ZONE AT CROSSING")
    print("-" * 78)

    print(
        f"Zone reference frame      : "
        f"{int(nearest_zone_row['frame'])}"
    )

    print(
        f"Horizontal zone           : "
        f"{zone_left:.1f} -> "
        f"{zone_right:.1f}"
    )

    print(
        f"Vertical zone             : "
        f"{zone_top:.1f} -> "
        f"{zone_bottom:.1f}"
    )

    print()
    print("DECISION")
    print("-" * 78)

    print(
        f"Horizontal inside         : "
        f"{horizontal_inside}"
    )

    print(
        f"Vertical inside           : "
        f"{vertical_inside}"
    )

    print(
        f"Evidence quality          : "
        f"{evidence_quality}"
    )

    print(
        f"FINAL CLASSIFICATION      : "
        f"{classification}"
    )

    print()
    print("IMPORTANT")
    print("-" * 78)

    print(
        "This is an estimated 2D plate-crossing classification."
    )

    print(
        "It is not an umpire-grade or calibrated BALL/STRIKE call."
    )

    print(
        "Only real YOLO ball observations were used for the fit."
    )

    print(
        "Kalman-only positions were NOT treated as measurements."
    )

    print()
    print(
        f"Output CSV                : "
        f"{output_csv}"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()