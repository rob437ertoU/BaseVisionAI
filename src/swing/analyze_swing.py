from pathlib import Path
import pandas as pd
import numpy as np
import math


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "batter_pose_smoothed_v1.csv"
)

OUTPUT_FRAME_CSV = (
    PROJECT_ROOT
    / "results"
    / "swing_analysis_v1_3.csv"
)

OUTPUT_SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "swing_analysis_summary_v1_3.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

FPS = 30.0

MIN_SWING_POSE_QUALITY = 0.65

# First valid frames used to define the batter's initial stance
BASELINE_FRAMES = 10


# ============================================================
# BASIC GEOMETRY
# ============================================================

def point(row, keypoint):

    x_col = f"{keypoint}_smooth_x"
    y_col = f"{keypoint}_smooth_y"

    if x_col not in row or y_col not in row:
        return None

    x = row[x_col]
    y = row[y_col]

    if pd.isna(x) or pd.isna(y):
        return None

    return float(x), float(y)


def midpoint(p1, p2):

    if p1 is None or p2 is None:
        return None

    return (
        (p1[0] + p2[0]) / 2.0,
        (p1[1] + p2[1]) / 2.0
    )


def distance(p1, p2):

    if p1 is None or p2 is None:
        return np.nan

    return math.hypot(
        p2[0] - p1[0],
        p2[1] - p1[1]
    )


def line_angle(p1, p2):

    if p1 is None or p2 is None:
        return np.nan

    return math.degrees(
        math.atan2(
            p2[1] - p1[1],
            p2[0] - p1[0]
        )
    )


def joint_angle(a, b, c):

    if a is None or b is None or c is None:
        return np.nan

    ba = np.array(
        [
            a[0] - b[0],
            a[1] - b[1]
        ],
        dtype=float
    )

    bc = np.array(
        [
            c[0] - b[0],
            c[1] - b[1]
        ],
        dtype=float
    )

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return np.nan

    cosine = np.dot(
        ba,
        bc
    ) / (
        norm_ba
        *
        norm_bc
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return math.degrees(
        math.acos(cosine)
    )


def speed(current, previous, frame_delta):

    if (
        current is None
        or previous is None
        or frame_delta <= 0
    ):
        return np.nan

    return (
        distance(
            current,
            previous
        )
        /
        frame_delta
    )


# ============================================================
# AXIAL ANGLES
# ============================================================

def axial_difference_deg(a, b):
    """
    Difference between two undirected lines.

    A shoulder/hip line at 10° is equivalent to 190°.

    Output:
        approximately [-90°, +90°]
    """

    if pd.isna(a) or pd.isna(b):
        return np.nan

    return (
        (a - b + 90.0)
        % 180.0
        - 90.0
    )


# ============================================================
# RELATIVE POINT
# ============================================================

def relative_point(p, origin):

    if p is None or origin is None:
        return None

    return (
        p[0] - origin[0],
        p[1] - origin[1]
    )


# ============================================================
# FRAME ANALYSIS
# ============================================================

def analyze_frame(row, previous_row):

    # ========================================================
    # CURRENT KEYPOINTS
    # ========================================================

    left_shoulder = point(
        row,
        "left_shoulder"
    )

    right_shoulder = point(
        row,
        "right_shoulder"
    )

    left_elbow = point(
        row,
        "left_elbow"
    )

    right_elbow = point(
        row,
        "right_elbow"
    )

    left_wrist = point(
        row,
        "left_wrist"
    )

    right_wrist = point(
        row,
        "right_wrist"
    )

    left_hip = point(
        row,
        "left_hip"
    )

    right_hip = point(
        row,
        "right_hip"
    )

    left_knee = point(
        row,
        "left_knee"
    )

    right_knee = point(
        row,
        "right_knee"
    )

    left_ankle = point(
        row,
        "left_ankle"
    )

    right_ankle = point(
        row,
        "right_ankle"
    )

    nose = point(
        row,
        "nose"
    )

    # ========================================================
    # BODY CENTERS
    # ========================================================

    shoulder_center = midpoint(
        left_shoulder,
        right_shoulder
    )

    hip_center = midpoint(
        left_hip,
        right_hip
    )

    # ========================================================
    # BODY ANGLES
    # ========================================================

    shoulder_angle = line_angle(
        left_shoulder,
        right_shoulder
    )

    hip_angle = line_angle(
        left_hip,
        right_hip
    )

    torso_angle = line_angle(
        hip_center,
        shoulder_center
    )

    shoulder_hip_separation = (
        axial_difference_deg(
            shoulder_angle,
            hip_angle
        )
    )

    # ========================================================
    # JOINT ANGLES
    # ========================================================

    left_elbow_angle = joint_angle(
        left_shoulder,
        left_elbow,
        left_wrist
    )

    right_elbow_angle = joint_angle(
        right_shoulder,
        right_elbow,
        right_wrist
    )

    left_knee_angle = joint_angle(
        left_hip,
        left_knee,
        left_ankle
    )

    right_knee_angle = joint_angle(
        right_hip,
        right_knee,
        right_ankle
    )

    # ========================================================
    # BODY MEASUREMENTS
    # ========================================================

    shoulder_width = distance(
        left_shoulder,
        right_shoulder
    )

    hip_width = distance(
        left_hip,
        right_hip
    )

    wrist_separation = distance(
        left_wrist,
        right_wrist
    )

    ankle_height_difference = np.nan

    if (
        left_ankle is not None
        and
        right_ankle is not None
    ):
        ankle_height_difference = (
            left_ankle[1]
            -
            right_ankle[1]
        )

    knee_height_difference = np.nan

    if (
        left_knee is not None
        and
        right_knee is not None
    ):
        knee_height_difference = (
            left_knee[1]
            -
            right_knee[1]
        )

    # ========================================================
    # MOTION
    # ========================================================

    left_wrist_speed = np.nan
    right_wrist_speed = np.nan

    hip_center_speed = np.nan
    shoulder_center_speed = np.nan
    head_speed = np.nan

    left_wrist_relative_speed = np.nan
    right_wrist_relative_speed = np.nan

    # ========================================================
    # PREVIOUS FRAME MOTION
    # ========================================================

    if previous_row is not None:

        frame_delta = (
            int(row["frame"])
            -
            int(previous_row["frame"])
        )

        previous_left_shoulder = point(
            previous_row,
            "left_shoulder"
        )

        previous_right_shoulder = point(
            previous_row,
            "right_shoulder"
        )

        previous_left_wrist = point(
            previous_row,
            "left_wrist"
        )

        previous_right_wrist = point(
            previous_row,
            "right_wrist"
        )

        previous_left_hip = point(
            previous_row,
            "left_hip"
        )

        previous_right_hip = point(
            previous_row,
            "right_hip"
        )

        previous_nose = point(
            previous_row,
            "nose"
        )

        previous_shoulder_center = midpoint(
            previous_left_shoulder,
            previous_right_shoulder
        )

        previous_hip_center = midpoint(
            previous_left_hip,
            previous_right_hip
        )

        # ----------------------------------------------------
        # Absolute wrist motion
        # ----------------------------------------------------

        left_wrist_speed = speed(
            left_wrist,
            previous_left_wrist,
            frame_delta
        )

        right_wrist_speed = speed(
            right_wrist,
            previous_right_wrist,
            frame_delta
        )

        # ----------------------------------------------------
        # Body translation
        # ----------------------------------------------------

        hip_center_speed = speed(
            hip_center,
            previous_hip_center,
            frame_delta
        )

        shoulder_center_speed = speed(
            shoulder_center,
            previous_shoulder_center,
            frame_delta
        )

        head_speed = speed(
            nose,
            previous_nose,
            frame_delta
        )

        # ----------------------------------------------------
        # Wrist relative to shoulder center
        # ----------------------------------------------------

        current_left_wrist_relative = relative_point(
            left_wrist,
            shoulder_center
        )

        previous_left_wrist_relative = relative_point(
            previous_left_wrist,
            previous_shoulder_center
        )

        current_right_wrist_relative = relative_point(
            right_wrist,
            shoulder_center
        )

        previous_right_wrist_relative = relative_point(
            previous_right_wrist,
            previous_shoulder_center
        )

        left_wrist_relative_speed = speed(
            current_left_wrist_relative,
            previous_left_wrist_relative,
            frame_delta
        )

        right_wrist_relative_speed = speed(
            current_right_wrist_relative,
            previous_right_wrist_relative,
            frame_delta
        )

    # ========================================================
    # RETURN FRAME METRICS
    # ========================================================

    return {
        "frame":
            int(row["frame"]),

        "time_seconds":
            (int(row["frame"]) - 1)
            /
            FPS,

        "swing_pose_quality":
            row.get(
                "swing_pose_quality",
                np.nan
            ),

        # ----------------------------------------------------
        # Raw orientation
        # ----------------------------------------------------

        "shoulder_angle_raw_deg":
            shoulder_angle,

        "hip_angle_raw_deg":
            hip_angle,

        "torso_angle_deg":
            torso_angle,

        "shoulder_hip_separation_deg":
            shoulder_hip_separation,

        # ----------------------------------------------------
        # Joint angles
        # ----------------------------------------------------

        "left_elbow_angle_deg":
            left_elbow_angle,

        "right_elbow_angle_deg":
            right_elbow_angle,

        "left_knee_angle_deg":
            left_knee_angle,

        "right_knee_angle_deg":
            right_knee_angle,

        # ----------------------------------------------------
        # Body measurements
        # ----------------------------------------------------

        "wrist_separation_px":
            wrist_separation,

        "shoulder_width_px":
            shoulder_width,

        "hip_width_px":
            hip_width,

        "ankle_height_difference_px":
            ankle_height_difference,

        "knee_height_difference_px":
            knee_height_difference,

        # ----------------------------------------------------
        # Absolute motion
        # ----------------------------------------------------

        "left_wrist_speed_px_frame":
            left_wrist_speed,

        "right_wrist_speed_px_frame":
            right_wrist_speed,

        "hip_center_speed_px_frame":
            hip_center_speed,

        "shoulder_center_speed_px_frame":
            shoulder_center_speed,

        "head_speed_px_frame":
            head_speed,

        # ----------------------------------------------------
        # Wrist motion relative to upper body
        # ----------------------------------------------------

        "left_wrist_relative_speed_px_frame":
            left_wrist_relative_speed,

        "right_wrist_relative_speed_px_frame":
            right_wrist_relative_speed,

        # ----------------------------------------------------
        # Centers
        # ----------------------------------------------------

        "shoulder_center_x":
            (
                shoulder_center[0]
                if shoulder_center
                is not None
                else np.nan
            ),

        "shoulder_center_y":
            (
                shoulder_center[1]
                if shoulder_center
                is not None
                else np.nan
            ),

        "hip_center_x":
            (
                hip_center[0]
                if hip_center
                is not None
                else np.nan
            ),

        "hip_center_y":
            (
                hip_center[1]
                if hip_center
                is not None
                else np.nan
            )
    }


# ============================================================
# BASELINE
# ============================================================

def baseline_mean(df, column):

    valid = df[
        (
            df["swing_pose_quality"]
            >= MIN_SWING_POSE_QUALITY
        )
        &
        df[column].notna()
    ]

    if valid.empty:
        return np.nan

    return float(
        valid
        .head(BASELINE_FRAMES)[column]
        .median()
    )


# ============================================================
# SUMMARY HELPERS
# ============================================================

def safe_max(df, column):

    values = df[
        column
    ].dropna()

    if values.empty:
        return np.nan

    return float(
        values.max()
    )


def safe_min(df, column):

    values = df[
        column
    ].dropna()

    if values.empty:
        return np.nan

    return float(
        values.min()
    )


def frame_of_max(df, column):

    valid = df.dropna(
        subset=[column]
    )

    if valid.empty:
        return np.nan

    idx = valid[
        column
    ].idxmax()

    return int(
        valid.loc[
            idx,
            "frame"
        ]
    )


def frame_of_max_abs(df, column):

    valid = df.dropna(
        subset=[column]
    )

    if valid.empty:
        return np.nan

    idx = (
        valid[
            column
        ]
        .abs()
        .idxmax()
    )

    return int(
        valid.loc[
            idx,
            "frame"
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("BaseVisionAI - Swing Analytics V1.3")
    print("=" * 80)

    print(
        f"\nInput:\n{INPUT_CSV}"
    )

    # ========================================================
    # LOAD FILE
    # ========================================================

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"File not found:\n"
            f"{INPUT_CSV}"
        )

    df = pd.read_csv(
        INPUT_CSV
    )

    df = (
        df
        .sort_values("frame")
        .reset_index(drop=True)
    )

    print(
        f"\nFrames loaded: {len(df)}"
    )

    # ========================================================
    # FRAME-BY-FRAME ANALYSIS
    # ========================================================

    rows = []

    previous_row = None

    for _, row in df.iterrows():

        metrics = analyze_frame(
            row,
            previous_row
        )

        rows.append(
            metrics
        )

        previous_row = row

    result = pd.DataFrame(
        rows
    )

    # ========================================================
    # VALID BASELINE FRAMES
    # ========================================================

    valid_for_baseline = result[
        result[
            "swing_pose_quality"
        ]
        >=
        MIN_SWING_POSE_QUALITY
    ].head(
        BASELINE_FRAMES
    )

    if valid_for_baseline.empty:

        raise ValueError(
            "No valid frames available "
            "for baseline calculation."
        )

    # ========================================================
    # AXIAL ORIENTATION BASELINES
    # ========================================================

    baseline_shoulder = float(
        valid_for_baseline[
            "shoulder_angle_raw_deg"
        ].median()
    )

    baseline_hip = float(
        valid_for_baseline[
            "hip_angle_raw_deg"
        ].median()
    )

    # ========================================================
    # LEG BASELINES
    # ========================================================

    baseline_ankle = baseline_mean(
        result,
        "ankle_height_difference_px"
    )

    baseline_knee = baseline_mean(
        result,
        "knee_height_difference_px"
    )

    # ========================================================
    # RELATIVE AXIAL ROTATION
    # ========================================================

    result[
        "shoulder_rotation_relative_deg"
    ] = result[
        "shoulder_angle_raw_deg"
    ].apply(
        lambda angle:
            axial_difference_deg(
                angle,
                baseline_shoulder
            )
            if not pd.isna(angle)
            else np.nan
    )

    result[
        "hip_rotation_relative_deg"
    ] = result[
        "hip_angle_raw_deg"
    ].apply(
        lambda angle:
            axial_difference_deg(
                angle,
                baseline_hip
            )
            if not pd.isna(angle)
            else np.nan
    )

    # ========================================================
    # LEG KICK RELATIVE TO BASELINE
    # ========================================================

    result[
        "ankle_change_from_baseline_px"
    ] = (
        result[
            "ankle_height_difference_px"
        ]
        -
        baseline_ankle
    )

    result[
        "knee_change_from_baseline_px"
    ] = (
        result[
            "knee_height_difference_px"
        ]
        -
        baseline_knee
    )

    # ========================================================
    # VALID ANALYSIS FRAMES
    # ========================================================

    valid = result[
        result[
            "swing_pose_quality"
        ]
        >=
        MIN_SWING_POSE_QUALITY
    ].copy()

    # Don't search leg-kick peak inside the baseline itself.
    post_baseline = valid[
        valid["frame"]
        >
        BASELINE_FRAMES
    ].copy()

    # ========================================================
    # ROTATION VALUES
    # ========================================================

    shoulder_min = safe_min(
        valid,
        "shoulder_rotation_relative_deg"
    )

    shoulder_max = safe_max(
        valid,
        "shoulder_rotation_relative_deg"
    )

    hip_min = safe_min(
        valid,
        "hip_rotation_relative_deg"
    )

    hip_max = safe_max(
        valid,
        "hip_rotation_relative_deg"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "analyzed_frames":
            len(valid),

        "avg_pose_quality":
            float(
                valid[
                    "swing_pose_quality"
                ].mean()
            ),

        # ----------------------------------------------------
        # Absolute wrist speed
        # ----------------------------------------------------

        "max_left_wrist_speed_px_frame":
            safe_max(
                valid,
                "left_wrist_speed_px_frame"
            ),

        "max_left_wrist_speed_frame":
            frame_of_max(
                valid,
                "left_wrist_speed_px_frame"
            ),

        "max_right_wrist_speed_px_frame":
            safe_max(
                valid,
                "right_wrist_speed_px_frame"
            ),

        "max_right_wrist_speed_frame":
            frame_of_max(
                valid,
                "right_wrist_speed_px_frame"
            ),

        # ----------------------------------------------------
        # Relative wrist speed
        # ----------------------------------------------------

        "max_left_wrist_relative_speed_px_frame":
            safe_max(
                valid,
                "left_wrist_relative_speed_px_frame"
            ),

        "max_left_wrist_relative_speed_frame":
            frame_of_max(
                valid,
                "left_wrist_relative_speed_px_frame"
            ),

        "max_right_wrist_relative_speed_px_frame":
            safe_max(
                valid,
                "right_wrist_relative_speed_px_frame"
            ),

        "max_right_wrist_relative_speed_frame":
            frame_of_max(
                valid,
                "right_wrist_relative_speed_px_frame"
            ),

        # ----------------------------------------------------
        # Body translation
        # ----------------------------------------------------

        "max_hip_center_speed_px_frame":
            safe_max(
                valid,
                "hip_center_speed_px_frame"
            ),

        "max_hip_center_speed_frame":
            frame_of_max(
                valid,
                "hip_center_speed_px_frame"
            ),

        "max_shoulder_center_speed_px_frame":
            safe_max(
                valid,
                "shoulder_center_speed_px_frame"
            ),

        "max_shoulder_center_speed_frame":
            frame_of_max(
                valid,
                "shoulder_center_speed_px_frame"
            ),

        # ----------------------------------------------------
        # Shoulder rotation
        # ----------------------------------------------------

        "baseline_shoulder_angle_deg":
            baseline_shoulder,

        "shoulder_rotation_min_deg":
            shoulder_min,

        "shoulder_rotation_max_deg":
            shoulder_max,

        "shoulder_rotation_range_deg":
            shoulder_max
            -
            shoulder_min,

        # ----------------------------------------------------
        # Hip rotation
        # ----------------------------------------------------

        "baseline_hip_angle_deg":
            baseline_hip,

        "hip_rotation_min_deg":
            hip_min,

        "hip_rotation_max_deg":
            hip_max,

        "hip_rotation_range_deg":
            hip_max
            -
            hip_min,

        # ----------------------------------------------------
        # Shoulder / hip separation
        # ----------------------------------------------------

        "max_abs_shoulder_hip_separation_deg":
            float(
                valid[
                    "shoulder_hip_separation_deg"
                ]
                .abs()
                .max()
            ),

        "max_shoulder_hip_separation_frame":
            frame_of_max_abs(
                valid,
                "shoulder_hip_separation_deg"
            ),

        # ----------------------------------------------------
        # Leg kick
        # ----------------------------------------------------

        "max_abs_ankle_change_from_baseline_px":
            float(
                post_baseline[
                    "ankle_change_from_baseline_px"
                ]
                .abs()
                .max()
            ),

        "max_ankle_change_frame":
            frame_of_max_abs(
                post_baseline,
                "ankle_change_from_baseline_px"
            ),

        "max_abs_knee_change_from_baseline_px":
            float(
                post_baseline[
                    "knee_change_from_baseline_px"
                ]
                .abs()
                .max()
            ),

        "max_knee_change_frame":
            frame_of_max_abs(
                post_baseline,
                "knee_change_from_baseline_px"
            )
    }

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_FRAME_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FRAME_CSV,
        index=False
    )

    pd.DataFrame(
        [summary]
    ).to_csv(
        OUTPUT_SUMMARY_CSV,
        index=False
    )

    # ========================================================
    # TERMINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 80)
    print("SWING SUMMARY V1.3")
    print("=" * 80)

    print("\nABSOLUTE WRIST SPEED")

    print(
        f"Left : "
        f"{summary['max_left_wrist_speed_px_frame']:.2f} "
        f"px/frame @ "
        f"{summary['max_left_wrist_speed_frame']}"
    )

    print(
        f"Right: "
        f"{summary['max_right_wrist_speed_px_frame']:.2f} "
        f"px/frame @ "
        f"{summary['max_right_wrist_speed_frame']}"
    )

    print("\nRELATIVE WRIST SPEED")

    print(
        f"Left : "
        f"{summary['max_left_wrist_relative_speed_px_frame']:.2f} "
        f"px/frame @ "
        f"{summary['max_left_wrist_relative_speed_frame']}"
    )

    print(
        f"Right: "
        f"{summary['max_right_wrist_relative_speed_px_frame']:.2f} "
        f"px/frame @ "
        f"{summary['max_right_wrist_relative_speed_frame']}"
    )

    print("\nBODY TRANSLATION")

    print(
        f"Hip peak: "
        f"{summary['max_hip_center_speed_px_frame']:.2f} "
        f"px/frame @ "
        f"{summary['max_hip_center_speed_frame']}"
    )

    print(
        f"Shoulder peak: "
        f"{summary['max_shoulder_center_speed_px_frame']:.2f} "
        f"px/frame @ "
        f"{summary['max_shoulder_center_speed_frame']}"
    )

    print("\nCORRECTED AXIAL ROTATION")

    print(
        f"Shoulder range: "
        f"{summary['shoulder_rotation_range_deg']:.2f} deg"
    )

    print(
        f"Hip range: "
        f"{summary['hip_rotation_range_deg']:.2f} deg"
    )

    print(
        f"Max shoulder/hip separation: "
        f"{summary['max_abs_shoulder_hip_separation_deg']:.2f} deg "
        f"@ frame "
        f"{summary['max_shoulder_hip_separation_frame']}"
    )

    print("\nLEG KICK")

    print(
        f"Ankle change: "
        f"{summary['max_abs_ankle_change_from_baseline_px']:.2f} px "
        f"@ frame "
        f"{summary['max_ankle_change_frame']}"
    )

    print(
        f"Knee change: "
        f"{summary['max_abs_knee_change_from_baseline_px']:.2f} px "
        f"@ frame "
        f"{summary['max_knee_change_frame']}"
    )

    print("\n" + "=" * 80)
    print("SWING ANALYTICS V1.3 COMPLETED")
    print("=" * 80)

    print(
        f"\nFrame analysis:\n"
        f"{OUTPUT_FRAME_CSV}"
    )

    print(
        f"\nSummary:\n"
        f"{OUTPUT_SUMMARY_CSV}"
    )


if __name__ == "__main__":
    main()