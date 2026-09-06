from pathlib import Path
import pandas as pd
import numpy as np
import math


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "pitch_analysis_summary.csv"
)

POINTS_CSV = (
    PROJECT_ROOT
    / "results"
    / "pitch_analysis_points.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "contact_detection.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

# Number of points used at the end/start of each track
WINDOW_SIZE = 4

# Maximum temporal separation between tracks that can be
# considered part of the same baseball event.
MAX_GAP_FRAMES = 6

# Minimum direction change that starts becoming interesting.
MIN_ANGLE_CHANGE = 25.0

# Strong direction change.
STRONG_ANGLE_CHANGE = 45.0

# Minimum confidence score to classify as POSSIBLE_CONTACT.
CONTACT_SCORE_THRESHOLD = 0.60


# ============================================================
# FUNCTIONS
# ============================================================

def calculate_angle(dx, dy):
    """
    Calculate vector angle in degrees.

    Image coordinates:
        +X = right
        +Y = down
    """

    return math.degrees(
        math.atan2(dy, dx)
    )


def normalize_angle_difference(angle1, angle2):
    """
    Return smallest difference between two angles.

    Output range:
        0 - 180 degrees
    """

    difference = abs(
        angle2 - angle1
    )

    if difference > 180:
        difference = 360 - difference

    return difference


def calculate_direction_vector(points):
    """
    Estimate movement direction using several points instead
    of only two consecutive detections.

    This makes the direction estimate less sensitive to
    individual noisy detections.
    """

    if len(points) < 2:
        return None

    first = points.iloc[0]
    last = points.iloc[-1]

    dx = (
        last["filtered_x"]
        -
        first["filtered_x"]
    )

    dy = (
        last["filtered_y"]
        -
        first["filtered_y"]
    )

    angle = calculate_angle(
        dx,
        dy
    )

    magnitude = math.sqrt(
        dx ** 2
        +
        dy ** 2
    )

    return {
        "dx": float(dx),
        "dy": float(dy),
        "angle": float(angle),
        "magnitude": float(magnitude)
    }


def calculate_average_speed(points):
    """
    Average image-plane speed for a set of trajectory points.
    """

    if len(points) < 2:
        return np.nan

    speeds = []

    for i in range(
        1,
        len(points)
    ):

        previous = points.iloc[i - 1]
        current = points.iloc[i]

        frame_delta = (
            current["frame"]
            -
            previous["frame"]
        )

        if frame_delta <= 0:
            continue

        dx = (
            current["filtered_x"]
            -
            previous["filtered_x"]
        )

        dy = (
            current["filtered_y"]
            -
            previous["filtered_y"]
        )

        distance = math.sqrt(
            dx ** 2
            +
            dy ** 2
        )

        speed = (
            distance
            /
            frame_delta
        )

        speeds.append(
            speed
        )

    if not speeds:
        return np.nan

    return float(
        np.mean(speeds)
    )


def get_track_points(
    points_df,
    track_id
):
    """
    Get valid trajectory points belonging to one track.
    """

    track = points_df[
        points_df["analysis_track_id"]
        ==
        track_id
    ].copy()

    track = track.dropna(
        subset=[
            "filtered_x",
            "filtered_y"
        ]
    )

    track = track.sort_values(
        "frame"
    )

    track = track.reset_index(
        drop=True
    )

    return track


def score_contact(
    gap_frames,
    angle_change,
    previous_speed,
    next_speed
):
    """
    Generate an interpretable contact score from 0 to 1.

    This is NOT a trained ML probability.

    It is a heuristic evidence score.
    """

    # ========================================================
    # 1. TEMPORAL SCORE
    # ========================================================

    if gap_frames <= 1:
        temporal_score = 1.0

    elif gap_frames <= 3:
        temporal_score = 0.9

    elif gap_frames <= MAX_GAP_FRAMES:
        temporal_score = 0.6

    else:
        temporal_score = 0.0

    # ========================================================
    # 2. DIRECTION CHANGE SCORE
    # ========================================================

    if angle_change >= 70:
        angle_score = 1.0

    elif angle_change >= STRONG_ANGLE_CHANGE:
        angle_score = 0.85

    elif angle_change >= MIN_ANGLE_CHANGE:
        angle_score = 0.60

    else:
        angle_score = 0.20

    # ========================================================
    # 3. SPEED CHANGE SCORE
    # ========================================================

    speed_change_ratio = np.nan

    if (
        not pd.isna(previous_speed)
        and
        not pd.isna(next_speed)
        and
        previous_speed > 0
    ):

        speed_change_ratio = (
            next_speed
            /
            previous_speed
        )

        # Significant change in apparent image speed
        if (
            speed_change_ratio >= 1.35
            or
            speed_change_ratio <= 0.65
        ):
            speed_score = 1.0

        elif (
            speed_change_ratio >= 1.15
            or
            speed_change_ratio <= 0.85
        ):
            speed_score = 0.65

        else:
            speed_score = 0.30

    else:

        speed_score = 0.0

    # ========================================================
    # FINAL SCORE
    # ========================================================
    #
    # Direction is the strongest evidence.
    #
    # Temporal proximity is also important.
    #
    # Pixel speed is weaker because perspective can strongly
    # affect apparent image speed.
    # ========================================================

    final_score = (
        angle_score * 0.50
        +
        temporal_score * 0.35
        +
        speed_score * 0.15
    )

    return {
        "score": float(final_score),
        "angle_score": float(angle_score),
        "temporal_score": float(temporal_score),
        "speed_score": float(speed_score),
        "speed_change_ratio": speed_change_ratio
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BaseVisionAI - Possible Contact Detection V1")
    print("=" * 70)

    # ========================================================
    # CHECK FILES
    # ========================================================

    if not SUMMARY_CSV.exists():

        raise FileNotFoundError(
            f"Summary CSV not found:\n"
            f"{SUMMARY_CSV}"
        )

    if not POINTS_CSV.exists():

        raise FileNotFoundError(
            f"Points CSV not found:\n"
            f"{POINTS_CSV}"
        )

    # ========================================================
    # LOAD DATA
    # ========================================================

    summary_df = pd.read_csv(
        SUMMARY_CSV
    )

    points_df = pd.read_csv(
        POINTS_CSV
    )

    summary_df = summary_df.sort_values(
        "start_frame"
    ).reset_index(
        drop=True
    )

    print(
        f"\nTracks found: "
        f"{len(summary_df)}"
    )

    # ========================================================
    # CONTACT CANDIDATES
    # ========================================================

    events = []

    for i in range(
        len(summary_df) - 1
    ):

        previous_summary = (
            summary_df.iloc[i]
        )

        next_summary = (
            summary_df.iloc[i + 1]
        )

        previous_track_id = int(
            previous_summary["track_id"]
        )

        next_track_id = int(
            next_summary["track_id"]
        )

        print(
            "\n"
            +
            "-" * 70
        )

        print(
            f"Comparing Track "
            f"{previous_track_id} "
            f"-> Track "
            f"{next_track_id}"
        )

        # ====================================================
        # GET TRAJECTORY POINTS
        # ====================================================

        previous_track = get_track_points(
            points_df,
            previous_track_id
        )

        next_track = get_track_points(
            points_df,
            next_track_id
        )

        if (
            len(previous_track) < 2
            or
            len(next_track) < 2
        ):

            print(
                "Not enough trajectory points."
            )

            continue

        # ====================================================
        # LOCAL WINDOWS
        # ====================================================

        previous_window = (
            previous_track.tail(
                WINDOW_SIZE
            )
        )

        next_window = (
            next_track.head(
                WINDOW_SIZE
            )
        )

        # ====================================================
        # LOCAL DIRECTION
        # ====================================================

        previous_vector = (
            calculate_direction_vector(
                previous_window
            )
        )

        next_vector = (
            calculate_direction_vector(
                next_window
            )
        )

        previous_angle = (
            previous_vector["angle"]
        )

        next_angle = (
            next_vector["angle"]
        )

        angle_change = (
            normalize_angle_difference(
                previous_angle,
                next_angle
            )
        )

        # ====================================================
        # LOCAL SPEED
        # ====================================================

        previous_speed = (
            calculate_average_speed(
                previous_window
            )
        )

        next_speed = (
            calculate_average_speed(
                next_window
            )
        )

        # ====================================================
        # TEMPORAL GAP
        # ====================================================

        previous_end_frame = int(
            previous_track.iloc[-1][
                "frame"
            ]
        )

        next_start_frame = int(
            next_track.iloc[0][
                "frame"
            ]
        )

        # Frames between both tracks
        gap_frames = max(
            0,
            next_start_frame
            -
            previous_end_frame
            -
            1
        )

        # ====================================================
        # ESTIMATE EVENT FRAME
        # ====================================================

        event_frame = int(
            round(
                (
                    previous_end_frame
                    +
                    next_start_frame
                )
                /
                2
            )
        )

        # ====================================================
        # SCORE
        # ====================================================

        scoring = score_contact(
            gap_frames,
            angle_change,
            previous_speed,
            next_speed
        )

        contact_score = (
            scoring["score"]
        )

        # ====================================================
        # CLASSIFICATION
        # ====================================================

        if (
            gap_frames <= MAX_GAP_FRAMES
            and
            angle_change >= MIN_ANGLE_CHANGE
            and
            contact_score
            >=
            CONTACT_SCORE_THRESHOLD
        ):

            classification = (
                "POSSIBLE_CONTACT"
            )

        else:

            classification = (
                "NO_CONTACT_EVIDENCE"
            )

        # ====================================================
        # CONFIDENCE LABEL
        # ====================================================

        if contact_score >= 0.80:
            evidence_level = "HIGH"

        elif contact_score >= 0.60:
            evidence_level = "MEDIUM"

        else:
            evidence_level = "LOW"

        # ====================================================
        # TERMINAL OUTPUT
        # ====================================================

        print(
            f"Previous track ends: "
            f"frame {previous_end_frame}"
        )

        print(
            f"Next track starts: "
            f"frame {next_start_frame}"
        )

        print(
            f"Gap: "
            f"{gap_frames} frames"
        )

        print(
            f"\nPrevious local angle: "
            f"{previous_angle:.2f} deg"
        )

        print(
            f"Next local angle: "
            f"{next_angle:.2f} deg"
        )

        print(
            f"Direction change: "
            f"{angle_change:.2f} deg"
        )

        print(
            f"\nPrevious local speed: "
            f"{previous_speed:.2f} px/frame"
        )

        print(
            f"Next local speed: "
            f"{next_speed:.2f} px/frame"
        )

        speed_ratio = (
            scoring[
                "speed_change_ratio"
            ]
        )

        if not pd.isna(
            speed_ratio
        ):

            print(
                f"Speed ratio: "
                f"{speed_ratio:.2f}x"
            )

        print(
            f"\nContact score: "
            f"{contact_score:.3f}"
        )

        print(
            f"Evidence: "
            f"{evidence_level}"
        )

        print(
            f"Classification: "
            f"{classification}"
        )

        print(
            f"Estimated event frame: "
            f"{event_frame}"
        )

        # ====================================================
        # SAVE EVENT
        # ====================================================

        events.append(
            {
                "previous_track_id":
                    previous_track_id,

                "next_track_id":
                    next_track_id,

                "previous_end_frame":
                    previous_end_frame,

                "next_start_frame":
                    next_start_frame,

                "gap_frames":
                    gap_frames,

                "estimated_event_frame":
                    event_frame,

                "previous_local_angle_deg":
                    previous_angle,

                "next_local_angle_deg":
                    next_angle,

                "angle_change_deg":
                    angle_change,

                "previous_speed_px_frame":
                    previous_speed,

                "next_speed_px_frame":
                    next_speed,

                "speed_change_ratio":
                    scoring[
                        "speed_change_ratio"
                    ],

                "angle_score":
                    scoring[
                        "angle_score"
                    ],

                "temporal_score":
                    scoring[
                        "temporal_score"
                    ],

                "speed_score":
                    scoring[
                        "speed_score"
                    ],

                "contact_score":
                    contact_score,

                "evidence_level":
                    evidence_level,

                "classification":
                    classification
            }
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    events_df = pd.DataFrame(
        events
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    events_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "CONTACT DETECTION COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"\nOutput:\n"
        f"{OUTPUT_CSV}"
    )

    if len(events_df) > 0:

        possible_contacts = (
            events_df[
                events_df[
                    "classification"
                ]
                ==
                "POSSIBLE_CONTACT"
            ]
        )

        print(
            f"\nPossible contacts detected: "
            f"{len(possible_contacts)}"
        )

        for _, event in (
            possible_contacts.iterrows()
        ):

            print(
                f"\nTrack "
                f"{int(event['previous_track_id'])}"
                f" -> "
                f"{int(event['next_track_id'])}"
            )

            print(
                f"Estimated frame: "
                f"{int(event['estimated_event_frame'])}"
            )

            print(
                f"Score: "
                f"{event['contact_score']:.3f}"
            )


if __name__ == "__main__":
    main()