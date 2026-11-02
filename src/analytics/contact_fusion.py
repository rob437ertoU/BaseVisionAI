from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BALL_TRACK_CSV = (
    PROJECT_ROOT
    / "results"
    / "ball_tracking_v3_1.csv"
)

BALL_CONTACT_CSV = (
    PROJECT_ROOT
    / "results"
    / "contact_detection.csv"
)

SWING_PHASES_CSV = (
    PROJECT_ROOT
    / "results"
    / "swing_phases_v1_1.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "contact_fusion_v1_1.csv"
)

OUTPUT_SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "contact_fusion_summary_v1_1.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

SEARCH_PHASES = {
    "SWING",
    "PEAK_SWING"
}

BALL_EVENT_WINDOW = 6
TRACK_TRANSITION_WINDOW = 6

MIN_CONTACT_SCORE = 0.55


# ============================================================
# HELPERS
# ============================================================

def normalize_series(series):

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    valid = values.dropna()

    if valid.empty:
        return pd.Series(
            0.0,
            index=series.index
        )

    low = valid.min()
    high = valid.max()

    if abs(high - low) < 1e-9:
        return pd.Series(
            0.0,
            index=series.index
        )

    normalized = (
        (values - low)
        /
        (high - low)
    )

    return normalized.clip(
        0.0,
        1.0
    )


def temporal_score(
    frame,
    event_frame,
    max_distance
):

    if event_frame is None:
        return 0.0

    distance = abs(
        int(frame)
        -
        int(event_frame)
    )

    if distance > max_distance:
        return 0.0

    return 1.0 - (
        distance
        /
        max_distance
    )


# ============================================================
# CONTACT CSV EVENT
# ============================================================

def find_contact_csv_event(contact_df):

    if contact_df.empty:
        return None

    preferred_columns = [
        "event_frame",
        "candidate_event_frame",
        "contact_frame",
        "estimated_contact_frame",
        "estimated_event_frame"
    ]

    for column in preferred_columns:

        if column not in contact_df.columns:
            continue

        values = pd.to_numeric(
            contact_df[column],
            errors="coerce"
        ).dropna()

        if not values.empty:

            return int(
                round(
                    float(values.iloc[0])
                )
            )

    # --------------------------------------------------------
    # Fallback:
    # search any column containing "frame"
    # --------------------------------------------------------

    frame_columns = [
        column
        for column in contact_df.columns
        if "frame" in column.lower()
    ]

    print(
        "\nContact CSV frame columns:",
        frame_columns
    )

    return None


# ============================================================
# BALL TRACK TRANSITION
# ============================================================

def find_track_transition(ball_df):

    if "track_id" not in ball_df.columns:

        print(
            "\nNo track_id column found."
        )

        return None, None, None

    valid = ball_df.copy()

    valid["track_id"] = pd.to_numeric(
        valid["track_id"],
        errors="coerce"
    )

    valid = valid.dropna(
        subset=[
            "frame",
            "track_id"
        ]
    )

    if valid.empty:

        return None, None, None

    valid["frame"] = (
        valid["frame"]
        .astype(int)
    )

    valid["track_id"] = (
        valid["track_id"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Ignore WAITING / RESET / LOST if state is available.
    # Keep physical tracking and Kalman predictions.
    # --------------------------------------------------------

    if "state" in valid.columns:

        useful_states = {
            "TRACKING",
            "KALMAN"
        }

        valid = valid[
            valid["state"]
            .astype(str)
            .str.upper()
            .isin(useful_states)
        ]

    if valid.empty:

        return None, None, None

    # --------------------------------------------------------
    # Build one segment per track
    # --------------------------------------------------------

    segments = (
        valid
        .groupby("track_id")
        ["frame"]
        .agg(
            start_frame="min",
            end_frame="max",
            observations="count"
        )
        .reset_index()
        .sort_values("start_frame")
        .reset_index(drop=True)
    )

    print(
        "\nBall track segments:"
    )

    print(
        segments.to_string(
            index=False
        )
    )

    if len(segments) < 2:

        return None, None, None

    best_transition = None
    best_gap = None

    # --------------------------------------------------------
    # Find closest consecutive track segments
    # --------------------------------------------------------

    for i in range(
        len(segments) - 1
    ):

        current_track = segments.iloc[i]
        next_track = segments.iloc[i + 1]

        current_end = int(
            current_track[
                "end_frame"
            ]
        )

        next_start = int(
            next_track[
                "start_frame"
            ]
        )

        gap = (
            next_start
            -
            current_end
        )

        if gap < 0:
            continue

        if (
            best_gap is None
            or
            gap < best_gap
        ):

            best_gap = gap

            best_transition = (
                current_end,
                next_start
            )

    if best_transition is None:

        return None, None, None

    previous_end = (
        best_transition[0]
    )

    next_start = (
        best_transition[1]
    )

    # --------------------------------------------------------
    # Midpoint of the discontinuity
    # --------------------------------------------------------

    transition_frame = int(
        round(
            (
                previous_end
                +
                next_start
            )
            /
            2
        )
    )

    return (
        transition_frame,
        previous_end,
        next_start
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("BaseVisionAI - Contact Fusion V1.1")
    print("=" * 80)

    # ========================================================
    # CHECK FILES
    # ========================================================

    for path in [
        BALL_TRACK_CSV,
        BALL_CONTACT_CSV,
        SWING_PHASES_CSV
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing input file:\n"
                f"{path}"
            )

    # ========================================================
    # LOAD
    # ========================================================

    ball_df = pd.read_csv(
        BALL_TRACK_CSV
    )

    contact_df = pd.read_csv(
        BALL_CONTACT_CSV
    )

    swing_df = pd.read_csv(
        SWING_PHASES_CSV
    )

    print(
        f"\nBall frames: "
        f"{len(ball_df)}"
    )

    print(
        f"Swing frames: "
        f"{len(swing_df)}"
    )

    # ========================================================
    # CONTACT HEURISTIC EVENT
    # ========================================================

    contact_csv_event = (
        find_contact_csv_event(
            contact_df
        )
    )

    print(
        "\nContact heuristic event frame:",
        contact_csv_event
    )

    # ========================================================
    # BALL TRACK TRANSITION
    # ========================================================

    (
        track_transition_frame,
        previous_track_end,
        next_track_start

    ) = find_track_transition(
        ball_df
    )

    print(
        "\nTrack transition:"
    )

    print(
        f"Previous track end: "
        f"{previous_track_end}"
    )

    print(
        f"Next track start: "
        f"{next_track_start}"
    )

    print(
        f"Estimated transition frame: "
        f"{track_transition_frame}"
    )

    # ========================================================
    # DECIDE PRIMARY BALL EVENT
    # ========================================================

    if contact_csv_event is not None:

        primary_ball_event = (
            contact_csv_event
        )

        ball_event_source = (
            "CONTACT_HEURISTIC"
        )

    elif track_transition_frame is not None:

        primary_ball_event = (
            track_transition_frame
        )

        ball_event_source = (
            "TRACK_TRANSITION_FALLBACK"
        )

    else:

        primary_ball_event = None

        ball_event_source = (
            "NO_BALL_EVENT"
        )

    print(
        "\nPrimary ball event:"
    )

    print(
        f"Frame: {primary_ball_event}"
    )

    print(
        f"Source: {ball_event_source}"
    )

    # ========================================================
    # SWING SCORE
    # ========================================================

    if (
        "swing_activity_score"
        not in swing_df.columns
    ):

        raise ValueError(
            "Missing swing_activity_score "
            "in swing phases CSV."
        )

    swing_df[
        "swing_score_normalized"
    ] = normalize_series(
        swing_df[
            "swing_activity_score"
        ]
    )

    # ========================================================
    # FUSION
    # ========================================================

    rows = []

    for _, row in swing_df.iterrows():

        frame = int(
            row["frame"]
        )

        phase = str(
            row["swing_phase"]
        )

        swing_score = float(
            row[
                "swing_score_normalized"
            ]
        )

        ball_score = temporal_score(
            frame,
            primary_ball_event,
            BALL_EVENT_WINDOW
        )

        transition_score = temporal_score(
            frame,
            track_transition_frame,
            TRACK_TRANSITION_WINDOW
        )

        # ----------------------------------------------------
        # Important:
        #
        # If the contact heuristic exists, use three signals.
        #
        # If not, track transition is only one ball source,
        # so avoid counting it twice.
        # ----------------------------------------------------

        if contact_csv_event is not None:

            fusion_score = (
                0.40
                *
                swing_score

                +

                0.40
                *
                ball_score

                +

                0.20
                *
                transition_score
            )

        elif track_transition_frame is not None:

            fusion_score = (
                0.45
                *
                swing_score

                +

                0.55
                *
                transition_score
            )

        else:

            fusion_score = (
                swing_score
            )

        valid_phase = (
            phase
            in
            SEARCH_PHASES
        )

        if not valid_phase:

            fusion_score *= 0.20

        rows.append(
            {
                "frame":
                    frame,

                "phase":
                    phase,

                "swing_score":
                    swing_score,

                "ball_event_score":
                    ball_score,

                "track_transition_score":
                    transition_score,

                "fusion_score":
                    fusion_score,

                "valid_contact_phase":
                    valid_phase
            }
        )

    result = pd.DataFrame(
        rows
    )

    # ========================================================
    # FIND BEST FRAME
    # ========================================================

    valid_result = result[
        result[
            "valid_contact_phase"
        ]
    ].copy()

    if valid_result.empty:

        raise RuntimeError(
            "No SWING / PEAK_SWING "
            "frames were found."
        )

    best_index = (
        valid_result[
            "fusion_score"
        ]
        .idxmax()
    )

    best_row = (
        result.loc[
            best_index
        ]
    )

    predicted_frame = int(
        best_row[
            "frame"
        ]
    )

    fusion_score = float(
        best_row[
            "fusion_score"
        ]
    )

    # ========================================================
    # STATUS
    # ========================================================

    if fusion_score >= MIN_CONTACT_SCORE:

        status = (
            "CONTACT_CANDIDATE"
        )

    else:

        status = (
            "LOW_CONFIDENCE_EVENT"
        )

    # ========================================================
    # SAVE FRAME SCORES
    # ========================================================

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = pd.DataFrame(
        [
            {
                "predicted_contact_frame":
                    predicted_frame,

                "fusion_score":
                    fusion_score,

                "status":
                    status,

                "phase":
                    best_row[
                        "phase"
                    ],

                "swing_score":
                    best_row[
                        "swing_score"
                    ],

                "ball_event_score":
                    best_row[
                        "ball_event_score"
                    ],

                "track_transition_score":
                    best_row[
                        "track_transition_score"
                    ],

                "contact_csv_event_frame":
                    contact_csv_event,

                "track_transition_frame":
                    track_transition_frame,

                "previous_track_end":
                    previous_track_end,

                "next_track_start":
                    next_track_start,

                "ball_event_source":
                    ball_event_source
            }
        ]
    )

    summary.to_csv(
        OUTPUT_SUMMARY_CSV,
        index=False
    )

    # ========================================================
    # TERMINAL
    # ========================================================

    print(
        "\n"
        +
        "=" * 80
    )

    print(
        "CONTACT FUSION RESULT V1.1"
    )

    print(
        "=" * 80
    )

    print(
        f"\nPredicted contact frame: "
        f"{predicted_frame}"
    )

    print(
        f"Fusion score: "
        f"{fusion_score:.3f}"
    )

    print(
        f"Phase: "
        f"{best_row['phase']}"
    )

    print(
        f"Swing score: "
        f"{best_row['swing_score']:.3f}"
    )

    print(
        f"Ball event score: "
        f"{best_row['ball_event_score']:.3f}"
    )

    print(
        f"Track transition score: "
        f"{best_row['track_transition_score']:.3f}"
    )

    print(
        f"Ball event source: "
        f"{ball_event_source}"
    )

    print(
        f"Status: "
        f"{status}"
    )

    print(
        "\n"
        +
        "=" * 80
    )

    print(
        "CONTACT FUSION V1.1 COMPLETED"
    )

    print(
        "=" * 80
    )

    print(
        f"\nFrame scores:\n"
        f"{OUTPUT_CSV}"
    )

    print(
        f"\nSummary:\n"
        f"{OUTPUT_SUMMARY_CSV}"
    )


if __name__ == "__main__":
    main()