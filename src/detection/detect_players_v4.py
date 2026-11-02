from ultralytics import YOLO
from pathlib import Path
import argparse
import cv2
import csv
import math


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# YOLO
# ============================================================

MODEL_NAME = "yolo26n.pt"

PERSON_CLASS_ID = 0

CONF_THRESHOLD = 0.30

IMG_SIZE = 1280


# ============================================================
# FIELD FILTER
# ============================================================

FIELD_MIN_Y = 0.28


# ============================================================
# HOME PLATE REGION
# ============================================================

HOME_X_MIN = 0.36
HOME_X_MAX = 0.60

HOME_Y_MIN = 0.28
HOME_Y_MAX = 0.66


# ============================================================
# PITCHER REGION
# ============================================================

PITCHER_X_MIN = 0.25
PITCHER_X_MAX = 0.55

PITCHER_Y_MIN = 0.48
PITCHER_Y_MAX = 0.92


# ============================================================
# TEMPORAL PARAMETERS
# ============================================================

MAX_ROLE_JUMP_PX = 220

TEMPORAL_WEIGHT = 0.35

SPATIAL_WEIGHT = 0.65


# ============================================================
# HOME PLATE PARAMETERS V4
# ============================================================

EXPECTED_HOME_X = 0.49
EXPECTED_HOME_Y = 0.45

HOME_CLUSTER_SIZE = 3

# Minimum horizontal overlap ratio to consider two people
# part of catcher/umpire vertical pair.
MIN_HORIZONTAL_OVERLAP = 0.15

# Temporal information is important once roles have
# already been initialized.
ROLE_TEMPORAL_BONUS = 0.25


# ============================================================
# DRAWING
# ============================================================

DRAW_OTHER_FIELD_PLAYERS = False


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2
        +
        (p1[1] - p2[1]) ** 2
    )


def normalized_distance(
    p1,
    p2,
    width,
    height
):

    dx = (
        p1[0] - p2[0]
    ) / width

    dy = (
        p1[1] - p2[1]
    ) / height

    return math.sqrt(
        dx ** 2
        +
        dy ** 2
    )


def is_inside_region(
    detection,
    xmin,
    xmax,
    ymin,
    ymax
):

    nx = detection["normalized_x"]
    ny = detection["normalized_y"]

    return (
        xmin <= nx <= xmax
        and
        ymin <= ny <= ymax
    )


# ============================================================
# FIELD FILTER
# ============================================================

def filter_field_people(detections):

    field_people = []

    for detection in detections:

        if (
            detection["normalized_y"]
            >=
            FIELD_MIN_Y
        ):

            field_people.append(
                detection
            )

    return field_people


# ============================================================
# PITCHER
# ============================================================

def select_pitcher(
    detections,
    previous_roles,
    frame_width,
    frame_height
):

    candidates = [
        d
        for d in detections
        if is_inside_region(
            d,
            PITCHER_X_MIN,
            PITCHER_X_MAX,
            PITCHER_Y_MIN,
            PITCHER_Y_MAX
        )
    ]

    if not candidates:
        return None

    expected_x = 0.37
    expected_y = 0.68

    previous_pitcher = (
        previous_roles.get(
            "PITCHER"
        )
    )

    best = None
    best_score = float("inf")

    for candidate in candidates:

        nx = candidate["normalized_x"]
        ny = candidate["normalized_y"]

        spatial_score = math.sqrt(
            (nx - expected_x) ** 2
            +
            (ny - expected_y) ** 2
        )

        relative_height = (
            candidate["box_height"]
            /
            frame_height
        )

        size_bonus = min(
            relative_height,
            0.30
        )

        spatial_score -= (
            size_bonus
            *
            0.15
        )

        temporal_score = 0.0

        if previous_pitcher is not None:

            temporal_score = (
                normalized_distance(
                    candidate["center"],
                    previous_pitcher,
                    frame_width,
                    frame_height
                )
            )

        score = (
            spatial_score
            *
            SPATIAL_WEIGHT
            +
            temporal_score
            *
            TEMPORAL_WEIGHT
        )

        if score < best_score:

            best_score = score
            best = candidate

    return best


# ============================================================
# HOME PLATE PEOPLE
# ============================================================

def select_home_plate_people(
    detections,
    pitcher
):

    candidates = []

    for detection in detections:

        if (
            pitcher is not None
            and
            detection is pitcher
        ):
            continue

        if is_inside_region(
            detection,
            HOME_X_MIN,
            HOME_X_MAX,
            HOME_Y_MIN,
            HOME_Y_MAX
        ):

            candidates.append(
                detection
            )

    return candidates


# ============================================================
# HOME CLUSTER
# ============================================================

def select_home_cluster(
    candidates,
    frame_width,
    frame_height
):

    if (
        len(candidates)
        <=
        HOME_CLUSTER_SIZE
    ):

        return list(
            candidates
        )

    expected_home = (
        EXPECTED_HOME_X
        *
        frame_width,

        EXPECTED_HOME_Y
        *
        frame_height
    )

    sorted_candidates = sorted(
        candidates,
        key=lambda d:
            distance(
                d["center"],
                expected_home
            )
    )

    return sorted_candidates[
        :HOME_CLUSTER_SIZE
    ]


# ============================================================
# BBOX HELPERS
# ============================================================

def box_area(detection):

    return (
        detection["box_width"]
        *
        detection["box_height"]
    )


def horizontal_overlap_ratio(
    detection_a,
    detection_b
):
    """
    Horizontal overlap between two bounding boxes.

    0.0 = no horizontal overlap
    1.0 = smallest box fully overlaps horizontally
    """

    ax1, ay1, ax2, ay2 = (
        detection_a["bbox"]
    )

    bx1, by1, bx2, by2 = (
        detection_b["bbox"]
    )

    overlap = max(
        0,
        min(
            ax2,
            bx2
        )
        -
        max(
            ax1,
            bx1
        )
    )

    width_a = max(
        1,
        ax2 - ax1
    )

    width_b = max(
        1,
        bx2 - bx1
    )

    denominator = min(
        width_a,
        width_b
    )

    return (
        overlap
        /
        denominator
    )


def vertical_center_difference(
    detection_a,
    detection_b
):

    return abs(
        detection_a["center"][1]
        -
        detection_b["center"][1]
    )


# ============================================================
# TEMPORAL HELPERS
# ============================================================

def temporal_distance(
    detection,
    previous_position
):

    if previous_position is None:

        return None

    return distance(
        detection["center"],
        previous_position
    )


def closest_to_previous(
    candidates,
    previous_position
):

    if (
        previous_position is None
        or
        not candidates
    ):

        return None

    return min(
        candidates,
        key=lambda d:
            distance(
                d["center"],
                previous_position
            )
    )


# ============================================================
# CATCHER / UMPIRE PAIR SCORE
# ============================================================

def catcher_umpire_pair_score(
    detection_a,
    detection_b,
    previous_roles
):
    """
    Higher score = more likely catcher + umpire pair.

    Main evidence:
    - horizontal overlap
    - vertical alignment
    - temporal consistency
    """

    overlap = (
        horizontal_overlap_ratio(
            detection_a,
            detection_b
        )
    )

    center_a = (
        detection_a["center"]
    )

    center_b = (
        detection_b["center"]
    )

    # Smaller horizontal center separation is useful
    horizontal_distance = abs(
        center_a[0]
        -
        center_b[0]
    )

    average_width = (
        detection_a["box_width"]
        +
        detection_b["box_width"]
    ) / 2.0

    normalized_horizontal_distance = (
        horizontal_distance
        /
        max(
            1.0,
            average_width
        )
    )

    horizontal_closeness = max(
        0.0,
        1.0
        -
        normalized_horizontal_distance
    )

    # Catcher/umpire should not have exactly the same Y center.
    vertical_difference = (
        vertical_center_difference(
            detection_a,
            detection_b
        )
    )

    average_height = (
        detection_a["box_height"]
        +
        detection_b["box_height"]
    ) / 2.0

    normalized_vertical_difference = (
        vertical_difference
        /
        max(
            1.0,
            average_height
        )
    )

    vertical_score = min(
        normalized_vertical_difference,
        1.0
    )

    score = (
        overlap * 0.55
        +
        horizontal_closeness * 0.25
        +
        vertical_score * 0.20
    )

    # ========================================================
    # Temporal consistency
    # ========================================================

    previous_catcher = (
        previous_roles.get(
            "CATCHER"
        )
    )

    previous_umpire = (
        previous_roles.get(
            "UMPIRE"
        )
    )

    if (
        previous_catcher is not None
        and
        previous_umpire is not None
    ):

        cost_ab = (
            distance(
                detection_a["center"],
                previous_catcher
            )
            +
            distance(
                detection_b["center"],
                previous_umpire
            )
        )

        cost_ba = (
            distance(
                detection_b["center"],
                previous_catcher
            )
            +
            distance(
                detection_a["center"],
                previous_umpire
            )
        )

        best_cost = min(
            cost_ab,
            cost_ba
        )

        temporal_bonus = max(
            0.0,
            1.0
            -
            (
                best_cost
                /
                (
                    2.0
                    *
                    MAX_ROLE_JUMP_PX
                )
            )
        )

        score += (
            temporal_bonus
            *
            ROLE_TEMPORAL_BONUS
        )

    return score


# ============================================================
# FIND CATCHER/UMPIRE PAIR
# ============================================================

def find_catcher_umpire_pair(
    candidates,
    previous_roles
):

    if len(candidates) < 2:

        return None

    best_pair = None
    best_score = -float("inf")

    for i in range(
        len(candidates)
    ):

        for j in range(
            i + 1,
            len(candidates)
        ):

            a = candidates[i]
            b = candidates[j]

            score = (
                catcher_umpire_pair_score(
                    a,
                    b,
                    previous_roles
                )
            )

            overlap = (
                horizontal_overlap_ratio(
                    a,
                    b
                )
            )

            # We allow low overlap only if temporal information
            # already strongly supports the pair.
            if (
                overlap
                <
                MIN_HORIZONTAL_OVERLAP
            ):

                previous_catcher = (
                    previous_roles.get(
                        "CATCHER"
                    )
                )

                previous_umpire = (
                    previous_roles.get(
                        "UMPIRE"
                    )
                )

                if (
                    previous_catcher is None
                    or
                    previous_umpire is None
                ):

                    score -= 0.30

            if score > best_score:

                best_score = score

                best_pair = (
                    a,
                    b
                )

    return best_pair


# ============================================================
# ASSIGN CATCHER AND UMPIRE
# ============================================================

def assign_catcher_umpire(
    pair,
    previous_roles
):

    result = {
        "CATCHER": None,
        "UMPIRE": None
    }

    if pair is None:

        return result

    if len(pair) == 1:

        candidate = pair[0]

        previous_catcher = (
            previous_roles.get(
                "CATCHER"
            )
        )

        previous_umpire = (
            previous_roles.get(
                "UMPIRE"
            )
        )

        if (
            previous_catcher is not None
            or
            previous_umpire is not None
        ):

            catcher_dist = (
                float("inf")
            )

            umpire_dist = (
                float("inf")
            )

            if previous_catcher is not None:

                catcher_dist = distance(
                    candidate["center"],
                    previous_catcher
                )

            if previous_umpire is not None:

                umpire_dist = distance(
                    candidate["center"],
                    previous_umpire
                )

            if catcher_dist <= umpire_dist:

                result[
                    "CATCHER"
                ] = candidate

            else:

                result[
                    "UMPIRE"
                ] = candidate

        else:

            result[
                "CATCHER"
            ] = candidate

        return result

    a = pair[0]
    b = pair[1]

    previous_catcher = (
        previous_roles.get(
            "CATCHER"
        )
    )

    previous_umpire = (
        previous_roles.get(
            "UMPIRE"
        )
    )

    # ========================================================
    # TEMPORAL ASSIGNMENT
    # ========================================================

    if (
        previous_catcher is not None
        and
        previous_umpire is not None
    ):

        cost_ab = (
            distance(
                a["center"],
                previous_catcher
            )
            +
            distance(
                b["center"],
                previous_umpire
            )
        )

        cost_ba = (
            distance(
                b["center"],
                previous_catcher
            )
            +
            distance(
                a["center"],
                previous_umpire
            )
        )

        if cost_ab <= cost_ba:

            result[
                "CATCHER"
            ] = a

            result[
                "UMPIRE"
            ] = b

        else:

            result[
                "CATCHER"
            ] = b

            result[
                "UMPIRE"
            ] = a

        return result

    # ========================================================
    # VERTICAL INITIALIZATION
    # ========================================================
    #
    # Image coordinates:
    # smaller y = visually higher
    # larger y  = visually lower
    #
    # Umpire generally stands behind/above catcher.
    # Catcher generally crouches lower.
    # ========================================================

    if (
        a["center"][1]
        >
        b["center"][1]
    ):

        catcher = a
        umpire = b

    else:

        catcher = b
        umpire = a

    result[
        "CATCHER"
    ] = catcher

    result[
        "UMPIRE"
    ] = umpire

    return result


# ============================================================
# BATTER TEMPORAL CORRECTION
# ============================================================

def stabilize_batter(
    batter,
    cluster,
    catcher,
    umpire,
    previous_roles
):

    previous_batter = (
        previous_roles.get(
            "BATTER"
        )
    )

    if previous_batter is None:

        return batter

    if batter is not None:

        current_jump = distance(
            batter["center"],
            previous_batter
        )

        if (
            current_jump
            <=
            MAX_ROLE_JUMP_PX
        ):

            return batter

    # Find closest unused candidate to previous batter

    available = []

    for candidate in cluster:

        if (
            catcher is not None
            and
            candidate is catcher
        ):
            continue

        if (
            umpire is not None
            and
            candidate is umpire
        ):
            continue

        available.append(
            candidate
        )

    candidate = (
        closest_to_previous(
            available,
            previous_batter
        )
    )

    if candidate is None:

        return batter

    candidate_jump = distance(
        candidate["center"],
        previous_batter
    )

    if (
        candidate_jump
        <=
        MAX_ROLE_JUMP_PX
    ):

        return candidate

    return batter


# ============================================================
# HOME ROLE ASSIGNMENT V4
# ============================================================

def assign_home_roles(
    candidates,
    previous_roles,
    frame_width,
    frame_height
):

    roles = {
        "BATTER": None,
        "CATCHER": None,
        "UMPIRE": None
    }

    if not candidates:

        return roles

    cluster = (
        select_home_cluster(
            candidates,
            frame_width,
            frame_height
        )
    )

    # ========================================================
    # 3 PEOPLE
    # ========================================================

    if len(cluster) >= 3:

        pair = (
            find_catcher_umpire_pair(
                cluster,
                previous_roles
            )
        )

        if pair is None:

            return roles

        pair_roles = (
            assign_catcher_umpire(
                pair,
                previous_roles
            )
        )

        catcher = (
            pair_roles[
                "CATCHER"
            ]
        )

        umpire = (
            pair_roles[
                "UMPIRE"
            ]
        )

        roles[
            "CATCHER"
        ] = catcher

        roles[
            "UMPIRE"
        ] = umpire

        remaining = [
            detection
            for detection
            in cluster
            if (
                detection is not catcher
                and
                detection is not umpire
            )
        ]

        batter = None

        if remaining:

            batter = (
                remaining[0]
            )

        batter = (
            stabilize_batter(
                batter,
                cluster,
                catcher,
                umpire,
                previous_roles
            )
        )

        roles[
            "BATTER"
        ] = batter

        return roles

    # ========================================================
    # 2 PEOPLE
    # ========================================================

    if len(cluster) == 2:

        a = cluster[0]
        b = cluster[1]

        previous_batter = (
            previous_roles.get(
                "BATTER"
            )
        )

        # ----------------------------------------------------
        # Preserve previously known batter if possible
        # ----------------------------------------------------

        if previous_batter is not None:

            distance_a = distance(
                a["center"],
                previous_batter
            )

            distance_b = distance(
                b["center"],
                previous_batter
            )

            if (
                min(
                    distance_a,
                    distance_b
                )
                <=
                MAX_ROLE_JUMP_PX
            ):

                if distance_a <= distance_b:

                    roles[
                        "BATTER"
                    ] = a

                    other = b

                else:

                    roles[
                        "BATTER"
                    ] = b

                    other = a

                pair_roles = (
                    assign_catcher_umpire(
                        [other],
                        previous_roles
                    )
                )

                roles[
                    "CATCHER"
                ] = (
                    pair_roles[
                        "CATCHER"
                    ]
                )

                roles[
                    "UMPIRE"
                ] = (
                    pair_roles[
                        "UMPIRE"
                    ]
                )

                return roles

        # ----------------------------------------------------
        # Without batter history:
        # determine whether these two look like
        # catcher + umpire.
        # ----------------------------------------------------

        overlap = (
            horizontal_overlap_ratio(
                a,
                b
            )
        )

        if (
            overlap
            >=
            MIN_HORIZONTAL_OVERLAP
        ):

            pair_roles = (
                assign_catcher_umpire(
                    [a, b],
                    previous_roles
                )
            )

            roles[
                "CATCHER"
            ] = (
                pair_roles[
                    "CATCHER"
                ]
            )

            roles[
                "UMPIRE"
            ] = (
                pair_roles[
                    "UMPIRE"
                ]
            )

            return roles

        # ----------------------------------------------------
        # Otherwise likely batter + one home plate person.
        #
        # Batter is generally more horizontally separated
        # from the expected center of plate.
        # ----------------------------------------------------

        expected_home_x_px = (
            EXPECTED_HOME_X
            *
            frame_width
        )

        offset_a = abs(
            a["center"][0]
            -
            expected_home_x_px
        )

        offset_b = abs(
            b["center"][0]
            -
            expected_home_x_px
        )

        if offset_a >= offset_b:

            batter = a
            other = b

        else:

            batter = b
            other = a

        roles[
            "BATTER"
        ] = batter

        pair_roles = (
            assign_catcher_umpire(
                [other],
                previous_roles
            )
        )

        roles[
            "CATCHER"
        ] = (
            pair_roles[
                "CATCHER"
            ]
        )

        roles[
            "UMPIRE"
        ] = (
            pair_roles[
                "UMPIRE"
            ]
        )

        return roles

    # ========================================================
    # 1 PERSON
    # ========================================================

    candidate = (
        cluster[0]
    )

    role_distances = []

    for role_name in [
        "BATTER",
        "CATCHER",
        "UMPIRE"
    ]:

        previous_position = (
            previous_roles.get(
                role_name
            )
        )

        if previous_position is None:
            continue

        role_distances.append(
            (
                distance(
                    candidate["center"],
                    previous_position
                ),
                role_name
            )
        )

    if role_distances:

        role_distances.sort(
            key=lambda item:
                item[0]
        )

        best_distance = (
            role_distances[0][0]
        )

        best_role = (
            role_distances[0][1]
        )

        if (
            best_distance
            <=
            MAX_ROLE_JUMP_PX
        ):

            roles[
                best_role
            ] = candidate

            return roles

    horizontal_offset = abs(
        candidate["normalized_x"]
        -
        EXPECTED_HOME_X
    )

    if (
        horizontal_offset
        >=
        0.045
    ):

        roles[
            "BATTER"
        ] = candidate

    else:

        roles[
            "CATCHER"
        ] = candidate

    return roles


# ============================================================
# DRAW ROLE
# ============================================================

def draw_role(
    frame,
    detection,
    role
):

    if detection is None:
        return

    x1, y1, x2, y2 = (
        detection["bbox"]
    )

    confidence = (
        detection[
            "confidence"
        ]
    )

    center = (
        detection[
            "center"
        ]
    )

    cv2.rectangle(
        frame,
        (
            x1,
            y1
        ),
        (
            x2,
            y2
        ),
        (
            255,
            255,
            255
        ),
        3
    )

    cv2.circle(
        frame,
        center,
        5,
        (
            255,
            255,
            255
        ),
        -1
    )

    label = (
        f"{role} "
        f"{confidence:.2f}"
    )

    cv2.putText(
        frame,
        label,
        (
            x1,
            max(
                30,
                y1 - 10
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (
            255,
            255,
            255
        ),
        2
    )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = (
        argparse.ArgumentParser(
            description=(
                "BaseVisionAI - Player Detection V4"
            )
        )
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Input video path"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where results will be saved"
    )

    return (
        parser.parse_args()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = (
        parse_args()
    )

    # ========================================================
    # INPUT
    # ========================================================

    video_path = Path(
        args.video
    )

    if not video_path.is_absolute():

        video_path = (
            PROJECT_ROOT
            /
            video_path
        )

    video_path = (
        video_path.resolve()
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    output_dir = Path(
        args.output_dir
    )

    if not output_dir.is_absolute():

        output_dir = (
            PROJECT_ROOT
            /
            output_dir
        )

    output_dir = (
        output_dir.resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_video = (
        output_dir
        /
        "player_detection_v4.mp4"
    )

    output_csv = (
        output_dir
        /
        "player_detection_v4.csv"
    )

    # ========================================================
    # HEADER
    # ========================================================

    print("=" * 70)

    print(
        "BaseVisionAI - Player Detection V4"
    )

    print("=" * 70)

    print(
        f"Video: "
        f"{video_path}"
    )

    print(
        f"Output directory: "
        f"{output_dir}"
    )

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video not found:\n"
            f"{video_path}"
        )

    # ========================================================
    # MODEL
    # ========================================================

    model = YOLO(
        MODEL_NAME
    )

    # ========================================================
    # OPEN VIDEO
    # ========================================================

    cap = cv2.VideoCapture(
        str(
            video_path
        )
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n"
            f"{video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print(
        f"FPS: "
        f"{fps:.2f}"
    )

    print(
        f"Resolution: "
        f"{width}x{height}"
    )

    print(
        f"Frames: "
        f"{total_frames}"
    )

    # ========================================================
    # VIDEO WRITER
    # ========================================================

    fourcc = (
        cv2.VideoWriter_fourcc(
            *"mp4v"
        )
    )

    writer = cv2.VideoWriter(
        str(
            output_video
        ),
        fourcc,
        fps,
        (
            width,
            height
        )
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            f"Could not create output video:\n"
            f"{output_video}"
        )

    # ========================================================
    # HISTORY
    # ========================================================

    previous_roles = {
        "PITCHER": None,
        "BATTER": None,
        "CATCHER": None,
        "UMPIRE": None
    }

    # ========================================================
    # CSV
    # ========================================================

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        csv_writer = (
            csv.writer(
                csv_file
            )
        )

        csv_writer.writerow(
            [
                "frame",
                "role",
                "confidence",

                "x1",
                "y1",
                "x2",
                "y2",

                "center_x",
                "center_y",

                "box_width",
                "box_height",

                "normalized_x",
                "normalized_y"
            ]
        )

        frame_number = 0

        # ====================================================
        # FRAME LOOP
        # ====================================================

        while True:

            ret, frame = (
                cap.read()
            )

            if not ret:
                break

            frame_number += 1

            # =================================================
            # YOLO
            # =================================================

            results = (
                model.predict(
                    frame,
                    imgsz=IMG_SIZE,
                    conf=CONF_THRESHOLD,
                    device=0,
                    classes=[
                        PERSON_CLASS_ID
                    ],
                    verbose=False
                )
            )

            boxes = (
                results[0].boxes
            )

            all_detections = []

            if boxes is not None:

                for box in boxes:

                    confidence = float(
                        box.conf[0]
                    )

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[
                            0
                        ].tolist()
                    )

                    center_x = int(
                        (
                            x1
                            +
                            x2
                        )
                        /
                        2
                    )

                    center_y = int(
                        (
                            y1
                            +
                            y2
                        )
                        /
                        2
                    )

                    box_width = (
                        x2 - x1
                    )

                    box_height = (
                        y2 - y1
                    )

                    normalized_x = (
                        center_x
                        /
                        width
                    )

                    normalized_y = (
                        center_y
                        /
                        height
                    )

                    all_detections.append(
                        {
                            "confidence":
                                confidence,

                            "bbox":
                                (
                                    x1,
                                    y1,
                                    x2,
                                    y2
                                ),

                            "center":
                                (
                                    center_x,
                                    center_y
                                ),

                            "box_width":
                                box_width,

                            "box_height":
                                box_height,

                            "normalized_x":
                                normalized_x,

                            "normalized_y":
                                normalized_y
                        }
                    )

            # =================================================
            # FIELD FILTER
            # =================================================

            field_people = (
                filter_field_people(
                    all_detections
                )
            )

            # =================================================
            # PITCHER
            # =================================================

            pitcher = (
                select_pitcher(
                    field_people,
                    previous_roles,
                    width,
                    height
                )
            )

            # =================================================
            # HOME PLATE
            # =================================================

            home_candidates = (
                select_home_plate_people(
                    field_people,
                    pitcher
                )
            )

            home_roles = (
                assign_home_roles(
                    home_candidates,
                    previous_roles,
                    width,
                    height
                )
            )

            roles = {
                "PITCHER":
                    pitcher,

                "BATTER":
                    home_roles[
                        "BATTER"
                    ],

                "CATCHER":
                    home_roles[
                        "CATCHER"
                    ],

                "UMPIRE":
                    home_roles[
                        "UMPIRE"
                    ]
            }

            # =================================================
            # UPDATE TEMPORAL HISTORY
            # =================================================

            for (
                role_name,
                detection
            ) in roles.items():

                if detection is not None:

                    previous_roles[
                        role_name
                    ] = (
                        detection[
                            "center"
                        ]
                    )

            # =================================================
            # DEBUG ROI
            # =================================================

            field_line_y = int(
                FIELD_MIN_Y
                *
                height
            )

            cv2.line(
                frame,
                (
                    0,
                    field_line_y
                ),
                (
                    width,
                    field_line_y
                ),
                (
                    120,
                    120,
                    120
                ),
                1
            )

            hx1 = int(
                HOME_X_MIN
                *
                width
            )

            hx2 = int(
                HOME_X_MAX
                *
                width
            )

            hy1 = int(
                HOME_Y_MIN
                *
                height
            )

            hy2 = int(
                HOME_Y_MAX
                *
                height
            )

            cv2.rectangle(
                frame,
                (
                    hx1,
                    hy1
                ),
                (
                    hx2,
                    hy2
                ),
                (
                    120,
                    120,
                    120
                ),
                1
            )

            # =================================================
            # DRAW ROLES
            # =================================================

            for (
                role_name,
                detection
            ) in roles.items():

                draw_role(
                    frame,
                    detection,
                    role_name
                )

            # =================================================
            # CSV
            # =================================================

            for (
                role_name,
                detection
            ) in roles.items():

                if detection is None:
                    continue

                x1, y1, x2, y2 = (
                    detection[
                        "bbox"
                    ]
                )

                center_x, center_y = (
                    detection[
                        "center"
                    ]
                )

                csv_writer.writerow(
                    [
                        frame_number,

                        role_name,

                        detection[
                            "confidence"
                        ],

                        x1,
                        y1,
                        x2,
                        y2,

                        center_x,
                        center_y,

                        detection[
                            "box_width"
                        ],

                        detection[
                            "box_height"
                        ],

                        detection[
                            "normalized_x"
                        ],

                        detection[
                            "normalized_y"
                        ]
                    ]
                )

            # =================================================
            # STATUS
            # =================================================

            cv2.putText(
                frame,
                f"Frame: {frame_number}",
                (
                    20,
                    35
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    255,
                    255,
                    255
                ),
                2
            )

            cv2.putText(
                frame,
                (
                    f"YOLO people: "
                    f"{len(all_detections)}"
                ),
                (
                    20,
                    65
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (
                    255,
                    255,
                    255
                ),
                2
            )

            cv2.putText(
                frame,
                (
                    f"Field people: "
                    f"{len(field_people)}"
                ),
                (
                    20,
                    95
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (
                    255,
                    255,
                    255
                ),
                2
            )

            detected_roles = [
                role
                for (
                    role,
                    detection
                )
                in roles.items()
                if detection is not None
            ]

            cv2.putText(
                frame,
                (
                    "Roles: "
                    +
                    ", ".join(
                        detected_roles
                    )
                ),
                (
                    20,
                    125
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (
                    255,
                    255,
                    255
                ),
                2
            )

            # =================================================
            # SAVE VIDEO
            # =================================================

            writer.write(
                frame
            )

            # =================================================
            # TERMINAL
            # =================================================

            print(
                f"Frame "
                f"{frame_number:03d}/"
                f"{total_frames} | "
                f"YOLO: "
                f"{len(all_detections)} | "
                f"Field: "
                f"{len(field_people)} | "
                f"Roles: "
                f"{detected_roles}"
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    writer.release()

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "PLAYER DETECTION V4 COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        "\nVideo:"
    )

    print(
        output_video
    )

    print(
        "\nCSV:"
    )

    print(
        output_csv
    )


if __name__ == "__main__":
    main()