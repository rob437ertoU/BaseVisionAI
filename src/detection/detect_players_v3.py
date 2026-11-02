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
# TEMPORAL STABILIZATION
# ============================================================

MAX_ROLE_JUMP_PX = 220

TEMPORAL_WEIGHT = 0.35
SPATIAL_WEIGHT = 0.65


# ============================================================
# HOME PLATE ROLE SETTINGS V3
# ============================================================
#
# Main idea:
#
# Batter should NOT be identified by "left/right".
#
# Batter is generally separated horizontally from the compact
# catcher + umpire pair.
#
# Catcher and umpire usually overlap or remain relatively close.
#
# Temporal positions are then used to stabilize identities.
# ============================================================

EXPECTED_HOME_X = 0.49
EXPECTED_HOME_Y = 0.45

# Maximum people retained around home plate.
HOME_CLUSTER_SIZE = 3


# ============================================================
# DRAWING
# ============================================================

DRAW_OTHER_FIELD_PLAYERS = False


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def distance(p1, p2):
    """
    Euclidean pixel distance.
    """

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
    """
    Distance normalized relative to image dimensions.
    """

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
    """
    Check if detection center is inside normalized region.
    """

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
    """
    Remove most spectators.

    People above FIELD_MIN_Y are usually spectators
    for this broadcast camera family.
    """

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
    """
    Select pitcher using spatial + temporal information.
    """

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

    best = None
    best_score = float("inf")

    previous_pitcher = (
        previous_roles.get(
            "PITCHER"
        )
    )

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
            size_bonus * 0.15
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
    """
    Get people around home plate.
    """

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
    """
    Retain the people closest to the expected home plate area.

    This avoids accidentally including another nearby fielder.
    """

    if len(candidates) <= HOME_CLUSTER_SIZE:
        return list(candidates)

    expected_home = (
        EXPECTED_HOME_X * frame_width,
        EXPECTED_HOME_Y * frame_height
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
# CLOSEST PAIR
# ============================================================

def find_closest_pair(
    detections,
    frame_width,
    frame_height
):
    """
    Find the two most spatially compact people.

    Around home plate this pair is normally catcher + umpire.

    The remaining person is the batter.
    """

    if len(detections) < 2:
        return None, None

    best_pair = None
    best_distance = float("inf")

    for i in range(
        len(detections)
    ):

        for j in range(
            i + 1,
            len(detections)
        ):

            d = normalized_distance(
                detections[i]["center"],
                detections[j]["center"],
                frame_width,
                frame_height
            )

            if d < best_distance:

                best_distance = d

                best_pair = (
                    detections[i],
                    detections[j]
                )

    return (
        best_pair,
        best_distance
    )


# ============================================================
# TEMPORAL HELPER
# ============================================================

def closest_to_previous(
    candidates,
    previous_position
):
    """
    Return candidate closest to previous role position.
    """

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
# CATCHER / UMPIRE ASSIGNMENT
# ============================================================

def assign_catcher_umpire(
    pair,
    previous_roles
):
    """
    Assign catcher and umpire inside the compact pair.

    If temporal history exists, use it.

    Otherwise use box geometry only as initialization.

    NOTE:
    The downstream swing pipeline mainly depends on BATTER,
    so catcher/umpire distinction is secondary here.
    """

    roles = {
        "CATCHER": None,
        "UMPIRE": None
    }

    if not pair:
        return roles

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
            and
            previous_umpire is not None
        ):

            catcher_distance = distance(
                candidate["center"],
                previous_catcher
            )

            umpire_distance = distance(
                candidate["center"],
                previous_umpire
            )

            if catcher_distance <= umpire_distance:
                roles["CATCHER"] = candidate
            else:
                roles["UMPIRE"] = candidate

        elif previous_catcher is not None:

            roles["CATCHER"] = candidate

        elif previous_umpire is not None:

            roles["UMPIRE"] = candidate

        else:

            roles["CATCHER"] = candidate

        return roles

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
    # Both role histories available
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

            roles["CATCHER"] = a
            roles["UMPIRE"] = b

        else:

            roles["CATCHER"] = b
            roles["UMPIRE"] = a

        return roles

    # ========================================================
    # Only catcher history
    # ========================================================

    if previous_catcher is not None:

        catcher = min(
            pair,
            key=lambda d:
                distance(
                    d["center"],
                    previous_catcher
                )
        )

        other = (
            b
            if catcher is a
            else a
        )

        roles["CATCHER"] = catcher
        roles["UMPIRE"] = other

        return roles

    # ========================================================
    # Only umpire history
    # ========================================================

    if previous_umpire is not None:

        umpire = min(
            pair,
            key=lambda d:
                distance(
                    d["center"],
                    previous_umpire
                )
        )

        other = (
            b
            if umpire is a
            else a
        )

        roles["UMPIRE"] = umpire
        roles["CATCHER"] = other

        return roles

    # ========================================================
    # No temporal history yet
    # ========================================================
    #
    # Initialize using bounding-box area.
    #
    # Catcher often occupies the larger foreground bbox due
    # to crouched equipment/body visibility.
    #
    # This is only an initialization heuristic.
    # ========================================================

    area_a = (
        a["box_width"]
        *
        a["box_height"]
    )

    area_b = (
        b["box_width"]
        *
        b["box_height"]
    )

    if area_a >= area_b:

        roles["CATCHER"] = a
        roles["UMPIRE"] = b

    else:

        roles["CATCHER"] = b
        roles["UMPIRE"] = a

    return roles


# ============================================================
# HOME ROLE ASSIGNMENT V3
# ============================================================

def assign_home_roles(
    candidates,
    previous_roles,
    frame_width,
    frame_height
):
    """
    Assign BATTER / CATCHER / UMPIRE.

    V3 IMPORTANT CHANGE:

    V2 assumed:
        left   = catcher
        middle = umpire
        right  = batter

    This fails when batter stands on the opposite side.

    V3 instead uses:

        1. Select compact home plate cluster.
        2. Find closest pair.
        3. Closest pair -> catcher + umpire.
        4. Remaining person -> batter.
        5. Use temporal history for stabilization.
    """

    roles = {
        "BATTER": None,
        "CATCHER": None,
        "UMPIRE": None
    }

    if not candidates:
        return roles

    cluster = select_home_cluster(
        candidates,
        frame_width,
        frame_height
    )

    # ========================================================
    # THREE PEOPLE
    # ========================================================

    if len(cluster) >= 3:

        pair, pair_distance = (
            find_closest_pair(
                cluster,
                frame_width,
                frame_height
            )
        )

        if pair is not None:

            remaining = [
                d
                for d in cluster
                if (
                    d is not pair[0]
                    and
                    d is not pair[1]
                )
            ]

            # ------------------------------------------------
            # BATTER = person outside catcher/umpire pair
            # ------------------------------------------------

            if remaining:

                batter = remaining[0]

                previous_batter = (
                    previous_roles.get(
                        "BATTER"
                    )
                )

                # --------------------------------------------
                # Temporal safety
                # --------------------------------------------

                if previous_batter is not None:

                    batter_jump = distance(
                        batter["center"],
                        previous_batter
                    )

                    # If new spatial assignment causes an
                    # extreme jump, check all cluster members
                    # for a more temporally plausible batter.
                    if (
                        batter_jump
                        >
                        MAX_ROLE_JUMP_PX
                    ):

                        temporal_batter = (
                            closest_to_previous(
                                cluster,
                                previous_batter
                            )
                        )

                        if temporal_batter is not None:

                            temporal_jump = distance(
                                temporal_batter["center"],
                                previous_batter
                            )

                            if (
                                temporal_jump
                                <
                                batter_jump
                                and
                                temporal_jump
                                <=
                                MAX_ROLE_JUMP_PX
                            ):

                                batter = temporal_batter

                                pair = tuple(
                                    d
                                    for d in cluster
                                    if d is not batter
                                )

                roles["BATTER"] = batter

            # ------------------------------------------------
            # CATCHER / UMPIRE
            # ------------------------------------------------

            pair_roles = (
                assign_catcher_umpire(
                    list(pair),
                    previous_roles
                )
            )

            roles["CATCHER"] = (
                pair_roles[
                    "CATCHER"
                ]
            )

            roles["UMPIRE"] = (
                pair_roles[
                    "UMPIRE"
                ]
            )

        return roles

    # ========================================================
    # TWO PEOPLE
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
        # If batter was already known, preserve it temporally.
        # ----------------------------------------------------

        if previous_batter is not None:

            dist_a = distance(
                a["center"],
                previous_batter
            )

            dist_b = distance(
                b["center"],
                previous_batter
            )

            if min(
                dist_a,
                dist_b
            ) <= MAX_ROLE_JUMP_PX:

                if dist_a <= dist_b:

                    roles["BATTER"] = a
                    remaining = [b]

                else:

                    roles["BATTER"] = b
                    remaining = [a]

                pair_roles = (
                    assign_catcher_umpire(
                        remaining,
                        previous_roles
                    )
                )

                roles["CATCHER"] = (
                    pair_roles[
                        "CATCHER"
                    ]
                )

                roles["UMPIRE"] = (
                    pair_roles[
                        "UMPIRE"
                    ]
                )

                return roles

        # ----------------------------------------------------
        # No batter temporal reference.
        #
        # If two people are very close, assume they are the
        # catcher/umpire pair and avoid inventing a batter.
        # ----------------------------------------------------

        pair_distance = (
            normalized_distance(
                a["center"],
                b["center"],
                frame_width,
                frame_height
            )
        )

        if pair_distance < 0.075:

            pair_roles = (
                assign_catcher_umpire(
                    cluster,
                    previous_roles
                )
            )

            roles["CATCHER"] = (
                pair_roles[
                    "CATCHER"
                ]
            )

            roles["UMPIRE"] = (
                pair_roles[
                    "UMPIRE"
                ]
            )

            return roles

        # ----------------------------------------------------
        # Otherwise choose the person more horizontally
        # separated from expected home center as batter.
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

            roles["BATTER"] = a

            pair_roles = (
                assign_catcher_umpire(
                    [b],
                    previous_roles
                )
            )

        else:

            roles["BATTER"] = b

            pair_roles = (
                assign_catcher_umpire(
                    [a],
                    previous_roles
                )
            )

        roles["CATCHER"] = (
            pair_roles[
                "CATCHER"
            ]
        )

        roles["UMPIRE"] = (
            pair_roles[
                "UMPIRE"
            ]
        )

        return roles

    # ========================================================
    # ONE PERSON
    # ========================================================

    candidate = cluster[0]

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
            key=lambda x:
                x[0]
        )

        best_distance, best_role = (
            role_distances[0]
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

    # --------------------------------------------------------
    # No history:
    # horizontal separation from center -> likely batter.
    # --------------------------------------------------------

    horizontal_offset = abs(
        candidate["normalized_x"]
        -
        EXPECTED_HOME_X
    )

    if horizontal_offset >= 0.045:

        roles["BATTER"] = candidate

    else:

        roles["CATCHER"] = candidate

    return roles


# ============================================================
# DRAW ROLE
# ============================================================

def draw_role(
    frame,
    detection,
    role
):
    """
    Draw assigned player role.
    """

    if detection is None:
        return

    x1, y1, x2, y2 = (
        detection["bbox"]
    )

    confidence = (
        detection["confidence"]
    )

    center = (
        detection["center"]
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        3
    )

    cv2.circle(
        frame,
        center,
        5,
        (255, 255, 255),
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
        (255, 255, 255),
        2
    )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "BaseVisionAI - Player Detection V3"
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

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # ========================================================
    # INPUT VIDEO
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
    # OUTPUT DIRECTORY
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
        "player_detection_v3.mp4"
    )

    output_csv = (
        output_dir
        /
        "player_detection_v3.csv"
    )

    # ========================================================
    # HEADER
    # ========================================================

    print("=" * 70)
    print(
        "BaseVisionAI - Player Detection V3"
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
    # VIDEO
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
        f"FPS: {fps:.2f}"
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
    # ROLE HISTORY
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

        csv_writer = csv.writer(
            csv_file
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

            results = model.predict(
                frame,
                imgsz=IMG_SIZE,
                conf=CONF_THRESHOLD,
                device=0,
                classes=[
                    PERSON_CLASS_ID
                ],
                verbose=False
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
                            x1 + x2
                        )
                        /
                        2
                    )

                    center_y = int(
                        (
                            y1 + y2
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
            # FIELD PEOPLE
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
            # HOME GROUP
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
            # UPDATE HISTORY
            # =================================================

            for role_name, detection in (
                roles.items()
            ):

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

            for role_name, detection in (
                roles.items()
            ):

                draw_role(
                    frame,
                    detection,
                    role_name
                )

            # =================================================
            # OPTIONAL FIELD PLAYERS
            # =================================================

            if DRAW_OTHER_FIELD_PLAYERS:

                assigned = [
                    d
                    for d in roles.values()
                    if d is not None
                ]

                for detection in (
                    field_people
                ):

                    if detection in assigned:
                        continue

                    x1, y1, x2, y2 = (
                        detection[
                            "bbox"
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
                            100,
                            100,
                            100
                        ),
                        1
                    )

            # =================================================
            # CSV
            # =================================================

            for role_name, detection in (
                roles.items()
            ):

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
                for role, detection
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
            # WRITE VIDEO
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
        "PLAYER DETECTION V3 COMPLETED"
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