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
#
# Based on the current broadcast camera.
#
# People whose center is above this normalized Y are mostly
# spectators in the stands.
#
# 0.0 = top of image
# 1.0 = bottom
# ============================================================

FIELD_MIN_Y = 0.28


# ============================================================
# HOME PLATE REGION
# ============================================================
#
# Region where batter / catcher / umpire are expected.
#
# These are normalized coordinates so they work at different
# video resolutions with the same camera framing.
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

# How much previous position matters when selecting same role
TEMPORAL_WEIGHT = 0.35

# Spatial baseball role logic weight
SPATIAL_WEIGHT = 0.65


# ============================================================
# DRAWING
# ============================================================

DRAW_OTHER_FIELD_PLAYERS = False


# ============================================================
# AUXILIARY FUNCTIONS
# ============================================================

def distance(p1, p2):
    """
    Euclidean distance between two points.
    """

    return math.sqrt(
        (p1[0] - p2[0]) ** 2
        +
        (p1[1] - p2[1]) ** 2
    )


def normalized_distance(p1, p2, width, height):
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
    Check whether detection center is inside normalized region.
    """

    nx = detection["normalized_x"]
    ny = detection["normalized_y"]

    return (
        xmin <= nx <= xmax
        and
        ymin <= ny <= ymax
    )


# ============================================================
# FILTER SPECTATORS
# ============================================================

def filter_field_people(detections):
    """
    Remove most spectators using their vertical position.

    In this camera, spectators are mostly in the upper
    portion of the image.
    """

    field_people = []

    for detection in detections:

        if (
            detection["normalized_y"]
            >= FIELD_MIN_Y
        ):

            field_people.append(
                detection
            )

    return field_people


# ============================================================
# PITCHER SELECTION
# ============================================================

def select_pitcher(
    detections,
    previous_roles,
    frame_width,
    frame_height
):
    """
    Select pitcher.

    Main assumptions:
    - pitcher is relatively low in image
    - near center-left
    - generally isolated from home plate group
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

    # Expected approximate pitcher location
    expected_x = 0.37
    expected_y = 0.68

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

        # Bigger person boxes are generally more relevant
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

        temporal_score = 0

        previous_pitcher = (
            previous_roles.get(
                "PITCHER"
            )
        )

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
# HOME PLATE GROUP
# ============================================================

def select_home_plate_people(
    detections,
    pitcher
):
    """
    Get people around home plate.

    Pitcher is excluded.
    """

    candidates = []

    for detection in detections:

        if pitcher is not None:

            if detection is pitcher:
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
# ROLE ASSIGNMENT AROUND HOME PLATE
# ============================================================

def assign_home_roles(
    candidates,
    previous_roles,
    frame_width,
    frame_height
):
    """
    Assign BATTER / CATCHER / UMPIRE using relative positions.

    For the current camera:

        catcher  -> generally left side
        umpire   -> middle
        batter   -> right side

    Temporal position is used to reduce frame-to-frame swaps.
    """

    roles = {
        "BATTER": None,
        "CATCHER": None,
        "UMPIRE": None
    }

    if not candidates:
        return roles

    # Sort primarily from left to right
    sorted_candidates = sorted(
        candidates,
        key=lambda d:
            d["center"][0]
    )

    # ========================================================
    # If we have at least 3 people near home plate
    # ========================================================

    if len(sorted_candidates) >= 3:

        # We only want the three people most likely belonging
        # to the actual home plate cluster.
        #
        # Avoid selecting random fielders by measuring distance
        # to expected home-plate center.

        expected_home = (
            0.49 * frame_width,
            0.45 * frame_height
        )

        cluster_candidates = sorted(
            sorted_candidates,
            key=lambda d:
                distance(
                    d["center"],
                    expected_home
                )
        )[:3]

        # Re-sort selected cluster left -> right
        cluster_candidates = sorted(
            cluster_candidates,
            key=lambda d:
                d["center"][0]
        )

        roles["CATCHER"] = (
            cluster_candidates[0]
        )

        roles["UMPIRE"] = (
            cluster_candidates[1]
        )

        roles["BATTER"] = (
            cluster_candidates[2]
        )

    # ========================================================
    # Two people visible
    # ========================================================

    elif len(sorted_candidates) == 2:

        left = sorted_candidates[0]
        right = sorted_candidates[1]

        roles["CATCHER"] = left
        roles["BATTER"] = right

    # ========================================================
    # Only one visible
    # ========================================================

    else:

        candidate = (
            sorted_candidates[0]
        )

        nx = (
            candidate[
                "normalized_x"
            ]
        )

        if nx >= 0.50:
            roles["BATTER"] = candidate

        else:
            roles["CATCHER"] = candidate

    # ========================================================
    # TEMPORAL CORRECTION
    # ========================================================
    #
    # Prevent obvious role swaps.
    #
    # If current role jumps very far but another candidate is
    # close to previous role position, use closer candidate.
    # ========================================================

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

        current = roles[
            role_name
        ]

        if current is None:
            continue

        current_distance = distance(
            current["center"],
            previous_position
        )

        if (
            current_distance
            <=
            MAX_ROLE_JUMP_PX
        ):
            continue

        available_candidates = [
            d
            for d in candidates
            if distance(
                d["center"],
                previous_position
            )
            <
            current_distance
        ]

        if available_candidates:

            replacement = min(
                available_candidates,
                key=lambda d:
                    distance(
                        d["center"],
                        previous_position
                    )
            )

            # Avoid duplicate assignment
            already_used = [
                roles[r]
                for r in roles
                if (
                    r != role_name
                    and
                    roles[r]
                    is not None
                )
            ]

            if replacement not in already_used:

                roles[
                    role_name
                ] = replacement

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
    Draw player role.
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

def parse_args():
    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Player Detection V2"
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

    video_path = Path(args.video)

    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path

    video_path = video_path.resolve()

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    output_dir = Path(args.output_dir)

    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    output_dir = output_dir.resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_video = output_dir / "player_detection_v2.mp4"
    output_csv = output_dir / "player_detection_v2.csv"

    print("=" * 70)
    print("BaseVisionAI - Player Detection V2")
    print("=" * 70)

    print(f"Video: {video_path}")
    print(f"Output directory: {output_dir}")

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video not found:\n"
            f"{video_path}"
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = YOLO(
        MODEL_NAME
    )

    # ========================================================
    # OPEN VIDEO
    # ========================================================

    cap = cv2.VideoCapture(
        str(
            video_path        )
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
    # VIDEO OUTPUT
    # ========================================================

    output_video.parent.mkdir(
        parents=True,
        exist_ok=True
    )

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
            # YOLO PERSON DETECTION
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
            # REMOVE STANDS
            # =================================================

            field_people = (
                filter_field_people(
                    all_detections
                )
            )

            # =================================================
            # SELECT PITCHER
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
            # HOME PLATE GROUP
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
            # SAVE PREVIOUS ROLE POSITIONS
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
            # DRAW ROI DEBUG
            # =================================================

            # Field cutoff
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

            # Home plate ROI
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
            # OPTIONAL OTHER FIELD PLAYERS
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
            # WRITE CSV
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
        "PLAYER DETECTION V2 COMPLETED"
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