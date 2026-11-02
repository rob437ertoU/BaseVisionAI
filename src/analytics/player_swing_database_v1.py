from pathlib import Path
import argparse

import pandas as pd


# ============================================================
# BASEVISIONAI - PLAYER SWING DATABASE V1
# ============================================================
#
# Purpose:
#   Append one Player Swing Profile V1 row to a persistent
#   swing-level database CSV.
#
# Input:
#   player_swing_profile_v1.csv
#
# Output:
#   player_swing_database_v1.csv
#
# ============================================================


def parse_args():

    parser = argparse.ArgumentParser(
        description="BaseVisionAI - Player Swing Database V1"
    )

    parser.add_argument(
        "--profile",
        required=True,
        help="Player Swing Profile V1 CSV"
    )

    parser.add_argument(
        "--database",
        required=True,
        help="Persistent swing database CSV"
    )

    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help=(
            "Replace an existing row with the same "
            "player_id/session_id/video_id"
        )
    )

    return parser.parse_args()


def main():

    args = parse_args()

    profile_path = Path(args.profile)
    database_path = Path(args.database)

    if not profile_path.exists():
        raise FileNotFoundError(
            f"Profile not found:\n{profile_path}"
        )

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    profile_df = pd.read_csv(
        profile_path
    )

    if len(profile_df) != 1:
        raise ValueError(
            "Profile CSV must contain exactly one row."
        )

    new_row = profile_df.iloc[[0]].copy()

    required_columns = [
        "video_id",
        "player_id",
        "session_id",
    ]

    for column in required_columns:

        if column not in new_row.columns:
            raise ValueError(
                f"Profile requires column: {column}"
            )

    # ========================================================
    # LOAD OR CREATE DATABASE
    # ========================================================

    if database_path.exists():

        database_df = pd.read_csv(
            database_path
        )

    else:

        database_df = pd.DataFrame(
            columns=new_row.columns
        )

    # ========================================================
    # ALIGN COLUMNS
    # ========================================================

    all_columns = list(
        dict.fromkeys(
            list(database_df.columns)
            +
            list(new_row.columns)
        )
    )

    database_df = database_df.reindex(
        columns=all_columns
    )

    new_row = new_row.reindex(
        columns=all_columns
    )

    # ========================================================
    # DUPLICATE KEY
    # ========================================================

    video_id = str(
        new_row.iloc[0]["video_id"]
    )

    player_id = str(
        new_row.iloc[0]["player_id"]
    )

    session_id = str(
        new_row.iloc[0]["session_id"]
    )

    if len(database_df) > 0:

        duplicate_mask = (
            database_df["video_id"].astype(str).eq(video_id)
            &
            database_df["player_id"].astype(str).eq(player_id)
            &
            database_df["session_id"].astype(str).eq(session_id)
        )

    else:

        duplicate_mask = pd.Series(
            False,
            index=database_df.index
        )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    # ========================================================
    # HANDLE DUPLICATES
    # ========================================================

    action = "APPENDED"

    if duplicate_count > 0:

        if args.replace_existing:

            database_df = database_df[
                ~duplicate_mask
            ].copy()

            action = "REPLACED"

        else:

            print()
            print("=" * 80)
            print(
                "PLAYER SWING DATABASE V1"
            )
            print("=" * 80)

            print(
                "Profile already exists in database."
            )

            print(
                f"Video ID   : {video_id}"
            )

            print(
                f"Player ID  : {player_id}"
            )

            print(
                f"Session ID : {session_id}"
            )

            print()
            print(
                "No changes were made."
            )

            print(
                "Use --replace-existing to overwrite it."
            )

            print("=" * 80)

            return

    # ========================================================
    # APPEND
    # ========================================================

    database_df = pd.concat(
        [
            database_df,
            new_row,
        ],
        ignore_index=True
    )

    # ========================================================
    # SORT
    # ========================================================

    sort_columns = [
        column
        for column in [
            "player_id",
            "session_id",
            "video_id",
        ]
        if column in database_df.columns
    ]

    if sort_columns:

        database_df = database_df.sort_values(
            sort_columns
        ).reset_index(
            drop=True
        )

    # ========================================================
    # SAVE
    # ========================================================

    database_df.to_csv(
        database_path,
        index=False
    )

    # ========================================================
    # REPORT
    # ========================================================

    player_rows = int(
        (
            database_df[
                "player_id"
            ].astype(str)
            ==
            player_id
        ).sum()
    )

    session_rows = int(
        (
            (
                database_df[
                    "player_id"
                ].astype(str)
                ==
                player_id
            )
            &
            (
                database_df[
                    "session_id"
                ].astype(str)
                ==
                session_id
            )
        ).sum()
    )

    print()
    print("=" * 80)
    print(
        "PLAYER SWING DATABASE V1 COMPLETE"
    )
    print("=" * 80)

    print()
    print("ACTION")
    print("-" * 80)

    print(
        f"{action}"
    )

    print()
    print("PROFILE")
    print("-" * 80)

    print(
        f"Video ID                  : {video_id}"
    )

    print(
        f"Player ID                 : {player_id}"
    )

    print(
        f"Session ID                : {session_id}"
    )

    print()
    print("DATABASE")
    print("-" * 80)

    print(
        f"Total swings              : {len(database_df)}"
    )

    print(
        f"Swings for player         : {player_rows}"
    )

    print(
        f"Swings in this session    : {session_rows}"
    )

    print()
    print("OUTPUT")
    print("-" * 80)

    print(
        database_path.resolve()
    )

    print()
    print("IMPORTANT")
    print("-" * 80)

    print(
        "Each row represents one analyzed swing/video profile."
    )

    print(
        "Metrics remain comparative 2D computer-vision features."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()