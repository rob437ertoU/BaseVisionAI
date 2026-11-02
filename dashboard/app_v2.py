from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# BASEVISION AI - DASHBOARD V2
# ============================================================
#
# Features:
#   - Player/session filters
#   - VALID_SWING filtering
#   - Swing A vs Swing B comparison
#   - Side-by-side metrics
#   - Difference table
#   - Radar comparison
#   - Timing comparison
#   - Embedded Full Analysis videos
#
# IMPORTANT:
#   - No model inference
#   - No hit/strike/ball classification
#   - No calibrated biomechanics
# ============================================================


st.set_page_config(
    page_title="BaseVision AI - Swing Analytics V2",
    page_icon="⚾",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

DATABASE_PATH = Path(
    "results/player_profiles/player_swing_database_v1.csv"
)

VALIDITY_PATH = Path(
    "results/swing_validity/swing_validity_v1.csv"
)

FULL_ANALYSIS_ROOT = Path(
    "results/full_analysis_v3"
)


# ============================================================
# METRIC LABELS
# ============================================================

METRIC_LABELS = {
    "left_wrist_peak_px_frame":
        "Left Wrist Peak",

    "right_wrist_peak_px_frame":
        "Right Wrist Peak",

    "left_wrist_peak_torsoscale_frame":
        "Left Wrist Peak Normalized",

    "right_wrist_peak_torsoscale_frame":
        "Right Wrist Peak Normalized",

    "hip_center_peak_px_frame":
        "Hip Center Peak",

    "shoulder_center_peak_px_frame":
        "Shoulder Center Peak",

    "shoulder_rotation_projected_deg":
        "Shoulder Rotation",

    "hip_rotation_projected_deg":
        "Hip Rotation",

    "hip_shoulder_separation_projected_deg":
        "Hip-Shoulder Separation",

    "head_prepeak_torso":
        "Head Pre-Peak",

    "left_ankle_prepeak_torso":
        "Left Ankle Pre-Peak",

    "right_ankle_prepeak_torso":
        "Right Ankle Pre-Peak",

    "load_duration_sec":
        "Load Duration",

    "swing_duration_sec":
        "Swing Duration",

    "peak_duration_sec":
        "Peak Duration",

    "follow_duration_sec":
        "Follow Duration",

    "load_to_peak_sec":
        "Load → Peak",

    "swing_to_peak_sec":
        "Swing → Peak",

    "mean_pose_quality":
        "Pose Quality",
}


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    database_df = pd.read_csv(
        DATABASE_PATH
    )

    validity_df = pd.read_csv(
        VALIDITY_PATH
    )

    merge_keys = [
        "video_id",
        "player_id",
        "session_id",
    ]

    validity_columns = merge_keys + [
        "validity_status",
        "evidence_quality",
        "reason",
    ]

    merged = database_df.merge(
        validity_df[
            validity_columns
        ],
        on=merge_keys,
        how="left"
    )

    merged["validity_status"] = (
        merged["validity_status"]
        .fillna("NOT_EVALUATED")
    )

    merged["evidence_quality"] = (
        merged["evidence_quality"]
        .fillna("UNKNOWN")
    )

    return merged


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return np.nan


def fmt(
    value,
    digits=3
):

    value = safe_float(
        value
    )

    if not np.isfinite(value):
        return "N/A"

    return f"{value:.{digits}f}"


def percent_difference(
    a,
    b
):

    a = safe_float(a)
    b = safe_float(b)

    if (
        not np.isfinite(a)
        or
        not np.isfinite(b)
    ):
        return np.nan

    if abs(a) < 1e-9:
        return np.nan

    return (
        (b - a)
        /
        abs(a)
        *
        100.0
    )


def find_video(video_id):

    search_roots = [
        Path("results/full_analysis_v3"),
        Path("results/full_analysis_v2"),
        Path("results/evaluation_v1"),
    ]

    preferred_names = [
        f"{video_id}_full_analysis_v3_web.mp4",
        f"{video_id}_full_analysis_v3.mp4",
        f"{video_id}_full_analysis_v2_1.mp4",
        f"{video_id}_full_analysis_v2.mp4",
        f"{video_id}.mp4",
    ]

    for root in search_roots:

        if not root.exists():
            continue

        for name in preferred_names:

            candidate = root / name

            if candidate.exists():
                return candidate

    for root in search_roots:

        if not root.exists():
            continue

        matches = list(
            root.rglob(f"*{video_id}*.mp4")
        )

        if len(matches) > 0:
            return matches[0]

    return None

def normalize_pair(
    a,
    b
):

    a = safe_float(a)
    b = safe_float(b)

    if (
        not np.isfinite(a)
        or
        not np.isfinite(b)
    ):
        return np.nan, np.nan

    maximum = max(
        abs(a),
        abs(b)
    )

    if maximum < 1e-9:
        return 0.0, 0.0

    return (
        abs(a) / maximum,
        abs(b) / maximum
    )


# ============================================================
# INITIAL DATA
# ============================================================

if not DATABASE_PATH.exists():

    st.error(
        f"Database not found: {DATABASE_PATH}"
    )

    st.stop()


if not VALIDITY_PATH.exists():

    st.error(
        f"Validity file not found: {VALIDITY_PATH}"
    )

    st.stop()


df = load_data()


# ============================================================
# HEADER
# ============================================================

st.title(
    "⚾ BaseVision AI"
)

st.subheader(
    "Player Swing Analytics Dashboard V2"
)

st.caption(
    "2D comparative computer-vision analysis. "
    "Metrics are intended for swing-to-swing comparison."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Analysis Filters"
)


players = sorted(
    df["player_id"]
    .astype(str)
    .unique()
)


selected_player = st.sidebar.selectbox(
    "Player",
    players
)


player_df = df[
    df["player_id"].astype(str)
    ==
    str(selected_player)
].copy()


sessions = sorted(
    player_df["session_id"]
    .astype(str)
    .unique()
)


selected_session = st.sidebar.selectbox(
    "Session",
    ["ALL"] + sessions
)


if selected_session != "ALL":

    filtered_df = player_df[
        player_df["session_id"].astype(str)
        ==
        str(selected_session)
    ].copy()

else:

    filtered_df = player_df.copy()


evidence_filter = st.sidebar.radio(
    "Swing Evidence",
    [
        "VALID_SWING only",
        "All analyzed sequences",
    ]
)


if evidence_filter == "VALID_SWING only":

    filtered_df = filtered_df[
        filtered_df["validity_status"]
        ==
        "VALID_SWING"
    ].copy()


if len(filtered_df) == 0:

    st.warning(
        "No profiles match the current filters."
    )

    st.stop()


# ============================================================
# OVERVIEW
# ============================================================

st.header(
    "Session Overview"
)


c1, c2, c3, c4 = st.columns(
    4
)


c1.metric(
    "Profiles",
    len(filtered_df)
)


c2.metric(
    "VALID_SWING",
    int(
        (
            filtered_df[
                "validity_status"
            ]
            ==
            "VALID_SWING"
        ).sum()
    )
)


c3.metric(
    "Mean Pose Quality",
    fmt(
        pd.to_numeric(
            filtered_df[
                "mean_pose_quality"
            ],
            errors="coerce"
        ).mean()
    )
)


c4.metric(
    "Sessions",
    filtered_df[
        "session_id"
    ].nunique()
)


st.divider()


# ============================================================
# SWING SELECTORS
# ============================================================

st.header(
    "Swing A vs Swing B"
)


video_options = list(
    filtered_df[
        "video_id"
    ].astype(str)
)


col_a, col_b = st.columns(
    2
)


with col_a:

    swing_a_id = st.selectbox(
        "Swing A",
        video_options,
        index=0
    )


with col_b:

    default_b = (
        1
        if len(video_options) > 1
        else 0
    )

    swing_b_id = st.selectbox(
        "Swing B",
        video_options,
        index=default_b
    )


swing_a = filtered_df[
    filtered_df[
        "video_id"
    ].astype(str)
    ==
    swing_a_id
].iloc[0]


swing_b = filtered_df[
    filtered_df[
        "video_id"
    ].astype(str)
    ==
    swing_b_id
].iloc[0]


# ============================================================
# VIDEO COMPARISON
# ============================================================

st.subheader(
    "Video Comparison"
)


video_col_a, video_col_b = st.columns(
    2
)


with video_col_a:

    st.markdown(
        f"### Swing A — {swing_a_id}"
    )

    video_a = find_video(
        swing_a_id
    )

    if video_a is not None:

        st.video(
            str(video_a)
        )

    else:

        st.info(
            "Full Analysis V3 video not found."
        )


with video_col_b:

    st.markdown(
        f"### Swing B — {swing_b_id}"
    )

    video_b = find_video(
        swing_b_id
    )

    if video_b is not None:

        st.video(
            str(video_b)
        )

    else:

        st.info(
            "Full Analysis V3 video not found."
        )


st.caption(
    "Videos are visualization outputs only. "
    "The dashboard does not rerun the analysis pipeline."
)


st.divider()


# ============================================================
# KEY METRICS
# ============================================================

st.subheader(
    "Key Metrics"
)


metric_cols = st.columns(
    4
)


metric_cols[0].metric(
    "A — Pose Quality",
    fmt(
        swing_a[
            "mean_pose_quality"
        ]
    )
)


metric_cols[1].metric(
    "B — Pose Quality",
    fmt(
        swing_b[
            "mean_pose_quality"
        ]
    )
)


metric_cols[2].metric(
    "A — Swing Duration",
    f"{fmt(swing_a['swing_duration_sec'])} s"
)


metric_cols[3].metric(
    "B — Swing Duration",
    f"{fmt(swing_b['swing_duration_sec'])} s"
)


metric_cols_2 = st.columns(
    4
)


metric_cols_2[0].metric(
    "A — Load → Peak",
    f"{fmt(swing_a['load_to_peak_sec'])} s"
)


metric_cols_2[1].metric(
    "B — Load → Peak",
    f"{fmt(swing_b['load_to_peak_sec'])} s"
)


metric_cols_2[2].metric(
    "A — Right Wrist",
    fmt(
        swing_a[
            "right_wrist_peak_px_frame"
        ]
    )
)


metric_cols_2[3].metric(
    "B — Right Wrist",
    fmt(
        swing_b[
            "right_wrist_peak_px_frame"
        ]
    )
)


st.divider()


# ============================================================
# COMPARISON TABLE
# ============================================================

st.header(
    "Metric Differences"
)


comparison_metrics = [
    "left_wrist_peak_px_frame",
    "right_wrist_peak_px_frame",
    "left_wrist_peak_torsoscale_frame",
    "right_wrist_peak_torsoscale_frame",
    "hip_center_peak_px_frame",
    "shoulder_center_peak_px_frame",
    "shoulder_rotation_projected_deg",
    "hip_rotation_projected_deg",
    "hip_shoulder_separation_projected_deg",
    "head_prepeak_torso",
    "left_ankle_prepeak_torso",
    "right_ankle_prepeak_torso",
    "load_duration_sec",
    "swing_duration_sec",
    "load_to_peak_sec",
    "swing_to_peak_sec",
    "mean_pose_quality",
]


comparison_rows = []


for metric in comparison_metrics:

    if metric not in filtered_df.columns:
        continue

    a = safe_float(
        swing_a[
            metric
        ]
    )

    b = safe_float(
        swing_b[
            metric
        ]
    )

    difference = (
        b - a
        if (
            np.isfinite(a)
            and
            np.isfinite(b)
        )
        else np.nan
    )

    pct = percent_difference(
        a,
        b
    )

    comparison_rows.append(
        {
            "Metric":
                METRIC_LABELS.get(
                    metric,
                    metric
                ),

            "Swing A":
                a,

            "Swing B":
                b,

            "B - A":
                difference,

            "% Change":
                pct,
        }
    )


comparison_df = pd.DataFrame(
    comparison_rows
)


st.dataframe(
    comparison_df.style.format(
        {
            "Swing A": "{:.3f}",
            "Swing B": "{:.3f}",
            "B - A": "{:+.3f}",
            "% Change": "{:+.1f}%",
        },
        na_rep="N/A"
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# RADAR
# ============================================================

st.header(
    "Normalized Swing Profile"
)


radar_metrics = [
    "right_wrist_peak_torsoscale_frame",
    "left_wrist_peak_torsoscale_frame",
    "hip_center_peak_px_frame",
    "shoulder_center_peak_px_frame",
    "hip_shoulder_separation_projected_deg",
    "head_prepeak_torso",
    "load_to_peak_sec",
]


radar_labels = []
radar_a = []
radar_b = []


for metric in radar_metrics:

    if metric not in filtered_df.columns:
        continue

    a_norm, b_norm = normalize_pair(
        swing_a[
            metric
        ],
        swing_b[
            metric
        ]
    )

    if (
        not np.isfinite(a_norm)
        or
        not np.isfinite(b_norm)
    ):
        continue

    radar_labels.append(
        METRIC_LABELS.get(
            metric,
            metric
        )
    )

    radar_a.append(
        a_norm
    )

    radar_b.append(
        b_norm
    )


if len(radar_labels) >= 3:

    radar_labels_closed = (
        radar_labels
        +
        [
            radar_labels[0]
        ]
    )

    radar_a_closed = (
        radar_a
        +
        [
            radar_a[0]
        ]
    )

    radar_b_closed = (
        radar_b
        +
        [
            radar_b[0]
        ]
    )


    radar_fig = go.Figure()


    radar_fig.add_trace(
        go.Scatterpolar(
            r=radar_a_closed,
            theta=radar_labels_closed,
            fill="toself",
            name=f"Swing A — {swing_a_id}"
        )
    )


    radar_fig.add_trace(
        go.Scatterpolar(
            r=radar_b_closed,
            theta=radar_labels_closed,
            fill="toself",
            name=f"Swing B — {swing_b_id}"
        )
    )


    radar_fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[
                    0,
                    1
                ]
            )
        ),
        showlegend=True,
        title=(
            "Pair-Normalized Swing Comparison"
        )
    )


    st.plotly_chart(
        radar_fig,
        use_container_width=True
    )


    st.caption(
        "Radar values are normalized only between the two selected swings. "
        "They are not performance scores."
    )


# ============================================================
# TIMING
# ============================================================

st.header(
    "Swing Timing"
)


timing_metrics = [
    "load_duration_sec",
    "swing_duration_sec",
    "peak_duration_sec",
    "follow_duration_sec",
]


timing_rows = []


for video_id, row in [
    (
        swing_a_id,
        swing_a
    ),
    (
        swing_b_id,
        swing_b
    )
]:

    for metric in timing_metrics:

        if metric not in filtered_df.columns:
            continue

        value = safe_float(
            row[
                metric
            ]
        )

        timing_rows.append(
            {
                "video_id":
                    video_id,

                "phase":
                    METRIC_LABELS.get(
                        metric,
                        metric
                    ),

                "duration":
                    value,
            }
        )


timing_df = pd.DataFrame(
    timing_rows
)


timing_fig = px.bar(
    timing_df,
    x="video_id",
    y="duration",
    color="phase",
    barmode="stack",
    labels={
        "video_id":
            "Swing",

        "duration":
            "Duration (s)",

        "phase":
            "Phase",
    },
    title=(
        "Phase Duration Comparison"
    )
)


st.plotly_chart(
    timing_fig,
    use_container_width=True
)


# ============================================================
# WRIST MOTION
# ============================================================

st.header(
    "Wrist Motion"
)


wrist_rows = []


for video_id, row in [
    (
        swing_a_id,
        swing_a
    ),
    (
        swing_b_id,
        swing_b
    )
]:

    for metric in [
        "left_wrist_peak_px_frame",
        "right_wrist_peak_px_frame",
    ]:

        if metric not in filtered_df.columns:
            continue

        wrist_rows.append(
            {
                "video_id":
                    video_id,

                "metric":
                    METRIC_LABELS[
                        metric
                    ],

                "value":
                    safe_float(
                        row[
                            metric
                        ]
                    )
            }
        )


wrist_df = pd.DataFrame(
    wrist_rows
)


wrist_fig = px.bar(
    wrist_df,
    x="metric",
    y="value",
    color="video_id",
    barmode="group",
    labels={
        "metric":
            "Wrist",

        "value":
            "Peak motion (px/frame)",

        "video_id":
            "Swing",
    },
    title=(
        "Wrist Peak Motion"
    )
)


st.plotly_chart(
    wrist_fig,
    use_container_width=True
)


# ============================================================
# ROTATION
# ============================================================

st.header(
    "Projected Rotation"
)


rotation_rows = []


for video_id, row in [
    (
        swing_a_id,
        swing_a
    ),
    (
        swing_b_id,
        swing_b
    )
]:

    for metric in [
        "shoulder_rotation_projected_deg",
        "hip_rotation_projected_deg",
        "hip_shoulder_separation_projected_deg",
    ]:

        if metric not in filtered_df.columns:
            continue

        rotation_rows.append(
            {
                "video_id":
                    video_id,

                "metric":
                    METRIC_LABELS[
                        metric
                    ],

                "value":
                    safe_float(
                        row[
                            metric
                        ]
                    )
            }
        )


rotation_df = pd.DataFrame(
    rotation_rows
)


rotation_fig = px.bar(
    rotation_df,
    x="metric",
    y="value",
    color="video_id",
    barmode="group",
    labels={
        "metric":
            "Metric",

        "value":
            "Projected degrees",

        "video_id":
            "Swing",
    },
    title=(
        "Projected Rotation Comparison"
    )
)


st.plotly_chart(
    rotation_fig,
    use_container_width=True
)


st.caption(
    "Projected angles are 2D camera-dependent measurements, "
    "not true 3D biomechanical rotations."
)


# ============================================================
# VALIDITY
# ============================================================

st.divider()

st.header(
    "Swing Evidence"
)


validity_cols = st.columns(
    2
)


with validity_cols[0]:

    st.markdown(
        f"### {swing_a_id}"
    )

    st.write(
        "Validity:",
        swing_a[
            "validity_status"
        ]
    )

    st.write(
        "Evidence:",
        swing_a[
            "evidence_quality"
        ]
    )

    st.write(
        "Reason:",
        swing_a[
            "reason"
        ]
    )


with validity_cols[1]:

    st.markdown(
        f"### {swing_b_id}"
    )

    st.write(
        "Validity:",
        swing_b[
            "validity_status"
        ]
    )

    st.write(
        "Evidence:",
        swing_b[
            "evidence_quality"
        ]
    )

    st.write(
        "Reason:",
        swing_b[
            "reason"
        ]
    )


# ============================================================
# RAW DATA
# ============================================================

st.divider()

st.header(
    "Selected Swing Data"
)


raw_df = pd.DataFrame(
    [
        swing_a,
        swing_b
    ]
)


st.dataframe(
    raw_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "BaseVision AI — Dashboard V2 | "
    "Comparative 2D computer-vision swing analytics | "
    "No calibrated biomechanics or outcome classification."
)