from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# BASEVISION AI - DASHBOARD V1
# ============================================================
#
# Reads:
#   - Player Swing Database V1
#   - Swing Validity V1
#
# Purpose:
#   Interactive visualization of comparative 2D swing metrics.
#
# IMPORTANT:
#   - Does NOT rerun detection/tracking/pose/swing models.
#   - Does NOT classify strike/ball/hit.
#   - Does NOT produce calibrated biomechanics.
# ============================================================


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="BaseVision AI - Swing Analytics",
    page_icon="⚾",
    layout="wide",
)


DATABASE_PATH = Path(
    "results/player_profiles/player_swing_database_v1.csv"
)

VALIDITY_PATH = Path(
    "results/swing_validity/swing_validity_v1.csv"
)


# ============================================================
# METRICS
# ============================================================

METRIC_LABELS = {
    "left_wrist_peak_px_frame":
        "Left Wrist Peak (px/frame)",

    "right_wrist_peak_px_frame":
        "Right Wrist Peak (px/frame)",

    "left_wrist_peak_torsoscale_frame":
        "Left Wrist Peak (torso/frame)",

    "right_wrist_peak_torsoscale_frame":
        "Right Wrist Peak (torso/frame)",

    "hip_center_peak_px_frame":
        "Hip Center Peak (px/frame)",

    "shoulder_center_peak_px_frame":
        "Shoulder Center Peak (px/frame)",

    "shoulder_rotation_projected_deg":
        "Shoulder Rotation (2D projected deg)",

    "hip_rotation_projected_deg":
        "Hip Rotation (2D projected deg)",

    "hip_shoulder_separation_projected_deg":
        "Hip-Shoulder Separation (2D projected deg)",

    "head_prepeak_torso":
        "Head Pre-Peak Movement (torso lengths)",

    "left_ankle_prepeak_torso":
        "Left Ankle Pre-Peak Movement (torso lengths)",

    "right_ankle_prepeak_torso":
        "Right Ankle Pre-Peak Movement (torso lengths)",

    "head_follow_torso":
        "Head Follow-Through Movement (torso lengths)",

    "left_ankle_follow_torso":
        "Left Ankle Follow-Through Movement (torso lengths)",

    "right_ankle_follow_torso":
        "Right Ankle Follow-Through Movement (torso lengths)",

    "load_duration_sec":
        "Load Duration (s)",

    "swing_duration_sec":
        "Swing Duration (s)",

    "peak_duration_sec":
        "Peak Duration (s)",

    "follow_duration_sec":
        "Follow-Through Duration (s)",

    "load_to_peak_sec":
        "Load → Peak (s)",

    "swing_to_peak_sec":
        "Swing → Peak (s)",

    "mean_pose_quality":
        "Mean Pose Quality",
}


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():

    if not DATABASE_PATH.exists():

        raise FileNotFoundError(
            f"Database not found: {DATABASE_PATH}"
        )

    if not VALIDITY_PATH.exists():

        raise FileNotFoundError(
            f"Validity file not found: {VALIDITY_PATH}"
        )

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

    validity_cols = merge_keys + [
        "validity_status",
        "evidence_quality",
        "reason",
    ]

    merged_df = database_df.merge(
        validity_df[
            validity_cols
        ],
        on=merge_keys,
        how="left"
    )

    merged_df[
        "validity_status"
    ] = merged_df[
        "validity_status"
    ].fillna(
        "NOT_EVALUATED"
    )

    merged_df[
        "evidence_quality"
    ] = merged_df[
        "evidence_quality"
    ].fillna(
        "UNKNOWN"
    )

    return merged_df


# ============================================================
# HELPERS
# ============================================================

def safe_mean(series):

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        values.mean()
    )


def safe_std(series):

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) < 2:
        return np.nan

    return float(
        values.std(
            ddof=1
        )
    )


def safe_cv(series):

    mean = safe_mean(
        series
    )

    std = safe_std(
        series
    )

    if (
        not np.isfinite(mean)
        or
        not np.isfinite(std)
        or
        abs(mean) < 1e-9
    ):
        return np.nan

    return float(
        std
        /
        abs(mean)
        *
        100.0
    )


def format_value(
    value,
    decimals=3
):

    if not np.isfinite(value):
        return "N/A"

    return f"{value:.{decimals}f}"


def existing_metrics(df):

    return {
        key: label
        for key, label
        in METRIC_LABELS.items()
        if key in df.columns
    }


# ============================================================
# LOAD
# ============================================================

try:

    df = load_data()

except Exception as exc:

    st.error(
        f"Could not load BaseVision AI data.\n\n{exc}"
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title(
    "⚾ BaseVision AI"
)

st.subheader(
    "Player Swing Analytics Dashboard V1"
)

st.caption(
    "Comparative 2D computer-vision swing analysis. "
    "Metrics are not calibrated physical biomechanics."
)

st.divider()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header(
    "Analysis Filters"
)


players = sorted(
    df[
        "player_id"
    ]
    .astype(str)
    .unique()
)

selected_player = st.sidebar.selectbox(
    "Player",
    players
)


player_df = df[
    df[
        "player_id"
    ].astype(str)
    ==
    str(selected_player)
].copy()


sessions = sorted(
    player_df[
        "session_id"
    ]
    .astype(str)
    .unique()
)


session_options = [
    "ALL"
] + sessions


selected_session = st.sidebar.selectbox(
    "Session",
    session_options
)


if selected_session != "ALL":

    selected_df = player_df[
        player_df[
            "session_id"
        ].astype(str)
        ==
        str(selected_session)
    ].copy()

else:

    selected_df = player_df.copy()


validity_filter = st.sidebar.radio(
    "Swing Evidence",
    [
        "All analyzed sequences",
        "VALID_SWING only",
    ]
)


if validity_filter == "VALID_SWING only":

    filtered_df = selected_df[
        selected_df[
            "validity_status"
        ]
        ==
        "VALID_SWING"
    ].copy()

else:

    filtered_df = selected_df.copy()


st.sidebar.divider()

st.sidebar.caption(
    "VALID_SWING means structurally usable for comparison. "
    "It does not prove that a true baseball swing occurred."
)


# ============================================================
# EMPTY CHECK
# ============================================================

if len(filtered_df) == 0:

    st.warning(
        "No swing profiles match the current filters."
    )

    st.stop()


# ============================================================
# OVERVIEW
# ============================================================

st.header(
    "Overview"
)


total_sequences = len(
    selected_df
)

valid_sequences = int(
    (
        selected_df[
            "validity_status"
        ]
        ==
        "VALID_SWING"
    ).sum()
)

limited_sequences = int(
    (
        selected_df[
            "validity_status"
        ]
        ==
        "LOW_CONFIDENCE_SWING"
    ).sum()
)

no_clear_sequences = int(
    (
        selected_df[
            "validity_status"
        ]
        ==
        "NO_CLEAR_SWING"
    ).sum()
)

mean_pose_quality = safe_mean(
    filtered_df[
        "mean_pose_quality"
    ]
)


col1, col2, col3, col4, col5 = st.columns(
    5
)

col1.metric(
    "Analyzed Sequences",
    total_sequences
)

col2.metric(
    "VALID_SWING",
    valid_sequences
)

col3.metric(
    "Limited Evidence",
    limited_sequences
)

col4.metric(
    "No Clear Swing",
    no_clear_sequences
)

col5.metric(
    "Mean Pose Quality",
    format_value(
        mean_pose_quality,
        3
    )
)


# ============================================================
# VALIDITY TABLE
# ============================================================

st.subheader(
    "Swing Evidence Status"
)


validity_table_columns = [
    "video_id",
    "session_id",
    "validity_status",
    "evidence_quality",
    "reason",
]


validity_table_columns = [
    c
    for c
    in validity_table_columns
    if c in selected_df.columns
]


st.dataframe(
    selected_df[
        validity_table_columns
    ],
    use_container_width=True,
    hide_index=True
)


st.divider()


# ============================================================
# SWING MOTION
# ============================================================

st.header(
    "Swing Motion"
)


motion_metrics = [
    "left_wrist_peak_px_frame",
    "right_wrist_peak_px_frame",
    "hip_center_peak_px_frame",
    "shoulder_center_peak_px_frame",
]


motion_metrics = [
    m
    for m
    in motion_metrics
    if m in filtered_df.columns
]


motion_long = filtered_df[
    [
        "video_id"
    ]
    +
    motion_metrics
].melt(
    id_vars=[
        "video_id"
    ],
    var_name="metric",
    value_name="value"
)


motion_long[
    "metric_label"
] = motion_long[
    "metric"
].map(
    METRIC_LABELS
)


fig_motion = px.bar(
    motion_long,
    x="video_id",
    y="value",
    color="metric_label",
    barmode="group",
    labels={
        "video_id": "Swing / Video",
        "value": "Peak motion (px/frame)",
        "metric_label": "Metric",
    },
    title="Peak Motion by Swing"
)


st.plotly_chart(
    fig_motion,
    use_container_width=True
)


# ============================================================
# NORMALIZED WRIST MOTION
# ============================================================

normalized_metrics = [
    "left_wrist_peak_torsoscale_frame",
    "right_wrist_peak_torsoscale_frame",
]


normalized_metrics = [
    m
    for m
    in normalized_metrics
    if m in filtered_df.columns
]


if len(normalized_metrics) > 0:

    normalized_long = filtered_df[
        [
            "video_id"
        ]
        +
        normalized_metrics
    ].melt(
        id_vars=[
            "video_id"
        ],
        var_name="metric",
        value_name="value"
    )

    normalized_long[
        "metric_label"
    ] = normalized_long[
        "metric"
    ].map(
        METRIC_LABELS
    )

    fig_normalized = px.bar(
        normalized_long,
        x="video_id",
        y="value",
        color="metric_label",
        barmode="group",
        labels={
            "video_id": "Swing / Video",
            "value": "Normalized motion",
            "metric_label": "Metric",
        },
        title="Torso-Normalized Wrist Motion"
    )

    st.plotly_chart(
        fig_normalized,
        use_container_width=True
    )


st.divider()


# ============================================================
# TIMING
# ============================================================

st.header(
    "Swing Timing"
)


timing_metrics = [
    "load_duration_sec",
    "swing_duration_sec",
    "load_to_peak_sec",
    "swing_to_peak_sec",
]


timing_metrics = [
    m
    for m
    in timing_metrics
    if m in filtered_df.columns
]


timing_long = filtered_df[
    [
        "video_id"
    ]
    +
    timing_metrics
].melt(
    id_vars=[
        "video_id"
    ],
    var_name="metric",
    value_name="value"
)


timing_long[
    "metric_label"
] = timing_long[
    "metric"
].map(
    METRIC_LABELS
)


fig_timing = px.bar(
    timing_long,
    x="video_id",
    y="value",
    color="metric_label",
    barmode="group",
    labels={
        "video_id": "Swing / Video",
        "value": "Time (seconds)",
        "metric_label": "Metric",
    },
    title="Swing Timing Comparison"
)


st.plotly_chart(
    fig_timing,
    use_container_width=True
)


# ============================================================
# PROJECTED ROTATION
# ============================================================

st.header(
    "Projected Rotation"
)


rotation_metrics = [
    "shoulder_rotation_projected_deg",
    "hip_rotation_projected_deg",
    "hip_shoulder_separation_projected_deg",
]


rotation_metrics = [
    m
    for m
    in rotation_metrics
    if m in filtered_df.columns
]


rotation_long = filtered_df[
    [
        "video_id"
    ]
    +
    rotation_metrics
].melt(
    id_vars=[
        "video_id"
    ],
    var_name="metric",
    value_name="value"
)


rotation_long[
    "metric_label"
] = rotation_long[
    "metric"
].map(
    METRIC_LABELS
)


fig_rotation = px.bar(
    rotation_long,
    x="video_id",
    y="value",
    color="metric_label",
    barmode="group",
    labels={
        "video_id": "Swing / Video",
        "value": "Projected angle (degrees)",
        "metric_label": "Metric",
    },
    title="2D Projected Rotation Comparison"
)


st.plotly_chart(
    fig_rotation,
    use_container_width=True
)


st.caption(
    "Projected angles are camera-dependent comparative 2D measurements, "
    "not true 3D biomechanical joint rotations."
)


st.divider()


# ============================================================
# STABILITY
# ============================================================

st.header(
    "Pre-Peak Stability"
)


stability_metrics = [
    "head_prepeak_torso",
    "left_ankle_prepeak_torso",
    "right_ankle_prepeak_torso",
]


stability_metrics = [
    m
    for m
    in stability_metrics
    if m in filtered_df.columns
]


stability_long = filtered_df[
    [
        "video_id"
    ]
    +
    stability_metrics
].melt(
    id_vars=[
        "video_id"
    ],
    var_name="metric",
    value_name="value"
)


stability_long[
    "metric_label"
] = stability_long[
    "metric"
].map(
    METRIC_LABELS
)


fig_stability = px.bar(
    stability_long,
    x="video_id",
    y="value",
    color="metric_label",
    barmode="group",
    labels={
        "video_id": "Swing / Video",
        "value": "Movement (torso lengths)",
        "metric_label": "Metric",
    },
    title="Pre-Peak Stability Comparison"
)


st.plotly_chart(
    fig_stability,
    use_container_width=True
)


st.divider()


# ============================================================
# VARIABILITY
# ============================================================

st.header(
    "Variability"
)

st.caption(
    "Coefficient of variation (CV) measures statistical variability. "
    "It is not a swing quality score."
)


available_metric_map = existing_metrics(
    filtered_df
)


selected_metric = st.selectbox(
    "Metric for variability analysis",
    options=list(
        available_metric_map.keys()
    ),
    format_func=lambda x:
        available_metric_map[x]
)


selected_values = pd.to_numeric(
    filtered_df[
        selected_metric
    ],
    errors="coerce"
)


mean_value = safe_mean(
    selected_values
)

std_value = safe_std(
    selected_values
)

cv_value = safe_cv(
    selected_values
)


v1, v2, v3 = st.columns(
    3
)


v1.metric(
    "Mean",
    format_value(
        mean_value,
        3
    )
)

v2.metric(
    "Sample SD",
    format_value(
        std_value,
        3
    )
)

v3.metric(
    "CV",
    (
        f"{cv_value:.1f}%"
        if np.isfinite(cv_value)
        else "N/A"
    )
)


variability_df = filtered_df[
    [
        "video_id",
        selected_metric
    ]
].copy()


variability_df[
    selected_metric
] = pd.to_numeric(
    variability_df[
        selected_metric
    ],
    errors="coerce"
)


fig_variability = px.scatter(
    variability_df,
    x="video_id",
    y=selected_metric,
    labels={
        "video_id": "Swing / Video",
        selected_metric:
            METRIC_LABELS.get(
                selected_metric,
                selected_metric
            ),
    },
    title=(
        f"{METRIC_LABELS.get(selected_metric, selected_metric)} "
        "Across Swings"
    )
)


if np.isfinite(mean_value):

    fig_variability.add_hline(
        y=mean_value,
        line_dash="dash",
        annotation_text="Mean"
    )


st.plotly_chart(
    fig_variability,
    use_container_width=True
)


# ============================================================
# ALL VS VALID
# ============================================================

st.divider()

st.header(
    "ALL vs VALID_SWING"
)


all_df = selected_df.copy()

valid_df = selected_df[
    selected_df[
        "validity_status"
    ]
    ==
    "VALID_SWING"
].copy()


comparison_metrics = [
    "right_wrist_peak_px_frame",
    "left_wrist_peak_px_frame",
    "hip_center_peak_px_frame",
    "shoulder_center_peak_px_frame",
    "load_to_peak_sec",
    "swing_to_peak_sec",
    "head_prepeak_torso",
    "hip_shoulder_separation_projected_deg",
]


comparison_rows = []


for metric in comparison_metrics:

    if metric not in selected_df.columns:
        continue

    comparison_rows.append(
        {
            "metric":
                METRIC_LABELS.get(
                    metric,
                    metric
                ),

            "group":
                "ALL",

            "cv_percent":
                safe_cv(
                    all_df[
                        metric
                    ]
                ),
        }
    )

    comparison_rows.append(
        {
            "metric":
                METRIC_LABELS.get(
                    metric,
                    metric
                ),

            "group":
                "VALID_SWING",

            "cv_percent":
                safe_cv(
                    valid_df[
                        metric
                    ]
                ),
        }
    )


comparison_plot_df = pd.DataFrame(
    comparison_rows
)


fig_compare = px.bar(
    comparison_plot_df,
    x="metric",
    y="cv_percent",
    color="group",
    barmode="group",
    labels={
        "metric": "Metric",
        "cv_percent": "CV (%)",
        "group": "Dataset",
    },
    title="Variability: ALL vs VALID_SWING"
)


fig_compare.update_layout(
    xaxis_tickangle=-35
)


st.plotly_chart(
    fig_compare,
    use_container_width=True
)


# ============================================================
# INDIVIDUAL SWING INSPECTION
# ============================================================

st.divider()

st.header(
    "Individual Swing Profile"
)


video_options = list(
    filtered_df[
        "video_id"
    ].astype(str)
)


selected_video = st.selectbox(
    "Swing / Video",
    video_options
)


swing_row = filtered_df[
    filtered_df[
        "video_id"
    ].astype(str)
    ==
    str(selected_video)
].iloc[0]


c1, c2, c3, c4 = st.columns(
    4
)


c1.metric(
    "Validity",
    str(
        swing_row[
            "validity_status"
        ]
    )
)

c2.metric(
    "Evidence",
    str(
        swing_row[
            "evidence_quality"
        ]
    )
)

c3.metric(
    "Pose Quality",
    format_value(
        float(
            swing_row[
                "mean_pose_quality"
            ]
        ),
        3
    )
)

c4.metric(
    "Swing Duration",
    (
        f"{float(swing_row['swing_duration_sec']):.3f} s"
        if pd.notna(
            swing_row[
                "swing_duration_sec"
            ]
        )
        else "N/A"
    )
)


profile_metrics = [
    (
        "Left Wrist Peak",
        "left_wrist_peak_px_frame"
    ),

    (
        "Right Wrist Peak",
        "right_wrist_peak_px_frame"
    ),

    (
        "Hip Peak",
        "hip_center_peak_px_frame"
    ),

    (
        "Shoulder Peak",
        "shoulder_center_peak_px_frame"
    ),

    (
        "Hip-Shoulder Separation",
        "hip_shoulder_separation_projected_deg"
    ),

    (
        "Head Pre-Peak",
        "head_prepeak_torso"
    ),

    (
        "Load → Peak",
        "load_to_peak_sec"
    ),

    (
        "Swing → Peak",
        "swing_to_peak_sec"
    ),
]


profile_rows = []


for label, metric in profile_metrics:

    if metric not in filtered_df.columns:
        continue

    value = pd.to_numeric(
        pd.Series(
            [
                swing_row[
                    metric
                ]
            ]
        ),
        errors="coerce"
    ).iloc[0]

    profile_rows.append(
        {
            "Metric":
                label,

            "Value":
                value
        }
    )


profile_df = pd.DataFrame(
    profile_rows
)


fig_profile = px.bar(
    profile_df,
    x="Metric",
    y="Value",
    title=(
        f"Swing Profile: {selected_video}"
    )
)


fig_profile.update_layout(
    xaxis_tickangle=-30
)


st.plotly_chart(
    fig_profile,
    use_container_width=True
)


if (
    "reason"
    in swing_row.index
):

    st.info(
        f"Validity evidence: {swing_row['reason']}"
    )


# ============================================================
# DATA TABLE
# ============================================================

st.divider()

st.header(
    "Raw Profile Data"
)


display_columns = [
    "video_id",
    "player_id",
    "session_id",
    "validity_status",
    "evidence_quality",
    "left_wrist_peak_px_frame",
    "right_wrist_peak_px_frame",
    "hip_center_peak_px_frame",
    "shoulder_center_peak_px_frame",
    "hip_shoulder_separation_projected_deg",
    "head_prepeak_torso",
    "load_to_peak_sec",
    "swing_to_peak_sec",
    "mean_pose_quality",
]


display_columns = [
    c
    for c
    in display_columns
    if c in filtered_df.columns
]


st.dataframe(
    filtered_df[
        display_columns
    ],
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "BaseVision AI — Dashboard V1 | "
    "2D comparative computer-vision analytics | "
    "No calibrated biomechanical, strike, ball or hit classification."
)