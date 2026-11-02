from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# BASEVISION AI - DASHBOARD V2.1
# ============================================================
#
# Features:
#   - Player/session filters
#   - VALID_SWING filtering
#   - Swing A vs Swing B comparison
#   - Side-by-side Full Analysis videos
#   - Key metrics
#   - Metric difference table
#   - Pair-normalized radar
#   - Timing comparison
#   - Wrist motion comparison
#   - Projected rotation comparison
#   - Swing Feedback V1.1 integration
#   - Session-relative baseline comparison
#   - Measurement confidence
#   - Swing evidence
#   - Raw selected swing data
#
# IMPORTANT:
#   - No model inference
#   - No hit/strike/ball classification
#   - No calibrated biomechanics
#   - HIGH/LOW describe statistical position, not swing quality
# ============================================================


st.set_page_config(
    page_title="BaseVision AI - Swing Analytics V2.1",
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

FEEDBACK_ROOT = Path(
    "results/swing_feedback"
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
# FEEDBACK METRIC CONFIG
# ============================================================

FEEDBACK_METRICS = [

    {
        "metric": "right_wrist_peak_torsoscale_frame",
        "label": "Right wrist motion",
        "group": "WRIST MOTION",
        "unit": "torso scales/frame",
    },

    {
        "metric": "left_wrist_peak_torsoscale_frame",
        "label": "Left wrist motion",
        "group": "WRIST MOTION",
        "unit": "torso scales/frame",
    },

    {
        "metric": "hip_center_peak_px_frame",
        "label": "Hip center motion",
        "group": "BODY CENTER MOTION",
        "unit": "px/frame",
    },

    {
        "metric": "shoulder_center_peak_px_frame",
        "label": "Shoulder center motion",
        "group": "BODY CENTER MOTION",
        "unit": "px/frame",
    },

    {
        "metric": "shoulder_rotation_projected_deg",
        "label": "Shoulder rotation",
        "group": "PROJECTED ROTATION",
        "unit": "deg projected",
    },

    {
        "metric": "hip_rotation_projected_deg",
        "label": "Hip rotation",
        "group": "PROJECTED ROTATION",
        "unit": "deg projected",
    },

    {
        "metric": "hip_shoulder_separation_projected_deg",
        "label": "Hip-shoulder separation",
        "group": "PROJECTED ROTATION",
        "unit": "deg projected",
    },

    {
        "metric": "head_prepeak_torso",
        "label": "Head pre-peak movement",
        "group": "PRE-PEAK STABILITY",
        "unit": "torso lengths",
    },

    {
        "metric": "left_ankle_prepeak_torso",
        "label": "Left ankle pre-peak movement",
        "group": "PRE-PEAK STABILITY",
        "unit": "torso lengths",
    },

    {
        "metric": "right_ankle_prepeak_torso",
        "label": "Right ankle pre-peak movement",
        "group": "PRE-PEAK STABILITY",
        "unit": "torso lengths",
    },

    {
        "metric": "load_duration_sec",
        "label": "Load duration",
        "group": "TIMING",
        "unit": "s",
    },

    {
        "metric": "swing_duration_sec",
        "label": "Swing duration",
        "group": "TIMING",
        "unit": "s",
    },

    {
        "metric": "load_to_peak_sec",
        "label": "Load to peak timing",
        "group": "TIMING",
        "unit": "s",
    },

    {
        "metric": "swing_to_peak_sec",
        "label": "Swing to peak timing",
        "group": "TIMING",
        "unit": "s",
    },
]


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

    validity_columns = [
        col
        for col in (
            merge_keys
            + [
                "validity_status",
                "evidence_quality",
                "reason",
            ]
        )
        if col in validity_df.columns
    ]

    merged = database_df.merge(
        validity_df[
            validity_columns
        ],
        on=merge_keys,
        how="left"
    )

    if "validity_status" not in merged.columns:
        merged["validity_status"] = "NOT_EVALUATED"

    if "evidence_quality" not in merged.columns:
        merged["evidence_quality"] = "UNKNOWN"

    if "reason" not in merged.columns:
        merged["reason"] = "N/A"

    merged["validity_status"] = (
        merged["validity_status"]
        .fillna("NOT_EVALUATED")
    )

    merged["evidence_quality"] = (
        merged["evidence_quality"]
        .fillna("UNKNOWN")
    )

    merged["reason"] = (
        merged["reason"]
        .fillna("N/A")
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


def find_video(video_id):

    search_roots = [

        Path(
            "results/full_analysis_v3"
        ),

        Path(
            "results/full_analysis_v2"
        ),

        Path(
            "results/evaluation_v1"
        ),
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

            candidate = (
                root / name
            )

            if candidate.exists():
                return candidate

    for root in search_roots:

        if not root.exists():
            continue

        matches = list(
            root.rglob(
                f"*{video_id}*.mp4"
            )
        )

        if len(matches) > 0:
            return matches[0]

    return None


# ============================================================
# FEEDBACK HELPERS
# ============================================================

def classify_zscore(z):

    z = safe_float(z)

    if not np.isfinite(z):
        return "N/A"

    if z <= -1.0:
        return "LOW"

    if z <= -0.5:
        return "SLIGHTLY_LOW"

    if z < 0.5:
        return "CONSISTENT"

    if z < 1.0:
        return "SLIGHTLY_HIGH"

    return "HIGH"


def classification_feedback(
    label,
    classification
):

    mapping = {

        "LOW":
            f"{label}: below the player's typical session range.",

        "SLIGHTLY_LOW":
            f"{label}: slightly below the player's session baseline.",

        "CONSISTENT":
            f"{label}: consistent with the player's session baseline.",

        "SLIGHTLY_HIGH":
            f"{label}: slightly above the player's session baseline.",

        "HIGH":
            f"{label}: above the player's typical session range.",
    }

    return mapping.get(
        classification,
        f"{label}: comparison unavailable."
    )


def measurement_confidence(
    pose_quality
):

    pose_quality = safe_float(
        pose_quality
    )

    if not np.isfinite(
        pose_quality
    ):
        return "UNKNOWN"

    if pose_quality >= 0.90:
        return "HIGH"

    if pose_quality >= 0.75:
        return "MODERATE"

    return "LOW"


def build_feedback(
    target_row,
    all_df
):

    player_id = str(
        target_row[
            "player_id"
        ]
    )

    session_id = str(
        target_row[
            "session_id"
        ]
    )

    video_id = str(
        target_row[
            "video_id"
        ]
    )

    session_df = all_df[
        (
            all_df[
                "player_id"
            ].astype(str)
            ==
            player_id
        )
        &
        (
            all_df[
                "session_id"
            ].astype(str)
            ==
            session_id
        )
    ].copy()

    valid_df = session_df[
        session_df[
            "validity_status"
        ]
        ==
        "VALID_SWING"
    ].copy()

    baseline_df = valid_df[
        valid_df[
            "video_id"
        ].astype(str)
        !=
        video_id
    ].copy()

    baseline_mode = (
        "LEAVE_ONE_OUT"
    )

    target_excluded = True

    if len(baseline_df) == 0:

        baseline_df = valid_df.copy()

        baseline_mode = (
            "VALID_SESSION_BASELINE"
        )

        target_excluded = False

    rows = []

    for config in FEEDBACK_METRICS:

        metric = config[
            "metric"
        ]

        if metric not in all_df.columns:
            continue

        target = safe_float(
            target_row[
                metric
            ]
        )

        baseline_values = (
            pd.to_numeric(
                baseline_df[
                    metric
                ],
                errors="coerce"
            )
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
            .dropna()
        )

        if (
            not np.isfinite(target)
            or
            len(baseline_values) == 0
        ):
            continue

        baseline_mean = (
            baseline_values.mean()
        )

        if len(baseline_values) >= 2:

            baseline_sd = (
                baseline_values.std(
                    ddof=1
                )
            )

        else:

            baseline_sd = np.nan

        if (
            np.isfinite(
                baseline_mean
            )
            and
            abs(
                baseline_mean
            ) > 1e-9
        ):

            difference_pct = (
                (
                    target
                    -
                    baseline_mean
                )
                /
                abs(
                    baseline_mean
                )
                *
                100.0
            )

        else:

            difference_pct = np.nan

        if (
            np.isfinite(
                baseline_sd
            )
            and
            baseline_sd > 1e-9
        ):

            zscore = (
                target
                -
                baseline_mean
            ) / baseline_sd

        else:

            zscore = np.nan

        classification = (
            classify_zscore(
                zscore
            )
        )

        feedback_text = (
            classification_feedback(
                config[
                    "label"
                ],
                classification
            )
        )

        rows.append(
            {
                "group":
                    config[
                        "group"
                    ],

                "metric":
                    metric,

                "label":
                    config[
                        "label"
                    ],

                "unit":
                    config[
                        "unit"
                    ],

                "target":
                    target,

                "baseline":
                    baseline_mean,

                "baseline_sd":
                    baseline_sd,

                "difference_pct":
                    difference_pct,

                "zscore":
                    zscore,

                "classification":
                    classification,

                "feedback":
                    feedback_text,
            }
        )

    feedback_df = pd.DataFrame(
        rows
    )

    metadata = {

        "session_profiles":
            len(
                session_df
            ),

        "valid_profiles":
            len(
                valid_df
            ),

        "baseline_profiles":
            len(
                baseline_df
            ),

        "baseline_mode":
            baseline_mode,

        "target_excluded":
            target_excluded,
    }

    return (
        feedback_df,
        metadata
    )


def classification_counts(
    feedback_df
):

    if len(feedback_df) == 0:

        return {
            "consistent": 0,
            "slightly": 0,
            "outside": 0,
            "high": 0,
            "low": 0,
        }

    classes = (
        feedback_df[
            "classification"
        ]
        .astype(str)
    )

    consistent = int(
        (
            classes
            ==
            "CONSISTENT"
        ).sum()
    )

    slightly = int(
        classes.isin(
            [
                "SLIGHTLY_LOW",
                "SLIGHTLY_HIGH",
            ]
        ).sum()
    )

    high = int(
        (
            classes
            ==
            "HIGH"
        ).sum()
    )

    low = int(
        (
            classes
            ==
            "LOW"
        ).sum()
    )

    outside = (
        high
        +
        low
    )

    return {
        "consistent":
            consistent,

        "slightly":
            slightly,

        "outside":
            outside,

        "high":
            high,

        "low":
            low,
    }


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
    "Player Swing Analytics Dashboard V2.1"
)

st.caption(
    "Comparative 2D computer-vision swing analysis "
    "with session-relative feedback."
)

st.info(
    "Feedback describes how a swing differs from the player's "
    "own session baseline. HIGH/LOW are statistical descriptions "
    "and do not represent GOOD/BAD swing quality."
)

st.divider()


# ============================================================
# SIDEBAR
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


selected_player = (
    st.sidebar.selectbox(
        "Player",
        players
    )
)


player_df = df[
    df[
        "player_id"
    ].astype(str)
    ==
    str(
        selected_player
    )
].copy()


sessions = sorted(
    player_df[
        "session_id"
    ]
    .astype(str)
    .unique()
)


selected_session = (
    st.sidebar.selectbox(
        "Session",
        [
            "ALL"
        ]
        +
        sessions
    )
)


if selected_session != "ALL":

    filtered_df = player_df[
        player_df[
            "session_id"
        ].astype(str)
        ==
        str(
            selected_session
        )
    ].copy()

else:

    filtered_df = (
        player_df.copy()
    )


evidence_filter = (
    st.sidebar.radio(
        "Swing Evidence",
        [
            "VALID_SWING only",
            "All analyzed sequences",
        ]
    )
)


if (
    evidence_filter
    ==
    "VALID_SWING only"
):

    filtered_df = filtered_df[
        filtered_df[
            "validity_status"
        ]
        ==
        "VALID_SWING"
    ].copy()


if len(filtered_df) == 0:

    st.warning(
        "No profiles match the current filters."
    )

    st.stop()


# ============================================================
# SESSION OVERVIEW
# ============================================================

st.header(
    "Session Overview"
)


c1, c2, c3, c4 = (
    st.columns(
        4
    )
)


c1.metric(
    "Profiles",
    len(
        filtered_df
    )
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


col_a, col_b = (
    st.columns(
        2
    )
)


with col_a:

    swing_a_id = (
        st.selectbox(
            "Swing A",
            video_options,
            index=0
        )
    )


with col_b:

    default_b = (
        1
        if len(
            video_options
        ) > 1
        else 0
    )

    swing_b_id = (
        st.selectbox(
            "Swing B",
            video_options,
            index=default_b
        )
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


video_col_a, video_col_b = (
    st.columns(
        2
    )
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
            str(
                video_a
            )
        )

    else:

        st.info(
            "Full Analysis video not found."
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
            str(
                video_b
            )
        )

    else:

        st.info(
            "Full Analysis video not found."
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


metric_cols_2 = (
    st.columns(
        4
    )
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
            "right_wrist_peak_torsoscale_frame"
        ]
    )
)


metric_cols_2[3].metric(
    "B — Right Wrist",
    fmt(
        swing_b[
            "right_wrist_peak_torsoscale_frame"
        ]
    )
)


st.caption(
    "Wrist motion is shown normalized by torso scale "
    "for improved swing-to-swing comparison."
)


st.divider()


# ============================================================
# SWING FEEDBACK V1.1
# ============================================================

st.header(
    "Swing Feedback V1.1"
)

st.caption(
    "Each selected swing is compared against the player's "
    "other VALID_SWING profiles from the same session."
)


feedback_a, feedback_meta_a = (
    build_feedback(
        swing_a,
        df
    )
)

feedback_b, feedback_meta_b = (
    build_feedback(
        swing_b,
        df
    )
)


# ============================================================
# FEEDBACK SUMMARY
# ============================================================

feedback_col_a, feedback_col_b = (
    st.columns(
        2
    )
)


with feedback_col_a:

    st.markdown(
        f"### Swing A — {swing_a_id}"
    )

    counts_a = (
        classification_counts(
            feedback_a
        )
    )

    ca1, ca2, ca3 = (
        st.columns(
            3
        )
    )

    ca1.metric(
        "Consistent",
        counts_a[
            "consistent"
        ]
    )

    ca2.metric(
        "Slightly Different",
        counts_a[
            "slightly"
        ]
    )

    ca3.metric(
        "Outside Range",
        counts_a[
            "outside"
        ]
    )

    st.write(
        "**Baseline profiles:**",
        feedback_meta_a[
            "baseline_profiles"
        ]
    )

    st.write(
        "**Baseline mode:**",
        feedback_meta_a[
            "baseline_mode"
        ]
    )


with feedback_col_b:

    st.markdown(
        f"### Swing B — {swing_b_id}"
    )

    counts_b = (
        classification_counts(
            feedback_b
        )
    )

    cb1, cb2, cb3 = (
        st.columns(
            3
        )
    )

    cb1.metric(
        "Consistent",
        counts_b[
            "consistent"
        ]
    )

    cb2.metric(
        "Slightly Different",
        counts_b[
            "slightly"
        ]
    )

    cb3.metric(
        "Outside Range",
        counts_b[
            "outside"
        ]
    )

    st.write(
        "**Baseline profiles:**",
        feedback_meta_b[
            "baseline_profiles"
        ]
    )

    st.write(
        "**Baseline mode:**",
        feedback_meta_b[
            "baseline_mode"
        ]
    )


# ============================================================
# MEASUREMENT CONFIDENCE
# ============================================================

st.subheader(
    "Measurement Confidence"
)


conf_cols = st.columns(
    2
)


pose_a = safe_float(
    swing_a[
        "mean_pose_quality"
    ]
)

pose_b = safe_float(
    swing_b[
        "mean_pose_quality"
    ]
)


conf_cols[0].metric(
    f"Swing A — {swing_a_id}",
    measurement_confidence(
        pose_a
    ),
    f"Pose quality {fmt(pose_a)}"
)


conf_cols[1].metric(
    f"Swing B — {swing_b_id}",
    measurement_confidence(
        pose_b
    ),
    f"Pose quality {fmt(pose_b)}"
)


st.caption(
    "Measurement confidence describes pose-estimation "
    "reliability, not swing quality."
)


# ============================================================
# DETAILED FEEDBACK TABLE
# ============================================================

st.subheader(
    "Detailed Session-Relative Feedback"
)


feedback_view_a = (
    feedback_a.copy()
)

feedback_view_b = (
    feedback_b.copy()
)


if len(feedback_view_a) > 0:

    feedback_view_a[
        "Swing"
    ] = (
        f"A — {swing_a_id}"
    )


if len(feedback_view_b) > 0:

    feedback_view_b[
        "Swing"
    ] = (
        f"B — {swing_b_id}"
    )


feedback_combined = pd.concat(
    [
        feedback_view_a,
        feedback_view_b,
    ],
    ignore_index=True
)


if len(feedback_combined) > 0:

    display_feedback = (
        feedback_combined[
            [
                "Swing",
                "group",
                "label",
                "target",
                "baseline",
                "baseline_sd",
                "difference_pct",
                "zscore",
                "classification",
                "unit",
                "feedback",
            ]
        ]
        .rename(
            columns={
                "group":
                    "Group",

                "label":
                    "Metric",

                "target":
                    "Target",

                "baseline":
                    "Session Baseline",

                "baseline_sd":
                    "Baseline SD",

                "difference_pct":
                    "Difference %",

                "zscore":
                    "Z-score",

                "classification":
                    "Classification",

                "unit":
                    "Unit",

                "feedback":
                    "Feedback",
            }
        )
    )

    st.dataframe(
        display_feedback.style.format(
            {
                "Target":
                    "{:.3f}",

                "Session Baseline":
                    "{:.3f}",

                "Baseline SD":
                    "{:.3f}",

                "Difference %":
                    "{:+.1f}%",

                "Z-score":
                    "{:+.2f}",
            },
            na_rep="N/A"
        ),
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No feedback metrics available."
    )


# ============================================================
# FEEDBACK Z-SCORE VISUALIZATION
# ============================================================

if len(feedback_combined) > 0:

    st.subheader(
        "Session Deviation"
    )

    zscore_plot = (
        feedback_combined[
            [
                "Swing",
                "label",
                "zscore",
            ]
        ]
        .dropna()
        .rename(
            columns={
                "label":
                    "Metric",

                "zscore":
                    "Z-score",
            }
        )
    )

    if len(zscore_plot) > 0:

        z_fig = px.bar(
            zscore_plot,
            x="Metric",
            y="Z-score",
            color="Swing",
            barmode="group",
            title=(
                "Deviation from Session Baseline"
            )
        )

        z_fig.add_hline(
            y=1.0,
            line_dash="dash",
            annotation_text="+1 SD"
        )

        z_fig.add_hline(
            y=-1.0,
            line_dash="dash",
            annotation_text="-1 SD"
        )

        z_fig.add_hline(
            y=0.0,
            line_dash="dot"
        )

        st.plotly_chart(
            z_fig,
            use_container_width=True
        )

        st.caption(
            "Z = 0 represents the player's session baseline. "
            "Values near ±1 indicate approximately one baseline "
            "standard deviation from the session mean."
        )


st.divider()


# ============================================================
# METRIC DIFFERENCES
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


comparison_df = (
    pd.DataFrame(
        comparison_rows
    )
)


st.dataframe(
    comparison_df.style.format(
        {
            "Swing A":
                "{:.3f}",

            "Swing B":
                "{:.3f}",

            "B - A":
                "{:+.3f}",

            "% Change":
                "{:+.1f}%",
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

    a_norm, b_norm = (
        normalize_pair(
            swing_a[
                metric
            ],
            swing_b[
                metric
            ]
        )
    )

    if (
        not np.isfinite(
            a_norm
        )
        or
        not np.isfinite(
            b_norm
        )
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

    radar_fig = (
        go.Figure()
    )

    radar_fig.add_trace(
        go.Scatterpolar(
            r=radar_a_closed,
            theta=radar_labels_closed,
            fill="toself",
            name=(
                f"Swing A — {swing_a_id}"
            )
        )
    )

    radar_fig.add_trace(
        go.Scatterpolar(
            r=radar_b_closed,
            theta=radar_labels_closed,
            fill="toself",
            name=(
                f"Swing B — {swing_b_id}"
            )
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
        "Radar values are normalized only between the two "
        "selected swings. They are not performance scores."
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


timing_df = (
    pd.DataFrame(
        timing_rows
    )
)


if len(timing_df) > 0:

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

        "left_wrist_peak_torsoscale_frame",

        "right_wrist_peak_torsoscale_frame",

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


wrist_df = (
    pd.DataFrame(
        wrist_rows
    )
)


if len(wrist_df) > 0:

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
                "Peak motion (torso scales/frame)",

            "video_id":
                "Swing",
        },
        title=(
            "Normalized Wrist Peak Motion"
        )
    )

    st.plotly_chart(
        wrist_fig,
        use_container_width=True
    )

    st.caption(
        "Wrist motion is normalized by torso scale "
        "to reduce sensitivity to player-camera distance."
    )


# ============================================================
# BODY CENTER MOTION
# ============================================================

st.header(
    "Body Center Motion"
)


body_rows = []


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

        "hip_center_peak_px_frame",

        "shoulder_center_peak_px_frame",

    ]:

        if metric not in filtered_df.columns:
            continue

        body_rows.append(
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


body_df = (
    pd.DataFrame(
        body_rows
    )
)


if len(body_df) > 0:

    body_fig = px.bar(
        body_df,
        x="metric",
        y="value",
        color="video_id",
        barmode="group",
        labels={
            "metric":
                "Body Center",

            "value":
                "Peak motion (px/frame)",

            "video_id":
                "Swing",
        },
        title=(
            "Body Center Peak Motion"
        )
    )

    st.plotly_chart(
        body_fig,
        use_container_width=True
    )


# ============================================================
# PROJECTED ROTATION
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


rotation_df = (
    pd.DataFrame(
        rotation_rows
    )
)


if len(rotation_df) > 0:

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
# PRE-PEAK STABILITY
# ============================================================

st.header(
    "Pre-Peak Stability"
)


stability_rows = []


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

        "head_prepeak_torso",

        "left_ankle_prepeak_torso",

        "right_ankle_prepeak_torso",

    ]:

        if metric not in filtered_df.columns:
            continue

        stability_rows.append(
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


stability_df = (
    pd.DataFrame(
        stability_rows
    )
)


if len(stability_df) > 0:

    stability_fig = px.bar(
        stability_df,
        x="metric",
        y="value",
        color="video_id",
        barmode="group",
        labels={
            "metric":
                "Measurement",

            "value":
                "Movement (torso lengths)",

            "video_id":
                "Swing",
        },
        title=(
            "Pre-Peak Movement Comparison"
        )
    )

    st.plotly_chart(
        stability_fig,
        use_container_width=True
    )


# ============================================================
# SWING EVIDENCE
# ============================================================

st.divider()

st.header(
    "Swing Evidence"
)


validity_cols = (
    st.columns(
        2
    )
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
# FEEDBACK METHODOLOGY
# ============================================================

st.divider()

with st.expander(
    "Feedback Methodology"
):

    st.markdown(
        """
### Session-relative comparison

BaseVision AI compares each selected swing against the
player's other `VALID_SWING` profiles from the same session.

When possible, the target swing is excluded from its own
baseline using a leave-one-out comparison.

### Classification

The classification is based on the target's position relative
to the session baseline:

- `CONSISTENT`: approximately within ±0.5 baseline SD
- `SLIGHTLY_LOW`: approximately -0.5 to -1.0 SD
- `SLIGHTLY_HIGH`: approximately +0.5 to +1.0 SD
- `LOW`: at or below approximately -1.0 SD
- `HIGH`: at or above approximately +1.0 SD

These labels describe statistical differences only.

They do **not** indicate whether a swing is mechanically
correct, incorrect, better, or worse.

### Measurement confidence

Pose quality is reported separately as measurement
confidence and is not included as a swing-performance score.

### 2D limitation

Rotation metrics are projected 2D measurements and depend on
camera viewpoint. They must not be interpreted as calibrated
3D biomechanical joint or segment rotations.
        """
    )


# ============================================================
# RAW DATA
# ============================================================

st.divider()

st.header(
    "Selected Swing Data"
)


raw_df = (
    pd.DataFrame(
        [
            swing_a,
            swing_b
        ]
    )
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
    "BaseVision AI — Dashboard V2.1 | "
    "Comparative 2D computer-vision swing analytics | "
    "Session-relative feedback | "
    "No calibrated biomechanics or outcome classification."
)