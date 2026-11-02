"""
BaseVisionAI - Dashboard V2.2
=============================

Integrated baseball swing analysis dashboard.

Data sources
------------
- Player Swing Database V1
- Swing Validity V1.1
- Swing Feedback V1.2
- Swing Consistency V1.1
- Full Analysis V3 web-compatible videos

Important
---------
This dashboard presents comparative 2D swing analysis.

HIGH / LOW classifications describe statistical position relative
to the player's own session baseline.

They are NOT biomechanical GOOD / BAD judgments and are NOT
performance scores.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="BaseVisionAI",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"

PLAYER_DATABASE = (
    RESULTS_DIR
    / "player_profiles"
    / "player_swing_database_v1.csv"
)

VALIDITY_DATABASE = (
    RESULTS_DIR
    / "swing_validity"
    / "swing_validity_v1_1.csv"
)

FULL_ANALYSIS_DIR = (
    RESULTS_DIR
    / "full_analysis_v3"
)

FEEDBACK_ROOT = (
    RESULTS_DIR
    / "swing_feedback_v1_2"
)

CONSISTENCY_ROOT = (
    RESULTS_DIR
    / "swing_consistency"
)


# =============================================================================
# STYLE
# =============================================================================

st.markdown(
    """
<style>

/* ------------------------------------------------------------------------- */
/* GLOBAL                                                                    */
/* ------------------------------------------------------------------------- */

.block-container {
    padding-top: 1.3rem;
    padding-bottom: 3rem;
    max-width: 1550px;
}

h1, h2, h3 {
    letter-spacing: -0.02em;
}

[data-testid="stMetricValue"] {
    font-size: 1.65rem;
}

[data-testid="stMetricLabel"] {
    font-size: 0.85rem;
}


/* ------------------------------------------------------------------------- */
/* HEADER                                                                    */
/* ------------------------------------------------------------------------- */

.bv-header {
    padding: 1.25rem 1.4rem;
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 14px;
    margin-bottom: 1rem;
}

.bv-title {
    font-size: 2rem;
    font-weight: 750;
    margin-bottom: 0.15rem;
}

.bv-subtitle {
    opacity: 0.72;
    font-size: 0.95rem;
}


/* ------------------------------------------------------------------------- */
/* SECTION                                                                   */
/* ------------------------------------------------------------------------- */

.bv-section {
    margin-top: 1.8rem;
    margin-bottom: 0.65rem;
    font-size: 1.15rem;
    font-weight: 700;
}


/* ------------------------------------------------------------------------- */
/* STATUS BADGES                                                             */
/* ------------------------------------------------------------------------- */

.badge {
    display: inline-block;
    padding: 0.30rem 0.65rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    margin-right: 0.35rem;
    margin-bottom: 0.25rem;
}

.badge-valid {
    background: rgba(40, 180, 99, 0.15);
    border: 1px solid rgba(40, 180, 99, 0.40);
}

.badge-warning {
    background: rgba(241, 196, 15, 0.15);
    border: 1px solid rgba(241, 196, 15, 0.40);
}

.badge-danger {
    background: rgba(231, 76, 60, 0.15);
    border: 1px solid rgba(231, 76, 60, 0.40);
}

.badge-info {
    background: rgba(52, 152, 219, 0.15);
    border: 1px solid rgba(52, 152, 219, 0.40);
}


/* ------------------------------------------------------------------------- */
/* CARDS                                                                     */
/* ------------------------------------------------------------------------- */

.info-card {
    padding: 1rem;
    border: 1px solid rgba(128,128,128,0.22);
    border-radius: 12px;
    min-height: 130px;
}

.card-title {
    font-size: 0.78rem;
    opacity: 0.65;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

.card-value {
    font-size: 1.25rem;
    font-weight: 700;
}

.card-small {
    font-size: 0.82rem;
    opacity: 0.72;
    margin-top: 0.35rem;
}


/* ------------------------------------------------------------------------- */
/* FEEDBACK                                                                  */
/* ------------------------------------------------------------------------- */

.feedback-row {
    padding: 0.75rem 0.85rem;
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 10px;
    margin-bottom: 0.45rem;
}

.feedback-name {
    font-weight: 650;
}

.feedback-text {
    font-size: 0.88rem;
    opacity: 0.78;
    margin-top: 0.2rem;
}


/* ------------------------------------------------------------------------- */
/* FOOTER                                                                    */
/* ------------------------------------------------------------------------- */

.bv-footer {
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(128,128,128,0.20);
    opacity: 0.60;
    font-size: 0.78rem;
}

</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# HELPERS
# =============================================================================

def safe_float(value, default=np.nan):
    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except (TypeError, ValueError):
        pass

    return default


def safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_bool(value):
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def fmt(value, decimals=2):
    value = safe_float(value)

    if not np.isfinite(value):
        return "N/A"

    return f"{value:.{decimals}f}"


def fmt_percent(value, decimals=1):
    value = safe_float(value)

    if not np.isfinite(value):
        return "N/A"

    return f"{value:.{decimals}f}%"


def first_existing_column(row, columns, default=None):
    for column in columns:
        if column in row.index:
            value = row[column]

            if pd.notna(value):
                return value

    return default


def classification_icon(classification):
    mapping = {
        "CONSISTENT": "✓",
        "SLIGHTLY_HIGH": "↑",
        "SLIGHTLY_LOW": "↓",
        "HIGH": "▲",
        "LOW": "▼",
        "INSUFFICIENT_DATA": "—",
    }

    return mapping.get(
        str(classification).upper(),
        "•",
    )


def classification_label(classification):
    mapping = {
        "CONSISTENT": "Consistent",
        "SLIGHTLY_HIGH": "Slightly high",
        "SLIGHTLY_LOW": "Slightly low",
        "HIGH": "High",
        "LOW": "Low",
        "INSUFFICIENT_DATA": "Insufficient data",
    }

    return mapping.get(
        str(classification).upper(),
        str(classification),
    )


def status_badge(status):
    status = str(status).upper()

    if status == "VALID_SWING":
        css = "badge-valid"

    elif status in {
        "TEMPORAL_LOW_CONFIDENCE",
        "LOW_CONFIDENCE_SWING",
    }:
        css = "badge-warning"

    else:
        css = "badge-info"

    return (
        f'<span class="badge {css}">'
        f'{status.replace("_", " ")}'
        f"</span>"
    )


def confidence_badge(label, value):
    value = str(value).upper()

    if value in {"HIGH", "SUPPORTED", "SUFFICIENT"}:
        css = "badge-valid"

    elif value in {
        "MODERATE",
        "LIMITED",
        "LOW_CONFIDENCE",
    }:
        css = "badge-warning"

    elif value == "LOW":
        css = "badge-danger"

    else:
        css = "badge-info"

    return (
        f'<span class="badge {css}">'
        f"{label}: {value.replace('_', ' ')}"
        f"</span>"
    )


# =============================================================================
# LOAD DATA
# =============================================================================

@st.cache_data
def load_csv(path_string):
    path = Path(path_string)

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)

    except Exception:
        return None


@st.cache_data
def load_player_database():
    return load_csv(str(PLAYER_DATABASE))


@st.cache_data
def load_validity_database():
    return load_csv(str(VALIDITY_DATABASE))


# =============================================================================
# FILE PATH FUNCTIONS
# =============================================================================

def feedback_detail_path(
    player_id,
    session_id,
    video_id,
):
    return (
        FEEDBACK_ROOT
        / str(player_id)
        / str(session_id)
        / f"{video_id}_swing_feedback_v1_2.csv"
    )


def feedback_summary_path(
    player_id,
    session_id,
    video_id,
):
    return (
        FEEDBACK_ROOT
        / str(player_id)
        / str(session_id)
        / f"{video_id}_swing_feedback_v1_2_summary.csv"
    )


def consistency_directory(
    player_id,
    session_id,
):
    return (
        CONSISTENCY_ROOT
        / str(player_id)
        / f"{session_id}_v1_1"
    )


def consistency_metrics_path(
    player_id,
    session_id,
):
    return (
        consistency_directory(
            player_id,
            session_id,
        )
        / "swing_consistency_v1_1_metrics.csv"
    )


def consistency_summary_path(
    player_id,
    session_id,
):
    return (
        consistency_directory(
            player_id,
            session_id,
        )
        / "swing_consistency_v1_1_summary.csv"
    )


def consistency_swings_path(
    player_id,
    session_id,
):
    return (
        consistency_directory(
            player_id,
            session_id,
        )
        / "swing_consistency_v1_1_swings.csv"
    )


def consistency_groups_path(
    player_id,
    session_id,
):
    return (
        consistency_directory(
            player_id,
            session_id,
        )
        / "swing_consistency_v1_1_groups.csv"
    )


def analysis_video_path(video_id):
    """
    Prefer browser-compatible video.
    """

    web_video = (
        FULL_ANALYSIS_DIR
        / f"{video_id}_full_analysis_v3_web.mp4"
    )

    if web_video.exists():
        return web_video

    fallback = (
        FULL_ANALYSIS_DIR
        / f"{video_id}_full_analysis_v3.mp4"
    )

    if fallback.exists():
        return fallback

    return None


# =============================================================================
# SESSION DATA
# =============================================================================

database = load_player_database()
validity = load_validity_database()


if database is None:
    st.error(
        f"Player database not found:\n\n{PLAYER_DATABASE}"
    )
    st.stop()


if validity is None:
    st.error(
        f"Validity database not found:\n\n{VALIDITY_DATABASE}"
    )
    st.stop()


required_db_columns = {
    "video_id",
    "player_id",
    "session_id",
}

if not required_db_columns.issubset(database.columns):
    st.error(
        "Player database is missing required columns."
    )
    st.stop()


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.title("⚾ BaseVisionAI")

    st.caption(
        "Baseball Swing Analysis"
    )

    st.divider()

    player_options = sorted(
        database[
            "player_id"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not player_options:
        st.error("No players found.")
        st.stop()

    selected_player = st.selectbox(
        "Player",
        player_options,
    )

    player_df = database[
        database["player_id"].astype(str)
        == selected_player
    ].copy()

    session_options = sorted(
        player_df[
            "session_id"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not session_options:
        st.error(
            "No sessions found for this player."
        )
        st.stop()

    selected_session = st.selectbox(
        "Session",
        session_options,
    )

    session_df = player_df[
        player_df["session_id"].astype(str)
        == selected_session
    ].copy()

    video_options = (
        session_df[
            "video_id"
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    if not video_options:
        st.error(
            "No swings found for this session."
        )
        st.stop()

    selected_video = st.selectbox(
        "Swing",
        video_options,
    )

    st.divider()

    st.caption(
        "Analysis pipeline"
    )

    st.markdown(
        """
**Video → Pose → Ball → Phases → Metrics → Profile → Validity → Feedback**
"""
    )

    st.divider()

    if st.button(
        "Refresh data",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.rerun()


# =============================================================================
# TARGET PROFILE
# =============================================================================

target_rows = session_df[
    session_df["video_id"].astype(str)
    == selected_video
].copy()

if target_rows.empty:
    st.error(
        "Selected swing was not found in the player database."
    )
    st.stop()

target_profile = target_rows.iloc[0]


# =============================================================================
# VALIDITY
# =============================================================================

validity_rows = validity[
    (
        validity["player_id"].astype(str)
        == selected_player
    )
    &
    (
        validity["session_id"].astype(str)
        == selected_session
    )
    &
    (
        validity["video_id"].astype(str)
        == selected_video
    )
].copy()


if validity_rows.empty:

    validity_row = pd.Series(dtype=object)

    validity_status = "UNKNOWN"
    baseline_eligible = False
    temporal_confidence = "UNKNOWN"
    evidence_quality = "UNKNOWN"

else:

    validity_row = validity_rows.iloc[0]

    validity_status = first_existing_column(
        validity_row,
        [
            "validity_status_v1_1",
            "validity_status",
        ],
        "UNKNOWN",
    )

    baseline_eligible = to_bool(
        first_existing_column(
            validity_row,
            ["baseline_eligible"],
            False,
        )
    )

    temporal_confidence = first_existing_column(
        validity_row,
        ["temporal_confidence"],
        "UNKNOWN",
    )

    evidence_quality = first_existing_column(
        validity_row,
        [
            "evidence_quality_v1_1",
            "evidence_quality",
        ],
        "UNKNOWN",
    )


# =============================================================================
# FEEDBACK
# =============================================================================

feedback_df = load_csv(
    str(
        feedback_detail_path(
            selected_player,
            selected_session,
            selected_video,
        )
    )
)

feedback_summary_df = load_csv(
    str(
        feedback_summary_path(
            selected_player,
            selected_session,
            selected_video,
        )
    )
)

if (
    feedback_summary_df is not None
    and not feedback_summary_df.empty
):
    feedback_summary = (
        feedback_summary_df.iloc[0]
    )
else:
    feedback_summary = pd.Series(
        dtype=object
    )


# =============================================================================
# CONSISTENCY
# =============================================================================

consistency_metrics = load_csv(
    str(
        consistency_metrics_path(
            selected_player,
            selected_session,
        )
    )
)

consistency_summary_df = load_csv(
    str(
        consistency_summary_path(
            selected_player,
            selected_session,
        )
    )
)

consistency_swings = load_csv(
    str(
        consistency_swings_path(
            selected_player,
            selected_session,
        )
    )
)

consistency_groups = load_csv(
    str(
        consistency_groups_path(
            selected_player,
            selected_session,
        )
    )
)

if (
    consistency_summary_df is not None
    and not consistency_summary_df.empty
):
    consistency_summary = (
        consistency_summary_df.iloc[0]
    )
else:
    consistency_summary = pd.Series(
        dtype=object
    )


# =============================================================================
# REPRESENTATIVE SWING
# =============================================================================

representative_video = None
different_video = None


if (
    consistency_swings is not None
    and not consistency_swings.empty
):

    distance_column = None

    for candidate in [
        "robust_distance",
        "distance",
        "robust_session_distance",
    ]:

        if candidate in consistency_swings.columns:
            distance_column = candidate
            break

    if distance_column is not None:

        ranked = consistency_swings.copy()

        ranked[
            distance_column
        ] = pd.to_numeric(
            ranked[
                distance_column
            ],
            errors="coerce",
        )

        ranked = ranked.dropna(
            subset=[distance_column]
        )

        if not ranked.empty:

            representative_video = str(
                ranked.sort_values(
                    distance_column,
                    ascending=True,
                )
                .iloc[0]["video_id"]
            )

            different_video = str(
                ranked.sort_values(
                    distance_column,
                    ascending=False,
                )
                .iloc[0]["video_id"]
            )


# Fallback to summary if present

if representative_video is None:

    representative_video = (
        first_existing_column(
            consistency_summary,
            [
                "representative_video_id",
                "most_representative_swing",
                "representative_swing",
            ],
            None,
        )
    )


if different_video is None:

    different_video = (
        first_existing_column(
            consistency_summary,
            [
                "most_different_video_id",
                "most_different_swing",
            ],
            None,
        )
    )


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    """
<div class="bv-header">

<div class="bv-title">
⚾ BaseVisionAI
</div>

<div class="bv-subtitle">
AI-Powered Baseball Swing Analysis · Dashboard V2.2
</div>

</div>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# CURRENT SELECTION
# =============================================================================

st.markdown(
    '<div class="bv-section">Current Analysis</div>',
    unsafe_allow_html=True,
)


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Player",
    selected_player,
)

col2.metric(
    "Session",
    selected_session,
)

col3.metric(
    "Selected Swing",
    selected_video,
)

col4.metric(
    "Session Swings",
    len(session_df),
)


# =============================================================================
# VIDEO + STATUS
# =============================================================================

st.markdown(
    '<div class="bv-section">Swing Analysis</div>',
    unsafe_allow_html=True,
)


video_col, status_col = st.columns(
    [1.7, 1],
    gap="large",
)


# -------------------------------------------------------------------------
# VIDEO
# -------------------------------------------------------------------------

with video_col:

    st.subheader(
        "Full Analysis Video"
    )

    video_path = analysis_video_path(
        selected_video
    )

    if video_path is not None:

        try:

            with open(
                video_path,
                "rb",
            ) as video_file:

                video_bytes = (
                    video_file.read()
                )

            st.video(
                video_bytes,
                format="video/mp4",
            )

        except Exception as exc:

            st.error(
                f"Unable to load video: {exc}"
            )

    else:

        st.warning(
            "Full Analysis V3 video was not found."
        )


# -------------------------------------------------------------------------
# STATUS
# -------------------------------------------------------------------------

with status_col:

    st.subheader(
        "Swing Status"
    )

    st.markdown(
        status_badge(
            validity_status
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        confidence_badge(
            "Temporal",
            temporal_confidence,
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        confidence_badge(
            "Evidence",
            evidence_quality,
        ),
        unsafe_allow_html=True,
    )

    pose_quality = safe_float(
        target_profile.get(
            "mean_pose_quality",
            np.nan,
        )
    )

    measurement_confidence = (
        first_existing_column(
            feedback_summary,
            ["measurement_confidence"],
            "UNKNOWN",
        )
    )

    st.markdown(
        confidence_badge(
            "Pose",
            measurement_confidence,
        ),
        unsafe_allow_html=True,
    )

    st.write("")

    status_c1, status_c2 = st.columns(2)

    status_c1.metric(
        "Pose Quality",
        fmt(
            pose_quality,
            3,
        ),
    )

    status_c2.metric(
        "Baseline",
        "Eligible"
        if baseline_eligible
        else "Excluded",
    )

    coverage = safe_float(
        first_existing_column(
            feedback_summary,
            ["metric_coverage_percent"],
            np.nan,
        )
    )

    st.metric(
        "Metric Coverage",
        fmt_percent(
            coverage,
            1,
        ),
    )


# =============================================================================
# WARNINGS
# =============================================================================

if validity_status == "TEMPORAL_LOW_CONFIDENCE":

    st.warning(
        """
This swing has **low temporal confidence**.

Pose estimation may still be reliable, but the detected LOAD / SWING /
PEAK timing does not have enough temporal support to define the player's
session baseline.

The swing can still be analyzed against the eligible session baseline.
"""
    )


elif validity_status == "LOW_CONFIDENCE_SWING":

    st.warning(
        """
This swing has **limited structural evidence** and is excluded from the
session baseline.

Its measurements may still be displayed for exploratory comparison.
"""
    )


if (
    np.isfinite(coverage)
    and coverage < 100
):

    available_metrics = safe_int(
        first_existing_column(
            feedback_summary,
            ["metrics_available"],
            0,
        )
    )

    configured_metrics = safe_int(
        first_existing_column(
            feedback_summary,
            ["metrics_configured"],
            14,
        )
    )

    missing_metrics = max(
        configured_metrics
        - available_metrics,
        0,
    )

    st.info(
        f"""
Metric coverage for this swing is **{coverage:.1f}%**
({available_metrics}/{configured_metrics} metrics).

**{missing_metrics} metric(s)** are unavailable and are not imputed.
"""
    )


# =============================================================================
# SESSION OVERVIEW
# =============================================================================

st.markdown(
    '<div class="bv-section">Session Overview</div>',
    unsafe_allow_html=True,
)


validity_session = validity[
    (
        validity["player_id"].astype(str)
        == selected_player
    )
    &
    (
        validity["session_id"].astype(str)
        == selected_session
    )
].copy()


if "baseline_eligible" in validity_session.columns:

    baseline_count = int(
        validity_session[
            "baseline_eligible"
        ]
        .apply(to_bool)
        .sum()
    )

else:
    baseline_count = 0


excluded_count = (
    len(validity_session)
    - baseline_count
)


c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Swings",
    len(session_df),
)

c2.metric(
    "Baseline Eligible",
    baseline_count,
)

c3.metric(
    "Excluded",
    excluded_count,
)

c4.metric(
    "Metrics",
    safe_int(
        first_existing_column(
            feedback_summary,
            ["metrics_configured"],
            14,
        )
    ),
)


# =============================================================================
# FEEDBACK SUMMARY
# =============================================================================

st.markdown(
    '<div class="bv-section">Robust Feedback Summary</div>',
    unsafe_allow_html=True,
)


if feedback_summary.empty:

    st.warning(
        "Swing Feedback V1.2 summary was not found."
    )

else:

    consistent = safe_int(
        feedback_summary.get(
            "consistent",
            0,
        )
    )

    slightly = safe_int(
        feedback_summary.get(
            "slightly_different",
            0,
        )
    )

    outside = safe_int(
        feedback_summary.get(
            "outside_typical_range",
            0,
        )
    )

    high_count = safe_int(
        feedback_summary.get(
            "high",
            0,
        )
    )

    low_count = safe_int(
        feedback_summary.get(
            "low",
            0,
        )
    )

    mean_rz = safe_float(
        feedback_summary.get(
            "mean_abs_robust_z",
            np.nan,
        )
    )

    median_rz = safe_float(
        feedback_summary.get(
            "median_abs_robust_z",
            np.nan,
        )
    )

    f1, f2, f3, f4, f5 = st.columns(5)

    f1.metric(
        "Consistent",
        consistent,
    )

    f2.metric(
        "Slightly Different",
        slightly,
    )

    f3.metric(
        "Outside Range",
        outside,
    )

    f4.metric(
        "Mean |Robust Z|",
        fmt(
            mean_rz,
            2,
        ),
    )

    f5.metric(
        "Median |Robust Z|",
        fmt(
            median_rz,
            2,
        ),
    )

    st.caption(
        f"Outside range → HIGH: {high_count} · LOW: {low_count}"
    )


# =============================================================================
# DETAILED ROBUST FEEDBACK
# =============================================================================

st.markdown(
    '<div class="bv-section">Selected Swing vs Session Baseline</div>',
    unsafe_allow_html=True,
)


if (
    feedback_df is None
    or feedback_df.empty
):

    st.warning(
        "Swing Feedback V1.2 detail file was not found."
    )

else:

    feedback_display = (
        feedback_df.copy()
    )

    if "available" in feedback_display.columns:

        feedback_display[
            "available_bool"
        ] = (
            feedback_display[
                "available"
            ]
            .apply(to_bool)
        )

    else:

        feedback_display[
            "available_bool"
        ] = True


    # ---------------------------------------------------------------------
    # GROUP FILTER
    # ---------------------------------------------------------------------

    groups = (
        feedback_display[
            "group"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_group = st.selectbox(
        "Metric Group",
        ["ALL"] + groups,
    )

    if selected_group != "ALL":

        filtered_feedback = (
            feedback_display[
                feedback_display[
                    "group"
                ].astype(str)
                == selected_group
            ].copy()
        )

    else:

        filtered_feedback = (
            feedback_display.copy()
        )


    # ---------------------------------------------------------------------
    # FEEDBACK CARDS
    # ---------------------------------------------------------------------

    for _, row in filtered_feedback.iterrows():

        classification = str(
            row.get(
                "classification",
                "INSUFFICIENT_DATA",
            )
        )

        icon = classification_icon(
            classification
        )

        metric_name = row.get(
            "metric",
            "Metric",
        )

        feedback_text = row.get(
            "feedback",
            "",
        )

        st.markdown(
            f"""
<div class="feedback-row">

<div class="feedback-name">
{icon} {metric_name}
&nbsp;·&nbsp;
{classification_label(classification)}
</div>

<div class="feedback-text">
{feedback_text}
</div>

</div>
""",
            unsafe_allow_html=True,
        )


    # ---------------------------------------------------------------------
    # TABLE
    # ---------------------------------------------------------------------

    st.subheader(
        "Metric Comparison"
    )

    table = filtered_feedback.copy()

    table[
        "Target"
    ] = pd.to_numeric(
        table.get(
            "target_value",
            np.nan,
        ),
        errors="coerce",
    )

    table[
        "Median"
    ] = pd.to_numeric(
        table.get(
            "baseline_median",
            np.nan,
        ),
        errors="coerce",
    )

    table[
        "MAD"
    ] = pd.to_numeric(
        table.get(
            "baseline_mad",
            np.nan,
        ),
        errors="coerce",
    )

    table[
        "Robust Z"
    ] = pd.to_numeric(
        table.get(
            "robust_z_score",
            np.nan,
        ),
        errors="coerce",
    )

    table[
        "Difference %"
    ] = pd.to_numeric(
        table.get(
            "difference_percent",
            np.nan,
        ),
        errors="coerce",
    )

    display_columns = [
        "group",
        "metric",
        "unit",
        "Target",
        "Median",
        "MAD",
        "Robust Z",
        "Difference %",
        "classification",
    ]

    existing_columns = [
        c
        for c in display_columns
        if c in table.columns
    ]

    table_display = table[
        existing_columns
    ].rename(
        columns={
            "group": "Group",
            "metric": "Metric",
            "unit": "Unit",
            "classification": "Classification",
        }
    )

    st.dataframe(
        table_display,
        use_container_width=True,
        hide_index=True,
    )


# =============================================================================
# TARGET VS BASELINE CHART
# =============================================================================

if (
    feedback_df is not None
    and not feedback_df.empty
):

    st.markdown(
        '<div class="bv-section">Robust Statistical Position</div>',
        unsafe_allow_html=True,
    )

    chart_df = (
        feedback_df.copy()
    )

    chart_df[
        "robust_z_score"
    ] = pd.to_numeric(
        chart_df[
            "robust_z_score"
        ],
        errors="coerce",
    )

    chart_df = chart_df.dropna(
        subset=[
            "metric",
            "robust_z_score",
        ]
    )

    if not chart_df.empty:

        chart_data = (
            chart_df[
                [
                    "metric",
                    "robust_z_score",
                ]
            ]
            .set_index(
                "metric"
            )
        )

        st.bar_chart(
            chart_data,
            horizontal=True,
            height=500,
        )

        st.caption(
            """
Robust Z describes statistical position relative to the player's
session baseline. Positive and negative values do not represent
better or worse performance.
"""
        )


# =============================================================================
# CONSISTENCY
# =============================================================================

st.markdown(
    '<div class="bv-section">Session Consistency</div>',
    unsafe_allow_html=True,
)


if consistency_summary.empty:

    st.warning(
        "Swing Consistency V1.1 summary was not found."
    )

else:

    median_cv = safe_float(
        first_existing_column(
            consistency_summary,
            [
                "median_metric_cv_percent",
                "median_cv_percent",
                "median_cv",
            ],
            np.nan,
        )
    )

    median_robust_cv = safe_float(
        first_existing_column(
            consistency_summary,
            [
                "median_metric_robust_cv_percent",
                "median_robust_cv_percent",
                "median_robust_cv",
            ],
            np.nan,
        )
    )

    mean_cv = safe_float(
        first_existing_column(
            consistency_summary,
            [
                "mean_metric_cv_percent",
                "mean_cv_percent",
                "mean_cv",
            ],
            np.nan,
        )
    )

    eligible_profiles = safe_int(
        first_existing_column(
            consistency_summary,
            [
                "baseline_eligible_profiles",
                "valid_profiles",
                "profiles_analyzed",
            ],
            baseline_count,
        )
    )

    s1, s2, s3, s4 = st.columns(4)

    s1.metric(
        "Median CV",
        fmt_percent(
            median_cv,
            1,
        ),
    )

    s2.metric(
        "Median Robust CV",
        fmt_percent(
            median_robust_cv,
            1,
        ),
    )

    s3.metric(
        "Mean CV",
        fmt_percent(
            mean_cv,
            1,
        ),
    )

    s4.metric(
        "Baseline Swings",
        eligible_profiles,
    )

    st.caption(
        """
Lower variability indicates greater repeatability across the
player's eligible session swings. It does not indicate better
baseball performance.
"""
    )


# =============================================================================
# REPRESENTATIVE SWING
# =============================================================================

st.markdown(
    '<div class="bv-section">Session Reference Swings</div>',
    unsafe_allow_html=True,
)


r1, r2 = st.columns(2)

with r1:

    st.markdown(
        """
<div class="info-card">
<div class="card-title">Most Representative Swing</div>
""",
        unsafe_allow_html=True,
    )

    if representative_video:

        st.markdown(
            f"""
<div class="card-value">{representative_video}</div>
<div class="card-small">
Closest observed swing to the robust session profile.
</div>
</div>
""",
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
<div class="card-value">N/A</div>
<div class="card-small">
Representative swing information unavailable.
</div>
</div>
""",
            unsafe_allow_html=True,
        )


with r2:

    st.markdown(
        """
<div class="info-card">
<div class="card-title">Most Different Swing</div>
""",
        unsafe_allow_html=True,
    )

    if different_video:

        st.markdown(
            f"""
<div class="card-value">{different_video}</div>
<div class="card-small">
Largest robust statistical distance from the session profile.
</div>
</div>
""",
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
<div class="card-value">N/A</div>
<div class="card-small">
Difference information unavailable.
</div>
</div>
""",
            unsafe_allow_html=True,
        )


# =============================================================================
# REPRESENTATIVE VS SELECTED VIDEO
# =============================================================================

if representative_video:

    st.markdown(
        '<div class="bv-section">Selected vs Representative Swing</div>',
        unsafe_allow_html=True,
    )

    left_video, right_video = st.columns(
        2,
        gap="large",
    )


    # ---------------------------------------------------------------------
    # SELECTED
    # ---------------------------------------------------------------------

    with left_video:

        st.subheader(
            f"Selected · {selected_video}"
        )

        selected_path = (
            analysis_video_path(
                selected_video
            )
        )

        if selected_path:

            try:

                with open(
                    selected_path,
                    "rb",
                ) as f:

                    st.video(
                        f.read(),
                        format="video/mp4",
                    )

            except Exception as exc:

                st.error(
                    f"Unable to load selected video: {exc}"
                )

        else:

            st.warning(
                "Selected video unavailable."
            )


    # ---------------------------------------------------------------------
    # REPRESENTATIVE
    # ---------------------------------------------------------------------

    with right_video:

        st.subheader(
            f"Representative · {representative_video}"
        )

        representative_path = (
            analysis_video_path(
                representative_video
            )
        )

        if representative_path:

            try:

                with open(
                    representative_path,
                    "rb",
                ) as f:

                    st.video(
                        f.read(),
                        format="video/mp4",
                    )

            except Exception as exc:

                st.error(
                    f"Unable to load representative video: {exc}"
                )

        else:

            st.warning(
                "Representative video unavailable."
            )


# =============================================================================
# CONSISTENCY METRICS
# =============================================================================

if (
    consistency_metrics is not None
    and not consistency_metrics.empty
):

    st.markdown(
        '<div class="bv-section">Session Variability by Metric</div>',
        unsafe_allow_html=True,
    )

    cm = consistency_metrics.copy()

    possible_columns = [
        "group",
        "metric",
        "unit",
        "samples",
        "mean",
        "median",
        "sd",
        "mad",
        "cv_percent",
        "robust_cv_percent",
        "repeatability_index",
        "classification",
    ]

    existing = [
        column
        for column in possible_columns
        if column in cm.columns
    ]

    if existing:

        st.dataframe(
            cm[existing],
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.dataframe(
            cm,
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# VALIDITY TABLE
# =============================================================================

with st.expander(
    "Session Swing Validity"
):

    validity_display = (
        validity_session.copy()
    )

    validity_columns = [
        "video_id",
        "validity_status_v1_1",
        "evidence_quality_v1_1",
        "temporal_confidence",
        "baseline_eligible",
        "load_support_frames",
        "swing_support_frames",
        "load_to_peak_support_frames",
    ]

    validity_columns = [
        c
        for c in validity_columns
        if c in validity_display.columns
    ]

    if validity_columns:

        st.dataframe(
            validity_display[
                validity_columns
            ],
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.dataframe(
            validity_display,
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# RAW DATA
# =============================================================================

with st.expander(
    "Developer / Raw Analysis Data"
):

    st.subheader(
        "Selected Player Profile"
    )

    st.dataframe(
        pd.DataFrame(
            [target_profile]
        ),
        use_container_width=True,
        hide_index=True,
    )

    if feedback_df is not None:

        st.subheader(
            "Swing Feedback V1.2"
        )

        st.dataframe(
            feedback_df,
            use_container_width=True,
            hide_index=True,
        )

    if consistency_summary_df is not None:

        st.subheader(
            "Swing Consistency V1.1 Summary"
        )

        st.dataframe(
            consistency_summary_df,
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# METHODOLOGY
# =============================================================================

with st.expander(
    "How to interpret this dashboard"
):

    st.markdown(
        """
### Baseline Eligible

Only swings with sufficient structural and temporal support contribute
to the player's session baseline.

A swing excluded from the baseline may still be analyzed against that
baseline.

### Leave-One-Out

When the selected swing itself is baseline eligible, it is removed from
its own comparison baseline.

This prevents self-comparison bias.

### Robust Statistics

Swing Feedback V1.2 primarily uses:

- Session median
- Median Absolute Deviation (MAD)
- Robust Z-score

Traditional mean and standard deviation are preserved for traceability.

### Consistency

Consistency describes how repeatable the measured swing features are
across eligible swings in the session.

Lower variability means greater repeatability.

It does **not** mean that the swing mechanics are better.

### Measurement Confidence

Pose confidence describes the reliability of the pose-estimation
measurements.

It is separate from temporal confidence.

### Temporal Confidence

Temporal confidence describes whether detected LOAD, SWING and PEAK
phases contain enough temporal support for session-level comparison.

### Projected Rotation

Rotation measurements are derived from the 2D camera projection.

They are useful for within-session comparative analysis but should not
be interpreted as calibrated 3D biomechanical joint angles.

### Performance

BaseVisionAI V2.2 does not generate a GOOD/BAD swing classification or
a biomechanical performance score.
"""
    )


# =============================================================================
# FOOTER
# =============================================================================

st.markdown(
    """
<div class="bv-footer">

BaseVisionAI · Dashboard V2.2

Comparative 2D baseball swing analysis using computer vision,
pose estimation, temporal phase detection and robust session statistics.

</div>
""",
    unsafe_allow_html=True,
)