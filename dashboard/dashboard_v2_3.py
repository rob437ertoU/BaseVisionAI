# =============================================================================
# BaseVisionAI - Dashboard V2.3
# =============================================================================
# Main additions over V2.2:
#   - Session Consistency V1.1
#   - Swing Feedback V1.2
#   - Selected Swing vs Session Baseline
#   - Measurement Reliability
#   - Technical Details
#
# Project structure:
#
# BaseVisionAI/
# ├── dashboard/
# │   └── dashboard_v2_3.py
# ├── results/
# ├── src/
# └── ...
# =============================================================================

from pathlib import Path
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="BaseVisionAI - Swing Analytics",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# PROJECT PATHS
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
# CSS
# =============================================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1500px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        min-width: 260px;
        max-width: 260px;
    }

    .bv-header {
        border: 1px solid #30343b;
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 26px;
        background: rgba(255,255,255,0.01);
    }

    .bv-title {
        font-size: 28px;
        font-weight: 800;
        margin: 0;
    }

    .bv-subtitle {
        color: #a8adb7;
        font-size: 14px;
        margin-top: 5px;
    }

    .section-label {
        font-size: 14px;
        font-weight: 700;
        margin-top: 6px;
        margin-bottom: 18px;
    }

    .status-pill {
        display: inline-block;
        padding: 5px 11px;
        margin: 0 6px 8px 0;
        border-radius: 999px;
        border: 1px solid #147a45;
        background: rgba(16, 185, 129, 0.10);
        color: #f5f5f5;
        font-size: 12px;
        font-weight: 700;
    }

    .status-pill-warning {
        display: inline-block;
        padding: 5px 11px;
        margin: 0 6px 8px 0;
        border-radius: 999px;
        border: 1px solid #9a6b18;
        background: rgba(245, 158, 11, 0.10);
        color: #f5f5f5;
        font-size: 12px;
        font-weight: 700;
    }

    .status-pill-danger {
        display: inline-block;
        padding: 5px 11px;
        margin: 0 6px 8px 0;
        border-radius: 999px;
        border: 1px solid #9c3131;
        background: rgba(239, 68, 68, 0.10);
        color: #f5f5f5;
        font-size: 12px;
        font-weight: 700;
    }

    .feedback-card {
        border: 1px solid #30343b;
        border-radius: 12px;
        padding: 15px 17px;
        min-height: 112px;
        margin-bottom: 10px;
        background: rgba(255,255,255,0.012);
    }

    .feedback-card-title {
        font-size: 12px;
        color: #a8adb7;
        text-transform: uppercase;
        letter-spacing: .04em;
        font-weight: 700;
        margin-bottom: 7px;
    }

    .feedback-card-value {
        font-size: 19px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .feedback-card-text {
        color: #c4c7ce;
        font-size: 13px;
        line-height: 1.4;
    }

    .note-box {
        border-left: 3px solid #5d6572;
        background: rgba(255,255,255,0.025);
        padding: 10px 14px;
        color: #b9bdc6;
        font-size: 13px;
        margin: 8px 0 15px 0;
    }

    div[data-testid="stMetric"] {
        background: transparent;
    }

    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
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
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_text(value, default="N/A"):
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    value = str(value).strip()

    if not value:
        return default

    return value


def safe_bool(value):
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col

    return None


def get_value(row, candidates, default=np.nan):
    if row is None:
        return default

    for col in candidates:
        if col in row.index:
            value = row[col]

            try:
                if pd.isna(value):
                    continue
            except Exception:
                pass

            return value

    return default


def format_number(value, digits=3):
    value = safe_float(value)

    if not np.isfinite(value):
        return "N/A"

    return f"{value:.{digits}f}"


def format_percent(value, digits=1):
    value = safe_float(value)

    if not np.isfinite(value):
        return "N/A"

    return f"{value:.{digits}f}%"


def normalize_classification(value):
    return safe_text(value, "UNKNOWN").upper().replace(" ", "_")


def classification_family(value):
    value = normalize_classification(value)

    if value == "CONSISTENT":
        return "CONSISTENT"

    if value in {
        "SLIGHTLY_LOW",
        "SLIGHTLY_HIGH",
        "SLIGHTLY_DIFFERENT",
    }:
        return "SLIGHT"

    if value in {
        "LOW",
        "HIGH",
        "OUTSIDE_RANGE",
        "OUTSIDE_TYPICAL_RANGE",
    }:
        return "OUTSIDE"

    return "OTHER"


def status_pill(text, kind="good"):
    css_class = "status-pill"

    if kind == "warning":
        css_class = "status-pill-warning"
    elif kind == "danger":
        css_class = "status-pill-danger"

    return (
        f'<span class="{css_class}">'
        f'{safe_text(text)}'
        f'</span>'
    )


def feedback_card(title, value, text):
    st.markdown(
        f"""
        <div class="feedback-card">
            <div class="feedback-card-title">{title}</div>
            <div class="feedback-card-value">{value}</div>
            <div class="feedback-card-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_csv(path_string):
    path = Path(path_string)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def locate_video(video_id):
    candidates = [
        FULL_ANALYSIS_DIR / f"{video_id}_full_analysis_v3_web.mp4",
        FULL_ANALYSIS_DIR / f"{video_id}_full_analysis_v3.mp4",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def locate_feedback_files(player_id, session_id, video_id):
    folder = (
        FEEDBACK_ROOT
        / str(player_id)
        / str(session_id)
    )

    detail = (
        folder
        / f"{video_id}_swing_feedback_v1_2.csv"
    )

    summary = (
        folder
        / f"{video_id}_swing_feedback_v1_2_summary.csv"
    )

    return detail, summary


def locate_consistency_folder(player_id, session_id):
    candidates = [
        CONSISTENCY_ROOT
        / str(player_id)
        / f"{session_id}_v1_1",

        CONSISTENCY_ROOT
        / str(player_id)
        / str(session_id),
    ]

    for folder in candidates:
        if folder.exists():
            return folder

    return candidates[0]


def get_validity_row(validity_df, video_id, player_id, session_id):
    if validity_df.empty:
        return None

    mask = validity_df["video_id"].astype(str) == str(video_id)

    if "player_id" in validity_df.columns:
        mask &= (
            validity_df["player_id"].astype(str)
            == str(player_id)
        )

    if "session_id" in validity_df.columns:
        mask &= (
            validity_df["session_id"].astype(str)
            == str(session_id)
        )

    rows = validity_df[mask]

    if rows.empty:
        return None

    return rows.iloc[0]


def get_profile_row(profile_df, video_id):
    rows = profile_df[
        profile_df["video_id"].astype(str)
        == str(video_id)
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


def get_feedback_summary_value(
    summary_row,
    candidates,
    default=np.nan,
):
    return get_value(
        summary_row,
        candidates,
        default,
    )


# =============================================================================
# LOAD CORE DATABASES
# =============================================================================

if not PLAYER_DATABASE.exists():
    st.error(
        "Player database not found:\n\n"
        f"{PLAYER_DATABASE}"
    )
    st.stop()

if not VALIDITY_DATABASE.exists():
    st.error(
        "Swing validity V1.1 database not found:\n\n"
        f"{VALIDITY_DATABASE}"
    )
    st.stop()


profiles_df = load_csv(str(PLAYER_DATABASE))
validity_df = load_csv(str(VALIDITY_DATABASE))


if profiles_df.empty:
    st.error("Player database is empty or could not be read.")
    st.stop()

if validity_df.empty:
    st.error("Validity database is empty or could not be read.")
    st.stop()


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    """
    <div class="bv-header">
        <div class="bv-title">⚾ BaseVisionAI</div>
        <div class="bv-subtitle">
            AI-Powered Baseball Swing Analysis · Dashboard V2.3
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("Analysis Filters")


players = sorted(
    profiles_df["player_id"]
    .dropna()
    .astype(str)
    .unique()
)

if not players:
    st.error("No players found in player database.")
    st.stop()


selected_player = st.sidebar.selectbox(
    "Player",
    players,
)


player_df = profiles_df[
    profiles_df["player_id"].astype(str)
    == str(selected_player)
].copy()


sessions = sorted(
    player_df["session_id"]
    .dropna()
    .astype(str)
    .unique()
)


if not sessions:
    st.error("No sessions found for selected player.")
    st.stop()


selected_session = st.sidebar.selectbox(
    "Session",
    sessions,
)


session_df = player_df[
    player_df["session_id"].astype(str)
    == str(selected_session)
].copy()


video_ids = (
    session_df["video_id"]
    .dropna()
    .astype(str)
    .tolist()
)


if not video_ids:
    st.error("No swings found for this session.")
    st.stop()


selected_video = st.sidebar.selectbox(
    "Swing",
    video_ids,
)


st.sidebar.divider()

st.sidebar.caption(
    "BaseVisionAI V2.3\n\n"
    "Session-relative swing analysis"
)


# =============================================================================
# SELECTED DATA
# =============================================================================

selected_profile = get_profile_row(
    session_df,
    selected_video,
)

selected_validity = get_validity_row(
    validity_df,
    selected_video,
    selected_player,
    selected_session,
)


feedback_detail_path, feedback_summary_path = (
    locate_feedback_files(
        selected_player,
        selected_session,
        selected_video,
    )
)

feedback_df = load_csv(
    str(feedback_detail_path)
)

feedback_summary_df = load_csv(
    str(feedback_summary_path)
)

feedback_summary_row = (
    feedback_summary_df.iloc[0]
    if not feedback_summary_df.empty
    else None
)


consistency_folder = locate_consistency_folder(
    selected_player,
    selected_session,
)

consistency_metrics_path = (
    consistency_folder
    / "swing_consistency_v1_1_metrics.csv"
)

consistency_swings_path = (
    consistency_folder
    / "swing_consistency_v1_1_swings.csv"
)

consistency_groups_path = (
    consistency_folder
    / "swing_consistency_v1_1_groups.csv"
)

consistency_summary_path = (
    consistency_folder
    / "swing_consistency_v1_1_summary.csv"
)


consistency_metrics_df = load_csv(
    str(consistency_metrics_path)
)

consistency_swings_df = load_csv(
    str(consistency_swings_path)
)

consistency_groups_df = load_csv(
    str(consistency_groups_path)
)

consistency_summary_df = load_csv(
    str(consistency_summary_path)
)

consistency_summary_row = (
    consistency_summary_df.iloc[0]
    if not consistency_summary_df.empty
    else None
)


# =============================================================================
# CURRENT ANALYSIS
# =============================================================================

st.markdown(
    '<div class="section-label">Current Analysis</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.caption("Player")
    st.markdown(f"### {selected_player}")

with c2:
    st.caption("Session")
    st.markdown(f"### {selected_session}")

with c3:
    st.caption("Selected Swing")
    st.markdown(f"### {selected_video}")

with c4:
    st.caption("Session Swings")
    st.markdown(f"### {len(session_df)}")


# =============================================================================
# SWING ANALYSIS
# =============================================================================

st.markdown("### Swing Analysis")

video_col, status_col = st.columns(
    [1.55, 0.85],
    gap="large",
)


# -----------------------------------------------------------------------------
# Video
# -----------------------------------------------------------------------------

with video_col:

    st.subheader("Full Analysis Video")

    video_path = locate_video(selected_video)

    if video_path is not None:
        st.video(str(video_path))
    else:
        st.info(
            f"Full Analysis V3 video not found for "
            f"{selected_video}."
        )


# -----------------------------------------------------------------------------
# Status
# -----------------------------------------------------------------------------

with status_col:

    st.subheader("Swing Status")

    if selected_validity is not None:

        validity_status = safe_text(
            get_value(
                selected_validity,
                [
                    "validity_status",
                    "validity",
                ],
                "UNKNOWN",
            )
        )

        temporal_confidence = safe_text(
            get_value(
                selected_validity,
                ["temporal_confidence"],
                "UNKNOWN",
            )
        )

        evidence_quality = safe_text(
            get_value(
                selected_validity,
                ["evidence_quality"],
                "UNKNOWN",
            )
        )

        baseline_eligible = safe_bool(
            get_value(
                selected_validity,
                ["baseline_eligible"],
                False,
            )
        )

        if validity_status == "VALID_SWING":
            validity_kind = "good"
        else:
            validity_kind = "warning"

        temporal_kind = (
            "good"
            if temporal_confidence == "SUPPORTED"
            else "warning"
        )

        evidence_kind = (
            "good"
            if evidence_quality == "SUFFICIENT"
            else "warning"
        )

        pose_quality = safe_float(
            get_value(
                selected_validity,
                ["mean_pose_quality"],
                np.nan,
            )
        )

        if np.isfinite(pose_quality):

            if pose_quality >= 0.90:
                pose_label = "HIGH"
                pose_kind = "good"

            elif pose_quality >= 0.75:
                pose_label = "MODERATE"
                pose_kind = "warning"

            else:
                pose_label = "LOW"
                pose_kind = "danger"

        else:
            pose_label = "UNKNOWN"
            pose_kind = "warning"

        st.markdown(
            status_pill(
                validity_status.replace("_", " "),
                validity_kind,
            )
            + "<br>"
            + status_pill(
                f"Temporal: {temporal_confidence}",
                temporal_kind,
            )
            + "<br>"
            + status_pill(
                f"Evidence: {evidence_quality}",
                evidence_kind,
            )
            + "<br>"
            + status_pill(
                f"Pose: {pose_label}",
                pose_kind,
            ),
            unsafe_allow_html=True,
        )

        st.write("")

        sc1, sc2 = st.columns(2)

        with sc1:
            st.caption("Pose Quality")
            st.markdown(
                f"### {format_number(pose_quality)}"
            )

        with sc2:
            st.caption("Baseline")
            st.markdown(
                "### Eligible"
                if baseline_eligible
                else "### Excluded"
            )

        metric_coverage = get_feedback_summary_value(
            feedback_summary_row,
            [
                "metric_coverage_percent",
                "metric_coverage",
            ],
            np.nan,
        )

        st.caption("Metric Coverage")

        if np.isfinite(safe_float(metric_coverage)):
            st.markdown(
                f"### {format_percent(metric_coverage)}"
            )
        else:
            st.markdown("### N/A")

    else:
        st.warning(
            "No validity record found for selected swing."
        )


# =============================================================================
# SESSION OVERVIEW
# =============================================================================

st.markdown("### Session Overview")

session_validity = validity_df[
    (
        validity_df["player_id"].astype(str)
        == str(selected_player)
    )
    &
    (
        validity_df["session_id"].astype(str)
        == str(selected_session)
    )
].copy()


eligible_count = 0

if "baseline_eligible" in session_validity.columns:
    eligible_count = int(
        session_validity["baseline_eligible"]
        .apply(safe_bool)
        .sum()
    )


excluded_count = max(
    len(session_validity) - eligible_count,
    0,
)


metric_count = (
    len(feedback_df)
    if not feedback_df.empty
    else 0
)


o1, o2, o3, o4 = st.columns(4)

o1.metric(
    "Swings",
    len(session_df),
)

o2.metric(
    "Baseline Eligible",
    eligible_count,
)

o3.metric(
    "Excluded",
    excluded_count,
)

o4.metric(
    "Metrics",
    metric_count,
)


# =============================================================================
# SESSION CONSISTENCY
# =============================================================================

st.divider()

st.header("Session Consistency")

st.caption(
    "Repeatability across baseline-eligible swings in the "
    "selected player session."
)


if (
    consistency_summary_row is None
    or consistency_metrics_df.empty
):

    st.info(
        "Swing Consistency V1.1 results were not found for "
        f"{selected_player} / {selected_session}.\n\n"
        f"Expected folder:\n{consistency_folder}"
    )

else:

    mean_cv = safe_float(
        get_value(
            consistency_summary_row,
            [
                "mean_metric_cv_percent",
                "mean_metric_cv",
                "mean_cv_percent",
                "mean_cv",
            ],
            np.nan,
        )
    )

    median_cv = safe_float(
        get_value(
            consistency_summary_row,
            [
                "median_metric_cv_percent",
                "median_metric_cv",
                "median_cv_percent",
                "median_cv",
            ],
            np.nan,
        )
    )

    repeatability = safe_float(
        get_value(
            consistency_summary_row,
            [
                "mean_repeatability_index",
                "repeatability_index",
                "session_repeatability_index",
            ],
            np.nan,
        )
    )

    valid_profiles = safe_int(
        get_value(
            consistency_summary_row,
            [
                "valid_swing_profiles",
                "baseline_eligible_profiles",
                "valid_profiles",
                "profiles_included",
            ],
            eligible_count,
        ),
        eligible_count,
    )

    cc1, cc2, cc3, cc4 = st.columns(4)

    cc1.metric(
        "Observed Repeatability Index",
        (
            f"{repeatability:.1f} / 100"
            if np.isfinite(repeatability)
            else "N/A"
        ),
    )

    cc2.metric(
        "Mean Metric CV",
        (
            f"{mean_cv:.1f}%"
            if np.isfinite(mean_cv)
            else "N/A"
        ),
    )

    cc3.metric(
        "Median Metric CV",
        (
            f"{median_cv:.1f}%"
            if np.isfinite(median_cv)
            else "N/A"
        ),
    )

    cc4.metric(
        "Baseline Swings",
        valid_profiles,
    )

    st.markdown(
        """
        <div class="note-box">
        The repeatability index describes similarity across
        baseline-eligible swings. It is <b>not</b> a baseball
        performance score.
        </div>
        """,
        unsafe_allow_html=True,
    )


    # -------------------------------------------------------------------------
    # Metric consistency chart
    # -------------------------------------------------------------------------

    repeat_col = first_existing_column(
        consistency_metrics_df,
        [
            "repeatability_index",
            "repeatability",
        ],
    )

    metric_name_col = first_existing_column(
        consistency_metrics_df,
        [
            "metric_label",
            "metric_name",
            "metric",
        ],
    )

    classification_col = first_existing_column(
        consistency_metrics_df,
        [
            "classification",
            "consistency_classification",
        ],
    )


    if (
        repeat_col is not None
        and metric_name_col is not None
    ):

        consistency_plot_df = (
            consistency_metrics_df[
                [
                    metric_name_col,
                    repeat_col,
                ]
            ]
            .copy()
        )

        consistency_plot_df[repeat_col] = (
            pd.to_numeric(
                consistency_plot_df[repeat_col],
                errors="coerce",
            )
        )

        consistency_plot_df = (
            consistency_plot_df
            .dropna(subset=[repeat_col])
            .sort_values(
                repeat_col,
                ascending=True,
            )
        )

        if not consistency_plot_df.empty:

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=consistency_plot_df[repeat_col],
                    y=consistency_plot_df[metric_name_col],
                    orientation="h",
                    text=[
                        f"{v:.1f}"
                        for v
                        in consistency_plot_df[repeat_col]
                    ],
                    textposition="outside",
                    hovertemplate=(
                        "%{y}<br>"
                        "Repeatability: %{x:.1f}/100"
                        "<extra></extra>"
                    ),
                )
            )

            fig.update_layout(
                title="Observed Metric Repeatability",
                xaxis_title="Repeatability Index",
                yaxis_title="",
                xaxis=dict(
                    range=[0, 105],
                ),
                height=max(
                    420,
                    34 * len(consistency_plot_df),
                ),
                margin=dict(
                    l=20,
                    r=50,
                    t=60,
                    b=30,
                ),
                showlegend=False,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


        # ---------------------------------------------------------------------
        # Most / least repeatable
        # ---------------------------------------------------------------------

        if len(consistency_plot_df) >= 1:

            best = consistency_plot_df.iloc[-1]
            worst = consistency_plot_df.iloc[0]

            rc1, rc2 = st.columns(2)

            with rc1:
                feedback_card(
                    "Most Repeatable Metric",
                    safe_text(
                        best[metric_name_col]
                    ),
                    (
                        f"Observed repeatability: "
                        f"{safe_float(best[repeat_col]):.1f}/100"
                    ),
                )

            with rc2:
                feedback_card(
                    "Highest Observed Variability",
                    safe_text(
                        worst[metric_name_col]
                    ),
                    (
                        f"Observed repeatability: "
                        f"{safe_float(worst[repeat_col]):.1f}/100"
                    ),
                )


# =============================================================================
# SELECTED SWING VS SESSION
# =============================================================================

st.divider()

st.header("Selected Swing vs Session")

st.caption(
    f"{selected_video} compared with the player's "
    "eligible session baseline."
)


if feedback_df.empty:

    st.info(
        "Swing Feedback V1.2 was not found for this swing.\n\n"
        f"Expected file:\n{feedback_detail_path}"
    )

else:

    class_col = first_existing_column(
        feedback_df,
        [
            "classification",
            "robust_classification",
        ],
    )

    metric_col = first_existing_column(
        feedback_df,
        [
            "metric_label",
            "metric_name",
            "metric",
        ],
    )

    z_col = first_existing_column(
        feedback_df,
        [
            "robust_z",
            "robust_z_score",
            "z_score",
            "zscore",
        ],
    )


    # -------------------------------------------------------------------------
    # Classification counts
    # -------------------------------------------------------------------------

    consistent_count = 0
    slight_count = 0
    outside_count = 0

    if class_col is not None:

        families = feedback_df[class_col].apply(
            classification_family
        )

        consistent_count = int(
            (families == "CONSISTENT").sum()
        )

        slight_count = int(
            (families == "SLIGHT").sum()
        )

        outside_count = int(
            (families == "OUTSIDE").sum()
        )


    fc1, fc2, fc3 = st.columns(3)

    fc1.metric(
        "Consistent",
        consistent_count,
    )

    fc2.metric(
        "Slightly Different",
        slight_count,
    )

    fc3.metric(
        "Outside Typical Range",
        outside_count,
    )


    # -------------------------------------------------------------------------
    # Z-score chart
    # -------------------------------------------------------------------------

    if (
        metric_col is not None
        and z_col is not None
    ):

        deviation_df = feedback_df[
            [
                metric_col,
                z_col,
            ]
        ].copy()

        deviation_df[z_col] = pd.to_numeric(
            deviation_df[z_col],
            errors="coerce",
        )

        deviation_df = deviation_df.dropna(
            subset=[z_col]
        )

        if not deviation_df.empty:

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=deviation_df[metric_col],
                    y=deviation_df[z_col],
                    text=[
                        f"{v:+.2f}"
                        for v in deviation_df[z_col]
                    ],
                    textposition="outside",
                    hovertemplate=(
                        "%{x}<br>"
                        "Robust deviation: %{y:+.2f}"
                        "<extra></extra>"
                    ),
                )
            )

            fig.add_hline(
                y=1.0,
                line_dash="dash",
                annotation_text="+1",
            )

            fig.add_hline(
                y=-1.0,
                line_dash="dash",
                annotation_text="-1",
            )

            fig.add_hline(
                y=0.0,
                line_width=1,
            )

            fig.update_layout(
                title="Deviation from Session Baseline",
                xaxis_title="Metric",
                yaxis_title="Robust Z-score",
                height=480,
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=100,
                ),
            )

            fig.update_xaxes(
                tickangle=-35,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.caption(
                "Positive and negative values indicate position "
                "relative to the session baseline. They do not "
                "represent better or worse swing mechanics."
            )


# =============================================================================
# SWING FEEDBACK
# =============================================================================

st.divider()

st.header("Swing Feedback")

st.caption(
    "Session-relative observations from Swing Feedback V1.2."
)


if feedback_df.empty:

    st.info(
        "No detailed feedback is available for the "
        "selected swing."
    )

else:

    feedback_text_col = first_existing_column(
        feedback_df,
        [
            "feedback",
            "feedback_text",
        ],
    )

    metric_col = first_existing_column(
        feedback_df,
        [
            "metric_label",
            "metric_name",
            "metric",
        ],
    )

    class_col = first_existing_column(
        feedback_df,
        [
            "classification",
            "robust_classification",
        ],
    )


    if (
        metric_col is not None
        and class_col is not None
    ):

        tmp = feedback_df.copy()

        tmp["_family"] = tmp[class_col].apply(
            classification_family
        )

        consistent_df = tmp[
            tmp["_family"] == "CONSISTENT"
        ]

        slight_df = tmp[
            tmp["_family"] == "SLIGHT"
        ]

        outside_df = tmp[
            tmp["_family"] == "OUTSIDE"
        ]


        fb1, fb2, fb3 = st.columns(3)


        # ---------------------------------------------------------------------
        # Consistent
        # ---------------------------------------------------------------------

        with fb1:

            st.subheader("Consistent")

            if consistent_df.empty:
                st.caption(
                    "No metrics classified as consistent."
                )

            else:
                for _, row in consistent_df.iterrows():

                    text = (
                        safe_text(
                            row[feedback_text_col]
                        )
                        if feedback_text_col
                        else safe_text(row[metric_col])
                    )

                    feedback_card(
                        "Consistent",
                        safe_text(row[metric_col]),
                        text,
                    )


        # ---------------------------------------------------------------------
        # Slight
        # ---------------------------------------------------------------------

        with fb2:

            st.subheader("Slight Deviation")

            if slight_df.empty:
                st.caption(
                    "No metrics with slight deviation."
                )

            else:
                for _, row in slight_df.iterrows():

                    text = (
                        safe_text(
                            row[feedback_text_col]
                        )
                        if feedback_text_col
                        else safe_text(row[metric_col])
                    )

                    feedback_card(
                        normalize_classification(
                            row[class_col]
                        ).replace("_", " "),
                        safe_text(row[metric_col]),
                        text,
                    )


        # ---------------------------------------------------------------------
        # Outside
        # ---------------------------------------------------------------------

        with fb3:

            st.subheader("Outside Typical Range")

            if outside_df.empty:
                st.caption(
                    "No metrics outside the typical range."
                )

            else:
                for _, row in outside_df.iterrows():

                    text = (
                        safe_text(
                            row[feedback_text_col]
                        )
                        if feedback_text_col
                        else safe_text(row[metric_col])
                    )

                    feedback_card(
                        normalize_classification(
                            row[class_col]
                        ).replace("_", " "),
                        safe_text(row[metric_col]),
                        text,
                    )


# =============================================================================
# MEASUREMENT RELIABILITY
# =============================================================================

st.divider()

st.header("Measurement Reliability")

st.caption(
    "Reliability indicators describe analysis confidence, "
    "not baseball performance."
)


pose_quality = safe_float(
    get_value(
        selected_profile,
        ["mean_pose_quality"],
        np.nan,
    )
)


if np.isfinite(pose_quality):

    if pose_quality >= 0.90:
        pose_confidence = "HIGH"

    elif pose_quality >= 0.75:
        pose_confidence = "MODERATE"

    else:
        pose_confidence = "LOW"

else:
    pose_confidence = "UNKNOWN"


temporal_confidence = safe_text(
    get_value(
        selected_validity,
        ["temporal_confidence"],
        "UNKNOWN",
    )
)


evidence_quality = safe_text(
    get_value(
        selected_validity,
        ["evidence_quality"],
        "UNKNOWN",
    )
)


metric_coverage = safe_float(
    get_feedback_summary_value(
        feedback_summary_row,
        [
            "metric_coverage_percent",
            "metric_coverage",
        ],
        np.nan,
    )
)


mr1, mr2, mr3, mr4 = st.columns(4)

mr1.metric(
    "Pose Confidence",
    pose_confidence,
    (
        f"Pose quality {pose_quality:.3f}"
        if np.isfinite(pose_quality)
        else None
    ),
)

mr2.metric(
    "Temporal Support",
    temporal_confidence,
)

mr3.metric(
    "Evidence",
    evidence_quality,
)

mr4.metric(
    "Metric Coverage",
    (
        f"{metric_coverage:.1f}%"
        if np.isfinite(metric_coverage)
        else "N/A"
    ),
)


# =============================================================================
# TECHNICAL DETAILS
# =============================================================================

st.divider()

with st.expander(
    "Technical Details",
    expanded=False,
):

    st.markdown(
        """
        ### Interpretation

        BaseVisionAI compares the selected swing against
        baseline-eligible swings from the same player session.

        For baseline-eligible target swings, leave-one-out
        comparison should be used whenever enough other
        eligible swings are available.

        **Important:** classifications describe statistical
        position relative to the observed session baseline.
        They do not indicate whether a swing is mechanically
        correct, incorrect, better, or worse.

        Projected rotation measurements remain 2D
        camera-dependent measurements and must not be
        interpreted as calibrated 3D biomechanics.
        """
    )

    if not feedback_df.empty:

        st.subheader("Swing Feedback V1.2")

        preferred_columns = [
            "group",
            "metric_label",
            "metric_name",
            "metric",
            "target",
            "target_value",
            "session_baseline",
            "baseline",
            "baseline_sd",
            "difference_percent",
            "difference_pct",
            "robust_z",
            "robust_z_score",
            "z_score",
            "classification",
            "unit",
            "feedback",
        ]

        columns_to_show = []

        for col in preferred_columns:
            if (
                col in feedback_df.columns
                and col not in columns_to_show
            ):
                columns_to_show.append(col)

        if columns_to_show:
            st.dataframe(
                feedback_df[columns_to_show],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.dataframe(
                feedback_df,
                use_container_width=True,
                hide_index=True,
            )


    if not consistency_metrics_df.empty:

        st.subheader("Swing Consistency V1.1")

        st.dataframe(
            consistency_metrics_df,
            use_container_width=True,
            hide_index=True,
        )


    st.subheader("Selected Player Profile")

    if selected_profile is not None:

        profile_display = pd.DataFrame(
            [selected_profile]
        )

        st.dataframe(
            profile_display,
            use_container_width=True,
            hide_index=True,
        )


    st.subheader("Validity V1.1")

    if selected_validity is not None:

        validity_display = pd.DataFrame(
            [selected_validity]
        )

        st.dataframe(
            validity_display,
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "BaseVisionAI Dashboard V2.3 · "
    "Comparative 2D baseball swing analysis · "
    "Session-relative metrics are not calibrated "
    "biomechanical or performance scores."
)