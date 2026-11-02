"""

BaseVisionAI Dashboard V2.6

===========================



Dashboard for:

- Swing video analysis

- Swing Validity V1.1

- Swing Feedback V1.2

- Swing Insights V1

- Swing Consistency V1.1

- Session Insights V1
- Session Report V1



Run from the BaseVisionAI project root:



    streamlit run dashboard/dashboard_v2_6.py

"""



from pathlib import Path



import numpy as np

import pandas as pd

import streamlit as st





# =============================================================================

# PAGE CONFIG

# =============================================================================



st.set_page_config(

    page_title="BaseVisionAI",

    page_icon="⚾",

    layout="wide",

    initial_sidebar_state="expanded",

)





# =============================================================================

# PROJECT PATHS

# =============================================================================



# dashboard/dashboard_v2_5.py

# parent          -> dashboard/

# parent.parent   -> BaseVisionAI/

PROJECT_ROOT = Path(__file__).resolve().parent.parent



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



SWING_INSIGHTS_ROOT = (

    RESULTS_DIR

    / "swing_insights_v1"

)



CONSISTENCY_ROOT = (

    RESULTS_DIR

    / "swing_consistency"

)



SESSION_INSIGHTS_ROOT = (

    RESULTS_DIR

    / "session_insights_v1"

)


SESSION_REPORT_ROOT = (

    RESULTS_DIR

    / "session_reports_v1"

)





# =============================================================================

# CSS

# =============================================================================



st.markdown(

    """

    <style>



    .block-container {

        max-width: 1450px;

        padding-top: 1.4rem;

        padding-bottom: 3rem;

    }



    [data-testid="stMetric"] {

        background-color: rgba(120,120,120,0.06);

        border: 1px solid rgba(120,120,120,0.16);

        border-radius: 12px;

        padding: 12px 14px;

    }



    .bva-card {

        border: 1px solid rgba(120,120,120,0.20);

        border-radius: 12px;

        padding: 16px 18px;

        margin-bottom: 12px;

        background-color: rgba(120,120,120,0.035);

    }



    .bva-card-title {

        font-size: 0.82rem;

        opacity: 0.70;

        margin-bottom: 5px;

    }



    .bva-card-value {

        font-size: 1.25rem;

        font-weight: 650;

    }



    .bva-muted {

        opacity: 0.70;

        font-size: 0.90rem;

    }



    .bva-small {

        opacity: 0.75;

        font-size: 0.82rem;

    }



    .bva-observation {

        border-left: 3px solid rgba(120,120,120,0.45);

        padding: 8px 14px;

        margin: 8px 0;

        background-color: rgba(120,120,120,0.035);

        border-radius: 0 8px 8px 0;

    }



    .bva-disclaimer {

        border: 1px solid rgba(120,120,120,0.18);

        border-radius: 10px;

        padding: 14px 16px;

        opacity: 0.78;

        font-size: 0.86rem;

    }



    </style>

    """,

    unsafe_allow_html=True,

)





# =============================================================================

# GENERAL UTILITIES

# =============================================================================



def safe_read_csv(path: Path):

    if not path.exists():

        return None



    try:

        return pd.read_csv(path)

    except Exception:

        return None





def clean_string(value, default="N/A"):

    if value is None:

        return default



    try:

        if pd.isna(value):

            return default

    except Exception:

        pass



    text = str(value).strip()



    if not text:

        return default



    return text





def as_float(value, default=np.nan):

    try:

        value = float(value)



        if np.isfinite(value):

            return value



    except Exception:

        pass



    return default





def as_int(value, default=0):

    try:

        return int(float(value))

    except Exception:

        return default





def as_bool(value):

    if isinstance(value, bool):

        return value



    if value is None:

        return False



    return str(value).strip().lower() in {

        "true",

        "1",

        "yes",

        "y",

        "t",

    }





def fmt(value, decimals=2, suffix=""):

    value = as_float(value)



    if not np.isfinite(value):

        return "N/A"



    return f"{value:.{decimals}f}{suffix}"





def first_existing_column(df, candidates):

    if df is None:

        return None



    for column in candidates:

        if column in df.columns:

            return column



    return None





def normalize_status(value):

    return clean_string(

        value,

        "",

    ).upper()





def metric_card(title, value, subtitle=None):

    subtitle_html = ""



    if subtitle:

        subtitle_html = (

            f'<div class="bva-small">{subtitle}</div>'

        )



    st.markdown(

        f"""

        <div class="bva-card">

            <div class="bva-card-title">{title}</div>

            <div class="bva-card-value">{value}</div>

            {subtitle_html}

        </div>

        """,

        unsafe_allow_html=True,

    )





# =============================================================================

# DATA LOADING

# =============================================================================



@st.cache_data

def load_player_database():

    return safe_read_csv(PLAYER_DATABASE)





@st.cache_data

def load_validity_database():

    return safe_read_csv(VALIDITY_DATABASE)





@st.cache_data

def load_csv_cached(path_string):

    return safe_read_csv(Path(path_string))





# =============================================================================

# PLAYER / SESSION DISCOVERY

# =============================================================================



player_db = load_player_database()



if player_db is None:

    st.error(

        "Player database not found:\n\n"

        f"`{PLAYER_DATABASE}`"

    )

    st.stop()





required_player_columns = {

    "video_id",

    "player_id",

    "session_id",

}



missing_player_columns = (

    required_player_columns

    - set(player_db.columns)

)



if missing_player_columns:

    st.error(

        "Player database is missing columns: "

        + ", ".join(

            sorted(missing_player_columns)

        )

    )

    st.stop()





player_db["player_id"] = (

    player_db["player_id"]

    .astype(str)

)



player_db["session_id"] = (

    player_db["session_id"]

    .astype(str)

)



player_db["video_id"] = (

    player_db["video_id"]

    .astype(str)

)





# =============================================================================

# HEADER

# =============================================================================



st.title("⚾ BaseVisionAI")



st.caption(

    "Comparative 2D baseball swing analysis • "

    "Swing-level feedback • Session repeatability"

)





# =============================================================================

# SIDEBAR

# =============================================================================



with st.sidebar:



    st.header("Analysis")



    players = sorted(

        player_db["player_id"]

        .dropna()

        .unique()

        .tolist()

    )



    selected_player = st.selectbox(

        "Player",

        players,

    )



    player_rows = player_db[

        player_db["player_id"]

        == selected_player

    ]



    sessions = sorted(

        player_rows["session_id"]

        .dropna()

        .unique()

        .tolist()

    )



    selected_session = st.selectbox(

        "Session",

        sessions,

    )



    session_rows = player_rows[

        player_rows["session_id"]

        == selected_session

    ].copy()



    videos = (

        session_rows["video_id"]

        .dropna()

        .unique()

        .tolist()

    )



    selected_video = st.selectbox(

        "Swing",

        videos,

    )



    st.divider()



    st.caption(

        "Dashboard V2.6"

    )



    st.caption(

        "Validity V1.1 • Feedback V1.2 • "

        "Consistency V1.1 • Session Report V1"

    )





# =============================================================================

# PATH RESOLUTION

# =============================================================================



feedback_dir = (

    FEEDBACK_ROOT

    / selected_player

    / selected_session

)



feedback_file = (

    feedback_dir

    / f"{selected_video}_swing_feedback_v1_2.csv"

)



feedback_summary_file = (

    feedback_dir

    / f"{selected_video}_swing_feedback_v1_2_summary.csv"

)



swing_insights_dir = (

    SWING_INSIGHTS_ROOT

    / selected_player

    / selected_session

)



swing_insights_file = (

    swing_insights_dir

    / f"{selected_video}_swing_insights_v1.csv"

)



swing_insights_summary_file = (

    swing_insights_dir

    / f"{selected_video}_swing_insights_v1_summary.csv"

)



consistency_dir = (

    CONSISTENCY_ROOT

    / selected_player

    / f"{selected_session}_v1_1"

)



consistency_metrics_file = (

    consistency_dir

    / "swing_consistency_v1_1_metrics.csv"

)



consistency_summary_file = (

    consistency_dir

    / "swing_consistency_v1_1_summary.csv"

)



consistency_swings_file = (

    consistency_dir

    / "swing_consistency_v1_1_swings.csv"

)



session_insights_dir = (

    SESSION_INSIGHTS_ROOT

    / selected_player

    / selected_session

)



session_insights_file = (

    session_insights_dir

    / "session_insights_v1.csv"

)



session_insights_summary_file = (

    session_insights_dir

    / "session_insights_v1_summary.csv"

)


session_report_dir = (

    SESSION_REPORT_ROOT

    / selected_player

    / selected_session

)


session_report_summary_file = (

    session_report_dir

    / "session_report_v1_summary.csv"

)


session_report_swings_file = (

    session_report_dir

    / "session_report_v1_swings.csv"

)


session_report_observations_file = (

    session_report_dir

    / "session_report_v1_observations.csv"

)





# =============================================================================

# LOAD SELECTED DATA

# =============================================================================



feedback = load_csv_cached(

    str(feedback_file)

)



feedback_summary = load_csv_cached(

    str(feedback_summary_file)

)



swing_insights = load_csv_cached(

    str(swing_insights_file)

)



swing_insights_summary = load_csv_cached(

    str(swing_insights_summary_file)

)



consistency_metrics = load_csv_cached(

    str(consistency_metrics_file)

)



consistency_summary = load_csv_cached(

    str(consistency_summary_file)

)



consistency_swings = load_csv_cached(

    str(consistency_swings_file)

)



session_insights = load_csv_cached(

    str(session_insights_file)

)



session_insights_summary = load_csv_cached(

    str(session_insights_summary_file)

)


session_report_summary = load_csv_cached(

    str(session_report_summary_file)

)


session_report_swings = load_csv_cached(

    str(session_report_swings_file)

)


session_report_observations = load_csv_cached(

    str(session_report_observations_file)

)



validity_db = load_validity_database()





# =============================================================================

# VALIDITY LOOKUP

# =============================================================================



selected_validity = None



if validity_db is not None:



    validity_required = {

        "video_id",

        "player_id",

        "session_id",

    }



    if validity_required.issubset(

        validity_db.columns

    ):



        validity_match = validity_db[

            (

                validity_db["video_id"]

                .astype(str)

                == selected_video

            )

            &

            (

                validity_db["player_id"]

                .astype(str)

                == selected_player

            )

            &

            (

                validity_db["session_id"]

                .astype(str)

                == selected_session

            )

        ]



        if not validity_match.empty:

            selected_validity = (

                validity_match.iloc[0]

            )





# =============================================================================

# VIDEO RESOLUTION

# =============================================================================



def find_analysis_video(video_id):



    preferred = (

        FULL_ANALYSIS_DIR

        / f"{video_id}_full_analysis_v3_web.mp4"

    )



    if preferred.exists():

        return preferred



    fallback = (

        FULL_ANALYSIS_DIR

        / f"{video_id}_full_analysis_v3.mp4"

    )



    if fallback.exists():

        return fallback



    return None





video_path = find_analysis_video(

    selected_video

)





# =============================================================================

# MAIN TABS

# =============================================================================



tab_swing, tab_session, tab_data = st.tabs(

    [

        "Swing Analysis",

        "Session Analysis",

        "Data & Traceability",

    ]

)





# =============================================================================

# TAB 1 — SWING ANALYSIS

# =============================================================================



with tab_swing:



    st.subheader(

        f"Swing Analysis — {selected_video}"

    )



    # =========================================================================

    # SWING STATUS

    # =========================================================================



    if selected_validity is not None:



        final_status_column = (

            "validity_status_v1_1"

            if "validity_status_v1_1"

            in selected_validity.index

            else "validity_status"

        )



        final_status = normalize_status(

            selected_validity.get(

                final_status_column,

                "",

            )

        )



        baseline_eligible = as_bool(

            selected_validity.get(

                "baseline_eligible",

                False,

            )

        )



        temporal_confidence = clean_string(

            selected_validity.get(

                "temporal_confidence",

                "N/A",

            )

        )



    else:



        final_status = "N/A"

        baseline_eligible = False

        temporal_confidence = "N/A"



    col1, col2, col3 = st.columns(3)



    with col1:

        st.metric(

            "Validity",

            final_status,

        )



    with col2:

        st.metric(

            "Baseline eligible",

            "Yes"

            if baseline_eligible

            else "No",

        )



    with col3:

        st.metric(

            "Temporal confidence",

            temporal_confidence,

        )



    st.divider()



    # =========================================================================

    # VIDEO + SWING SUMMARY

    # =========================================================================



    video_col, info_col = st.columns(

        [1.7, 1]

    )



    with video_col:



        st.markdown(

            "### Full Analysis Video"

        )



        if video_path is not None:

            st.video(

                str(video_path)

            )

        else:

            st.info(

                "Full-analysis video was not found "

                f"for `{selected_video}`."

            )



    with info_col:



        st.markdown(

            "### Swing Comparison"

        )



        if (

            swing_insights_summary

            is not None

            and not swing_insights_summary.empty

        ):



            row = (

                swing_insights_summary

                .iloc[0]

            )



            metric_card(

                "Comparison level",

                clean_string(

                    row.get(

                        "comparison_level"

                    )

                ).replace(

                    "_",

                    " ",

                ),

            )



            metric_card(

                "Metrics outside typical range",

                as_int(

                    row.get(

                        "outside_typical_range"

                    )

                ),

            )



            metric_card(

                "Metric coverage",

                fmt(

                    row.get(

                        "metric_coverage_percent"

                    ),

                    1,

                    "%",

                ),

            )



            metric_card(

                "Measurement confidence",

                clean_string(

                    row.get(

                        "measurement_confidence"

                    )

                ),

            )



        elif (

            feedback_summary is not None

            and not feedback_summary.empty

        ):



            row = feedback_summary.iloc[0]



            metric_card(

                "Metrics outside typical range",

                as_int(

                    row.get(

                        "outside_typical_range"

                    )

                ),

            )



            metric_card(

                "Metric coverage",

                fmt(

                    row.get(

                        "metric_coverage_percent"

                    ),

                    1,

                    "%",

                ),

            )



            metric_card(

                "Measurement confidence",

                clean_string(

                    row.get(

                        "measurement_confidence"

                    )

                ),

            )



        else:



            st.info(

                "Swing Insights summary was not found."

            )



    # =========================================================================

    # SWING INSIGHTS

    # =========================================================================



    st.divider()



    st.markdown(

        "### Swing Insights"

    )



    if (

        swing_insights is not None

        and not swing_insights.empty

    ):



        category_column = (

            first_existing_column(

                swing_insights,

                [

                    "category",

                    "insight_type",

                    "type",

                ],

            )

        )



        observation_column = (

            first_existing_column(

                swing_insights,

                [

                    "observation",

                    "feedback",

                    "insight",

                    "text",

                ],

            )

        )



        if observation_column:



            if category_column:



                for category in (

                    swing_insights[

                        category_column

                    ]

                    .dropna()

                    .unique()

                ):



                    category_rows = (

                        swing_insights[

                            swing_insights[

                                category_column

                            ]

                            == category

                        ]

                    )



                    st.markdown(

                        f"**{str(category).replace('_', ' ').title()}**"

                    )



                    for _, row in (

                        category_rows.iterrows()

                    ):



                        observation = (

                            clean_string(

                                row.get(

                                    observation_column

                                )

                            )

                        )



                        st.markdown(

                            f"""

                            <div class="bva-observation">

                            {observation}

                            </div>

                            """,

                            unsafe_allow_html=True,

                        )



            else:



                for _, row in (

                    swing_insights.iterrows()

                ):



                    observation = (

                        clean_string(

                            row.get(

                                observation_column

                            )

                        )

                    )



                    st.markdown(

                        f"""

                        <div class="bva-observation">

                        {observation}

                        </div>

                        """,

                        unsafe_allow_html=True,

                    )



        else:



            st.dataframe(

                swing_insights,

                use_container_width=True,

                hide_index=True,

            )



    else:



        st.info(

            "Swing Insights V1 data is not available "

            "for this swing."

        )



    # =========================================================================

    # FEEDBACK TABLE

    # =========================================================================



    st.divider()



    st.markdown(

        "### Comparative Metrics"

    )



    if feedback is not None:



        display_columns = [

            column

            for column in [

                "group",

                "metric",

                "target_value",

                "baseline_median",

                "difference_percent",

                "robust_z_score",

                "classification",

                "unit",

            ]

            if column in feedback.columns

        ]



        feedback_display = (

            feedback[

                display_columns

            ].copy()

        )



        rename_map = {

            "group": "Group",

            "metric": "Metric",

            "target_value": "Target",

            "baseline_median": "Baseline Median",

            "difference_percent": "Difference %",

            "robust_z_score": "Robust Z",

            "classification": "Classification",

            "unit": "Unit",

        }



        feedback_display.rename(

            columns=rename_map,

            inplace=True,

        )



        numeric_display_columns = [

            "Target",

            "Baseline Median",

            "Difference %",

            "Robust Z",

        ]



        for column in numeric_display_columns:



            if column in feedback_display.columns:



                feedback_display[column] = (

                    pd.to_numeric(

                        feedback_display[column],

                        errors="coerce",

                    ).round(3)

                )



        st.dataframe(

            feedback_display,

            use_container_width=True,

            hide_index=True,

        )



    else:



        st.info(

            "Swing Feedback V1.2 was not found "

            "for this swing."

        )





# =============================================================================
# TAB 2 — SESSION ANALYSIS
# =============================================================================

with tab_session:

    st.subheader(
        f"Session Analysis — {selected_session}"
    )

    st.caption(
        f"Player: {selected_player} • Consolidated by Session Report V1"
    )

    if (
        session_report_summary is None
        or session_report_summary.empty
    ):
        st.warning(
            "Session Report V1 summary was not found:\n\n"
            f"`{session_report_summary_file}`"
        )
        st.info(
            "Generate Session Report V1 for this player/session before "
            "using the consolidated Session Analysis view."
        )
    else:
        report_row = session_report_summary.iloc[0]

        # =====================================================================
        # SESSION EVIDENCE
        # =====================================================================
        st.markdown("### Session Evidence")

        evidence_cols = st.columns(6)
        evidence_values = [
            ("Profiles", as_int(report_row.get("profiles_analyzed"))),
            ("Valid swings", as_int(report_row.get("valid_swing_profiles"))),
            ("Baseline eligible", as_int(report_row.get("baseline_eligible_swings"))),
            ("Excluded", as_int(report_row.get("excluded_profiles"))),
            ("Temporal low confidence", as_int(report_row.get("temporal_low_confidence_swings"))),
            ("Low confidence", as_int(report_row.get("low_confidence_swings"))),
        ]

        for column, (label, value) in zip(evidence_cols, evidence_values):
            with column:
                st.metric(label, value)

        st.divider()

        # =====================================================================
        # SESSION VARIABILITY
        # =====================================================================
        st.markdown("### Observed Session Variability")

        variability_cols = st.columns(4)
        variability_values = [
            ("Mean Robust CV", fmt(report_row.get("mean_robust_cv_percent"), 1, "%")),
            ("Median Robust CV", fmt(report_row.get("median_robust_cv_percent"), 1, "%")),
            ("Mean conventional CV", fmt(report_row.get("mean_cv_percent"), 1, "%")),
            ("Median conventional CV", fmt(report_row.get("median_cv_percent"), 1, "%")),
        ]

        for column, (label, value) in zip(variability_cols, variability_values):
            with column:
                st.metric(label, value)

        # =====================================================================
        # SESSION PATTERN
        # =====================================================================
        st.markdown("### Session Pattern")

        pattern_left, pattern_right = st.columns(2)

        with pattern_left:
            metric_card(
                "Lowest observed variability",
                clean_string(report_row.get("lowest_variability_metric")),
                (
                    f"{clean_string(report_row.get('lowest_variability_group'))}"
                    f" • Robust CV "
                    f"{fmt(report_row.get('lowest_variability_robust_cv_percent'), 1, '%')}"
                ),
            )

        with pattern_right:
            metric_card(
                "Highest observed variability",
                clean_string(report_row.get("highest_variability_metric")),
                (
                    f"{clean_string(report_row.get('highest_variability_group'))}"
                    f" • Robust CV "
                    f"{fmt(report_row.get('highest_variability_robust_cv_percent'), 1, '%')}"
                ),
            )

        representative_col, different_col = st.columns(2)

        with representative_col:
            metric_card(
                "Representative swing",
                clean_string(report_row.get("representative_swing")),
                "Robust distance: " + fmt(report_row.get("representative_distance"), 3),
            )

        with different_col:
            metric_card(
                "Most different eligible swing",
                clean_string(report_row.get("most_different_eligible_swing")),
                "Robust distance: " + fmt(report_row.get("most_different_distance"), 3),
            )

        # =====================================================================
        # CONSOLIDATED SWING OVERVIEW
        # =====================================================================
        st.divider()
        st.markdown("### Swing Overview")
        st.caption(
            "All analyzed swings remain visible. Non-eligible swings are shown "
            "for traceability but do not define the session baseline."
        )

        if session_report_swings is not None and not session_report_swings.empty:
            swing_columns = [
                column for column in [
                    "video_id",
                    "validity_status",
                    "baseline_eligible",
                    "comparison_level",
                    "outside_typical_range",
                    "metric_coverage_percent",
                    "measurement_confidence",
                    "is_representative_swing",
                    "is_most_different_eligible_swing",
                ]
                if column in session_report_swings.columns
            ]

            swing_table = session_report_swings[swing_columns].copy()

            if "metric_coverage_percent" in swing_table.columns:
                swing_table["metric_coverage_percent"] = pd.to_numeric(
                    swing_table["metric_coverage_percent"], errors="coerce"
                ).round(1)

            if "baseline_eligible" in swing_table.columns:
                swing_table["baseline_eligible"] = swing_table["baseline_eligible"].apply(
                    lambda value: "Yes" if as_bool(value) else "No"
                )

            if "is_representative_swing" in swing_table.columns:
                swing_table["is_representative_swing"] = swing_table["is_representative_swing"].apply(
                    lambda value: "Yes" if as_bool(value) else ""
                )

            if "is_most_different_eligible_swing" in swing_table.columns:
                swing_table["is_most_different_eligible_swing"] = swing_table["is_most_different_eligible_swing"].apply(
                    lambda value: "Yes" if as_bool(value) else ""
                )

            swing_table.rename(
                columns={
                    "video_id": "Swing",
                    "validity_status": "Validity",
                    "baseline_eligible": "Baseline Eligible",
                    "comparison_level": "Comparison",
                    "outside_typical_range": "Outside Range",
                    "metric_coverage_percent": "Coverage %",
                    "measurement_confidence": "Measurement Confidence",
                    "is_representative_swing": "Representative",
                    "is_most_different_eligible_swing": "Most Different",
                },
                inplace=True,
            )

            st.dataframe(
                swing_table,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Session Report V1 swing table is not available.")

        # =====================================================================
        # CONSISTENCY CHART
        # =====================================================================
        st.divider()
        st.markdown("### Robust Variability by Metric")

        if consistency_metrics is not None and not consistency_metrics.empty:
            robust_cv_column = first_existing_column(
                consistency_metrics, ["robust_cv_percent", "robust_cv"]
            )
            metric_column = first_existing_column(
                consistency_metrics, ["metric", "metric_name"]
            )
            group_column = first_existing_column(
                consistency_metrics, ["group", "metric_group"]
            )

            if robust_cv_column and metric_column:
                chart_df = consistency_metrics[[metric_column, robust_cv_column]].copy()
                chart_df[robust_cv_column] = pd.to_numeric(
                    chart_df[robust_cv_column], errors="coerce"
                )
                chart_df.dropna(subset=[robust_cv_column], inplace=True)
                chart_df.sort_values(robust_cv_column, ascending=True, inplace=True)
                chart_df.rename(
                    columns={
                        metric_column: "Metric",
                        robust_cv_column: "Robust CV (%)",
                    },
                    inplace=True,
                )
                chart_df.set_index("Metric", inplace=True)
                st.bar_chart(
                    chart_df[["Robust CV (%)"]],
                    horizontal=True,
                    height=520,
                )

                table_columns = [metric_column, robust_cv_column]
                if group_column:
                    table_columns.insert(0, group_column)

                for column in [
                    "median",
                    "mad",
                    "robust_cv_classification",
                    "classification",
                ]:
                    if column in consistency_metrics.columns and column not in table_columns:
                        table_columns.append(column)

                with st.expander("View consistency metrics"):
                    st.dataframe(
                        consistency_metrics[table_columns],
                        use_container_width=True,
                        hide_index=True,
                    )
            else:
                st.warning(
                    "Could not identify metric and Robust CV columns in Consistency V1.1."
                )
        else:
            st.info("Swing Consistency V1.1 metrics were not found.")

        # =====================================================================
        # SESSION OBSERVATIONS
        # =====================================================================
        st.divider()
        st.markdown("### Session Observations")

        displayed_observations = 0
        summary_categories = {
            "SESSION_LOWEST_VARIABILITY",
            "SESSION_HIGHEST_VARIABILITY",
            "SESSION_REPRESENTATIVE_SWING",
            "SESSION_MOST_DIFFERENT_SWING",
            "SESSION_BASELINE_SUPPORT",
        }

        if (
            session_report_observations is not None
            and not session_report_observations.empty
            and "observation" in session_report_observations.columns
        ):
            observations_view = session_report_observations.copy()

            if "category" in observations_view.columns:
                preferred = observations_view[
                    observations_view["category"].astype(str).isin(summary_categories)
                ]
                if not preferred.empty:
                    observations_view = preferred

            for _, observation_row in observations_view.iterrows():
                observation = clean_string(observation_row.get("observation"), "")
                if not observation:
                    continue
                displayed_observations += 1
                st.markdown(
                    f"""
                    <div class="bva-observation">
                        <b>{displayed_observations}.</b> {observation}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if displayed_observations == 0:
            st.info("Session Report V1 observations are not available.")

        # =====================================================================
        # ELIGIBLE SWING COMPARISON
        # =====================================================================
        st.divider()
        st.markdown("### Eligible Swing Comparisons")

        comparison_col1, comparison_col2 = st.columns(2)
        with comparison_col1:
            st.metric(
                "Mean metric coverage",
                fmt(report_row.get("eligible_mean_metric_coverage_percent"), 1, "%"),
            )
        with comparison_col2:
            st.metric(
                "Mean outside-range metrics",
                fmt(report_row.get("eligible_mean_outside_typical_range"), 2),
            )

        # =====================================================================
        # INTERPRETATION NOTE
        # =====================================================================
        st.markdown(
            """
            <div class="bva-disclaimer">
            <b>Interpretation note</b><br><br>
            Session Report V1 consolidates previously generated BaseVisionAI
            analytics and does not recompute validity, baselines, consistency,
            or swing feedback.<br><br>
            Session consistency describes observed repeatability across
            baseline-eligible swings. Lower variability means greater observed
            similarity within the recorded session and does not imply better
            baseball mechanics.<br><br>
            The representative swing is the eligible swing closest to the
            robust session median profile; it is not the "best" swing. The most
            different eligible swing is not the "worst" swing.<br><br>
            TEMPORAL_LOW_CONFIDENCE and LOW_CONFIDENCE_SWING profiles may remain
            visible for traceability but are excluded from baseline estimation.
            Motion and projected-angle measurements remain comparative 2D
            camera-dependent measurements. No calibrated 3D biomechanics or
            performance score is inferred.
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================

# TAB 3 — DATA & TRACEABILITY

# =============================================================================



with tab_data:



    st.subheader(

        "Data & Traceability"

    )



    st.markdown(

        "### Active Analysis"

    )



    trace_col1, trace_col2, trace_col3 = (

        st.columns(3)

    )



    with trace_col1:

        st.metric(

            "Player",

            selected_player,

        )



    with trace_col2:

        st.metric(

            "Session",

            selected_session,

        )



    with trace_col3:

        st.metric(

            "Swing",

            selected_video,

        )



    st.divider()



    st.markdown(

        "### Pipeline"

    )



    st.code(

        """

Player Swing Database V1

        ↓

Swing Validity V1.1

        ↓

┌─────────────────────────────┐

│ Baseline-eligible swings    │

└─────────────────────────────┘

        ↓

Swing Consistency V1.1

        ↓

Swing Feedback V1.2

        ↓

Swing Insights V1

        ↓

Session Insights V1

        ↓

Session Report V1

        ↓

Dashboard V2.6

        """,

        language="text",

    )



    st.divider()



    st.markdown(

        "### Data Sources"

    )



    paths = {

        "Player database":

            PLAYER_DATABASE,



        "Validity V1.1":

            VALIDITY_DATABASE,



        "Feedback V1.2":

            feedback_file,



        "Swing Insights V1":

            swing_insights_file,



        "Consistency V1.1":

            consistency_metrics_file,



        "Session Insights V1":

            session_insights_summary_file,

        "Session Report V1 summary":
            session_report_summary_file,

        "Session Report V1 swings":
            session_report_swings_file,

        "Session Report V1 observations":
            session_report_observations_file,



        "Full analysis video":

            video_path,

    }



    path_rows = []



    for name, path in paths.items():



        if path is None:



            exists = False

            path_text = "Not found"



        else:



            exists = Path(path).exists()

            path_text = str(path)



        path_rows.append(

            {

                "Source": name,

                "Available": exists,

                "Path": path_text,

            }

        )



    st.dataframe(

        pd.DataFrame(path_rows),

        use_container_width=True,

        hide_index=True,

    )



    # =========================================================================

    # RAW DATA

    # =========================================================================



    st.divider()



    st.markdown(

        "### Raw Analysis Data"

    )



    with st.expander(

        "Player profile"

    ):



        selected_profile = (

            session_rows[

                session_rows[

                    "video_id"

                ]

                == selected_video

            ]

        )



        st.dataframe(

            selected_profile,

            use_container_width=True,

            hide_index=True,

        )



    with st.expander(

        "Swing Feedback V1.2"

    ):



        if feedback is not None:



            st.dataframe(

                feedback,

                use_container_width=True,

                hide_index=True,

            )



        else:



            st.info(

                "Feedback data unavailable."

            )



    with st.expander(

        "Swing Insights V1"

    ):



        if swing_insights is not None:



            st.dataframe(

                swing_insights,

                use_container_width=True,

                hide_index=True,

            )



        else:



            st.info(

                "Swing Insights data unavailable."

            )



    with st.expander(

        "Session Insights V1"

    ):



        if session_insights is not None:



            st.dataframe(

                session_insights,

                use_container_width=True,

                hide_index=True,

            )



        else:



            st.info(

                "Session Insights data unavailable."

            )



    with st.expander(
        "Session Report V1 — Summary"
    ):
        if session_report_summary is not None:
            st.dataframe(
                session_report_summary,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Session Report summary unavailable.")

    with st.expander(
        "Session Report V1 — Swings"
    ):
        if session_report_swings is not None:
            st.dataframe(
                session_report_swings,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Session Report swing data unavailable.")

    with st.expander(
        "Session Report V1 — Observations"
    ):
        if session_report_observations is not None:
            st.dataframe(
                session_report_observations,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Session Report observations unavailable.")

    with st.expander(

        "Consistency V1.1"

    ):



        if consistency_metrics is not None:



            st.dataframe(

                consistency_metrics,

                use_container_width=True,

                hide_index=True,

            )



        else:



            st.info(

                "Consistency data unavailable."

            )





# =============================================================================

# FOOTER

# =============================================================================



st.divider()



st.caption(

    "BaseVisionAI Dashboard V2.6 • "

    "Comparative 2D swing and session analysis"

)