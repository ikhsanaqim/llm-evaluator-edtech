"""
app.py — LLM Evaluator for Edtech Dashboard
Hugging Face Spaces deployment (Streamlit SDK)

Reads from data/responses/ and data/scores/ directories.
"""

import streamlit as st
import pandas as pd
import json
import glob
import os
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Evaluator — Edtech",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

.main { background-color: #0f1117; }

.metric-card {
    background: linear-gradient(135deg, #1a1d2e 0%, #16192a 100%);
    border: 1px solid #2a2d3e;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-bottom: 12px;
}

.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #7ee8fa;
}

.metric-label {
    font-size: 0.75rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}

.score-bar-container {
    background: #1a1d2e;
    border-radius: 6px;
    height: 8px;
    width: 100%;
    margin-top: 6px;
}

.score-bar {
    background: linear-gradient(90deg, #7ee8fa, #80ff72);
    border-radius: 6px;
    height: 8px;
}

.halluc-low { color: #80ff72; font-weight: 600; }
.halluc-medium { color: #ffd060; font-weight: 600; }
.halluc-high { color: #ff6b6b; font-weight: 600; }

.section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #7ee8fa;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    border-bottom: 1px solid #2a2d3e;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

.response-box {
    background: #1a1d2e;
    border: 1px solid #2a2d3e;
    border-radius: 8px;
    padding: 16px;
    font-size: 0.85rem;
    line-height: 1.6;
    color: #d1d5db;
    font-family: 'Sora', sans-serif;
    white-space: pre-wrap;
    max-height: 300px;
    overflow-y: auto;
}

.tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    margin-right: 4px;
}

.tag-sd { background: #1e3a5f; color: #7ee8fa; }
.tag-smp { background: #1e4620; color: #80ff72; }
.tag-sma { background: #4a1e1e; color: #ff9999; }
.tag-mtk { background: #2d1e4a; color: #c4b5fd; }
.tag-ipa { background: #1e3a2d; color: #6ee7b7; }
.tag-bin { background: #3a2d1e; color: #fbbf24; }

.stSelectbox label { color: #9ca3af !important; }
.stSlider label { color: #9ca3af !important; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_latest_responses():
    """Load the most recent pipeline response file."""
    files = glob.glob("data/responses/responses_*.json")
    if not files:
        return None
    latest = max(files, key=os.path.getmtime)
    with open(latest, encoding="utf-8") as f:
        return json.load(f), Path(latest).name


@st.cache_data
def load_scores_csv():
    """Load all scores CSVs and combine."""
    files = glob.glob("data/scores/scores_*.csv")
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True).drop_duplicates() if dfs else None


@st.cache_data
def load_judge_scores():
    """Load LLM-as-Judge scores if available."""
    files = glob.glob("data/scores/judged_*.json")
    if not files:
        return None
    latest = max(files, key=os.path.getmtime)
    with open(latest, encoding="utf-8") as f:
        return json.load(f)


def score_color(score):
    if score is None:
        return "#6b7280"
    if score >= 0.8:
        return "#80ff72"
    elif score >= 0.5:
        return "#ffd060"
    return "#ff6b6b"


# ── Load data ─────────────────────────────────────────────────────────────────
response_data, filename = load_latest_responses() or (None, None)
scores_df = load_scores_csv()
judge_data = load_judge_scores()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 LLM Evaluator")
    st.markdown("*Edtech Edition — Indonesia K-12*")
    st.divider()

    st.markdown("**Navigation**")
    page = st.radio(
        "",
        ["📊 Overview", "🔍 Question Explorer", "⚖️ LLM-as-Judge", "📈 Consistency"],
        label_visibility="collapsed"
    )

    st.divider()

    if scores_df is not None:
        models = scores_df["model_key"].unique().tolist()
        st.markdown("**Models**")
        for m in models:
            success = scores_df[scores_df["model_key"] == m]["error"].isna().sum()
            total = len(scores_df[scores_df["model_key"] == m])
            pct = int(success / total * 100) if total > 0 else 0
            color = "#80ff72" if pct > 80 else "#ffd060" if pct > 40 else "#ff6b6b"
            st.markdown(
                f'<span style="color:{color}">●</span> `{m}` {success}/{total}',
                unsafe_allow_html=True
            )

    st.divider()
    st.markdown(
        '<span style="color:#6b7280;font-size:0.75rem;">Built with LangChain · OpenRouter · LangSmith</span>',
        unsafe_allow_html=True
    )


# ── Page: Overview ────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.markdown("# LLM Evaluation Dashboard")
    st.markdown("*Benchmarking language models for Indonesian K-12 educational Q&A*")
    st.divider()

    if scores_df is None or response_data is None:
        st.warning("No data found. Run `python src/run_pipeline_v2.py` first.")
        st.stop()

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)

    total_q = len(response_data)
    total_models = scores_df["model_key"].nunique()
    valid_responses = scores_df["error"].isna().sum()
    total_responses = len(scores_df)
    avg_score = scores_df["composite_auto_score"].mean()

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_q}</div>
            <div class="metric-label">Questions</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_models}</div>
            <div class="metric-label">Models Tested</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{valid_responses}/{total_responses}</div>
            <div class="metric-label">Successful Responses</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        avg_str = f"{avg_score:.3f}" if pd.notna(avg_score) else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_str}</div>
            <div class="metric-label">Avg Auto Score</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Score breakdown by model
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-header">Average Score by Model</div>', unsafe_allow_html=True)
        model_scores = (
            scores_df[scores_df["error"].isna()]
            .groupby("model_key")[["composite_auto_score", "accuracy_score", "structure_score", "length_score"]]
            .mean()
            .round(3)
            .reset_index()
        )

        if not model_scores.empty:
            st.dataframe(
                model_scores.rename(columns={
                    "model_key": "Model",
                    "composite_auto_score": "Composite",
                    "accuracy_score": "Accuracy",
                    "structure_score": "Structure",
                    "length_score": "Length",
                }),
                use_container_width=True,
                hide_index=True,
            )

            # Bar chart
            st.bar_chart(
                model_scores.set_index("model_key")[["accuracy_score", "structure_score", "length_score"]],
                use_container_width=True,
                height=250,
            )

    with col_right:
        st.markdown('<div class="section-header">Score by Subject</div>', unsafe_allow_html=True)
        subject_scores = (
            scores_df[scores_df["error"].isna()]
            .groupby("subject")["composite_auto_score"]
            .mean()
            .round(3)
            .reset_index()
        )
        if not subject_scores.empty:
            st.dataframe(
                subject_scores.rename(columns={"subject": "Subject", "composite_auto_score": "Avg Score"}),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown('<div class="section-header" style="margin-top:24px">Score by Level</div>', unsafe_allow_html=True)
        level_scores = (
            scores_df[scores_df["error"].isna()]
            .groupby("level")["composite_auto_score"]
            .mean()
            .round(3)
            .reset_index()
        )
        if not level_scores.empty:
            st.dataframe(
                level_scores.rename(columns={"level": "Level", "composite_auto_score": "Avg Score"}),
                use_container_width=True,
                hide_index=True,
            )

    # Latency comparison
    st.divider()
    st.markdown('<div class="section-header">Latency by Model (ms)</div>', unsafe_allow_html=True)
    latency_df = (
        scores_df[scores_df["error"].isna() & scores_df["latency_ms"].notna()]
        .groupby("model_key")["latency_ms"]
        .agg(["mean", "min", "max"])
        .round(0)
        .astype(int)
        .reset_index()
        .rename(columns={"model_key": "Model", "mean": "Avg (ms)", "min": "Min (ms)", "max": "Max (ms)"})
    )
    if not latency_df.empty:
        st.dataframe(latency_df, use_container_width=True, hide_index=True)


# ── Page: Question Explorer ───────────────────────────────────────────────────
elif page == "🔍 Question Explorer":
    st.markdown("# Question Explorer")
    st.markdown("*Browse individual questions and compare model responses side by side*")
    st.divider()

    if response_data is None:
        st.warning("No response data found.")
        st.stop()

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        subjects = list(set(q["subject"] for q in response_data))
        subject_filter = st.selectbox("Subject", ["All"] + sorted(subjects))
    with col2:
        levels = list(set(q["level"] for q in response_data))
        level_filter = st.selectbox("Level", ["All"] + sorted(levels))
    with col3:
        q_ids = [q["question_id"] for q in response_data
                 if (subject_filter == "All" or q["subject"] == subject_filter)
                 and (level_filter == "All" or q["level"] == level_filter)]
        selected_id = st.selectbox("Question ID", q_ids)

    # Find selected question
    selected = next((q for q in response_data if q["question_id"] == selected_id), None)
    if not selected:
        st.warning("Question not found.")
        st.stop()

    st.divider()

    # Question info
    level_class = f"tag-{selected['level'].lower()}"
    subj_key = {"Matematika": "mtk", "IPA": "ipa", "Bahasa Indonesia": "bin"}.get(selected["subject"], "mtk")
    subj_class = f"tag-{subj_key}"

    st.markdown(
        f'<span class="tag {level_class}">{selected["level"]}</span>'
        f'<span class="tag {subj_class}">{selected["subject"]}</span>'
        f'<span class="tag" style="background:#1a1d2e;color:#9ca3af">{selected["grade"]}</span>',
        unsafe_allow_html=True
    )
    st.markdown(f"### {selected['question']}")

    with st.expander("Ground Truth"):
        st.markdown(selected["ground_truth"])

    st.divider()

    # Model responses
    st.markdown('<div class="section-header">Model Responses</div>', unsafe_allow_html=True)

    valid_responses = [r for r in selected["model_responses"] if r.get("response")]
    failed_responses = [r for r in selected["model_responses"] if not r.get("response")]

    if not valid_responses:
        st.error("No valid responses for this question.")
    else:
        cols = st.columns(len(valid_responses))
        for i, resp in enumerate(valid_responses):
            with cols[i]:
                ae = resp.get("auto_eval", {})
                score = ae.get("composite_auto_score")
                color = score_color(score)
                score_str = f"{score:.3f}" if score is not None else "N/A"

                st.markdown(
                    f'<div style="font-family:JetBrains Mono,monospace;font-size:0.8rem;color:#9ca3af">{resp["model_key"]}</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;font-size:1.4rem;color:{color};font-weight:600">{score_str}</div>'
                    f'<div style="color:#6b7280;font-size:0.7rem">{resp.get("latency_ms","N/A")} ms</div>',
                    unsafe_allow_html=True
                )

                # Score bars
                for dim, key in [("Accuracy", "accuracy"), ("Structure", "structure"), ("Length", "length")]:
                    dim_score = ae.get(key, {}).get("score", 0) or 0
                    st.markdown(
                        f'<div style="font-size:0.65rem;color:#6b7280;margin-top:8px">{dim}</div>'
                        f'<div class="score-bar-container"><div class="score-bar" style="width:{dim_score*100:.0f}%"></div></div>',
                        unsafe_allow_html=True
                    )

                st.markdown(
                    f'<div class="response-box" style="margin-top:12px">{resp["response"]}</div>',
                    unsafe_allow_html=True
                )

    if failed_responses:
        with st.expander(f"⚠️ {len(failed_responses)} model(s) failed (rate limit)"):
            for r in failed_responses:
                st.markdown(f'`{r["model_key"]}` — {r.get("error","unknown error")[:100]}')


# ── Page: LLM-as-Judge ───────────────────────────────────────────────────────
elif page == "⚖️ LLM-as-Judge":
    st.markdown("# LLM-as-Judge Scores")
    st.markdown("*Qualitative evaluation: reasoning quality, level appropriateness, hallucination risk*")
    st.divider()

    if judge_data is None:
        st.warning("No judge data found. Run `python src/run_judge.py` first.")
        st.stop()

    rows = []
    for item in judge_data:
        for resp in item["model_responses"]:
            judge = resp.get("llm_judge")
            if not judge or judge.get("composite_judge_score") is None:
                continue
            rq = judge.get("reasoning_quality", {}) or {}
            la = judge.get("level_appropriateness", {}) or {}
            hr = judge.get("hallucination_risk", {}) or {}
            rows.append({
                "Question": item["question_id"],
                "Subject": item["subject"],
                "Level": item["level"],
                "Model": resp["model_key"],
                "Composite": judge["composite_judge_score"],
                "Reasoning": rq.get("score_raw"),
                "Level Fit": la.get("score_raw"),
                "Hallucination": hr.get("level", "-"),
                "RQ Note": rq.get("reasoning", ""),
                "LA Note": la.get("reasoning", ""),
                "HR Note": hr.get("reasoning", ""),
            })

    if not rows:
        st.warning("Judge scores exist but could not be parsed.")
        st.stop()

    df = pd.DataFrame(rows)

    # Summary table
    st.markdown('<div class="section-header">Judge Score Summary</div>', unsafe_allow_html=True)

    def color_halluc(val):
        colors = {"low": "color: #80ff72", "medium": "color: #ffd060", "high": "color: #ff6b6b"}
        return colors.get(str(val).lower(), "")

    display_df = df[["Question", "Subject", "Level", "Model", "Composite", "Reasoning", "Level Fit", "Hallucination"]]
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # Detail view
    st.markdown('<div class="section-header">Detailed Reasoning</div>', unsafe_allow_html=True)
    selected_row = st.selectbox("Select row", range(len(df)), format_func=lambda i: f"{df.iloc[i]['Question']} — {df.iloc[i]['Model']}")

    row = df.iloc[selected_row]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Reasoning Quality**")
        st.markdown(f"Score: `{row['Reasoning']}/5`")
        st.markdown(row["RQ Note"] or "-")
    with col2:
        st.markdown("**Level Appropriateness**")
        st.markdown(f"Score: `{row['Level Fit']}/5`")
        st.markdown(row["LA Note"] or "-")
    with col3:
        halluc = str(row["Hallucination"]).lower()
        color_class = {"low": "halluc-low", "medium": "halluc-medium", "high": "halluc-high"}.get(halluc, "")
        st.markdown("**Hallucination Risk**")
        st.markdown(f'<span class="{color_class}">{row["Hallucination"].upper()}</span>', unsafe_allow_html=True)
        st.markdown(row["HR Note"] or "-")

    # Limitations note
    st.divider()
    st.info(
        "**Known Limitations of LLM-as-Judge:** "
        "The judge model (gpt-oss-120b) may exhibit self-preference bias, favoring responses stylistically similar to its own output. "
        "Scores are not reproducible across different judge models. "
        "Use alongside automated metrics for a balanced evaluation."
    )


# ── Page: Consistency ────────────────────────────────────────────────────────
elif page == "📈 Consistency":
    st.markdown("# Consistency Analysis")
    st.markdown("*Each question asked 3× at temperature=0.3 — how stable are the answers?*")
    st.divider()

    consistency_files = glob.glob("data/scores/consistency_*.json")
    if not consistency_files:
        st.warning("No consistency data found. Run `python src/run_consistency.py` first.")
        st.stop()

    latest = max(consistency_files, key=os.path.getmtime)
    with open(latest, encoding="utf-8") as f:
        consistency_data = json.load(f)

    rows = []
    for item in consistency_data:
        c = item["consistency"]
        rows.append({
            "Question": item["question_id"],
            "Subject": item["subject"],
            "Level": item["level"],
            "Model": item["model_key"],
            "Consistency Score": c["consistency_score"],
            "Numeric Consistency": c.get("numeric_consistency"),
            "Concept Consistency": c.get("concept_consistency"),
            "Length Variance %": c.get("length_variance_pct"),
            "Valid Runs": c["valid_responses"],
            "Notes": c["reasoning"],
        })

    df = pd.DataFrame(rows)
    avg = df["Consistency Score"].mean()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Consistency Score", f"{avg:.3f}")
    with col2:
        st.metric("Questions Tested", len(df))
    with col3:
        st.metric("Repeats per Question", "3×")

    st.divider()
    st.dataframe(df[["Question", "Subject", "Level", "Model", "Consistency Score", "Numeric Consistency", "Concept Consistency", "Length Variance %"]], use_container_width=True, hide_index=True)

    st.divider()
    st.markdown('<div class="section-header">Consistency Score per Question</div>', unsafe_allow_html=True)
    st.bar_chart(df.set_index("Question")["Consistency Score"], use_container_width=True, height=250)

    st.divider()
    st.info(
        "**Metric Limitation:** Numeric consistency underperforms on essay questions "
        "(e.g. fotosintesis) because numbered list markers are counted as 'numbers', "
        "inflating the denominator. A future improvement would separate numeric vs. conceptual question types "
        "before applying this metric."
    )
