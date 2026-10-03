"""
CEREBRO Neural Sentiment Hackathon (IEM Kolkata)
Problem Statement 1: Detecting Academic Procrastination Patterns

Run with:  streamlit run app.py
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
st.set_page_config(page_title="ProcrastiScope | Academic Analytics",
                   page_icon="🧠", layout="wide")

REQUIRED_COLS = ["task_name", "assigned_date", "deadline", "submission_time"]
DATE_COLS = ["assigned_date", "deadline", "submission_time"]
PATTERN_COLORS = {
    "Early Bird": "#10B981",
    "Steady Worker": "#3B82F6",
    "Last-Minute Procrastinator": "#F59E0B",
    "Chronic Cruncher": "#EF4444",
}
PATTERN_ICONS = {"Early Bird": "🐦", "Steady Worker": "⚙️",
                 "Last-Minute Procrastinator": "⏳", "Chronic Cruncher": "🔥"}
PLOT_LAYOUT = dict(template="plotly_white", font=dict(family="Inter, sans-serif"),
                   margin=dict(l=10, r=10, t=50, b=10), height=380)

CSS = """
<style>
.block-container {padding-top: 1.2rem; max-width: 1400px;}
.hero {background: linear-gradient(120deg,#0F172A 0%,#1E3A8A 60%,#6D28D9 100%);
       padding: 1.6rem 2rem; border-radius: 16px; color: #fff; margin-bottom: 1.2rem;}
.hero h1 {margin: 0; font-size: 1.9rem; letter-spacing: .3px;}
.hero p {margin: .3rem 0 0; opacity: .8;}
.card {background:#fff; border:1px solid #E5E7EB; border-radius:14px;
       padding:1rem 1.2rem; box-shadow:0 1px 3px rgba(0,0,0,.06); height:100%;}
.card .lbl {font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; color:#6B7280;}
.card .val {font-size:1.7rem; font-weight:700; color:#111827; line-height:1.2;}
.card .sub {font-size:.8rem; color:#6B7280;}
.badge {display:inline-block; padding:.35rem .9rem; border-radius:999px;
        color:#fff; font-weight:600; font-size:.95rem;}
.explain {background:#F8FAFC; border-left:5px solid #6D28D9; border-radius:8px;
          padding:1.1rem 1.4rem; color:#1F2937; line-height:1.65;}
footer, #MainMenu {visibility:hidden;}
</style>
"""


# --------------------------------------------------------------------------
# Data generation
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_sample_data(seed: int = 42) -> pd.DataFrame:
    """Create a realistic multi-semester dataset for four student personas.

    Each persona has a mean 'urgency ratio' per semester (0 = submitted the
    moment the task was assigned, 1 = submitted exactly at the deadline).
    """
    rng = np.random.default_rng(seed)
    personas = {
        "S101": {"means": [0.80, 0.94, 0.99], "noise": 0.04},  # cruncher, worsening
        "S102": {"means": [0.20, 0.18, 0.22], "noise": 0.08},  # early bird
        "S103": {"means": [0.90, 0.68, 0.45], "noise": 0.07},  # improving
        "S104": {"means": [0.45, 0.50, 0.42], "noise": 0.07},  # steady
    }
    categories = {"Essay": (1200, 3000), "Lab Report": (800, 1800),
                  "Problem Set": (300, 900), "Project": (1500, 4000),
                  "Quiz Prep": (100, 400)}
    semesters = [pd.Timestamp("2023-08-14"), pd.Timestamp("2024-01-15"),
                 pd.Timestamp("2024-08-12")]
    rows = []
    for sid, cfg in personas.items():
        for s_idx, start in enumerate(semesters):
            for k in range(8):
                cat = rng.choice(list(categories))
                assigned = start + pd.Timedelta(days=int(k * 14 + rng.integers(0, 3)),
                                                hours=int(rng.integers(8, 18)))
                window_days = int(rng.integers(5, 15))
                deadline = (assigned + pd.Timedelta(days=window_days)).normalize() \
                    + pd.Timedelta(hours=23, minutes=59)
                window = (deadline - assigned).total_seconds()
                ratio = np.clip(rng.normal(cfg["means"][s_idx], cfg["noise"]), 0.03, 1.04)
                submitted = assigned + pd.Timedelta(seconds=ratio * window)
                lo, hi = categories[cat]
                rows.append({
                    "student_id": sid,
                    "task_name": f"{cat} {s_idx * 8 + k + 1}",
                    "task_category": cat,
                    "assigned_date": assigned,
                    "deadline": deadline,
                    "submission_time": submitted.round("min"),
                    "word_count": int(rng.integers(lo, hi)),
                })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def generate_cohort_data(seed: int = 2024) -> pd.DataFrame:
    """Larger 5-student, 4-semester cohort that uses 'effort_score'."""
    rng = np.random.default_rng(seed)
    personas = {
        "ENG2021-A": ([0.97, 0.99, 1.02, 1.04], 0.03),  # chronic, often late
        "ENG2021-B": ([0.15, 0.20, 0.12, 0.18], 0.06),  # early bird
        "ENG2021-C": ([0.50, 0.55, 0.48, 0.52], 0.07),  # steady
        "ENG2021-D": ([0.92, 0.80, 0.62, 0.40], 0.06),  # improving
        "ENG2021-E": ([0.30, 0.45, 0.65, 0.85], 0.06),  # declining
    }
    cats = {"Coding Assignment": (0, 0), "Research Paper": (2000, 5000),
            "Presentation": (300, 900), "Lab Journal": (600, 1500),
            "Midterm Project": (1500, 4000), "Reading Response": (250, 700)}
    starts = [pd.Timestamp(x) for x in
              ["2022-01-17", "2022-08-15", "2023-01-16", "2023-08-14"]]
    rows = []
    for sid, (means, noise) in personas.items():
        for si, start in enumerate(starts):
            for k in range(7):
                cat = rng.choice(list(cats))
                a = start + pd.Timedelta(days=int(k * 15 + rng.integers(0, 4)),
                                         hours=int(rng.integers(8, 19)))
                d = (a + pd.Timedelta(days=int(rng.integers(4, 18)))).normalize() \
                    + pd.Timedelta(hours=23, minutes=59)
                r = np.clip(rng.normal(means[si], noise), 0.02, 1.08)
                sub = (a + pd.Timedelta(seconds=r * (d - a).total_seconds())).round("min")
                lo, hi = cats[cat]
                effort = (int(rng.integers(40, 100)) if hi == 0
                          else int(rng.integers(lo, hi) / 50))
                rows.append({"student_id": sid, "task_name": f"{cat} #{si * 7 + k + 1}",
                             "task_category": cat, "assigned_date": a, "deadline": d,
                             "submission_time": sub, "effort_score": effort})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def generate_messy_data() -> pd.DataFrame:
    """Deliberately flawed data to demonstrate the error-handling layer."""
    base = generate_cohort_data()
    m = base[base.student_id.isin(["ENG2021-A", "ENG2021-D"])].head(40).copy().astype(str)
    m.loc[m.index[3], "deadline"] = "not a date"
    m.loc[m.index[7], "submission_time"] = ""
    m.loc[m.index[11], "deadline"] = m.loc[m.index[11], "assigned_date"]
    m.columns = [c.title().replace("_", " ") for c in m.columns]  # odd headers
    return m.reset_index(drop=True)


DATASETS = {
    "Sample 1: Mixed personas (4 students)": ("sample_1_mixed.csv", generate_sample_data),
    "Sample 2: Large cohort (5 students)": ("sample_2_cohort.csv", generate_cohort_data),
    "Sample 3: Messy data (error-handling demo)": ("sample_3_messy.csv", generate_messy_data),
}


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Clean an arbitrary DataFrame. Raises ValueError for fatal problems."""
    notes: list[str] = []
    df = raw.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    if "student_id" not in df.columns:
        df["student_id"] = "STUDENT"
        notes.append("No 'student_id' column found; treating all rows as one student.")
    if "task_category" not in df.columns:
        df["task_category"] = "General"
    # Effort metric: prefer effort_score, fall back to word_count.
    if "effort_score" in df.columns:
        df["effort"] = pd.to_numeric(df["effort_score"], errors="coerce")
    elif "word_count" in df.columns:
        df["effort"] = pd.to_numeric(df["word_count"], errors="coerce")
    else:
        df["effort"] = np.nan

    for col in DATE_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=DATE_COLS)
    df = df[df["deadline"] > df["assigned_date"]]
    if len(df) < before:
        notes.append(f"Dropped {before - len(df)} row(s) with invalid or inconsistent dates.")
    if df.empty:
        raise ValueError("No valid rows remain after cleaning. Check date formats.")

    df["student_id"] = df["student_id"].astype(str)
    df["task_category"] = df["task_category"].fillna("General").astype(str)
    return df.reset_index(drop=True), notes


# --------------------------------------------------------------------------
# Analytics engine
# --------------------------------------------------------------------------
def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add Lead Time, Urgency Ratio, Procrastination Index and semester labels.

    Lead Time (h)   = deadline - submission (negative => late)
    Urgency Ratio   = (submission - assigned) / (deadline - assigned)
    Proc. Index     = 90 * clip(ratio, 0, 1) + 10 if late   (range 0-100)
    """
    out = df.sort_values(["student_id", "assigned_date"]).copy()
    window_h = (out["deadline"] - out["assigned_date"]).dt.total_seconds() / 3600
    out["window_hours"] = window_h
    out["lead_time_hours"] = (out["deadline"] - out["submission_time"]).dt.total_seconds() / 3600
    out["urgency_ratio"] = ((out["submission_time"] - out["assigned_date"])
                            .dt.total_seconds() / 3600) / window_h
    out["is_late"] = out["lead_time_hours"] < 0
    out["procrastination_index"] = (90 * out["urgency_ratio"].clip(0, 1)
                                    + np.where(out["is_late"], 10, 0)).clip(0, 100).round(1)
    term = np.where(out["assigned_date"].dt.month <= 6, "Spring", "Fall")
    out["semester"] = out["assigned_date"].dt.year.astype(str) + " " + term
    out["task_order"] = out.groupby("student_id").cumcount() + 1
    return out


def classify(sdf: pd.DataFrame) -> str:
    """Rule-based behaviour classification for one student's history."""
    mean_pi = sdf["procrastination_index"].mean()
    crunch_share = (sdf["urgency_ratio"] >= 0.90).mean()
    if mean_pi >= 70 and crunch_share >= 0.5:
        return "Chronic Cruncher"
    if mean_pi >= 55:
        return "Last-Minute Procrastinator"
    if mean_pi >= 30:
        return "Steady Worker"
    return "Early Bird"


def analyze_trend(sdf: pd.DataFrame) -> dict:
    """Detect whether behaviour is improving or worsening over time."""
    n = len(sdf)
    if n < 4:
        return {"label": "Insufficient data", "slope": 0.0, "delta": 0.0}
    y = sdf["procrastination_index"].to_numpy()
    slope = float(np.polyfit(np.arange(n), y, 1)[0])
    half = n // 2
    delta = float(y[half:].mean() - y[:half].mean())
    if delta >= 5:
        label = "Worsening"
    elif delta <= -5:
        label = "Improving"
    else:
        label = "Stable"
    return {"label": label, "slope": slope, "delta": delta}


def generate_explanation(sid: str, sdf: pd.DataFrame, pattern: str, trend: dict) -> str:
    """Build a professional, rule-based narrative from the student's data."""
    n = len(sdf)
    mean_pi = sdf["procrastination_index"].mean()
    med_lead = sdf["lead_time_hours"].median()
    final_10 = (sdf["urgency_ratio"] >= 0.90).mean() * 100
    late = sdf["is_late"].mean() * 100
    mean_ratio = sdf["urgency_ratio"].mean() * 100

    openings = {
        "Early Bird": "consistently starts and completes work well ahead of deadlines",
        "Steady Worker": "distributes effort fairly evenly across the available window",
        "Last-Minute Procrastinator": "tends to defer work until the closing stretch of each window",
        "Chronic Cruncher": "habitually delivers work in the final hours before the deadline",
    }
    text = (f"**{sid}** is classified as a **{pattern}** based on {n} recorded tasks: "
            f"the student {openings[pattern]}. The average Procrastination Index is "
            f"**{mean_pi:.0f}/100**, with submissions typically made after "
            f"**{mean_ratio:.0f}%** of the allowed time had elapsed. ")
    if med_lead >= 0:
        text += f"The median lead time is **{med_lead:.1f} hours** before the deadline, "
    else:
        text += f"The median submission is **{abs(med_lead):.1f} hours past the deadline**, "
    text += (f"**{final_10:.0f}%** of tasks were submitted in the final 10% of the window "
             f"and **{late:.0f}%** were late. ")

    cat = sdf.groupby("task_category")["procrastination_index"].mean()
    if len(cat) > 1:
        text += (f"Delay is most pronounced for **{cat.idxmax()}** tasks "
                 f"(avg index {cat.max():.0f}), while **{cat.idxmin()}** tasks are handled "
                 f"best ({cat.min():.0f}). ")

    if trend["label"] == "Worsening":
        text += (f"Behaviour is **worsening**: the later half of the history scores "
                 f"{trend['delta']:+.1f} points higher than the earlier half. ")
    elif trend["label"] == "Improving":
        text += (f"Behaviour is **improving**: the later half of the history scores "
                 f"{abs(trend['delta']):.1f} points lower than the earlier half. ")
    elif trend["label"] == "Stable":
        text += "Habits have remained **stable** across the observed period. "

    valid = sdf[["effort", "procrastination_index"]].dropna()
    if len(valid) >= 6 and valid["effort"].std() > 0:
        corr = valid["effort"].corr(valid["procrastination_index"])
        if corr >= 0.35:
            text += "Larger tasks are associated with later submission, suggesting difficulty planning big workloads. "
        elif corr <= -0.35:
            text += "Larger tasks are started earlier, which suggests the student prioritises high-effort work. "

    advice = {
        "Early Bird": "Recommendation: maintain current habits and consider peer-mentoring.",
        "Steady Worker": "Recommendation: introduce mid-point checkpoints to push the index lower.",
        "Last-Minute Procrastinator": "Recommendation: split tasks into milestones with interim deadlines.",
        "Chronic Cruncher": "Recommendation: urgent intervention, with enforced milestones, "
                            "advisor check-ins and structured time-blocking.",
    }
    return text + advice[pattern]


# --------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------
def metric_card(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="card"><div class="lbl">{label}</div>'
            f'<div class="val">{value}</div><div class="sub">{sub}</div></div>')


def gauge(value: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={"text": "Procrastination Index"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#111827"},
               "steps": [{"range": [0, 30], "color": "#A7F3D0"},
                         {"range": [30, 55], "color": "#BFDBFE"},
                         {"range": [55, 75], "color": "#FDE68A"},
                         {"range": [75, 100], "color": "#FCA5A5"}]}))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def gantt_chart(sdf: pd.DataFrame) -> go.Figure:
    """Timeline of assignment window with submission markers."""
    d = sdf.tail(15)
    fig = px.timeline(d, x_start="assigned_date", x_end="deadline", y="task_name",
                      color="task_category", opacity=0.55,
                      title="Task Windows vs. Actual Submissions (latest 15 tasks)")
    fig.update_yaxes(autorange="reversed", title="")
    fig.add_trace(go.Scatter(
        x=d["submission_time"], y=d["task_name"], mode="markers", name="Submitted",
        marker=dict(size=11, symbol="diamond", line=dict(width=1, color="white"),
                    color=np.where(d["is_late"], "#EF4444", "#111827")),
        hovertemplate="%{y}<br>Submitted: %{x}<extra></extra>"))
    fig.update_layout(**{**PLOT_LAYOUT, "height": 480})
    return fig


def lead_time_hist(sdf: pd.DataFrame) -> go.Figure:
    fig = px.histogram(sdf, x="lead_time_hours", nbins=25,
                       title="Distribution of Hours Before Deadline",
                       color_discrete_sequence=["#6D28D9"])
    fig.add_vline(x=0, line_dash="dash", line_color="#EF4444",
                  annotation_text="Deadline")
    fig.update_layout(**PLOT_LAYOUT, xaxis_title="Lead time (hours)", yaxis_title="Tasks")
    return fig


def trend_chart(sdf: pd.DataFrame) -> go.Figure:
    y = sdf["procrastination_index"].to_numpy()
    x = sdf["task_order"].to_numpy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="Task index",
                             line=dict(color="#6D28D9"), text=sdf["task_name"]))
    if len(y) >= 3:
        m, b = np.polyfit(x, y, 1)
        fig.add_trace(go.Scatter(x=x, y=m * x + b, mode="lines", name="Trend",
                                 line=dict(color="#EF4444", dash="dash")))
    fig.add_hrect(y0=75, y1=100, fillcolor="#EF4444", opacity=0.06, line_width=0)
    fig.update_layout(**PLOT_LAYOUT, title="Procrastination Index Over Time",
                      xaxis_title="Task sequence", yaxis=dict(range=[0, 100], title="Index"))
    return fig


def semester_chart(sdf: pd.DataFrame) -> go.Figure:
    g = sdf.groupby("semester", sort=False)["procrastination_index"].mean().reset_index()
    fig = px.bar(g, x="semester", y="procrastination_index", text_auto=".0f",
                 title="Average Index by Semester", color="procrastination_index",
                 color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
                 range_color=[0, 100])
    fig.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False, yaxis_range=[0, 100])
    return fig


def category_chart(sdf: pd.DataFrame) -> go.Figure:
    g = sdf.groupby("task_category")["procrastination_index"].mean().sort_values().reset_index()
    fig = px.bar(g, x="procrastination_index", y="task_category", orientation="h",
                 text_auto=".0f", title="Average Index by Task Category",
                 color_discrete_sequence=["#3B82F6"])
    fig.update_layout(**PLOT_LAYOUT, xaxis_range=[0, 100], xaxis_title="Index", yaxis_title="")
    return fig


def hour_chart(sdf: pd.DataFrame) -> go.Figure:
    hours = sdf["submission_time"].dt.hour.value_counts().reindex(range(24), fill_value=0)
    fig = px.bar(x=hours.index, y=hours.values, title="Submissions by Hour of Day",
                 color_discrete_sequence=["#F59E0B"])
    fig.update_layout(**PLOT_LAYOUT, xaxis_title="Hour (24h)", yaxis_title="Submissions")
    return fig


# --------------------------------------------------------------------------
# Main application
# --------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, list[str]]:
    """Sidebar data source selector; always returns a valid DataFrame."""
    st.sidebar.header("📂 Data Source")
    options = list(DATASETS) + ["Upload your own CSV"]
    mode = st.sidebar.radio("Choose dataset", options)
    raw = generate_sample_data()
    if mode in DATASETS:
        raw = DATASETS[mode][1]()
    else:
        file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        st.sidebar.caption("Required: task_name, assigned_date, deadline, submission_time. "
                           "Optional: student_id, task_category, word_count / effort_score.")
        if file is not None:
            try:
                raw = pd.read_csv(file)
            except Exception as exc:  # noqa: BLE001 - never crash on bad uploads
                st.sidebar.error(f"Could not read file: {exc}")
                st.sidebar.info("Falling back to sample data.")
                raw = generate_sample_data()
        else:
            st.sidebar.info("No file yet - showing Sample 1.")

    with st.sidebar.expander("⬇️ Download sample datasets"):
        for label, (fname, fn) in DATASETS.items():
            st.download_button(label, fn().to_csv(index=False), fname, "text/csv",
                               key=f"dl_{fname}", use_container_width=True)
    try:
        return validate_data(raw)
    except ValueError as exc:
        st.sidebar.error(f"Data problem: {exc}")
        st.sidebar.info("Falling back to sample data.")
        return validate_data(generate_sample_data())


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1>🧠 ProcrastiScope</h1>'
                '<p>Academic procrastination pattern detection &amp; behavioural analytics '
                '| CEREBRO Neural Sentiment Hackathon</p></div>', unsafe_allow_html=True)

    df, notes = load_data()
    for note in notes:
        st.sidebar.warning(note)
    try:
        metrics = compute_metrics(df)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Analysis failed: {exc}")
        st.stop()

    students = sorted(metrics["student_id"].unique())
    sid = st.sidebar.selectbox("🎓 Student", students)
    sdf = metrics[metrics["student_id"] == sid].reset_index(drop=True)
    sdf["task_order"] = np.arange(1, len(sdf) + 1)

    pattern = classify(sdf)
    trend = analyze_trend(sdf)
    mean_pi = sdf["procrastination_index"].mean()

    # --- KPI row ---
    med_lead = sdf["lead_time_hours"].median()
    cols = st.columns(5)
    cards = [
        ("Behaviour Pattern",
         f'<span class="badge" style="background:{PATTERN_COLORS[pattern]}">'
         f'{PATTERN_ICONS[pattern]} {pattern}</span>', f"{len(sdf)} tasks analysed"),
        ("Avg Procrastination Index", f"{mean_pi:.0f}/100", "0 = early, 100 = last minute"),
        ("Median Lead Time", f"{med_lead:.1f} h", "before deadline"),
        ("Avg Urgency Ratio", f"{sdf['urgency_ratio'].mean():.2f}", "share of window used"),
        ("Trend", {"Worsening": "📈 Worsening", "Improving": "📉 Improving",
                   "Stable": "➖ Stable"}.get(trend["label"], "n/a"),
         f"{trend['delta']:+.1f} pts (2nd vs 1st half)"),
    ]
    for col, (lbl, val, sub) in zip(cols, cards):
        col.markdown(metric_card(lbl, val, sub), unsafe_allow_html=True)

    st.write("")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Overview", "📈 Trends", "🧾 AI Explanation", "🗂️ Data & Cohort"])

    with tab1:
        c1, c2 = st.columns([1, 2])
        c1.plotly_chart(gauge(mean_pi), use_container_width=True)
        c2.plotly_chart(lead_time_hist(sdf), use_container_width=True)
        st.plotly_chart(gantt_chart(sdf), use_container_width=True)
        c3, c4 = st.columns(2)
        c3.plotly_chart(category_chart(sdf), use_container_width=True)
        c4.plotly_chart(hour_chart(sdf), use_container_width=True)

    with tab2:
        st.plotly_chart(trend_chart(sdf), use_container_width=True)
        st.plotly_chart(semester_chart(sdf), use_container_width=True)

    with tab3:
        st.subheader("Why this classification?")
        st.markdown(generate_explanation(sid, sdf, pattern, trend))
        with st.expander("Scoring methodology"):
            st.markdown(
                "- **Lead Time** = deadline - submission (hours; negative = late)\n"
                "- **Urgency Ratio** = time elapsed at submission / total allowed time\n"
                "- **Procrastination Index** = 90 x clipped ratio, +10 if late (0-100)\n"
                "- **Chronic Cruncher**: mean index >= 70 and >= 50% of tasks in last 10% of window\n"
                "- **Last-Minute Procrastinator**: mean >= 55 | **Steady Worker**: >= 30 | "
                "**Early Bird**: < 30\n"
                "- **Trend**: second-half vs first-half mean index (+/-5 pts threshold)")

    with tab4:
        st.subheader("Cohort comparison")
        rows = []
        for s in students:
            g = metrics[metrics["student_id"] == s]
            rows.append({"Student": s, "Pattern": classify(g), "Tasks": len(g),
                         "Avg Index": round(g["procrastination_index"].mean(), 1),
                         "Late %": round(g["is_late"].mean() * 100, 1),
                         "Trend": analyze_trend(g)["label"]})
        cohort = pd.DataFrame(rows)
        st.dataframe(cohort, use_container_width=True, hide_index=True)
        if len(cohort) > 1:
            fig = px.bar(cohort, x="Student", y="Avg Index", color="Pattern",
                         color_discrete_map=PATTERN_COLORS, text_auto=".0f",
                         title="Average Procrastination Index by Student")
            fig.update_layout(**PLOT_LAYOUT, yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
        st.subheader(f"Computed metrics: {sid}")
        show = sdf[["task_name", "task_category", "assigned_date", "deadline",
                    "submission_time", "lead_time_hours", "urgency_ratio",
                    "procrastination_index"]].round(2)
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export results (CSV)", show.to_csv(index=False),
                           f"{sid}_analysis.csv", "text/csv")


if __name__ == "__main__":
    main()