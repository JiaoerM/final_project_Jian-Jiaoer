"""
dashboard.py  —  Food Desert & Health Outcomes Explorer
Run with:  streamlit run dashboard.py
Requires:  pip install streamlit plotly pandas numpy scipy statsmodels
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Food Desert & Health Outcomes",
    page_icon="🥦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }

  /* Dark sidebar */
  [data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
  }
  [data-testid="stSidebar"] * {
    color: #e6edf3 !important;
  }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stMultiSelect label,
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stRadio label {
    color: #8b949e !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  /* Main background */
  .main .block-container {
    background: #f6f8fa;
    padding-top: 2rem;
  }

  /* Hero header */
  .hero {
    background: #0d1117;
    border-radius: 12px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(88,166,255,0.12) 0%, transparent 70%);
  }
  .hero-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #58a6ff;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
  }
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #e6edf3;
    line-height: 1.1;
    margin-bottom: 0.75rem;
  }
  .hero-subtitle {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 1rem;
    color: #8b949e;
    max-width: 680px;
    line-height: 1.6;
  }

  /* Stat cards */
  .stat-card {
    background: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    text-align: center;
  }
  .stat-number {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #0d1117;
    line-height: 1;
  }
  .stat-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #8b949e;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.4rem;
  }
  .stat-delta {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 13px;
    margin-top: 0.5rem;
  }
  .delta-up   { color: #da3633; }
  .delta-down { color: #3fb950; }

  /* Section headers */
  .section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #0d1117;
    margin: 0.5rem 0 1.2rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid #0d1117;
  }

  /* Chart card */
  .chart-card {
    background: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
  }

  /* Insight box */
  .insight-box {
    background: #ddf4ff;
    border-left: 4px solid #58a6ff;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
  }
  .insight-box p {
    font-size: 14px;
    color: #0d1117;
    margin: 0;
    line-height: 1.6;
  }
  .insight-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #58a6ff;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
  }

  /* Regression table */
  .reg-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .reg-table th {
    background: #0d1117;
    color: #e6edf3;
    padding: 8px 12px;
    text-align: left;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.05em;
  }
  .reg-table td { padding: 7px 12px; border-bottom: 1px solid #d0d7de; }
  .reg-table tr:hover td { background: #f6f8fa; }
  .sig-star { color: #da3633; font-weight: 700; }

  /* Tab styling */
  [data-testid="stTab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
  }

  /* Hide Streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/majiaoer/Desktop/final_project_Jian-Jiaoer/data/derived/merged_tract_data.csv", dtype={"CensusTract": str})

    # Derived columns
    df["FoodDesert"] = df["LILATracts_1And10"].map({0: "Non-Food Desert", 1: "Food Desert"})
    df["UrbanLabel"] = df["Urban"].map({1: "Urban", 0: "Rural"})
    df["income_quartile"] = pd.qcut(
        df["MedianFamilyIncome"], q=4,
        labels=["Q1 · Lowest", "Q2", "Q3", "Q4 · Highest"]
    )

    # State-level summary
    state_df = df.groupby("State").agg(
        n_tracts=("CensusTract", "count"),
        pct_desert=("LILATracts_1And10", "mean"),
        mean_diabetes=("DIABETES", "mean"),
        mean_obesity=("OBESITY", "mean"),
        mean_bphigh=("BPHIGH", "mean"),
        mean_csmoking=("CSMOKING", "mean"),
        mean_income=("MedianFamilyIncome", "mean"),
    ).reset_index().round(3)

    return df, state_df

df, state_df = load_data()

# Constants
OUTCOMES = {
    "DIABETES":  "Diabetes Prevalence (%)",
    "OBESITY":   "Obesity Prevalence (%)",
    "BPHIGH":    "High Blood Pressure (%)",
    "CSMOKING":  "Current Smoking (%)",
}
OUTCOME_COLORS = {
    "DIABETES": "#e36209",
    "OBESITY":  "#8250df",
    "BPHIGH":   "#da3633",
    "CSMOKING": "#1a7f37",
}
DESERT_COLORS = {"Food Desert": "#e36209", "Non-Food Desert": "#0969da"}
ALL_STATES = ["All States"] + sorted(df["State"].unique().tolist())


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 0.5rem 0;'>
      <div style='font-family: Syne, sans-serif; font-size: 1.1rem; font-weight: 800;
                  color: #e6edf3; margin-bottom: 0.25rem;'>🥦 Food Desert</div>
      <div style='font-family: IBM Plex Mono, monospace; font-size: 10px;
                  color: #58a6ff; letter-spacing: 0.1em;'>HEALTH EXPLORER</div>
    </div>
    <hr style='border-color: #21262d; margin: 0.75rem 0 1.25rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown("**HEALTH OUTCOME**")
    selected_outcome = st.selectbox(
        "outcome", list(OUTCOMES.keys()),
        format_func=lambda x: OUTCOMES[x], label_visibility="collapsed"
    )

    st.markdown("**STATE FILTER**")
    selected_state = st.selectbox("state", ALL_STATES, label_visibility="collapsed")

    st.markdown("**GEOGRAPHY**")
    selected_urban = st.radio(
        "urban", ["All", "Urban Only", "Rural Only"], label_visibility="collapsed"
    )

    st.markdown("**INCOME RANGE (USD)**")
    inc_min = int(df["MedianFamilyIncome"].dropna().min())
    inc_max = int(df["MedianFamilyIncome"].dropna().max())
    income_range = st.slider(
        "income", inc_min, inc_max, (inc_min, inc_max),
        step=5000, format="$%d", label_visibility="collapsed"
    )

    st.markdown("**SCATTER SAMPLE SIZE**")
    scatter_n = st.slider(
        "sample", 1000, 20000, 8000, step=1000, label_visibility="collapsed"
    )

    st.markdown("""
    <hr style='border-color: #21262d; margin: 1.5rem 0 0.75rem 0;'>
    <div style='font-family: IBM Plex Mono, monospace; font-size: 10px; color: #484f58;
                line-height: 1.7;'>
      USDA Food Access Research Atlas 2019<br>
      CDC PLACES 2022<br>
      N = 56,327 census tracts<br>
      Harris School · DAP Final Project
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FILTER DATA
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def filter_data(state, urban, inc_low, inc_high):
    d = df.copy()
    if state != "All States":
        d = d[d["State"] == state]
    if urban == "Urban Only":
        d = d[d["Urban"] == 1]
    elif urban == "Rural Only":
        d = d[d["Urban"] == 0]
    d = d[d["MedianFamilyIncome"].between(inc_low, inc_high)]
    return d

filtered = filter_data(selected_state, selected_urban, income_range[0], income_range[1])
outcome_label = OUTCOMES[selected_outcome]
outcome_color = OUTCOME_COLORS[selected_outcome]


# ══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <div class="hero-tag">Public Health · Census Tract Analysis · USDA × CDC</div>
  <div class="hero-title">Food Deserts &amp;<br>Health Outcomes</div>
  <div class="hero-subtitle">
    Exploring how geographic food access correlates with chronic disease rates
    across {len(df):,} U.S. census tracts. Currently viewing:
    <strong style="color:#58a6ff;">{outcome_label}</strong>
    {f'in <strong style="color:#58a6ff;">{selected_state}</strong>' if selected_state != "All States" else "nationally"}.
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# KPI CARDS
# ══════════════════════════════════════════════════════════════════════════════
desert_mean    = filtered[filtered["LILATracts_1And10"]==1][selected_outcome].mean()
nondesert_mean = filtered[filtered["LILATracts_1And10"]==0][selected_outcome].mean()
overall_mean   = filtered[selected_outcome].mean()
pct_desert     = filtered["LILATracts_1And10"].mean() * 100
gap            = desert_mean - nondesert_mean
t_stat, p_val  = stats.ttest_ind(
    filtered[filtered["LILATracts_1And10"]==1][selected_outcome].dropna(),
    filtered[filtered["LILATracts_1And10"]==0][selected_outcome].dropna()
)
sig_label = "p < 0.001 ***" if p_val < 0.001 else (f"p = {p_val:.3f}" + (" **" if p_val < 0.01 else " *" if p_val < 0.05 else ""))

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-number">{len(filtered):,}</div>
      <div class="stat-label">Tracts in View</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-number">{pct_desert:.1f}%</div>
      <div class="stat-label">Food Desert Tracts</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-number">{overall_mean:.1f}%</div>
      <div class="stat-label">Mean {selected_outcome}</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-number" style="color: #da3633;">{desert_mean:.1f}%</div>
      <div class="stat-label">Food Desert Mean</div>
      <div class="stat-delta delta-up">vs {nondesert_mean:.1f}% non-desert</div>
    </div>""", unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-number" style="color: #e36209;">+{gap:.1f}pp</div>
      <div class="stat-label">Desert Gap</div>
      <div class="stat-delta" style="color:#8b949e;">{sig_label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Correlation Explorer",
    "🗺️  State Map",
    "📦  Group Comparison",
    "📈  Regression",
    "🔍  Tract Explorer",
])


# ════════════════════════════════════════════════
# TAB 1 · CORRELATION SCATTER
# ════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Food Access vs. Health Outcome</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 1])

    with col_right:
        st.markdown("**X-AXIS MEASURE**")
        x_measure = st.selectbox("x", {
            "lapop1":           "Pop. >1mi from Grocery",
            "lalowi1":          "Low Income & Low Access Pop.",
            "MedianFamilyIncome": "Median Family Income",
            "TractHUNV":        "Households w/ No Vehicle",
            "TractSNAP":        "SNAP Households",
        }.keys(), format_func=lambda k: {
            "lapop1": "Pop. >1mi from Grocery",
            "lalowi1": "Low Income & Low Access Pop.",
            "MedianFamilyIncome": "Median Family Income",
            "TractHUNV": "Households w/ No Vehicle",
            "TractSNAP": "SNAP Households",
        }[k], label_visibility="collapsed")

        x_labels = {
            "lapop1": "Population >1 Mile from Grocery",
            "lalowi1": "Low Income & Low Access Population",
            "MedianFamilyIncome": "Median Family Income (USD)",
            "TractHUNV": "Households Without a Vehicle",
            "TractSNAP": "SNAP Recipient Households",
        }

        st.markdown("**COLOR BY**")
        color_by = st.radio("color", ["Food Desert Status", "Urban/Rural"], label_visibility="collapsed")
        show_reg = st.checkbox("Show regression lines", value=True)
        log_x    = st.checkbox("Log-scale X axis", value=False)

    with col_left:
        scatter_data = filtered.dropna(subset=[x_measure, selected_outcome])
        if len(scatter_data) > scatter_n:
            scatter_data = scatter_data.sample(n=scatter_n, random_state=42)

        color_col   = "FoodDesert" if color_by == "Food Desert Status" else "UrbanLabel"
        color_map   = DESERT_COLORS if color_by == "Food Desert Status" else {"Urban": "#0969da", "Rural": "#8250df"}

        fig = px.scatter(
            scatter_data, x=x_measure, y=selected_outcome,
            color=color_col, color_discrete_map=color_map,
            opacity=0.45, size_max=6,
            labels={x_measure: x_labels[x_measure], selected_outcome: outcome_label},
            log_x=log_x,
        )

        if show_reg:
            for group_val, group_color in color_map.items():
                col_key = "FoodDesert" if color_by == "Food Desert Status" else "UrbanLabel"
                sub = scatter_data[scatter_data[col_key] == group_val][[x_measure, selected_outcome]].dropna()
                if len(sub) > 30:
                    m, b = np.polyfit(np.log1p(sub[x_measure]) if log_x else sub[x_measure], sub[selected_outcome], 1)
                    x_sorted = np.linspace(sub[x_measure].min(), sub[x_measure].max(), 300)
                    y_line   = m * (np.log1p(x_sorted) if log_x else x_sorted) + b
                    fig.add_trace(go.Scatter(
                        x=x_sorted, y=y_line, mode="lines",
                        line=dict(color=group_color, width=2.5, dash="solid"),
                        name=f"{group_val} trend", showlegend=True
                    ))

        # Pearson r
        r, p = stats.pearsonr(scatter_data[x_measure].dropna(), scatter_data[selected_outcome].dropna())

        fig.update_layout(
            height=500, paper_bgcolor="white", plot_bgcolor="white",
            font_family="IBM Plex Sans",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(showgrid=True, gridcolor="#e8e8e8", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#e8e8e8"),
            title=dict(
                text=f"Pearson r = <b>{r:.3f}</b>  |  {'p < 0.001' if p < 0.001 else f'p = {p:.4f}'}  |  n = {len(scatter_data):,}",
                font_size=13, x=0
            )
        )
        fig.update_traces(marker=dict(size=5))
        st.plotly_chart(fig, use_container_width=True)

    # Insight
    direction = "positive" if r > 0 else "negative"
    strength  = "strong" if abs(r) > 0.5 else ("moderate" if abs(r) > 0.3 else "weak")
    st.markdown(f"""
    <div class="insight-box">
      <div class="insight-label">📌 Interpretation</div>
      <p>There is a <strong>{strength} {direction}</strong> correlation (r = {r:.3f}) between
      <em>{x_labels[x_measure]}</em> and <em>{outcome_label}</em> across {len(scatter_data):,}
      sampled census tracts. Food desert tracts (orange) show a mean of
      <strong>{desert_mean:.1f}%</strong> vs <strong>{nondesert_mean:.1f}%</strong> for non-desert
      tracts — a gap of <strong>{gap:+.1f} percentage points</strong> ({sig_label}).</p>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════
# TAB 2 · STATE MAP
# ════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">State-Level Overview</div>', unsafe_allow_html=True)

    map_col1, map_col2 = st.columns([2, 1])

    with map_col2:
        map_metric = st.selectbox("Map color shows:", {
            "pct_desert":    "% Food Desert Tracts",
            "mean_diabetes": "Mean Diabetes Rate",
            "mean_obesity":  "Mean Obesity Rate",
            "mean_bphigh":   "Mean High BP Rate",
            "mean_csmoking": "Mean Smoking Rate",
            "mean_income":   "Mean Family Income",
        }.keys(), format_func=lambda k: {
            "pct_desert":    "% Food Desert Tracts",
            "mean_diabetes": "Mean Diabetes Rate",
            "mean_obesity":  "Mean Obesity Rate",
            "mean_bphigh":   "Mean High BP Rate",
            "mean_csmoking": "Mean Smoking Rate",
            "mean_income":   "Mean Family Income",
        }[k])

        metric_labels = {
            "pct_desert":    "Food Desert %",
            "mean_diabetes": "Diabetes %",
            "mean_obesity":  "Obesity %",
            "mean_bphigh":   "High BP %",
            "mean_csmoking": "Smoking %",
            "mean_income":   "Median Income",
        }

        map_color = "Oranges" if "desert" in map_metric else ("Blues_r" if map_metric == "mean_income" else "RdPu")

    with map_col1:
        # Plotly choropleth requires 2-letter abbreviations, not full state names
        STATE_ABBREV = {
            "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
            "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
            "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
            "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
            "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
            "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
            "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
            "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
            "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
            "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
            "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
            "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
            "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
        }
        state_df_display = state_df.copy()
        state_df_display["StateAbbr"] = state_df_display["State"].map(STATE_ABBREV)
        state_df_display = state_df_display.dropna(subset=["StateAbbr"])

        if map_metric == "pct_desert":
            state_df_display["display"] = (state_df_display["pct_desert"] * 100).round(1)
            fmt = ".1f"
        else:
            state_df_display["display"] = state_df_display[map_metric].round(2)
            fmt = ".2f"

        fig_map = px.choropleth(
            state_df_display,
            locations="StateAbbr",
            locationmode="USA-states",
            color="display",
            scope="usa",
            color_continuous_scale=map_color,
            labels={"display": metric_labels[map_metric]},
            hover_name="State",
            hover_data={
                "display": f":.{fmt[1]}f",
                "n_tracts": ":,",
                "pct_desert": ":.1%",
            },
        )
        fig_map.update_layout(
            height=460,
            paper_bgcolor="white",
            geo=dict(bgcolor="white", lakecolor="#e8f4f8"),
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_colorbar=dict(
                title=metric_labels[map_metric],
                thickness=14, len=0.6,
                tickfont=dict(family="IBM Plex Mono", size=10),
            )
        )
        st.plotly_chart(fig_map, use_container_width=True)

    # State ranking table
    st.markdown('<div class="section-header">State Rankings</div>', unsafe_allow_html=True)
    rank_df = state_df[["State", "n_tracts", "pct_desert",
                          "mean_diabetes", "mean_obesity", "mean_bphigh", "mean_income"]].copy()
    rank_df["pct_desert"] = (rank_df["pct_desert"] * 100).round(1)
    rank_df.columns = ["State", "Tracts", "Food Desert %", "Diabetes %", "Obesity %", "High BP %", "Median Income"]
    rank_df = rank_df.sort_values("Food Desert %", ascending=False).reset_index(drop=True)
    rank_df.index += 1
    st.dataframe(rank_df, use_container_width=True, height=350)


# ════════════════════════════════════════════════
# TAB 3 · GROUP COMPARISON
# ════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Food Desert vs. Non-Food Desert</div>', unsafe_allow_html=True)

    g_col1, g_col2 = st.columns(2)

    # ── Bar chart: all 4 outcomes ──
    with g_col1:
        g_desert     = filtered[filtered["LILATracts_1And10"]==1][list(OUTCOMES.keys())].mean()
        g_nondesert  = filtered[filtered["LILATracts_1And10"]==0][list(OUTCOMES.keys())].mean()

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Non-Food Desert", x=list(OUTCOMES.values()), y=g_nondesert.values,
            marker_color="#0969da", opacity=0.85,
        ))
        fig_bar.add_trace(go.Bar(
            name="Food Desert", x=list(OUTCOMES.values()), y=g_desert.values,
            marker_color="#e36209", opacity=0.85,
        ))
        fig_bar.update_layout(
            barmode="group", height=380,
            paper_bgcolor="white", plot_bgcolor="white",
            font_family="IBM Plex Sans",
            legend=dict(orientation="h", y=1.1),
            yaxis_title="Prevalence (%)",
            title="Mean Health Outcomes by Food Desert Status",
            xaxis=dict(tickfont=dict(size=11)),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Violin / box by outcome ──
    with g_col2:
        violin_data = filtered[["FoodDesert", selected_outcome]].dropna()
        fig_violin = px.violin(
            violin_data, x="FoodDesert", y=selected_outcome,
            color="FoodDesert", color_discrete_map=DESERT_COLORS,
            box=True, points=False,
            labels={"FoodDesert": "", selected_outcome: outcome_label},
            title=f"Distribution of {outcome_label}",
        )
        fig_violin.update_layout(
            height=380, paper_bgcolor="white", plot_bgcolor="white",
            font_family="IBM Plex Sans", showlegend=False,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_violin, use_container_width=True)

    # ── Income quartile breakdown ──
    st.markdown('<div class="section-header">By Income Quartile — Does the Food Desert Gap Persist?</div>', unsafe_allow_html=True)

    quartile_data = filtered.dropna(subset=["income_quartile", selected_outcome, "LILATracts_1And10"])
    quartile_means = quartile_data.groupby(["income_quartile", "FoodDesert"])[selected_outcome].mean().reset_index()

    fig_quart = px.bar(
        quartile_means, x="income_quartile", y=selected_outcome,
        color="FoodDesert", barmode="group",
        color_discrete_map=DESERT_COLORS,
        labels={"income_quartile": "Income Quartile", selected_outcome: outcome_label, "FoodDesert": ""},
        title=f"{outcome_label} by Income Quartile and Food Desert Status",
    )
    fig_quart.update_layout(
        height=360, paper_bgcolor="white", plot_bgcolor="white",
        font_family="IBM Plex Sans",
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_quart, use_container_width=True)

    st.markdown(f"""
    <div class="insight-box">
      <div class="insight-label">💡 Key Test</div>
      <p>If food desert tracts show <em>higher</em> {outcome_label.lower()} than non-desert
      tracts even within the <strong>Q4 (highest income)</strong> group, this suggests the food
      access effect is <strong>independent of poverty</strong> — a stronger policy argument.
      If the gap disappears at higher incomes, income may be the primary driver.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Urban vs Rural ──
    st.markdown('<div class="section-header">Urban vs. Rural Subgroup</div>', unsafe_allow_html=True)
    urban_data = filtered.groupby(["UrbanLabel", "FoodDesert"])[selected_outcome].mean().reset_index()
    fig_urban = px.bar(
        urban_data, x="UrbanLabel", y=selected_outcome,
        color="FoodDesert", barmode="group",
        color_discrete_map=DESERT_COLORS,
        labels={"UrbanLabel": "", selected_outcome: outcome_label, "FoodDesert": ""},
        title=f"{outcome_label} by Urban/Rural and Food Desert Status",
    )
    fig_urban.update_layout(
        height=320, paper_bgcolor="white", plot_bgcolor="white",
        font_family="IBM Plex Sans",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_urban, use_container_width=True)


# ════════════════════════════════════════════════
# TAB 4 · REGRESSION
# ════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">OLS Regression Analysis</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-box">
      <div class="insight-label">Model Specification</div>
      <p><strong>{outcome_label}</strong> ~ LILATracts_1And10 + lapop1 + MedianFamilyIncome
      + TractHUNV + TractSNAP + TractBlack + TractHispanic + Urban</p>
    </div>
    """, unsafe_allow_html=True)

    reg_data = filtered.dropna(subset=[
        selected_outcome, "LILATracts_1And10", "lapop1",
        "MedianFamilyIncome", "TractHUNV", "TractSNAP",
        "TractBlack", "TractHispanic", "Urban"
    ])

    if len(reg_data) > 50:
        formula = (f"{selected_outcome} ~ LILATracts_1And10 + lapop1 + MedianFamilyIncome "
                   f"+ TractHUNV + TractSNAP + TractBlack + TractHispanic + Urban")
        model = smf.ols(formula, data=reg_data).fit()

        r1, r2, r3 = st.columns(3)
        r1.metric("R²", f"{model.rsquared:.4f}")
        r2.metric("Adj. R²", f"{model.rsquared_adj:.4f}")
        r3.metric("N (tracts)", f"{int(model.nobs):,}")

        # Coefficient table
        var_labels = {
            "Intercept":            "Intercept",
            "LILATracts_1And10":    "Food Desert Flag (LILA)",
            "lapop1":               "Pop. >1mi from Grocery",
            "MedianFamilyIncome":   "Median Family Income",
            "TractHUNV":            "No-Vehicle Households",
            "TractSNAP":            "SNAP Households",
            "TractBlack":           "Black Population",
            "TractHispanic":        "Hispanic Population",
            "Urban":                "Urban Tract",
        }

        rows = ""
        for var in model.params.index:
            coef  = model.params[var]
            se    = model.bse[var]
            t     = model.tvalues[var]
            p     = model.pvalues[var]
            ci_lo = model.conf_int().loc[var, 0]
            ci_hi = model.conf_int().loc[var, 1]
            sig   = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            highlight = ' style="background:#fff8f0;"' if var == "LILATracts_1And10" else ""
            rows += f"""
            <tr{highlight}>
              <td><strong>{var_labels.get(var, var)}</strong></td>
              <td>{coef:.4f}</td>
              <td>{se:.4f}</td>
              <td>[{ci_lo:.4f}, {ci_hi:.4f}]</td>
              <td>{t:.2f}</td>
              <td>{p:.4f}</td>
              <td><span class="sig-star">{sig}</span></td>
            </tr>"""

        st.markdown(f"""
        <table class="reg-table">
          <thead>
            <tr>
              <th>Variable</th><th>Coef.</th><th>Std Err</th>
              <th>95% CI</th><th>t</th><th>p-value</th><th>Sig.</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        <p style="font-family: IBM Plex Mono; font-size: 10px; color: #8b949e; margin-top: 0.5rem;">
          *** p &lt; 0.001  ** p &lt; 0.01  * p &lt; 0.05  |  Highlighted row = main variable of interest
        </p>
        """, unsafe_allow_html=True)

        # Coefficient plot
        st.markdown("<br>", unsafe_allow_html=True)
        coef_df = pd.DataFrame({
            "Variable": [var_labels.get(v, v) for v in model.params.index[1:]],
            "Coef":     model.params.values[1:],
            "CI_lo":    model.conf_int().values[1:, 0],
            "CI_hi":    model.conf_int().values[1:, 1],
        })
        coef_df["Error_lo"] = coef_df["Coef"] - coef_df["CI_lo"]
        coef_df["Error_hi"] = coef_df["CI_hi"] - coef_df["Coef"]
        coef_df["Color"]    = coef_df["Coef"].apply(lambda x: "#da3633" if x > 0 else "#1a7f37")

        fig_coef = go.Figure()
        fig_coef.add_vline(x=0, line_dash="dash", line_color="#8b949e", line_width=1)
        fig_coef.add_trace(go.Scatter(
            x=coef_df["Coef"], y=coef_df["Variable"],
            mode="markers",
            marker=dict(color=coef_df["Color"], size=10, symbol="circle"),
            error_x=dict(
                type="data", symmetric=False,
                array=coef_df["Error_hi"], arrayminus=coef_df["Error_lo"],
                color="#8b949e", thickness=1.5, width=5
            ),
            hovertemplate="<b>%{y}</b><br>Coef: %{x:.4f}<extra></extra>"
        ))
        fig_coef.update_layout(
            height=380, paper_bgcolor="white", plot_bgcolor="white",
            font_family="IBM Plex Sans",
            title="Coefficient Plot with 95% Confidence Intervals",
            xaxis_title="Coefficient (percentage points)",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        fig_coef.update_xaxes(showgrid=True, gridcolor="#e8e8e8")
        st.plotly_chart(fig_coef, use_container_width=True)

        lila_coef = model.params["LILATracts_1And10"]
        lila_p    = model.pvalues["LILATracts_1And10"]
        st.markdown(f"""
        <div class="insight-box">
          <div class="insight-label">📌 Key Finding</div>
          <p>After controlling for income, race, vehicle access, SNAP enrollment, and urban/rural status,
          being in a food desert is associated with a
          <strong>{lila_coef:+.3f} percentage point</strong> {'increase' if lila_coef > 0 else 'decrease'}
          in {outcome_label.lower()} ({"p < 0.001 ***" if lila_p < 0.001 else f"p = {lila_p:.4f}"}).
          The model explains <strong>{model.rsquared*100:.1f}%</strong> of the variance (R² = {model.rsquared:.4f}).</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("Not enough data to fit a regression with the current filters. Try broadening your selection.")


# ════════════════════════════════════════════════
# TAB 5 · TRACT EXPLORER
# ════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Census Tract Data Explorer</div>', unsafe_allow_html=True)

    t_col1, t_col2 = st.columns([1, 1])
    with t_col1:
        sort_by = st.selectbox("Sort by:", list(OUTCOMES.keys()) + ["MedianFamilyIncome", "lapop1"],
                               format_func=lambda k: OUTCOMES.get(k, k))
    with t_col2:
        show_only_desert = st.checkbox("Show food desert tracts only", value=False)

    display_df = filtered.copy()
    if show_only_desert:
        display_df = display_df[display_df["LILATracts_1And10"] == 1]

    display_df = display_df[[
        "CensusTract", "State", "County", "FoodDesert", "UrbanLabel",
        "DIABETES", "OBESITY", "BPHIGH", "CSMOKING",
        "lapop1", "MedianFamilyIncome", "TractHUNV"
    ]].rename(columns={
        "CensusTract": "Tract", "FoodDesert": "Desert Status",
        "UrbanLabel": "Urban/Rural", "lapop1": "Pop >1mi Grocery",
        "MedianFamilyIncome": "Median Income", "TractHUNV": "No-Vehicle HH"
    }).sort_values(sort_by, ascending=False).reset_index(drop=True)

    # Cap rows to stay within Pandas Styler's render limit
    max_rows = 2000
    display_capped = display_df.head(max_rows)
    if len(display_df) > max_rows:
        st.caption(f"Showing top {max_rows:,} of {len(display_df):,} tracts. Narrow filters to see more.")

    pd.set_option("styler.render.max_elements", display_capped.size + 1000)
    st.dataframe(
        display_capped.style.background_gradient(subset=["DIABETES", "OBESITY", "BPHIGH"], cmap="OrRd"),
        use_container_width=True,
        height=450,
    )

    # Correlation matrix for current filter
    st.markdown('<div class="section-header">Correlation Matrix (Current Filter)</div>', unsafe_allow_html=True)
    corr_cols = ["DIABETES", "OBESITY", "BPHIGH", "CSMOKING",
                 "lapop1", "LILATracts_1And10", "MedianFamilyIncome", "TractHUNV"]
    corr_labels = {
        "DIABETES": "Diabetes", "OBESITY": "Obesity", "BPHIGH": "High BP",
        "CSMOKING": "Smoking", "lapop1": ">1mi Pop",
        "LILATracts_1And10": "Food Desert", "MedianFamilyIncome": "Income", "TractHUNV": "No Vehicle"
    }
    corr_data = filtered[corr_cols].dropna().corr()
    corr_data.index   = [corr_labels[c] for c in corr_data.index]
    corr_data.columns = [corr_labels[c] for c in corr_data.columns]

    fig_corr = px.imshow(
        corr_data, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Pearson Correlation Matrix"
    )
    fig_corr.update_layout(
        height=420, paper_bgcolor="white", font_family="IBM Plex Sans",
        margin=dict(l=10, r=10, t=50, b=10),
        coloraxis_colorbar=dict(thickness=12, len=0.7),
    )
    fig_corr.update_traces(textfont_size=11)
    st.plotly_chart(fig_corr, use_container_width=True)