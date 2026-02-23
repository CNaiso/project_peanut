import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Claims Intelligence Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%); }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1b2838 100%);
    border-right: 1px solid rgba(100,200,255,0.1);
}
[data-testid="stSidebar"] .css-1d391kg { padding-top: 1rem; }
.metric-card {
    background: linear-gradient(135deg, rgba(30,40,80,0.8), rgba(20,30,60,0.9));
    border: 1px solid rgba(100,200,255,0.15);
    border-radius: 16px; padding: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    backdrop-filter: blur(10px); text-align: center;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.metric-card:hover { transform: translateY(-4px); box-shadow: 0 12px 40px rgba(100,200,255,0.15); }
.metric-value { font-size: 2rem; font-weight: 800; background: linear-gradient(135deg, #00d2ff, #7b2ff7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.metric-label { font-size: 0.85rem; color: #8892b0; margin-top: 0.3rem; text-transform: uppercase; letter-spacing: 1px; }
.section-header {
    font-size: 1.8rem; font-weight: 700;
    background: linear-gradient(90deg, #00d2ff, #7b2ff7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 2rem 0 1rem 0; padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(100,200,255,0.2);
}
.insight-box {
    background: rgba(0,210,255,0.05); border-left: 4px solid #00d2ff;
    padding: 1rem 1.5rem; border-radius: 0 12px 12px 0; margin: 1rem 0;
    color: #ccd6f6; font-size: 0.95rem;
}
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(30,40,80,0.6); border-radius: 8px;
    color: #8892b0; padding: 8px 20px;
    border: 1px solid rgba(100,200,255,0.1);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #00d2ff33, #7b2ff733) !important;
    color: #00d2ff !important; border-color: #00d2ff !important;
}
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(30,40,80,0.7), rgba(20,30,60,0.8));
    border: 1px solid rgba(100,200,255,0.12); border-radius: 12px; padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─── PLOTLY THEME ───────────────────────────────────────────────────────────────
COLORS = ["#00d2ff","#7b2ff7","#ff6b6b","#feca57","#48dbfb","#ff9ff3",
          "#54a0ff","#5f27cd","#01a3a4","#f368e0","#ff9f43","#ee5a24"]
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#ccd6f6"),
    xaxis=dict(gridcolor="rgba(100,200,255,0.07)", zerolinecolor="rgba(100,200,255,0.1)"),
    yaxis=dict(gridcolor="rgba(100,200,255,0.07)", zerolinecolor="rgba(100,200,255,0.1)"),
    margin=dict(l=40, r=40, t=50, b=40),
    hoverlabel=dict(bgcolor="#1a1a2e", font_color="#ccd6f6", bordercolor="#00d2ff"),
)

def styled_fig(fig, h=480):
    fig.update_layout(**PLOT_LAYOUT, height=h)
    return fig

# ─── DATA LOADING ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    grp = pd.read_csv("cleaned_group_death_claims.csv")
    ind = pd.read_csv("cleaned_individual_death_claims.csv")
    combined = pd.concat([grp, ind], ignore_index=True)
    # Standardize insurer names
    name_map = {
        "ABSL": "Aditya Birla Life", "Baj Alz": "Bajaj Allianz",
        "Can HSBC": "Canara HSBC OBC", "Edelws": "Edelweiss Tokio",
        "Fut Genli": "Future Generali", "HDFC": "HDFC Life",
        "ICICI": "ICICI Prudential", "Indiafirst": "India First",
        "Kotak": "Kotak Mahindra", "Max": "Max Life",
        "PNB Metlife": "PNB Met Life", "Pramerica": "Pramerica Life",
        "Reliance": "Reliance Nippon", "SUD": "Star Union",
        "Sahara": "Sahara Life", "Exide": "Exide Life",
        "Ageas": "Ageas Federal", "Aegon": "Aegon",
        "PVT.": "Private Total", "Industry": "Industry Total",
    }
    for df in [grp, ind, combined]:
        df["life_insurer"] = df["life_insurer"].replace(name_map)
    aggregate_labels = ["Industry Total", "Private Total"]
    grp_clean = grp[~grp["life_insurer"].isin(aggregate_labels)]
    ind_clean = ind[~ind["life_insurer"].isin(aggregate_labels)]
    combined_clean = combined[~combined["life_insurer"].isin(aggregate_labels)]
    return grp, ind, combined, grp_clean, ind_clean, combined_clean

grp, ind, combined, grp_clean, ind_clean, combined_clean = load_data()

# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ **Claims Intelligence**")
    st.markdown("---")
    page = st.radio("📑 **Navigation**", [
        "🏠 Executive Dashboard",
        "📊 Exploratory Analysis",
        "🏢 Insurer Deep-Dive",
        "📈 Trend & Time-Series",
        "🔬 Statistical Analysis",
        "🤖 Predictive Modelling",
    ], index=0)
    st.markdown("---")
    all_years = sorted(combined_clean["year"].unique())
    selected_years = st.multiselect("📅 **Filter Years**", all_years, default=all_years)
    all_insurers = sorted(combined_clean["life_insurer"].unique())
    selected_insurers = st.multiselect("🏢 **Filter Insurers**", all_insurers, default=all_insurers)
    st.markdown("---")
    st.caption("Data: IRDAI Life Insurance Death Claims · 2017-2022")

# Apply filters
mask = (combined_clean["year"].isin(selected_years)) & (combined_clean["life_insurer"].isin(selected_insurers))
df = combined_clean[mask].copy()
grp_f = df[df["category"] == "Group Death Claims"]
ind_f = df[df["category"] == "Individual Death Claims"]

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
if page == "🏠 Executive Dashboard":
    st.markdown('<div class="section-header">🏠 Executive Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">High-level overview of life insurance death claims across India — covering both Group and Individual categories.</div>', unsafe_allow_html=True)

    # KPI row
    total_claims = df["total_claims_no"].sum()
    total_paid = df["claims_paid_no"].sum()
    avg_paid_ratio = df["claims_paid_ratio_no"].mean() * 100
    total_amt = df["total_claims_amt"].sum()
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1, c2, c3, c4],
        [f"{total_claims:,.0f}", f"{total_paid:,.0f}", f"{avg_paid_ratio:.1f}%", f"₹{total_amt:,.0f} Cr"],
        ["Total Claims Filed", "Total Claims Paid", "Avg Settlement Rate", "Total Claim Amount"]
    ):
        col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    # Donut: Group vs Individual split
    with col1:
        cat_summary = df.groupby("category").agg(total=("total_claims_no","sum")).reset_index()
        fig = px.pie(cat_summary, values="total", names="category", hole=0.55,
                     color_discrete_sequence=["#00d2ff","#7b2ff7"], title="Claims Split: Group vs Individual")
        fig.update_traces(textinfo="percent+label", textfont_size=13)
        st.plotly_chart(styled_fig(fig, 400), use_container_width=True)

    # Bar: Year-over-year total claims
    with col2:
        yoy = df.groupby("year").agg(total=("total_claims_no","sum"), paid=("claims_paid_no","sum")).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=yoy["year"], y=yoy["total"], name="Total Claims", marker_color="#7b2ff7"))
        fig.add_trace(go.Bar(x=yoy["year"], y=yoy["paid"], name="Claims Paid", marker_color="#00d2ff"))
        fig.update_layout(title="Year-over-Year Claims Volume", barmode="group", legend=dict(x=0.01, y=0.99))
        st.plotly_chart(styled_fig(fig, 400), use_container_width=True)

    col3, col4 = st.columns(2)
    # Settlement rate trend
    with col3:
        trend = df.groupby(["year","category"]).agg(ratio=("claims_paid_ratio_no","mean")).reset_index()
        fig = px.line(trend, x="year", y="ratio", color="category", markers=True,
                      color_discrete_sequence=["#00d2ff","#ff6b6b"],
                      title="Settlement Rate Trend by Category")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(styled_fig(fig, 400), use_container_width=True)

    # Top 10 insurers by volume
    with col4:
        top10 = df.groupby("life_insurer").agg(t=("total_claims_no","sum")).nlargest(10,"t").reset_index()
        fig = px.bar(top10, x="t", y="life_insurer", orientation="h",
                     color="t", color_continuous_scale=["#7b2ff7","#00d2ff"],
                     title="Top 10 Insurers by Claim Volume")
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(styled_fig(fig, 400), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORATORY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📊 Exploratory Analysis":
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["Distributions","Correlations","Composition","Outliers"])

    with tab1:
        st.subheader("Distribution of Key Metrics")
        metric = st.selectbox("Select metric", [
            "claims_paid_ratio_no","claims_paid_ratio_amt","claims_repudiated_rejected_ratio_no",
            "claims_pending_ratio_no","total_claims_no","claims_paid_no","total_claims_amt"])
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x=metric, color="category", nbins=30, marginal="box",
                               color_discrete_sequence=["#00d2ff","#ff6b6b"],
                               title=f"Distribution of {metric}")
            st.plotly_chart(styled_fig(fig), use_container_width=True)
        with c2:
            fig = px.violin(df, y=metric, x="category", color="category", box=True, points="all",
                            color_discrete_sequence=["#00d2ff","#ff6b6b"],
                            title=f"Violin Plot — {metric}")
            st.plotly_chart(styled_fig(fig), use_container_width=True)

    with tab2:
        st.subheader("Correlation Matrix")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto",
                        title="Pearson Correlation Heatmap")
        st.plotly_chart(styled_fig(fig, 700), use_container_width=True)

        st.subheader("Scatter: Claims Paid vs Repudiated")
        fig = px.scatter(df, x="claims_paid_ratio_no", y="claims_repudiated_rejected_ratio_no",
                         color="category", size="total_claims_no", hover_name="life_insurer",
                         color_discrete_sequence=["#00d2ff","#ff6b6b"],
                         title="Paid Rate vs Rejection Rate (bubble = volume)")
        st.plotly_chart(styled_fig(fig), use_container_width=True)

    with tab3:
        st.subheader("Claims Composition Breakdown")
        comp = df.groupby("year").agg(
            paid=("claims_paid_no","sum"), repudiated=("claims_repudiated_no","sum"),
            rejected=("claims_rejected_no","sum"), unclaimed=("claims_unclaimed_no","sum"),
            pending=("claims_pending_end_no","sum")).reset_index()
        fig = px.bar(comp, x="year", y=["paid","repudiated","rejected","unclaimed","pending"],
                     color_discrete_sequence=COLORS, title="Claims Outcome Composition by Year")
        fig.update_layout(barmode="stack", legend_title="Outcome")
        st.plotly_chart(styled_fig(fig), use_container_width=True)

        st.subheader("Amount-Weighted Composition")
        comp2 = df.groupby("year").agg(
            paid=("claims_paid_amt","sum"), repudiated=("claims_repudiated_amt","sum"),
            rejected=("claims_rejected_amt","sum"), unclaimed=("claims_unclaimed_amt","sum"),
            pending=("claims_pending_end_amt","sum")).reset_index()
        fig = px.bar(comp2, x="year", y=["paid","repudiated","rejected","unclaimed","pending"],
                     color_discrete_sequence=COLORS, title="Claim Amount Composition by Year (₹ Cr)")
        fig.update_layout(barmode="stack", legend_title="Outcome")
        st.plotly_chart(styled_fig(fig), use_container_width=True)

    with tab4:
        st.subheader("Outlier Detection — Box Plots")
        out_metric = st.selectbox("Metric for outlier analysis", [
            "claims_paid_ratio_no","total_claims_no","claims_pending_ratio_no",
            "claims_repudiated_rejected_ratio_no","total_claims_amt"], key="out_m")
        fig = px.box(df, x="year", y=out_metric, color="category", points="all",
                     color_discrete_sequence=["#00d2ff","#ff6b6b"],
                     title=f"Outliers in {out_metric} by Year")
        st.plotly_chart(styled_fig(fig, 500), use_container_width=True)

        # IQR-based outlier table
        Q1, Q3 = df[out_metric].quantile(0.25), df[out_metric].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[out_metric] < Q1 - 1.5*IQR) | (df[out_metric] > Q3 + 1.5*IQR)]
        st.markdown(f"**{len(outliers)} outlier records** detected (IQR method)")
        if len(outliers) > 0:
            st.dataframe(outliers[["life_insurer","year","category",out_metric]].sort_values(out_metric, ascending=False), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — INSURER DEEP-DIVE
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🏢 Insurer Deep-Dive":
    st.markdown('<div class="section-header">🏢 Insurer Deep-Dive</div>', unsafe_allow_html=True)

    chosen = st.selectbox("Select an insurer", sorted(df["life_insurer"].unique()))
    ins_df = df[df["life_insurer"] == chosen]
    st.markdown(f'<div class="insight-box">Showing {len(ins_df)} records for <b>{chosen}</b> across {ins_df["year"].nunique()} years.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Settlement Rate", f"{ins_df['claims_paid_ratio_no'].mean()*100:.1f}%")
    c2.metric("Total Claims", f"{ins_df['total_claims_no'].sum():,.0f}")
    c3.metric("Total Paid Amount", f"₹{ins_df['claims_paid_amt'].sum():,.1f} Cr")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(ins_df, x="year", y=["claims_paid_no","claims_repudiated_no","claims_rejected_no","claims_pending_end_no"],
                     color_discrete_sequence=COLORS, barmode="stack", facet_col="category",
                     title=f"{chosen} — Claims Outcomes Over Time")
        st.plotly_chart(styled_fig(fig, 450), use_container_width=True)
    with col2:
        fig = px.line(ins_df, x="year", y="claims_paid_ratio_no", color="category", markers=True,
                      color_discrete_sequence=["#00d2ff","#ff6b6b"],
                      title=f"{chosen} — Settlement Rate Trend")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(styled_fig(fig, 450), use_container_width=True)

    # Radar chart for latest year
    st.subheader("Performance Radar — Latest Year")
    latest = ins_df[ins_df["year"] == ins_df["year"].max()]
    if not latest.empty:
        radar_metrics = ["claims_paid_ratio_no","claims_repudiated_rejected_ratio_no","claims_pending_ratio_no"]
        labels = ["Settlement Rate","Rejection Rate","Pending Rate"]
        for _, row in latest.iterrows():
            vals = [row[m] for m in radar_metrics] + [row[radar_metrics[0]]]
            fig = go.Figure(go.Scatterpolar(r=vals, theta=labels+[labels[0]], fill='toself',
                                            fillcolor="rgba(0,210,255,0.15)", line_color="#00d2ff",
                                            name=row["category"]))
        fig.update_layout(title=f"Radar — {chosen}", polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, gridcolor="rgba(100,200,255,0.1)", tickformat=".0%"),
            angularaxis=dict(gridcolor="rgba(100,200,255,0.1)")))
        st.plotly_chart(styled_fig(fig, 420), use_container_width=True)

    # Comparative ranking
    st.subheader("Comparative Ranking (All Insurers — Latest Year)")
    latest_all = df[df["year"] == df["year"].max()]
    rank_df = latest_all.groupby("life_insurer").agg(avg_rate=("claims_paid_ratio_no","mean")).reset_index()
    rank_df = rank_df.sort_values("avg_rate", ascending=True)
    rank_df["color"] = rank_df["life_insurer"].apply(lambda x: "#00d2ff" if x == chosen else "#3a3f5c")
    fig = px.bar(rank_df, x="avg_rate", y="life_insurer", orientation="h",
                 color="color", color_discrete_map="identity",
                 title="Settlement Rate Ranking (Latest Year)")
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(showlegend=False)
    st.plotly_chart(styled_fig(fig, max(400, len(rank_df)*22)), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — TREND & TIME-SERIES
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📈 Trend & Time-Series":
    st.markdown('<div class="section-header">📈 Trend & Time-Series Analysis</div>', unsafe_allow_html=True)

    # Multi-insurer trend
    st.subheader("Multi-Insurer Trend Comparison")
    trend_metric = st.selectbox("Trend metric", [
        "claims_paid_ratio_no","total_claims_no","claims_paid_amt",
        "claims_repudiated_rejected_ratio_no","claims_pending_ratio_no"])
    compare = st.multiselect("Compare insurers", sorted(df["life_insurer"].unique()),
                             default=sorted(df["life_insurer"].unique())[:5])
    if compare:
        tdf = df[df["life_insurer"].isin(compare)]
        fig = px.line(tdf, x="year", y=trend_metric, color="life_insurer", markers=True,
                      color_discrete_sequence=COLORS, title=f"{trend_metric} Trend Comparison")
        if "ratio" in trend_metric:
            fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(styled_fig(fig, 480), use_container_width=True)

    # Heatmap: Insurers x Years
    st.subheader("Heatmap: Insurer × Year")
    heat_metric = st.selectbox("Heatmap metric", [
        "claims_paid_ratio_no","total_claims_no","claims_repudiated_rejected_ratio_no"], key="hm")
    pivot = df.groupby(["life_insurer","year"])[heat_metric].mean().reset_index()
    pivot = pivot.pivot(index="life_insurer", columns="year", values=heat_metric).fillna(0)
    fig = px.imshow(pivot, text_auto=".2f", color_continuous_scale="Viridis", aspect="auto",
                    title=f"{heat_metric} — Insurer × Year Heatmap")
    st.plotly_chart(styled_fig(fig, max(500, len(pivot)*20)), use_container_width=True)

    # Growth rate analysis
    st.subheader("Year-on-Year Growth Rate — Total Claims")
    yoy = df.groupby("year").agg(t=("total_claims_no","sum")).reset_index()
    yoy["growth"] = yoy["t"].pct_change() * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(x=yoy["year"], y=yoy["growth"], marker_color=[
        "#ff6b6b" if v < 0 else "#00d2ff" for v in yoy["growth"].fillna(0)]))
    fig.update_layout(title="YoY Growth Rate (%)", yaxis_title="Growth %")
    st.plotly_chart(styled_fig(fig, 380), use_container_width=True)

    # Area chart — cumulative claims
    st.subheader("Cumulative Claims Over Time")
    cum = df.groupby(["year","category"]).agg(total=("total_claims_no","sum")).reset_index()
    cum = cum.sort_values("year")
    for cat in cum["category"].unique():
        mask_c = cum["category"] == cat
        cum.loc[mask_c, "cumulative"] = cum.loc[mask_c, "total"].cumsum()
    fig = px.area(cum, x="year", y="cumulative", color="category",
                  color_discrete_sequence=["#00d2ff","#7b2ff7"],
                  title="Cumulative Claims Filed Over Time")
    st.plotly_chart(styled_fig(fig, 420), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 5 — STATISTICAL ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Statistical Analysis":
    st.markdown('<div class="section-header">🔬 Statistical Analysis</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Descriptive Stats","Hypothesis Testing","Regression Diagnostics"])

    with tab1:
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe().T.style.format("{:.4f}").background_gradient(cmap="Blues"), use_container_width=True)
        st.subheader("Skewness & Kurtosis")
        num_cols = df.select_dtypes(include=np.number).columns
        sk = pd.DataFrame({"Skewness": df[num_cols].skew(), "Kurtosis": df[num_cols].kurtosis()})
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(sk.reset_index(), x="index", y="Skewness", color="Skewness",
                         color_continuous_scale="RdBu_r", title="Feature Skewness")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(styled_fig(fig, 420), use_container_width=True)
        with c2:
            fig = px.bar(sk.reset_index(), x="index", y="Kurtosis", color="Kurtosis",
                         color_continuous_scale="Plasma", title="Feature Kurtosis")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(styled_fig(fig, 420), use_container_width=True)

    with tab2:
        st.subheader("Hypothesis Tests")
        st.markdown("**Mann-Whitney U Test:** Is the settlement rate distribution different between Group & Individual claims?")
        grp_rates = grp_f["claims_paid_ratio_no"].dropna()
        ind_rates = ind_f["claims_paid_ratio_no"].dropna()
        if len(grp_rates) > 0 and len(ind_rates) > 0:
            stat_u, p_u = stats.mannwhitneyu(grp_rates, ind_rates, alternative="two-sided")
            st.markdown(f"U-statistic = **{stat_u:.2f}**, p-value = **{p_u:.6f}**")
            st.markdown("✅ Significant difference" if p_u < 0.05 else "❌ No significant difference", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Kruskal-Wallis Test:** Does settlement rate differ across years?")
        groups = [g["claims_paid_ratio_no"].dropna().values for _, g in df.groupby("year")]
        if len(groups) > 1:
            stat_k, p_k = stats.kruskal(*groups)
            st.markdown(f"H-statistic = **{stat_k:.2f}**, p-value = **{p_k:.6f}**")
            st.markdown("✅ Significant difference across years" if p_k < 0.05 else "❌ No significant difference across years")

        st.markdown("---")
        st.markdown("**Shapiro-Wilk Normality Test** on Settlement Rate")
        sample = df["claims_paid_ratio_no"].dropna()
        if len(sample) > 3:
            stat_s, p_s = stats.shapiro(sample[:5000])
            st.markdown(f"W-statistic = **{stat_s:.4f}**, p-value = **{p_s:.6f}**")
            st.markdown("❌ Not normally distributed" if p_s < 0.05 else "✅ Approximately normal")

    with tab3:
        st.subheader("OLS Regression Diagnostics")
        st.markdown("Regressing `claims_paid_ratio_no` on numeric predictors.")
        features = ["total_claims_no","claims_intimated_no","claims_pending_start_no"]
        X = df[features].dropna()
        y = df.loc[X.index, "claims_paid_ratio_no"]
        if len(X) > 10:
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression().fit(X, y)
            preds = lr.predict(X)
            residuals = y - preds
            c1, c2 = st.columns(2)
            with c1:
                fig = px.scatter(x=preds, y=residuals, labels={"x":"Predicted","y":"Residual"},
                                 title="Residual Plot", color_discrete_sequence=["#00d2ff"])
                fig.add_hline(y=0, line_dash="dash", line_color="#ff6b6b")
                st.plotly_chart(styled_fig(fig, 400), use_container_width=True)
            with c2:
                fig = px.histogram(residuals, nbins=30, title="Residual Distribution",
                                   color_discrete_sequence=["#7b2ff7"], marginal="box")
                st.plotly_chart(styled_fig(fig, 400), use_container_width=True)

            st.markdown(f"**R² = {r2_score(y, preds):.4f}** · Coefficients: {dict(zip(features, lr.coef_.round(6)))}")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 6 — PREDICTIVE MODELLING
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Predictive Modelling":
    st.markdown('<div class="section-header">🤖 Predictive Modelling</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Build and compare ML models to predict the claims settlement rate from operational metrics.</div>', unsafe_allow_html=True)

    target = st.selectbox("🎯 Target Variable", ["claims_paid_ratio_no","claims_paid_ratio_amt"])
    feature_cols = ["total_claims_no","claims_intimated_no","claims_pending_start_no",
                    "claims_pending_start_amt","claims_intimated_amt","total_claims_amt",
                    "claims_repudiated_no","claims_rejected_no","claims_unclaimed_no",
                    "claims_pending_end_no"]
    selected_features = st.multiselect("🔧 Features", feature_cols, default=feature_cols[:6])

    if len(selected_features) < 2:
        st.warning("Select at least 2 features.")
        st.stop()

    # Encode category
    ml_df = df.copy()
    ml_df["is_group"] = (ml_df["category"] == "Group Death Claims").astype(int)
    feat_final = selected_features + ["is_group"]
    X = ml_df[feat_final].dropna()
    y = ml_df.loc[X.index, target].dropna()
    X = X.loc[y.index]

    test_size = st.slider("Test split %", 10, 40, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge (α=1)": Ridge(alpha=1),
        "Lasso (α=0.001)": Lasso(alpha=0.001),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
    }

    results = []
    preds_dict = {}
    with st.spinner("Training models..."):
        for name, model in models.items():
            use_scaled = name in ["Linear Regression","Ridge (α=1)","Lasso (α=0.001)"]
            model.fit(X_train_s if use_scaled else X_train, y_train)
            p = model.predict(X_test_s if use_scaled else X_test)
            preds_dict[name] = p
            results.append({
                "Model": name,
                "R²": r2_score(y_test, p),
                "RMSE": np.sqrt(mean_squared_error(y_test, p)),
                "MAE": mean_absolute_error(y_test, p),
            })

    res_df = pd.DataFrame(results).sort_values("R²", ascending=False)

    # Model comparison chart
    st.subheader("📊 Model Comparison")
    fig = make_subplots(rows=1, cols=3, subplot_titles=["R² Score","RMSE","MAE"])
    fig.add_trace(go.Bar(x=res_df["Model"], y=res_df["R²"], marker_color=COLORS[:5], showlegend=False), 1, 1)
    fig.add_trace(go.Bar(x=res_df["Model"], y=res_df["RMSE"], marker_color=COLORS[:5], showlegend=False), 1, 2)
    fig.add_trace(go.Bar(x=res_df["Model"], y=res_df["MAE"], marker_color=COLORS[:5], showlegend=False), 1, 3)
    fig.update_xaxes(tickangle=30)
    st.plotly_chart(styled_fig(fig, 450), use_container_width=True)

    st.dataframe(res_df.style.format({"R²":"{:.4f}","RMSE":"{:.6f}","MAE":"{:.6f}"}).background_gradient(
        subset=["R²"], cmap="Greens").background_gradient(subset=["RMSE","MAE"], cmap="Reds_r"),
        use_container_width=True)

    # Best model details
    best_name = res_df.iloc[0]["Model"]
    st.success(f"🏆 Best Model: **{best_name}** — R² = {res_df.iloc[0]['R²']:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        best_preds = preds_dict[best_name]
        fig = px.scatter(x=y_test, y=best_preds, labels={"x":"Actual","y":"Predicted"},
                         title=f"Actual vs Predicted — {best_name}", color_discrete_sequence=["#00d2ff"])
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                                 mode="lines", line=dict(dash="dash", color="#ff6b6b"), name="Perfect"))
        st.plotly_chart(styled_fig(fig, 440), use_container_width=True)

    with col2:
        resid = y_test.values - best_preds
        fig = px.histogram(resid, nbins=25, title=f"Prediction Error Distribution — {best_name}",
                           color_discrete_sequence=["#7b2ff7"], marginal="rug")
        st.plotly_chart(styled_fig(fig, 440), use_container_width=True)

    # Feature Importance (tree models)
    st.subheader("🌲 Feature Importance (Tree-Based Models)")
    for mname in ["Random Forest","Gradient Boosting"]:
        m = models[mname]
        imp = pd.DataFrame({"Feature": feat_final, "Importance": m.feature_importances_})
        imp = imp.sort_values("Importance", ascending=True)
        fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale=["#1a1a2e","#00d2ff"],
                     title=f"Feature Importance — {mname}")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(styled_fig(fig, 380), use_container_width=True)

    # Learning curves
    st.subheader("📉 Learning Curves")
    best_model = models[best_name]
    train_sizes = np.linspace(0.1, 1.0, 8)
    train_scores_list, test_scores_list = [], []
    for frac in train_sizes:
        n = max(2, int(len(X_train) * frac))
        Xsub, ysub = X_train.iloc[:n], y_train.iloc[:n]
        use_scaled = best_name in ["Linear Regression","Ridge (α=1)","Lasso (α=0.001)"]
        m_clone = type(best_model)(**best_model.get_params())
        if use_scaled:
            sc2 = StandardScaler().fit(Xsub)
            m_clone.fit(sc2.transform(Xsub), ysub)
            train_scores_list.append(r2_score(ysub, m_clone.predict(sc2.transform(Xsub))))
            test_scores_list.append(r2_score(y_test, m_clone.predict(sc2.transform(X_test))))
        else:
            m_clone.fit(Xsub, ysub)
            train_scores_list.append(r2_score(ysub, m_clone.predict(Xsub)))
            test_scores_list.append(r2_score(y_test, m_clone.predict(X_test)))

    lc_df = pd.DataFrame({"Training Size": (train_sizes * len(X_train)).astype(int),
                           "Train R²": train_scores_list, "Test R²": test_scores_list})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lc_df["Training Size"], y=lc_df["Train R²"],
                             mode="lines+markers", name="Train", line=dict(color="#00d2ff")))
    fig.add_trace(go.Scatter(x=lc_df["Training Size"], y=lc_df["Test R²"],
                             mode="lines+markers", name="Test", line=dict(color="#ff6b6b")))
    fig.update_layout(title=f"Learning Curve — {best_name}", xaxis_title="Training Samples", yaxis_title="R²")
    st.plotly_chart(styled_fig(fig, 400), use_container_width=True)

# ─── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#4a5568;font-size:0.8rem;">'
    '🛡️ Insurance Claims Intelligence Suite · Built with Streamlit & Plotly · IRDAI Data 2017-2022</p>',
    unsafe_allow_html=True)
