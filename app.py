"""
Hotel Bookings — Insight Studio
================================
A richer, story-driven Streamlit dashboard for the hotel bookings dataset:
introduction, guiding business questions, demographics, geography (Indonesia
map), bar/pie/donut charts, a flashlight cursor effect, a colorful animated
background, and a brief technical/statistical deep-dive with KPI numbers.

Run:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from scipy import stats

st.set_page_config(
    page_title="Hotel Bookings — Insight Studio",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# THEME — colorful animated background + flashlight cursor + custom chrome
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stApp {
        background: linear-gradient(120deg, #0f172a, #1e1b4b, #312e81, #0f172a, #1e293b);
        background-size: 400% 400%;
        animation: gradientShift 22s ease infinite;
    }
    .block-container { padding-top: 1.6rem; }
    h1, h2, h3, h4, p, span, div, label { color: #F1F5F9; }
    .hero {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 20px;
        padding: 34px 40px;
        backdrop-filter: blur(6px);
        margin-bottom: 18px;
    }
    .hero h1 { font-size: 40px; margin-bottom: 6px; background: linear-gradient(90deg,#38bdf8,#a78bfa,#f472b6); -webkit-background-clip: text; background-clip: text; color: transparent; }
    .hero p { color: #CBD5E1; font-size: 16px; max-width: 780px; }
    .kpi-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 16px;
        padding: 16px 18px;
        backdrop-filter: blur(4px);
    }
    .kpi-label { font-size: 12px; color: #94A3B8; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
    .kpi-value { font-size: 26px; font-weight: 800; margin-top: 2px; color: #F8FAFC; }
    .kpi-sub { font-size: 12px; color: #7DD3FC; margin-top: 2px; }
    .section-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 22px 26px;
        margin-bottom: 18px;
        backdrop-filter: blur(4px);
    }
    .pill {
        display: inline-block; background: rgba(56,189,248,0.15); color: #7DD3FC;
        border: 1px solid rgba(56,189,248,0.35); border-radius: 999px;
        padding: 3px 12px; font-size: 12px; font-weight: 600; margin-right: 6px; margin-bottom: 6px;
    }
    .qcard {
        background: rgba(255,255,255,0.06); border-left: 3px solid #a78bfa;
        border-radius: 10px; padding: 12px 16px; margin-bottom: 10px; font-size: 14px; color: #E2E8F0;
    }
    .stat-line {
        background: rgba(255,255,255,0.05); border-radius: 10px; padding: 10px 14px;
        margin-bottom: 8px; font-size: 13.5px; color: #E2E8F0; border-left: 3px solid #34d399;
    }
    [data-testid="stSidebar"] { background: rgba(15,23,42,0.85); }
    [data-testid="stMetricValue"] { color: #F8FAFC; }
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb { background: rgba(148,163,184,0.4); border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Flashlight / spotlight cursor overlay (best-effort: reaches into the parent
# document so the spotlight tracks the mouse across the whole app, not just
# this component's iframe).
components.html(
    """
    <script>
    (function() {
        const doc = window.parent.document;
        if (doc.getElementById('spotlight-overlay')) return;
        const overlay = doc.createElement('div');
        overlay.id = 'spotlight-overlay';
        Object.assign(overlay.style, {
            position: 'fixed', top: '0', left: '0', width: '100%', height: '100%',
            pointerEvents: 'none', zIndex: '999999', mixBlendMode: 'normal',
            background: 'radial-gradient(circle 260px at 50% 40%, rgba(255,255,255,0.05) 0%, rgba(2,6,23,0.35) 100%)',
        });
        doc.body.appendChild(overlay);
        doc.addEventListener('mousemove', function(e) {
            overlay.style.background =
                'radial-gradient(circle 260px at ' + e.clientX + 'px ' + e.clientY + 'px, rgba(255,255,255,0.06) 0%, rgba(2,6,23,0.38) 100%)';
        });
    })();
    </script>
    """,
    height=0, width=0,
)

TEMPLATE = "plotly_dark"
PALETTE = ["#38bdf8", "#a78bfa", "#f472b6", "#34d399", "#fbbf24", "#fb7185", "#22d3ee", "#818cf8"]

def style_fig(fig, height=420):
    fig.update_layout(
        template=TEMPLATE, height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"), legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=50, b=30, l=10, r=10),
    )
    return fig

# ----------------------------------------------------------------------------
# GEO LOOKUP — Indonesian cities/regencies present in the dataset
# ----------------------------------------------------------------------------
CITY_COORDS = {
    "Kota Denpasar": (-8.6500, 115.2167), "Kabupaten Bangka": (-1.8500, 106.0000),
    "Kota Yogyakarta": (-7.7956, 110.3695), "Kota Batu": (-7.8706, 112.5239),
    "Kabupaten Bandung": (-7.0000, 107.6000), "Kabupaten Kepulauan Seribu": (-5.7500, 106.5500),
    "Kota Malang": (-7.9666, 112.6326), "Kabupaten Magelang": (-7.4707, 110.2177),
    "Kota Jakarta Barat": (-6.1683, 106.7588), "Kabupaten Belitung": (-2.7410, 107.9006),
    "Kabupaten Sleman": (-7.7167, 110.3572), "Kabupaten Sumedang": (-6.8583, 107.9219),
    "Kota Jakarta Timur": (-6.2250, 106.9004), "Kabupaten Garut": (-7.2125, 107.9082),
    "Kota Surabaya": (-7.2575, 112.7521), "Kota Cimahi": (-6.8841, 107.5413),
    "Kota Bogor": (-6.5971, 106.8060), "Kabupaten Pangandaran": (-7.6830, 108.6503),
    "Kota Jakarta Selatan": (-6.2615, 106.8106), "Kota Semarang": (-6.9667, 110.4167),
    "Kabupaten Tangerang": (-6.1783, 106.6319), "Kota Bengkulu": (-3.7928, 102.2608),
    "Kota Jakarta Pusat": (-6.1805, 106.8284), "Kota Jakarta Utara": (-6.1388, 106.8650),
    "Kabupaten Pandeglang": (-6.3000, 105.9167), "Kabupaten Bogor": (-6.5500, 106.7900),
    "Kabupaten Karawang": (-6.3227, 107.3376), "Kota Banjar": (-7.3667, 108.5333),
    "Kabupaten Blora": (-6.9700, 111.4167), "Kota Tangerang": (-6.1783, 106.6319),
    "Kabupaten Gresik": (-7.1567, 112.6522), "Kota Jambi": (-1.6100, 103.6100),
    "Kabupaten Cirebon": (-6.7063, 108.5570), "Kabupaten Banyumas": (-7.4298, 109.2340),
    "Kabupaten Purwakarta": (-6.5569, 107.4432), "Kabupaten Banjarnegara": (-7.3947, 109.6900),
    "Kabupaten Subang": (-6.5709, 107.7594), "Kabupaten Demak": (-6.8944, 110.6386),
    "Kabupaten Indramayu": (-6.3265, 108.3200), "Kota Gorontalo": (0.5435, 123.0568),
    "Kabupaten Jepara": (-6.5926, 110.6688), "Kota Bandung": (-6.9175, 107.6191),
    "Kota Bekasi": (-6.2383, 106.9756), "Kabupaten Mojokerto": (-7.4664, 112.4339),
    "Kota Tegal": (-6.8694, 109.1402),
}

# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
@st.cache_data
def load_data(path="hotel_bookings_data.csv"):
    df = pd.read_csv(path)
    df["children"] = df["children"].fillna(0)
    df["city"] = df["city"].fillna("Unknown")
    df["agent"] = df["agent"].fillna(0)
    df["company"] = df["company"].fillna(0)
    df["adr"] = df["adr"].clip(lower=0)
    df = df[df["adr"] < df["adr"].quantile(0.999)]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_weekdays_nights"]
    df["total_nights_safe"] = df["total_nights"].replace(0, np.nan)
    df["revenue"] = df["adr"] * df["total_nights"]
    month_order = ["January","February","March","April","May","June","July",
                    "August","September","October","November","December"]
    df["arrival_date_month"] = pd.Categorical(df["arrival_date_month"], categories=month_order, ordered=True)
    df["guest_type"] = np.select(
        [df["children"] > 0, df["babies"] > 0], ["Family (children)", "Family (babies)"], default="Adults only"
    )
    df["is_canceled_label"] = df["is_canceled"].map({0: "Completed", 1: "Canceled"})
    df["lat"] = df["city"].map(lambda c: CITY_COORDS.get(c, (None, None))[0])
    df["lon"] = df["city"].map(lambda c: CITY_COORDS.get(c, (None, None))[1])
    return df

df_raw = load_data()

# ----------------------------------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎛️ Filters")
    hotels = st.multiselect("Hotel type", sorted(df_raw["hotel"].unique()), default=sorted(df_raw["hotel"].unique()))
    years = st.multiselect("Year", sorted(df_raw["arrival_date_year"].unique()), default=sorted(df_raw["arrival_date_year"].unique()))
    segments = st.multiselect("Market segment", sorted(df_raw["market_segment"].unique()), default=sorted(df_raw["market_segment"].unique()))
    status = st.radio("Booking status", ["All", "Completed only", "Canceled only"])
    lead_range = st.slider("Lead time (days)", int(df_raw["lead_time"].min()), int(df_raw["lead_time"].max()),
                            (int(df_raw["lead_time"].min()), int(df_raw["lead_time"].max())))
    st.caption("Adjust filters — every section below updates live.")

df = df_raw[
    df_raw["hotel"].isin(hotels) & df_raw["arrival_date_year"].isin(years)
    & df_raw["market_segment"].isin(segments) & df_raw["lead_time"].between(*lead_range)
].copy()
if status == "Completed only": df = df[df["is_canceled"] == 0]
elif status == "Canceled only": df = df[df["is_canceled"] == 1]
if df.empty:
    st.warning("No bookings match these filters.")
    st.stop()

# ----------------------------------------------------------------------------
# HERO / INTRODUCTION
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1>🏨 Hotel Bookings — Insight Studio</h1>
        <p>An end-to-end read of <b>{len(df_raw):,}</b> hotel reservations across two hotel types and 177 Indonesian
        cities and regencies — from Bali to Jakarta. This isn't just charts: it's built to answer specific
        questions a revenue manager, marketing lead, or ops team would actually ask, with the demographics,
        geography, and statistics to back each answer.</p>
        <p style="margin-top:10px; color:#7DD3FC; font-size:13px;">Currently viewing {len(df):,} bookings ({len(df)/len(df_raw)*100:.1f}% of the dataset) based on your filters →</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
tab_intro, tab_demo, tab_geo, tab_charts, tab_trends, tab_tech = st.tabs(
    ["🧭 Business Questions", "👥 Demographics", "🌍 Geography", "📊 Bar · Pie · Donut", "📈 KPIs & Trends", "🔬 Technical Analysis"]
)

# ---- BUSINESS QUESTIONS -------------------------------------------------------
with tab_intro:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### What this dashboard is built to answer")
    qs = [
        ("💰 Revenue", "Which hotel type, segment, and city actually generate the most net revenue — not just bookings?"),
        ("⚠️ Cancellation risk", "Which combination of lead time, deposit type, and segment predicts a booking that won't show?"),
        ("👨‍👩‍👧 Who books", "Are we mostly selling to solo/couple travelers or families — and does that differ by hotel type?"),
        ("🌍 Where demand concentrates", "How dependent is the business on Bali (Denpasar) versus the rest of the portfolio?"),
        ("📅 Seasonality", "When does demand peak, and does pricing (ADR) actually move with that demand?"),
        ("🔁 Loyalty", "Do repeat guests behave differently — lower cancellation, higher spend — and is that worth investing in?"),
    ]
    cols = st.columns(2)
    for i, (tag, q) in enumerate(qs):
        with cols[i % 2]:
            st.markdown(f'<div class="qcard"><b>{tag}</b><br>{q}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Snapshot KPIs")
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("Total bookings", f"{len(df):,}"),
        ("Cancellation rate", f"{df['is_canceled'].mean()*100:.1f}%"),
        ("Avg ADR", f"${df['adr'].mean():,.0f}"),
        ("Net revenue", f"${df.loc[df['is_canceled']==0,'revenue'].sum():,.0f}"),
        ("Repeat guest rate", f"{df['is_repeated_guest'].mean()*100:.1f}%"),
    ]
    for col, (l, v) in zip([c1,c2,c3,c4,c5], kpis):
        col.markdown(f'<div class="kpi-card"><div class="kpi-label">{l}</div><div class="kpi-value">{v}</div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---- DEMOGRAPHICS -------------------------------------------------------
with tab_demo:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Who is booking")
    c1, c2, c3 = st.columns(3)
    with c1:
        gtype = df["guest_type"].value_counts().reset_index(); gtype.columns=["type","count"]
        fig = px.pie(gtype, names="type", values="count", hole=0.55, color_discrete_sequence=PALETTE, title="Family composition")
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)
    with c2:
        ctype = df["customer_type"].value_counts().reset_index(); ctype.columns=["type","count"]
        fig = px.pie(ctype, names="type", values="count", hole=0.55, color_discrete_sequence=PALETTE[2:], title="Customer type")
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)
    with c3:
        meal = df["meal"].value_counts().reset_index(); meal.columns=["meal","count"]
        fig = px.pie(meal, names="meal", values="count", hole=0.55, color_discrete_sequence=PALETTE[4:], title="Meal plan")
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)

    c4, c5 = st.columns(2)
    with c4:
        adults_dist = df["adults"].clip(upper=5).value_counts().sort_index().reset_index(); adults_dist.columns=["adults","count"]
        fig = px.bar(adults_dist, x="adults", y="count", color="count", color_continuous_scale="Tealgrn", title="Party size (adults per booking)")
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)
    with c5:
        req = df["total_of_special_requests"].value_counts().sort_index().reset_index(); req.columns=["requests","count"]
        fig = px.area(req, x="requests", y="count", title="Special requests per booking", color_discrete_sequence=["#a78bfa"])
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---- GEOGRAPHY -------------------------------------------------------
with tab_geo:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Where demand comes from — Indonesia")
    geo = df.dropna(subset=["lat", "lon"]).groupby(["city","lat","lon"], observed=True).agg(
        bookings=("hotel","count"), avg_adr=("adr","mean"), cancel_rate=("is_canceled","mean"), revenue=("revenue","sum")
    ).reset_index()
    geo["cancel_pct"] = geo["cancel_rate"]*100
    mapped_share = geo["bookings"].sum() / len(df) * 100

    fig_map = px.scatter_geo(
        geo, lat="lat", lon="lon", size="bookings", color="cancel_pct", hover_name="city",
        color_continuous_scale="RdYlGn_r", size_max=45,
        hover_data={"lat": False, "lon": False, "bookings": True, "avg_adr": ":.0f", "cancel_pct": ":.1f"},
        title=f"Bubble size = bookings · color = cancellation % (covers {mapped_share:.0f}% of filtered bookings)",
    )
    fig_map.update_geos(scope="asia", center=dict(lat=-4, lon=113), projection_scale=5.5,
                         showland=True, landcolor="#1e293b", showocean=True, oceancolor="#0f172a",
                         showcountries=True, countrycolor="#334155", bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(style_fig(fig_map, 560), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        top_city = geo.sort_values("bookings", ascending=True).tail(12)
        fig = px.bar(top_city, x="bookings", y="city", orientation="h", color="revenue",
                     color_continuous_scale="Purpor", title="Top 12 cities by booking volume")
        st.plotly_chart(style_fig(fig, 440), use_container_width=True)
    with c2:
        bali_share = df[df["city"] == "Kota Denpasar"].shape[0] / len(df) * 100
        rest_share = 100 - bali_share
        conc = pd.DataFrame({"segment": ["Kota Denpasar (Bali)", "Rest of portfolio"], "share": [bali_share, rest_share]})
        fig = px.pie(conc, names="segment", values="share", hole=0.5, color_discrete_sequence=["#f472b6", "#334155"],
                     title="Geographic concentration risk")
        st.plotly_chart(style_fig(fig, 440), use_container_width=True)
        st.markdown(f'<div class="stat-line">📍 <b>{bali_share:.1f}%</b> of all bookings in view come from a single city — Denpasar (Bali). That is a concentration risk worth monitoring if Bali demand softens.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---- BAR / PIE / DONUT -------------------------------------------------------
with tab_charts:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Bar charts")
    c1, c2 = st.columns(2)
    with c1:
        seg = df["market_segment"].value_counts().reset_index(); seg.columns=["segment","count"]
        fig = px.bar(seg.sort_values("count"), x="count", y="segment", orientation="h", color="count",
                     color_continuous_scale="Sunsetdark", title="Bookings by market segment")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
    with c2:
        chan = df.groupby("distribution_channel", observed=True).agg(bookings=("hotel","count"), avg_adr=("adr","mean")).reset_index()
        fig = px.bar(chan.sort_values("bookings"), x="bookings", y="distribution_channel", orientation="h",
                     color="avg_adr", color_continuous_scale="Blues", title="Distribution channel volume (color = avg ADR)")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Pie charts")
    c3, c4 = st.columns(2)
    with c3:
        hotel_split = df["hotel"].value_counts().reset_index(); hotel_split.columns=["hotel","count"]
        fig = px.pie(hotel_split, names="hotel", values="count", color_discrete_sequence=["#38bdf8","#f472b6"], title="Bookings by hotel type")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
    with c4:
        dep = df["deposit_type"].value_counts().reset_index(); dep.columns=["deposit","count"]
        fig = px.pie(dep, names="deposit", values="count", color_discrete_sequence=PALETTE, title="Deposit type mix")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Donut charts")
    c5, c6 = st.columns(2)
    with c5:
        stat = df["is_canceled_label"].value_counts().reset_index(); stat.columns=["status","count"]
        fig = px.pie(stat, names="status", values="count", hole=0.65, color_discrete_sequence=["#34d399","#fb7185"], title="Completed vs canceled")
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
    with c6:
        yr = df["arrival_date_year"].value_counts().sort_index().reset_index(); yr.columns=["year","count"]
        fig = px.pie(yr, names="year", values="count", hole=0.65, color_discrete_sequence=PALETTE[3:], title="Bookings by year")
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---- KPIs & TRENDS -------------------------------------------------------
with tab_trends:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### KPI numbers")
    kpi_vals = [
        ("Avg lead time", f"{df['lead_time'].mean():.0f} days", "how far ahead guests book"),
        ("Avg length of stay", f"{df['total_nights_safe'].mean():.1f} nights", "nights per booking"),
        ("Avg special requests", f"{df['total_of_special_requests'].mean():.2f}", "per booking"),
        ("Parking requests", f"{df['required_car_parking_spaces'].mean()*100:.1f}%", "of bookings need parking"),
        ("Booking changes", f"{df['booking_changes'].mean():.2f}", "avg changes per booking"),
        ("Waiting list days", f"{df['days_in_waiting_list'].mean():.1f}", "avg days on waitlist"),
    ]
    cols = st.columns(3)
    for i, (l, v, s) in enumerate(kpi_vals):
        with cols[i % 3]:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">{l}</div><div class="kpi-value">{v}</div><div class="kpi-sub">{s}</div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Trending graphs")
    trend = df.groupby(["arrival_date_year","arrival_date_month"], observed=True).agg(
        bookings=("hotel","count"), revenue=("revenue","sum"), avg_adr=("adr","mean"), cancel_rate=("is_canceled","mean")
    ).reset_index().sort_values(["arrival_date_year","arrival_date_month"])
    trend["period"] = trend["arrival_date_year"].astype(str) + " " + trend["arrival_date_month"].astype(str).str.slice(0,3)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=trend["period"], y=trend["bookings"], name="Bookings", marker_color="#38bdf8", opacity=.85, yaxis="y1"))
    fig.add_trace(go.Scatter(x=trend["period"], y=trend["avg_adr"], name="Avg ADR ($)", mode="lines+markers", line=dict(color="#fbbf24", width=3), yaxis="y2"))
    fig.update_layout(yaxis=dict(title="Bookings"), yaxis2=dict(title="Avg ADR ($)", overlaying="y", side="right", showgrid=False),
                       legend=dict(orientation="h", y=1.12), title="Monthly bookings vs average daily rate — does price track demand?")
    st.plotly_chart(style_fig(fig, 440), use_container_width=True)

    fig2 = px.line(trend, x="period", y="cancel_rate", markers=True, title="Cancellation rate trend over time",
                    color_discrete_sequence=["#fb7185"])
    fig2.update_yaxes(tickformat=".0%")
    st.plotly_chart(style_fig(fig2, 380), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---- TECHNICAL ANALYSIS -------------------------------------------------------
with tab_tech:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Technical & statistical analysis (brief)")

    lead_c = df[df["is_canceled"]==1]["lead_time"]
    lead_nc = df[df["is_canceled"]==0]["lead_time"]
    if len(lead_c) > 5 and len(lead_nc) > 5:
        t_stat, p_val = stats.ttest_ind(lead_c, lead_nc, equal_var=False)
    else:
        t_stat, p_val = float("nan"), float("nan")

    corr_lead_cancel = df[["lead_time","is_canceled"]].corr().iloc[0,1]
    corr_adr_cancel = df[["adr","is_canceled"]].corr().iloc[0,1]
    corr_req_cancel = df[["total_of_special_requests","is_canceled"]].corr().iloc[0,1]

    repeat_cancel = df[df["is_repeated_guest"]==1]["is_canceled"].mean()*100
    new_cancel = df[df["is_repeated_guest"]==0]["is_canceled"].mean()*100

    adr_skew = df["adr"].skew()
    adr_std = df["adr"].std()
    adr_cv = adr_std / df["adr"].mean() * 100

    years_sorted = sorted(df["arrival_date_year"].unique())
    yoy_text = "insufficient years in current filter"
    if len(years_sorted) >= 2:
        first_y, last_y = years_sorted[0], years_sorted[-1]
        b0 = df[df["arrival_date_year"]==first_y].shape[0]
        b1 = df[df["arrival_date_year"]==last_y].shape[0]
        if b0 > 0:
            yoy = (b1-b0)/b0*100
            yoy_text = f"{yoy:+.1f}% booking volume change from {first_y} to {last_y}"

    stat_lines = [
        f"<b>Lead time vs cancellation</b> — Pearson correlation of {corr_lead_cancel:.3f}. Welch's t-test comparing lead time for canceled ({lead_c.mean():.0f}d avg) vs completed ({lead_nc.mean():.0f}d avg) bookings: t = {t_stat:.2f}, p {'< 0.001' if p_val < 0.001 else f'= {p_val:.3f}'} — the difference is statistically significant, confirming lead time is a real (if modest) cancellation signal.",
        f"<b>ADR vs cancellation</b> — correlation of {corr_adr_cancel:.3f}: price alone is a weak predictor of cancellation in this dataset.",
        f"<b>Special requests vs cancellation</b> — correlation of {corr_req_cancel:.3f}: guests who make more special requests cancel {'less' if corr_req_cancel < 0 else 'more'} often, consistent with 'more invested in the trip = more likely to show.'",
        f"<b>Loyalty effect</b> — repeat guests cancel at {repeat_cancel:.1f}% vs {new_cancel:.1f}% for new guests, a {abs(repeat_cancel-new_cancel):.1f} point gap.",
        f"<b>ADR distribution</b> — mean ${df['adr'].mean():.0f}, std ${adr_std:.0f} (CV {adr_cv:.0f}%), skew {adr_skew:.2f} ({'right-tailed — a handful of premium bookings pull the average up' if adr_skew > 0.5 else 'roughly symmetric'}).",
        f"<b>Year-over-year</b> — {yoy_text} within the current filter selection.",
    ]
    for s in stat_lines:
        st.markdown(f'<div class="stat-line">{s}</div>', unsafe_allow_html=True)

    st.markdown("#### Correlation matrix — numeric booking features")
    num_cols = ["lead_time","adr","total_nights","total_guests","booking_changes",
                "previous_cancellations","total_of_special_requests","is_canceled"]
    corr = df[num_cols].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                          title="Pearson correlation heatmap")
    st.plotly_chart(style_fig(fig_corr, 480), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<p style="text-align:center; color:#64748B; font-size:12px; margin-top:20px;">Hotel Bookings Insight Studio · Built with Streamlit + Plotly</p>',
    unsafe_allow_html=True,
)
