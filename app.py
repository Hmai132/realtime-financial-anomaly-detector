import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time


st.set_page_config(
    page_title="Fintech Anomaly Detection System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    div[data-testid="column"]:nth-of-type(4) div[data-testid="stMetricValue"] > div {
        font-size: 1.45rem !important;
        white-space: nowrap !important;
    }
</style>
""", unsafe_allow_html=True)

if "market_data" not in st.session_state:
    st.session_state.market_data = []

st.title("⚡ Real-Time Financial Telemetry & Anomaly Detector")
st.caption("Hệ thống kiểm định Z-Score, dải Bollinger Bands và giám sát biến động dữ liệu trực tiếp qua Public API")

with st.sidebar:
    st.header("⚙️ Cấu hình thuật toán")
    window_size = st.slider("Cửa sổ trượt (Window N)", min_value=5, max_value=30, value=12)
    k_threshold = st.slider("Ngưỡng độ lệch (|Z| > k)", min_value=1.0, max_value=3.0, value=1.8, step=0.1)
    refresh_rate = st.slider("Chu kỳ quét API (giây)", min_value=2, max_value=10, value=3)
    
    st.markdown("---")
    with st.expander("📖 Cơ sở lý thuyết & Mô hình", expanded=True):
        st.markdown("**1. Công thức chuẩn hóa Z-Score:**")
        st.latex(r"Z_t = \frac{x_t - \mu_t}{\sigma_t}")
        st.markdown("""
        * $\mu_t$: Trung bình động ($SMA_N$)
        * $\sigma_t$: Độ lệch chuẩn động ($STD_N$)
        """)
        st.markdown("**2. Điều kiện kích hoạt cảnh báo:**")
        st.info("Kích hoạt cảnh báo khi giá trị tuyệt đối $|Z_t| > k$ (vượt ngoài dải biên Bollinger Bands).")

def get_live_market_data():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_vol=true"
    try:
        res = requests.get(url, timeout=2.5).json()
        price = float(res["bitcoin"]["usd"])
        vol = float(res["bitcoin"]["usd_24h_vol"])
        return price, vol
    except Exception:
        prev_price = st.session_state.market_data[-1]["Price"] if st.session_state.market_data else 65000.0
        return float(prev_price + np.random.normal(0, 8)), 32000000000.0

current_price, current_vol = get_live_market_data()
current_timestamp = datetime.now().strftime("%H:%M:%S")

st.session_state.market_data.append({
    "Timestamp": current_timestamp,
    "Price": current_price,
    "Volume_24h": current_vol
})

if len(st.session_state.market_data) > 40:
    st.session_state.market_data.pop(0)

df = pd.DataFrame(st.session_state.market_data)

df["SMA"] = df["Price"].rolling(window=window_size, min_periods=1).mean()
df["STD"] = df["Price"].rolling(window=window_size, min_periods=1).std().fillna(0)
df["Upper_Band"] = df["SMA"] + (k_threshold * df["STD"])
df["Lower_Band"] = df["SMA"] - (k_threshold * df["STD"])
df["Z_Score"] = (df["Price"] - df["SMA"]) / np.where(df["STD"] == 0, 1.0, df["STD"])
df["Is_Anomaly"] = df["Z_Score"].abs() > k_threshold
df["Delta"] = df["Price"].diff().fillna(0)

latest = df.iloc[-1]
c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.1, 1.2])

c1.metric("Giá Bitcoin (USD)", f"${latest['Price']:,.2f}", delta=f"{latest['Delta']:+,.2f} USD")
c2.metric("Trung bình trượt (SMA)", f"${latest['SMA']:,.2f}")
c3.metric("Độ biến động (σ)", f"±{latest['STD']:.2f}")

status_text = "🚨 DỊ BIỆT" if latest["Is_Anomaly"] else "BÌNH THƯỜNG"
c4.metric(
    "Kiểm định Z-Score", 
    status_text, 
    delta=f"Z = {latest['Z_Score']:.2f}"
)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["Timestamp"], y=df["Upper_Band"],
    mode="lines",
    line=dict(color="rgba(148, 163, 184, 0.35)", width=1),
    showlegend=False
))
fig.add_trace(go.Scatter(
    x=df["Timestamp"], y=df["Lower_Band"],
    mode="lines",
    line=dict(color="rgba(148, 163, 184, 0.35)", width=1),
    fill='tonexty',
    fillcolor='rgba(56, 189, 248, 0.12)',
    name="Vùng an toàn thống kê",
    hoverinfo="skip"
))
point_colors = np.where(df["Is_Anomaly"], "#ef4444", "#38bdf8")
fig.add_trace(go.Scatter(
    x=df["Timestamp"], y=df["Price"],
    mode="markers+lines",
    marker=dict(size=8, color=point_colors, line=dict(width=1.5, color="#ffffff")),
    line=dict(color="rgba(56, 189, 248, 0.5)", width=1),
    name="Dữ liệu quan trắc"
))
anomalies = df[df["Is_Anomaly"]]
if not anomalies.empty:
    fig.add_trace(go.Scatter(
        x=anomalies["Timestamp"], y=anomalies["Price"],
        mode="markers+text",
        marker=dict(symbol="star", size=15, color="#ef4444"),
        text=["Dị biệt" for _ in range(len(anomalies))],
        textposition="top center",
        name="Anomaly Points"
    ))

fig.update_layout(
    template="plotly_dark",
    height=420,
    margin=dict(l=15, r=15, t=30, b=20),
    xaxis=dict(title="Mốc thời gian", gridcolor="#1e293b"),
    yaxis=dict(title="Giá USD ($)", gridcolor="#1e293b"),
    legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
)
st.plotly_chart(fig, width='stretch')

st.subheader("📋 Bảng kiểm soát dữ liệu thời gian thực (Audit Log)")
display_df = df[["Timestamp", "Price", "SMA", "STD", "Z_Score", "Is_Anomaly"]].copy()
display_df["Price"] = display_df["Price"].map("${:,.2f}".format)
display_df["SMA"] = display_df["SMA"].map("${:,.2f}".format)
display_df["STD"] = display_df["STD"].map("{:.2f}".format)
display_df["Z_Score"] = display_df["Z_Score"].map("{:.2f}".format)
display_df["Is_Anomaly"] = display_df["Is_Anomaly"].map({True: "Dị biệt", False: "Bình thường"})

col_table, col_export = st.columns([0.8, 0.2])
with col_table:
    st.dataframe(display_df.iloc[::-1].head(8), width='stretch')

with col_export:
    st.markdown("**Xuất báo cáo:**")
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Tải file CSV",
        data=csv_data,
        file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        width='stretch'
    )

time.sleep(refresh_rate)
st.rerun()
