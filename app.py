import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

# Thiết lập giao diện
st.set_page_config(
    page_title="Real-Time Anomaly Detection Engine",
    page_icon="⚡",
    layout="wide"
)

# Khởi tạo dữ liệu lưu trữ
if "history" not in st.session_state:
    st.session_state.history = []

st.title("⚡ Real-Time Statistical Anomaly Detector")
st.caption("Hệ thống phát hiện bất thường thị trường theo thời gian thực (Z-Score & Bollinger Bands)")

# Thanh điều khiển bên trái
with st.sidebar:
    st.header("⚙️ Tham số phân tích")
    window_size = st.slider("Cửa sổ trượt (Rolling Window)", 5, 30, 10)
    k_factor = st.slider("Ngưỡng cảnh báo (|Z| > k)", 1.0, 3.0, 1.8, step=0.1)
    refresh_rate = st.slider("Tần suất quét API (giây)", 2, 8, 3)
    st.markdown("---")
    st.latex(r"Z = \frac{x_t - \mu}{\sigma}")

# Lấy dữ liệu trực tiếp qua API CoinGecko
def fetch_live_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_vol=true"
    try:
        r = requests.get(url, timeout=2.5).json()
        return float(r["bitcoin"]["usd"]), float(r["bitcoin"]["usd_24h_vol"])
    except:
        prev = st.session_state.history[-1]["price"] if st.session_state.history else 65000.0
        return prev + np.random.normal(0, 15), 35000000000.0

current_p, current_vol = fetch_live_price()
current_t = datetime.now().strftime("%H:%M:%S")

st.session_state.history.append({
    "time": current_t,
    "price": current_p,
    "volume": current_vol
})

if len(st.session_state.history) > 35:
    st.session_state.history.pop(0)

df = pd.DataFrame(st.session_state.history)

# Tính toán khoa học: Bollinger Bands & Z-Score
df["SMA"] = df["price"].rolling(window=window_size, min_periods=1).mean()
df["STD"] = df["price"].rolling(window=window_size, min_periods=1).std().fillna(0)
df["Upper_Band"] = df["SMA"] + (k_factor * df["STD"])
df["Lower_Band"] = df["SMA"] - (k_factor * df["STD"])
df["Z_Score"] = (df["price"] - df["SMA"]) / np.where(df["STD"] == 0, 1, df["STD"])
df["Is_Anomaly"] = df["Z_Score"].abs() > k_factor

# Thẻ KPI
latest = df.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Giá Bitcoin (USD)", f"${latest['price']:,.2f}")
c2.metric("Trung bình trượt SMA", f"${latest['SMA']:,.2f}")
c3.metric("Độ biến động σ", f"±{latest['STD']:.2f}")
c4.metric("Trạng thái", "🚨 BẤT THƯỜNG" if latest["Is_Anomaly"] else "✅ BÌNH THƯỜNG")

# Vẽ biểu đồ khoa học Bollinger Bands
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["time"], y=df["Upper_Band"], mode="lines", line=dict(color="rgba(148, 163, 184, 0.4)", width=1), showlegend=False))
fig.add_trace(go.Scatter(x=df["time"], y=df["Lower_Band"], mode="lines", line=dict(color="rgba(148, 163, 184, 0.4)", width=1), fill='tonexty', fillcolor='rgba(56, 189, 248, 0.12)', name="Vùng tin cậy thống kê"))
fig.add_trace(go.Scatter(x=df["time"], y=df["price"], mode="markers+lines", marker=dict(size=8, color=np.where(df["Is_Anomaly"], "#ef4444", "#38bdf8")), line=dict(color="#38bdf8", width=1), name="Giá thực tế"))

anomalies = df[df["Is_Anomaly"]]
if not anomalies.empty:
    fig.add_trace(go.Scatter(x=anomalies["time"], y=anomalies["price"], mode="markers+text", marker=dict(symbol="star", size=14, color="#ef4444"), text=["Dị biệt" for _ in range(len(anomalies))], textposition="top center", name="Điểm dị biệt"))

fig.update_layout(template="plotly_dark", height=430, margin=dict(l=10, r=10, t=30, b=20), xaxis=dict(title="Thời gian"), yaxis=dict(title="USD ($)"))
st.plotly_chart(fig, use_container_width=True)

# Tự động cập nhật (Real-time loop)
time.sleep(refresh_rate)
st.rerun()
