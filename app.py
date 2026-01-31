import os
import streamlit as st
import pandas as pd
from mongo import collection, ping, db
from vaex_loader import load_vaex_from_mongo, DASHBOARD_COLUMNS
from analytics_vaex import (
    trips_by_zone,
    revenue_by_zone,
    trips_by_hour,
    avg_fare_by_vehicle,
    surge_vs_fare,
    weather_impact,
)
from ml_inference import load_model, load_model_from_file, predict_fare

st.set_page_config("Urban Mobility Analytics", layout="wide")
st.title("Urban Mobility Analytics")

try:
    ping()
    st.success("MongoDB connected")
except Exception as e:
    st.warning("MongoDB chưa kết nối được. Tab Analytics cần DB; Fare Prediction (từ file) vẫn dùng được.")

# Mặc định 1000 dòng cho Analytics (server 1GB RAM); có thể tăng trong sidebar
DEFAULT_DATA_LIMIT = 1_000


@st.cache_data(ttl=300)
def get_vaex_data(limit: int):
    return load_vaex_from_mongo(collection, limit, columns=DASHBOARD_COLUMNS)


@st.cache_data(ttl=300)
def get_analytics_results(limit: int):
    """Chỉ chạy khi mở tab Analytics; cache theo limit."""
    df = get_vaex_data(limit)
    return {
        "trips_zone": trips_by_zone(df).set_index("pickup_zone"),
        "revenue_zone": revenue_by_zone(df).set_index("pickup_zone"),
        "trips_hour": trips_by_hour(df).set_index("hour"),
        "avg_fare_vehicle": avg_fare_by_vehicle(df).set_index("vehicle_type"),
        "surge_fare": surge_vs_fare(df).set_index("surge_multiplier"),
        "weather": weather_impact(df),
    }


# Ưu tiên load từ file (nhanh, không cần MongoDB cho model); không có file mới load từ MongoDB
MODEL_FILE = os.getenv("MODEL_PATH", "fare_model.joblib")


@st.cache_resource
def get_model():
    result = load_model_from_file(MODEL_FILE)
    if result is not None:
        return result
    model_col = db["ml_models"]
    return load_model(model_col)


# Lazy: mặc định Fare Prediction → mở app nhanh; Analytics mặc định 1000 dòng
page = st.radio(
    "Chọn trang",
    ["Fare Prediction", "Analytics Dashboard"],
    horizontal=True,
    label_visibility="collapsed",
)
st.divider()

# ===== ANALYTICS – chỉ chạy khi chọn tab này =====
if page == "Analytics Dashboard":
    limit = st.sidebar.number_input(
        "Số dòng tối đa từ DB",
        min_value=500,
        max_value=10000,
        value=DEFAULT_DATA_LIMIT,
        step=500,
        help="Mặc định 1000 dòng (nhẹ cho server). Có thể tăng nếu RAM đủ.",
    )
    try:
        with st.spinner("Đang tải dữ liệu và tính toán...s"):
            charts = get_analytics_results(limit)
    except Exception as e:
        st.error("Không tải được dữ liệu từ MongoDB. Kiểm tra kết nối và .env.")
        st.stop()
    st.subheader("Trips by zone")
    st.bar_chart(charts["trips_zone"])
    st.subheader("Revenue by zone")
    st.bar_chart(charts["revenue_zone"])
    st.subheader("Trips by hour")
    st.line_chart(charts["trips_hour"])
    st.subheader("Average fare by vehicle")
    st.bar_chart(charts["avg_fare_vehicle"])
    st.subheader("Surge vs fare")
    st.line_chart(charts["surge_fare"])
    st.subheader("Weather impact")
    st.dataframe(charts["weather"])

# ===== FARE PREDICTION – không đụng tới 100k dòng =====
elif page == "Fare Prediction":
    result = get_model()
    if result is None:
        st.warning("Chưa có model (không thấy file hoặc MongoDB).")
        st.markdown("Chạy train rồi lưu ra file (app sẽ ưu tiên load từ file):")
        st.code("python ml_train.py", language="bash")
        st.caption(f"Model mặc định: {MODEL_FILE}. Có thể set biến môi trường MODEL_PATH=đường/dẫn/file.joblib. Sau khi train xong, restart Streamlit để load model.")
    else:
        model, features, metrics = result
        st.info(f"Model RMSE: {metrics['rmse']:.2f}, R2: {metrics['r2']:.3f}")

        with st.form("predict"):
            distance = st.number_input("Distance (km)", 0.5, 50.0, 5.0)
            duration = st.number_input("Duration (min)", 1.0, 120.0, 15.0)
            hour = st.slider("Hour", 0, 23, 9)
            dow = st.slider("Day of week", 0, 6, 2)
            surge = st.selectbox("Surge", [1.0, 1.2, 1.5, 2.0])
            submit = st.form_submit_button("Predict")

        if submit:
            input_df = pd.DataFrame([{
                "distance_km": distance,
                "duration_min": duration,
                "hour": hour,
                "day_of_week": dow,
                "surge_multiplier": surge
            }])
            fare = predict_fare(model, features, input_df)
            st.success(f"Predicted fare: {fare:,.0f}")
