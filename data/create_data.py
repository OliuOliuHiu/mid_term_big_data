import numpy as np
import pandas as pd

np.random.seed(42)

n_rows = 300_000

zones = [
    "District_1", "District_3", "District_5", "Binh_Thanh",
    "Phu_Nhuan", "District_10", "District_7", "Thu_Duc"
]

vehicle_types = ["bike", "car", "suv"]
payment_methods = ["cash", "card", "wallet"]
weather_conditions = ["clear", "rain", "heavy_rain"]

pickup_time = pd.date_range(
    start="2025-01-01",
    periods=n_rows,
    freq="s"
)

duration_min = np.random.gamma(2.5, 6, n_rows)
distance_km = duration_min * np.random.uniform(0.3, 0.5, n_rows)

df = pd.DataFrame({
    "pickup_time": pickup_time,
    "pickup_zone": np.random.choice(zones, n_rows),
    "dropoff_zone": np.random.choice(zones, n_rows),
    "distance_km": np.round(distance_km, 2),
    "duration_min": np.round(duration_min, 1),
    "vehicle_type": np.random.choice(vehicle_types, n_rows, p=[0.5, 0.4, 0.1]),
    "payment_method": np.random.choice(payment_methods, n_rows),
    "weather_condition": np.random.choice(
        weather_conditions, n_rows, p=[0.6, 0.3, 0.1]
    )
})

# ===== Time-based features =====
df["hour"] = df["pickup_time"].dt.hour
df["day_of_week"] = df["pickup_time"].dt.weekday
df["is_peak_hour"] = df["hour"].isin([7,8,9,16,17,18,19]).astype(int)

# ===== Demand level =====
zone_demand_map = {
    zone: np.random.randint(1, 4)
    for zone in zones
}
df["zone_demand_level"] = df["pickup_zone"].map(zone_demand_map)

# ===== Trip distance bucket =====
df["trip_distance_bucket"] = pd.cut(
    df["distance_km"],
    bins=[0, 3, 7, 15, 100],
    labels=[0,1,2,3]
).astype(int)

# ===== Surge multiplier (target) =====
df["surge_multiplier"] = (
    1
    + df["is_peak_hour"] * np.random.uniform(0.2, 0.6, n_rows)
    + (df["weather_condition"] != "clear") * np.random.uniform(0.1, 0.5, n_rows)
    + (df["zone_demand_level"] / 5)
).round(2)

# ===== Fare =====
base_fare = 8000
df["fare"] = (
    base_fare + df["distance_km"] * 12000
) * df["surge_multiplier"]

df["fare"] = df["fare"].round(0)

# # ===== ML Target: high revenue =====
# threshold = df["fare"].quantile(0.75)
# df["high_revenue_trip"] = (df["fare"] > threshold).astype(int)

df.head()

file_path = "../data/urban_mobility_trips.csv"

df.to_csv(file_path, index = False)