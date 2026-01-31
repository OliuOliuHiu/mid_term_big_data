"""
Load dữ liệu từ MongoDB sang Vaex. Dùng projection chỉ lấy cột cần cho dashboard để giảm RAM & network.
"""
import pandas as pd
import numpy as np
import vaex

# Cột cần cho Analytics Dashboard (ít hơn = load nhanh hơn)
DASHBOARD_COLUMNS = [
    "pickup_time",
    "pickup_zone",
    "fare",
    "hour",
    "vehicle_type",
    "surge_multiplier",
    "weather_condition",
]


def load_vaex_from_mongo(collection, limit: int, columns=None):
    """
    Load từ MongoDB sang Vaex.
    :param collection: pymongo collection
    :param limit: số dòng tối đa
    :param columns: list tên cột cần load, hoặc None = load full (chậm hơn)
    """
    if columns is None:
        projection = {"_id": 0}
    else:
        projection = {"_id": 0, **{c: 1 for c in columns}}

    cursor = collection.find({}, projection).limit(limit)
    data = list(cursor)
    if not data:
        return vaex.from_pandas(pd.DataFrame(), copy_index=False)

    dfp = pd.DataFrame(data)

    if "pickup_time" in dfp.columns:
        dfp["pickup_time"] = pd.to_datetime(dfp["pickup_time"], errors="coerce")
    dfp = dfp.replace({np.nan: None})

    return vaex.from_pandas(dfp, copy_index=False)
