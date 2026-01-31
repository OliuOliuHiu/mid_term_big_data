"""
Train RandomForest dự đoán fare. Có thể lưu ra file (.joblib) và/hoặc MongoDB.
Chạy: python ml_train.py [--output file.joblib] [--limit N] [--sample N]
  --output    : đường dẫn file lưu model (app ưu tiên load từ file nếu có)
  --limit N   : max dòng từ MongoDB (mặc định 100000)
  --no-mongo  : không lưu vào MongoDB, chỉ ghi file
"""
import argparse
import pickle
import datetime
import numpy as np
import pandas as pd
import joblib
from pymongo import MongoClient
from gridfs import GridFS
from dotenv import load_dotenv
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

load_dotenv()

FEATURES = [
    "distance_km",
    "duration_min",
    "hour",
    "day_of_week",
    "surge_multiplier",
]
TARGET = "fare"
PROJECTION = {"_id": 0, **{f: 1 for f in FEATURES}, **{TARGET: 1}}


def main():
    parser = argparse.ArgumentParser(description="Train fare regression model, save to MongoDB")
    parser.add_argument("--output", "-o", type=str, default="fare_model.joblib", help="Path to save model file (default: fare_model.joblib)")
    parser.add_argument("--no-mongo", action="store_true", help="Only save to file, do not save to MongoDB")
    parser.add_argument("--limit", type=int, default=100_000, help="Max documents from MongoDB (0 = no limit)")
    parser.add_argument("--sample", type=int, default=0, help="After load, sample N rows for training (0 = use all)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--n-estimators", type=int, default=300, help="RandomForest n_estimators")
    args = parser.parse_args()

    print("Đang kết nối MongoDB...")
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client[os.getenv("MONGO_DB_NAME")]
    data_col = db["mobility_trips"]
    model_col = db["ml_models"]

    print("Đang tải dữ liệu từ MongoDB (có thể mất vài chục giây)...")
    cursor = data_col.find({}, PROJECTION)
    if args.limit > 0:
        cursor = cursor.limit(args.limit)
    df = pd.DataFrame(list(cursor))
    print(f"Đã tải {len(df):,} dòng.")

    df = df[FEATURES + [TARGET]].dropna()
    if len(df) == 0:
        raise SystemExit("No data after dropna. Check MongoDB collection and projection.")

    if args.sample > 0 and len(df) > args.sample:
        df = df.sample(n=args.sample, random_state=42)
        print(f"Sampled {args.sample} rows for training.")

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    print("Đang train RandomForest (vài phút, tùy số dòng & n_estimators)...")
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    metrics = {"rmse": rmse, "r2": r2}

    # Lưu ra file (app sẽ ưu tiên load từ file nếu tồn tại)
    print(f"Đang ghi file {args.output}...")
    joblib.dump(
        {"model": model, "features": FEATURES, "metrics": metrics},
        args.output,
    )
    print(f"Đã lưu model vào file: {args.output}")

    if not args.no_mongo:
        print("Đang lưu model vào MongoDB (GridFS)...")
        model_bytes = pickle.dumps(model)
        fs = GridFS(db)
        file_id = fs.put(model_bytes, filename="fare_regression.pkl")
        model_col.delete_many({"model_type": "fare_regression"})
        model_col.insert_one({
            "model_type": "fare_regression",
            "model_file_id": file_id,
            "features": FEATURES,
            "metrics": metrics,
            "trained_at": datetime.datetime.utcnow(),
        })
        print("Đã lưu vào MongoDB.")
    else:
        print("Bỏ qua MongoDB (--no-mongo).")

    print("Xong.")
    print(f"RMSE: {rmse:.2f} | R2: {r2:.3f} | rows used: {len(df):,}")


if __name__ == "__main__":
    main()
