import os
import pickle
import joblib
from gridfs import GridFS


def load_model_from_file(path: str):
    """
    Load model từ file .joblib (đã train bằng ml_train.py --output path).
    Trả về (model, features, metrics) hoặc None nếu file không tồn tại.
    """
    if not path or not os.path.isfile(path):
        return None
    data = joblib.load(path)
    return data["model"], data["features"], data["metrics"]


def load_model(model_collection):
    """
    Load model fare_regression từ MongoDB (hỗ trợ cả blob trong doc và GridFS).
    Trả về (model, features, metrics) hoặc None nếu chưa có model (chưa chạy ml_train.py).
    """
    doc = model_collection.find_one(
        {"model_type": "fare_regression"},
        sort=[("trained_at", -1)]
    )
    if doc is None:
        return None
    if "model_file_id" in doc:
        fs = GridFS(model_collection.database)
        model_bytes = fs.get(doc["model_file_id"]).read()
        model = pickle.loads(model_bytes)
    else:
        model = pickle.loads(doc["model_blob"])
    return model, doc["features"], doc["metrics"]

def predict_fare(model, features, input_df):
    X = input_df[features]
    return model.predict(X)[0]
