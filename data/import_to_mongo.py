"""
Import file CSV urban_mobility_trips.csv vào MongoDB.
Chạy từ thư mục gốc project: python data/import_to_mongo.py
Cần có .env với MONGO_URI, MONGO_DB_NAME, MONGO_COLLECTION_NAME.
"""
import os
import sys
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

# load .env từ thư mục gốc project
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME", "BigData")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "mobility_trips")

if not MONGO_URI:
    print("Lỗi: Chưa có MONGO_URI trong .env")
    sys.exit(1)

# CSV nằm cùng thư mục với script này
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "urban_mobility_trips.csv")

if not os.path.isfile(CSV_PATH):
    print(f"Lỗi: Không tìm thấy {CSV_PATH}. Chạy data/create_data.py trước.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
df["pickup_time"] = pd.to_datetime(df["pickup_time"], errors="coerce")
records = df.to_dict("records")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
coll = db[COLLECTION_NAME]
coll.delete_many({})
coll.insert_many(records)
print(f"Đã import {len(records)} dòng vào {DB_NAME}.{COLLECTION_NAME}")
