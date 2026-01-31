from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv(override=True)

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME", "BigData")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "mobility_trips")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI is missing. Check your .env file")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

def ping():
    client.admin.command("ping")