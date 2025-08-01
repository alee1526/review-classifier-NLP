import logging
from pymongo import MongoClient

def set_mongo_connection():
    # MongoDB connection
    uri = "mongodb+srv://alediaz1526:oINGhgHEvmzntC5m@cluster0.kj6lxoi.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

    try:
        client = MongoClient(uri)
        client.admin.command("ping")
        logging.info("Connected to MongoDB successfully.")
    except Exception as e:
        logging.error(f"Connection failed: {e}")
        exit(1)
    
    return client