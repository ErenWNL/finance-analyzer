from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Connect to MongoDB
client = MongoClient(os.getenv('MONGODB_URI'))
db = client[os.getenv('MONGODB_DB_NAME', 'finance_analyzer')]

# Test the connection and create database
try:
    # The ismaster command is cheap and does not require auth
    client.admin.command('ismaster')
    print("Successfully connected to MongoDB!")
    
    # Create a test collection and insert a document
    test_collection = db.test_collection
    test_collection.insert_one({"test": "connection"})
    print("Successfully created test document!")
    
    # List all databases
    print("\nAvailable databases:")
    for db_name in client.list_database_names():
        print(f"- {db_name}")
        
    # List collections in finance_analyzer
    print("\nCollections in finance_analyzer:")
    for collection in db.list_collection_names():
        print(f"- {collection}")
        
except Exception as e:
    print("Error connecting to MongoDB:", e) 