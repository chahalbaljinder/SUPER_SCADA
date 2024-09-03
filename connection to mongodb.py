import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

# MongoDB connection URI
mongo_uri = 'mongodb://localhost:27017/'  # Replace with your MongoDB URI

# Database and collection names
db_name = 'test_database'  # Replace with your database name
collection_name = 'test_collection'  # Replace with your collection name

try:
    # Establish connection to MongoDB
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)  # Timeout for connection
    print("Connected to MongoDB!")

    # Access the database and collection
    db = client[db_name]
    collection = db[collection_name]
    print(f"Accessing collection: {collection_name}")

    # Fetch all documents from the collection
    cursor = collection.find()

    # Convert the cursor to a list of dictionaries
    documents = list(cursor)

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(documents)
    print("Data converted to DataFrame.")

    # Display the head of the DataFrame
    print("Head of the DataFrame:")
    print(df.head())

except ServerSelectionTimeoutError as e:
    print("Failed to connect to MongoDB:", e)
