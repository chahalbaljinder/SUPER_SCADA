from pymongo import MongoClient
import pandas as pd

# Connect to MongoDB (replace the URI with your MongoDB connection string)
client = MongoClient("mongodb://localhost:27017/")  # Use your MongoDB URI
db = client["metro_data"]  # Replace with your database name
collection = db["7_211ObjEvent"]  # Replace with your collection name

# Read data from the collection
documents = collection.find()  # This retrieves all documents in the collection

# Check if any data is available
if collection.count_documents({}) > 0:  # Check if the collection has any documents
    print("Data Available")
    # Iterate over the documents and print them
else:
    print("No Data Available")
    
df = pd.DataFrame(documents)
print("Coverted to Dataframe Successfully")