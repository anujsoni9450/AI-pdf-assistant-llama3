import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Use the same local model as ingestion.py
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_retriever():
    """
    Connects to the persistent database and returns a retriever object.
    """
    # Load the database from the same folder where ingestion saved it
    if os.path.exists("./db_storage"):
        vector_db = Chroma(
            persist_directory="./db_storage",
            embedding_function=embeddings
        )
        # Search for the top 3 most relevant chunks
        return vector_db.as_retriever(search_kwargs={"k": 3})
    else:
        return None