from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve the OpenAI API key from environment variables
api_key = os.getenv('OPENAI_API_KEY')

# Check if the API key is available
if not api_key:
    raise ValueError("OpenAI API key is not set in environment variables.")

# Initialize OpenAI Embeddings with the API key
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Setup Qdrant client and vector store
url = "http://localhost:6333"
client = QdrantClient(url=url, prefer_grpc=False)
db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")

print("Vector DB Initialized")

# Example query
query = "What is Metastatic disease?"

# Perform similarity search
docs = db.similarity_search_with_score(query=query, k=2)

# Print the results
for i in docs:
    doc, score = i
    print({"score": score, "content": doc.page_content, "metadata": doc.metadata})
