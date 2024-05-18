import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

# Retrieve the OpenAI API key from environment variables
api_key = os.getenv('OPENAI_API_KEY')

print("API Key:", api_key)

# Check if the API key is available
if not api_key:
    raise ValueError("OpenAI API key is not set in environment variables.")

# Initialize OpenAI Embeddings with the API key
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

url = "http://localhost:6333"
client = QdrantClient(url=url, prefer_grpc=False)

loader = DirectoryLoader('data/', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name="vector_db"
)
print("Vector DB Successfully Created!")
