Commands to Start The project


Step 1: Pull docker image by running "docker pull qdrant/qdrant"

Step 2: Run the docker image by "docker run -p 6333:6333 qdrant/qdrant"

Step 3: Run in terminal "python ingest.py"

Step 4: Run in terminal "python .\retriever.py"

Step 5: Run in terminal uvicorn rag:app