from langchain import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
import os

from starlette.responses import JSONResponse

# Load environment variables
load_dotenv()

# Retrieve the OpenAI API key from environment variables
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OpenAI API key is not set in environment variables.")

print("Apikey", api_key)

app = FastAPI()

# Setup template and static files directories
templates = Jinja2Templates(directory="templates")
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration for embeddings
config = {
    'max_new_tokens': 1024,
    'context_length': 2048,
    'repetition_penalty': 1.1,
    'temperature': 0.1,
    'top_k': 50,
    'top_p': 0.9,
    'stream': True,
    'threads': int(os.cpu_count() / 2)
}


embeddings = OpenAIEmbeddings(api_key=api_key)

url = "http://localhost:6333"
client = QdrantClient(url=url, prefer_grpc=False)

db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")

prompt_template = """
Use the following pieces of information to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])


qa_chain = LLMChain(llm=ChatOpenAI(model="gpt-4", api_key = api_key, max_tokens=1024),
                    prompt=PromptTemplate.from_template(prompt_template))



@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/get_response")
async def get_response(query: str = Form(...)):
    relevant_docs = db.similarity_search(query)
    context = ""
    relevant_images = []
    for d in relevant_docs:
        if d.metadata['type'] == 'text':
            context += '[text]' + d.metadata['original_content']
        elif d.metadata['type'] == 'table':
            context += '[table]' + d.metadata['original_content']
        elif d.metadata['type'] == 'image':
            context += '[image]' + d.page_content
            relevant_images.append(d.metadata['original_content'])
    result = qa_chain.run({'context': context, 'question': query})
    return JSONResponse({"relevant_images": relevant_images[0], "result": result})