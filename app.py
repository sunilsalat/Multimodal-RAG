from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.document import Document
from langchain.vectorstores import FAISS
from langchain.retrievers.multi_vector import MultiVectorRetriever
import os
import uuid
import base64
from fastapi import FastAPI, Request, Form, Response, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
import json
from dotenv import load_dotenv
load_dotenv()

images = ['https://crickbook.s3.amazonaws.com/./media/images/figure-1-7.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVRUVVFJD2SCX2WNV%2F20240510%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20240510T070419Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=d7f9571499e2d2a410ccce636532fc34cfab1f4d458a0fdf8df90aa779748849', 'https://crickbook.s3.amazonaws.com/./media/images/figure-1-2.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVRUVVFJD2SCX2WNV%2F20240510%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20240510T070419Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=1ec73c24053d871bbf206a0daccf72b339bad259e8764a6dae33dd3ae60815f2', 'https://crickbook.s3.amazonaws.com/./media/images/figure-1-1.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVRUVVFJD2SCX2WNV%2F20240510%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20240510T070419Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=73641b9176741c2d7cebad8ac0f6cb96a009b544669f3deca642f8cca31677a8', 'https://crickbook.s3.amazonaws.com/./media/images/figure-1-10.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVRUVVFJD2SCX2WNV%2F20240510%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20240510T070419Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=fd6133d3a67bdab9ca2f159866bd77b7fadf49f0fabef320d85631b9dc10ecf2', 'https://crickbook.s3.amazonaws.com/./media/images/figure-1-5.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVRUVVFJD2SCX2WNV%2F20240510%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20240510T070419Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=69d0d4204bc89c7be7a7fd705b94cabffc7dc1603b3428bf5ec82f33da956b61', 'https://crickbook.s3.amazonaws.com/./media/images/figure-1-11.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVRUVVFJD2SCX2WNV%2F20240510%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20240510T070419Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=5b10d66b074c77afe1f64c6ba162b283e9c08d971c242964630a2539be64b447', 'https://crickbook.s3.amazonaws.com/./media/images/figure-1-3.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVRUVVFJD2SCX2WNV%2F20240510%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20240510T070419Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=3eeb706e5c673009b28ad08e376733f4a9f020bae1bacc8dbfa67425b6664aac', 'https://crickbook.s3.amazonaws.com/./media/images/figure-1-9.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVRUVVFJD2SCX2WNV%2F20240510%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20240510T070419Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=a3c2b82bf5325e2ef9dfda0a5dafc0a8454f29ad068aecfd3e61d203df03a31a', 'https://crickbook.s3.amazonaws.com/./media/images/figure-1-4.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVRUVVFJD2SCX2WNV%2F20240510%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20240510T070419Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=f25dcce7f34e210b1c0ea603a7dec28bc8142654cbe8ba820a3ecec7de05fa1c', 'https://crickbook.s3.amazonaws.com/./media/images/figure-1-6.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVRUVVFJD2SCX2WNV%2F20240510%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20240510T070419Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=66dcd43e99b2a0d7235f202c58966eec38e41f142361c73a37a1a6fddfe9ada5', 'https://crickbook.s3.amazonaws.com/./media/images/figure-1-8.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVRUVVFJD2SCX2WNV%2F20240510%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20240510T070419Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=6e5dfd30587b27673f74f2abe8bfa12d9e3fbbaae6d16957c75034e82e297976']

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_api_key = os.getenv("key_secret")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

prompt_template = """You are a vet doctor and an expert in analyzing dog's health.
Answer the question based only on the following context, which can include text, images and tables:
{context}
Question: {question}
Don't answer if you are not sure and decline to answer and say "Sorry, I don't have much information about it."
Just return the helpful answer in as much as detailed possible.
Answer:
"""

qa_chain = LLMChain(llm=ChatOpenAI(model="gpt-4", openai_api_key = openai_api_key, max_tokens=1024),
                    prompt=PromptTemplate.from_template(prompt_template))

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_answer")
async def get_answer(question: str = Form(...)):
    relevant_docs = db.similarity_search(question)
    print(relevant_docs)
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
    # result = qa_chain.run({'context': context, 'question': question})
    # return JSONResponse({"relevant_images": relevant_images[0], "result": result})
    return JSONResponse({"relevant_images": relevant_images[0], "result": context})
