from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import shutil
import sqlite3
from typing import List

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI()

# Initialize DB
def init_db():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_msg TEXT,
            assistant_msg TEXT
        );
    """)
    conn.commit()
    conn.close()

init_db()

# Helper functions
def save_chat(user_msg, assistant_msg):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chats (user_msg, assistant_msg) VALUES (?, ?)", (user_msg, assistant_msg))
    conn.commit()
    conn.close()

def get_chats():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT user_msg, assistant_msg FROM chats")
    rows = cursor.fetchall()
    conn.close()
    return rows

def delete_all_chats():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chats")
    conn.commit()
    conn.close()
    shutil.rmtree("vectorstore", ignore_errors=True)

def clean_text(text):
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")

def get_pdf_text(files):
    text = ""
    for file in files:
        reader = PdfReader(file.file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += clean_text(page_text)
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_texts(chunks, embedding=embeddings)
    db.save_local("vectorstore", index_name="index")

def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    path = os.path.join("vectorstore", "index.faiss")
    if not os.path.exists(path):
        return None
    return FAISS.load_local("vectorstore", embeddings, index_name="index", allow_dangerous_deserialization=True)

def get_conversational_chain():
    template = """
    Answer the questions as detailed as possible from the provided context.
    If the answer is not in the provided context, just say "Answer is not available in the context."

    Context:
    {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    model = ChatGroq(model_name="llama3-70b-8192", temperature=0.3)
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# API Models
class Question(BaseModel):
    query: str

@app.post("/upload-pdf")
def upload_pdfs(files: List[UploadFile] = File(...)):
    text = get_pdf_text(files)
    chunks = get_text_chunks(text)
    get_vector_store(chunks)
    return {"message": "Vector store created from uploaded PDFs."}

@app.post("/ask")
def ask_question(q: Question):
    chain = get_conversational_chain()
    db = load_vector_store()
    try:
        if db:
            docs = db.similarity_search(q.query)
            result = chain({"input_documents": docs, "question": q.query}, return_only_outputs=True)
            answer = result["output_text"]
        else:
            llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.3)
            answer = llm.invoke(q.query).content
        save_chat(q.query, answer)
        return {"answer": answer}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/history")
def history():
    chats = get_chats()
    return {"history": chats}

@app.delete("/history")
def clear_history():
    delete_all_chats()
    return {"message": "Chat history and vectorstore cleared."}
