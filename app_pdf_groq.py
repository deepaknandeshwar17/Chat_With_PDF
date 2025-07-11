

import streamlit as st
import os
import shutil
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import sqlite3

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# DB SETUP
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

# DB UTILS
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

# PDF Processing
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         reader = PdfReader(pdf)
#         for page in reader.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text
#     return text

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
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
        st.error("❗ No vector store found. Please upload and process PDFs.")
        st.stop()
    return FAISS.load_local("vectorstore", embeddings, index_name="index", allow_dangerous_deserialization=True)

# Chain builder
def get_conversational_chain():
    template = """
    You are a helpful assistant. Use the provided context to answer the question.
    If full information is not available, try to give a partial answer or summarize each side as best as you can.

    Answer the questions as detailed as possible from the provided context.
    If the answer is not in the provided context, just say "Answer is not available in the context.
    
    "

    Context:
    {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    model = ChatGroq(model_name="llama3-70b-8192", temperature=0.3)
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)
    # return load_qa_chain(model, chain_type="map_reduce", prompt=prompt)


# def get_conversational_chain():
#     question_prompt = PromptTemplate(
#         template="""
#         Use the following context to answer the question.
#         If you don't know the answer, just say that you don't know — don't make anything up.

#         Context:
#         {context}

#         Question: {question}
#         """,
#         input_variables=["context", "question"]
#     )

#     combine_prompt = PromptTemplate(
#         template="""
#         You are given the following extracted summaries from different documents.
#         Combine them to answer the original question in a detailed and thoughtful way.

#         Summaries:
#         {summaries}

#         Question: {question}

#         Final Answer:
#         """,
#         input_variables=["summaries", "question"]
#     )

#     model = ChatGroq(model_name="llama3-70b-8192", temperature=0.3)

#     return load_qa_chain(
#         model,
#         chain_type="map_reduce",
#         question_prompt=question_prompt,
#         combine_prompt=combine_prompt
#     )





# Response function
def respond_to_user(question):
    chain = get_conversational_chain()
    faiss_file = os.path.join("vectorstore", "index.faiss")

    try:
        if os.path.exists(faiss_file):
            db = load_vector_store()
            docs = db.similarity_search(question)
            result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
            answer = result["output_text"]
        else:
            # direct LLM fallback (Groq LLaMA3)
            llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.3)
            answer = llm.invoke(question).content

        save_chat(question, answer)
        return answer
    except Exception as e:
        return f"🚨 Error: {str(e)}"

# Streamlit UI
st.set_page_config("🧠 Chat with PDF - Groq LLaMA3")
st.title("📄 Chat with PDF using Groq LLaMA3")

with st.sidebar:
    st.subheader("📤 Upload PDFs")
    pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
    if st.button("📚 Process PDFs") and pdf_docs:
        with st.spinner("🔄 Processing PDFs..."):
            raw_text = get_pdf_text(pdf_docs)
            chunks = get_text_chunks(raw_text)
            get_vector_store(chunks)
            st.success("✅ Vector store created!")

    if st.button("🧹 Clear Chat History"):
        delete_all_chats()
        st.success("✅ Chat history cleared")
        st.experimental_rerun()

st.markdown("---")
st.subheader("💬 Ask Questions")

for user, bot in get_chats():
    with st.chat_message("user"):
        st.markdown(user)
    with st.chat_message("assistant"):
        st.markdown(bot)

# Chat input
if prompt := st.chat_input("Ask a question from the PDF or anything..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner("💬 Thinking..."):
        reply = respond_to_user(prompt)
    with st.chat_message("assistant"):
        st.markdown(reply)
