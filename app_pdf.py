# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os 

# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv


# load_dotenv()

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text= text + page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=100)
#     chunk = text_splitter.split_text(text)
#     return chunk

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding = embeddings)
#     vector_store.save_local("faiss_index", index_name="index")

# def get_conversational_chain():
#     prompt_template="""
    # Answer the questions as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in the provided context 
    # just say, "answer is not available in the context, don't provide the wrong answer\n\n
    # context:\n {context}?\n
    # Question: \n{question}\n

    # Answer:
#     """

#     model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt = prompt)
#     return chain

# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     new_db = FAISS.load_local("faiss_index", embeddings, index_name="index",allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)

#     chain = get_conversational_chain()

#     response = chain (
#         {"input_documents":docs, "question": user_question},
#         return_only_outputs=True
#     )

#     print(response)
#     st.write("Reply: ", response["output_text"])


# def main():
#     st.set_page_config("Chat with multiple PDF")
#     st.header("Chat with PDF using Gemini")

#     user_question = st.text_input("Ask a question from the PDF lines")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload Your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Done")

# if __name__=="__main__":
#     main()














# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# import re

# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# def clean_text(text):
#     return re.sub(r'[\ud800-\udfff]|[\x00-\x1F\x7F]', '', text)

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             try:
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += clean_text(page_text)
#             except Exception as e:
#                 print("Error reading page:", e)
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     return text_splitter.split_text(text)

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index", index_name="index")

# def get_conversational_chain():
#     prompt_template = """
#     Answer the questions as detailed as possible from the provided context.
#     If the answer is not in the context, say "Answer not available in the context."
#     Do not generate incorrect or assumed information.

#     Context:
#     {context}

#     Question:
#     {question}

#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, index_name="index", allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)

#     chain = get_conversational_chain()
#     response = chain(
#         {"input_documents": docs, "question": user_question},
#         return_only_outputs=True
#     )

#     st.write("Reply: ", response["output_text"])

# def main():
#     st.set_page_config("Chat with multiple PDF")
#     st.header("Chat with PDF using Gemini")

#     user_question = st.text_input("Ask a question from the PDF content")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader(
#             "Upload your PDF files and click on 'Submit & Process'",
#             accept_multiple_files=True
#         )

#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)

#                 if len(raw_text.strip()) < 50:
#                     st.warning("Couldn't extract enough readable text from the uploaded PDF(s). Try another.")
#                     return

#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("PDF processed successfully!")

# if __name__ == "__main__":
#     main()





















# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# import re

# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# # Load environment
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # === Utility Functions ===
# def clean_text(text):
#     return re.sub(r'[\ud800-\udfff]|[\x00-\x1F\x7F]', '', text)

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             try:
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += clean_text(page_text)
#             except Exception as e:
#                 print("Error reading page:", e)
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     return text_splitter.split_text(text)

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index", index_name="index")

# def get_conversational_chain():
#     prompt_template = """
#     Answer the questions as detailed as possible from the provided context.
#     If the answer is not in the context, say "Answer not available in the context."

#     Context:
#     {context}

#     Question:
#     {question}

#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# def generate_answer(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, index_name="index", allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain(
#         {"input_documents": docs, "question": user_question},
#         return_only_outputs=True
#     )
#     return response["output_text"]

# # === Streamlit UI ===
# st.set_page_config("PDF Chatbot", layout="wide")
# st.title("📄 Chat with your PDF using Gemini")

# # Upload section
# with st.sidebar:
#     st.header("📂 Upload your PDFs")
#     pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
#     if st.button("Submit & Process"):
#         with st.spinner("Extracting and indexing..."):
#             raw_text = get_pdf_text(pdf_docs)
#             if len(raw_text.strip()) < 50:
#                 st.warning("Couldn't extract enough readable text.")
#                 st.stop()
#             text_chunks = get_text_chunks(raw_text)
#             get_vector_store(text_chunks)
#             st.success("PDF processed successfully! Now start chatting!")

# # Initialize session chat history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Display previous messages
# for role, msg in st.session_state.chat_history:
#     with st.chat_message(role):
#         st.markdown(msg)

# # Chat input at the bottom
# if prompt := st.chat_input("Ask a question from your PDF..."):
#     # Display user message
#     st.chat_message("user").markdown(prompt)
#     st.session_state.chat_history.append(("user", prompt))

#     with st.chat_message("assistant"):
#         try:
#             answer = generate_answer(prompt)
#         except Exception as e:
#             answer = "⚠️ Error occurred while generating response: " + str(e)
#         st.markdown(answer)
#         st.session_state.chat_history.append(("assistant", answer))





















# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# import re
# import shutil

# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # === Utility Functions ===
# def clean_text(text):
#     return re.sub(r'[\ud800-\udfff]|[\x00-\x1F\x7F]', '', text)

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             try:
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += clean_text(page_text)
#             except Exception as e:
#                 print("Error reading page:", e)
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     return text_splitter.split_text(text)

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     # Try to load existing vector store
#     try:
#         vector_store = FAISS.load_local("faiss_index", embeddings, index_name="index", allow_dangerous_deserialization=True)
#         print("✅ Loaded existing index")
#     except:
#         vector_store = FAISS.from_texts([], embedding=embeddings)
#         print("📦 Created new index")

#     # Add new text chunks
#     new_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     vector_store.merge_from(new_store)

#     # Save updated store
#     vector_store.save_local("faiss_index", index_name="index")

# def get_conversational_chain():
#     prompt_template = """
#     Answer the questions as detailed as possible from the provided context.
#     If the answer is not in the context, say "Answer not available in the context."

#     Context:
#     {context}

#     Question:
#     {question}

#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# def generate_answer(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.load_local("faiss_index", embeddings, index_name="index", allow_dangerous_deserialization=True)
#     docs = vector_store.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain(
#         {"input_documents": docs, "question": user_question},
#         return_only_outputs=True
#     )
#     return response["output_text"]

# # === Streamlit UI ===
# st.set_page_config("PDF Chatbot", layout="wide")
# st.title("📄 Chat with your PDF using Gemini")

# # Reset Button
# with st.sidebar:
#     st.header("📂 Upload your PDFs")
#     pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)

#     if st.button("Submit & Process"):
#         with st.spinner("Extracting and indexing..."):
#             raw_text = get_pdf_text(pdf_docs)
#             if len(raw_text.strip()) < 50:
#                 st.warning("Couldn't extract enough readable text.")
#                 st.stop()
#             text_chunks = get_text_chunks(raw_text)
#             get_vector_store(text_chunks)
#             st.success("PDF processed and added to memory!")

#     if st.button("🧹 Clear Memory & Chat"):
#         if os.path.exists("faiss_index"):
#             shutil.rmtree("faiss_index")
#         st.session_state.chat_history = []
#         st.success("Memory and chat history cleared!")

# # Initialize chat history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Display chat history
# for role, msg in st.session_state.chat_history:
#     with st.chat_message(role):
#         st.markdown(msg)

# # Chat input
# if prompt := st.chat_input("Ask a question from your PDF..."):
#     st.chat_message("user").markdown(prompt)
#     st.session_state.chat_history.append(("user", prompt))

#     with st.chat_message("assistant"):
#         try:
#             answer = generate_answer(prompt)
#         except Exception as e:
#             answer = "⚠️ Error: " + str(e)
#         st.markdown(answer)
#         st.session_state.chat_history.append(("assistant", answer))

















# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# import re
# import shutil
# import sqlite3

# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# # ========== Load API Key ==========
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # ========== SQLite ==========
# DB_NAME = "chat_history.db"

# def init_db():
#     conn = sqlite3.connect(DB_NAME)
#     c = conn.cursor()
#     c.execute("""
#         CREATE TABLE IF NOT EXISTS chat (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             role TEXT NOT NULL,
#             message TEXT NOT NULL
#         )
#     """)
#     conn.commit()
#     conn.close()

# def save_chat_to_db(role, message):
#     conn = sqlite3.connect(DB_NAME)
#     c = conn.cursor()
#     c.execute("INSERT INTO chat (role, message) VALUES (?, ?)", (role, message))
#     conn.commit()
#     conn.close()

# def load_chat_from_db():
#     conn = sqlite3.connect(DB_NAME)
#     c = conn.cursor()
#     c.execute("SELECT role, message FROM chat ORDER BY id")
#     data = c.fetchall()
#     conn.close()
#     return data

# def clear_chat_history_db():
#     conn = sqlite3.connect(DB_NAME)
#     c = conn.cursor()
#     c.execute("DELETE FROM chat")
#     conn.commit()
#     conn.close()

# # ========== PDF Text Extraction ==========
# def clean_text(text):
#     return re.sub(r'[\ud800-\udfff]|[\x00-\x1F\x7F]', '', text)

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         reader = PdfReader(pdf)
#         for page in reader.pages:
#             try:
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += clean_text(page_text)
#             except:
#                 continue
#     return text.strip()

# def get_text_chunks(text):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     return splitter.split_text(text)

# # ========== FAISS Vector Store ==========
# def get_vector_store(text_chunks):
#     if not text_chunks:
#         st.warning("No valid text found in the PDFs to index.")
#         return

#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     try:
#         db = FAISS.load_local("faiss_index", embeddings, index_name="index", allow_dangerous_deserialization=True)
#         print("Loaded existing FAISS index.")
#     except:
#         db = None
#         print("No existing FAISS index found, creating new one.")

#     new_db = FAISS.from_texts(text_chunks, embedding=embeddings)

#     if db:
#         db.merge_from(new_db)
#     else:
#         db = new_db

#     db.save_local("faiss_index", index_name="index")

# # ========== Chain Setup ==========
# def get_conversational_chain():
#     prompt_template = """
#     Answer the questions as detailed as possible from the provided context.
#     If the answer is not in the context, say "Answer not available in the context."

#     Context:
#     {context}

#     Question:
#     {question}

#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# def generate_answer(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.load_local("faiss_index", embeddings, index_name="index", allow_dangerous_deserialization=True)
#     docs = vector_store.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     return response["output_text"]

# # ========== UI Setup ==========
# st.set_page_config("Semantic PDF Chat", layout="wide")
# st.title("📄 Chat with Your PDF — with Memory 🧠")

# init_db()

# # ========== Sidebar ==========
# with st.sidebar:
#     st.header("📂 Upload PDFs")
#     pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)

#     if st.button("Submit & Process"):
#         with st.spinner("Indexing PDFs..."):
#             raw_text = get_pdf_text(pdf_docs)
#             if len(raw_text.strip()) < 50:
#                 st.warning("Couldn't extract enough readable text.")
#                 st.stop()
#             text_chunks = get_text_chunks(raw_text)
#             get_vector_store(text_chunks)
#             st.success("PDF processed and added to memory!")

#     if st.button("🧹 Clear All (Chat + Memory)"):
#         if os.path.exists("faiss_index"):
#             shutil.rmtree("faiss_index")
#         clear_chat_history_db()
#         st.session_state.chat_history = []
#         st.success("Memory and chat history cleared.")

# # ========== Chat State ==========
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = load_chat_from_db()

# # ========== Show Chat ==========
# for role, message in st.session_state.chat_history:
#     with st.chat_message(role):
#         st.markdown(message)

# # ========== Chat Input ==========
# if prompt := st.chat_input("Ask something from your PDFs..."):
#     st.chat_message("user").markdown(prompt)
#     st.session_state.chat_history.append(("user", prompt))
#     save_chat_to_db("user", prompt)

#     with st.chat_message("assistant"):
#         try:
#             answer = generate_answer(prompt)
#         except Exception as e:
#             answer = "⚠️ Error: " + str(e)
#         st.markdown(answer)
#         st.session_state.chat_history.append(("assistant", answer))
#         save_chat_to_db("assistant", answer)




















# import streamlit as st
# import os
# import shutil
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_community.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import sqlite3
# import datetime

# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# # DB SETUP
# def init_db():
#     conn = sqlite3.connect("chat_history.db")
#     cursor = conn.cursor()
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS sessions (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             name TEXT NOT NULL,
#             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         );
#     """)
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS chats (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             session_id INTEGER,
#             timestamp TEXT,
#             user_msg TEXT,
#             assistant_msg TEXT,
#             FOREIGN KEY(session_id) REFERENCES sessions(id)
#         );
#     """)
#     conn.commit()
#     conn.close()

# init_db()

# # DB UTILS
# def create_session():
#     conn = sqlite3.connect("chat_history.db")
#     cursor = conn.cursor()
#     session_name = f"Session {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
#     cursor.execute("INSERT INTO sessions (name) VALUES (?)", (session_name,))
#     conn.commit()
#     session_id = cursor.lastrowid
#     conn.close()
#     return session_id

# def get_all_sessions():
#     conn = sqlite3.connect("chat_history.db")
#     cursor = conn.cursor()
#     cursor.execute("SELECT id, name FROM sessions ORDER BY created_at DESC")
#     rows = cursor.fetchall()
#     conn.close()
#     return rows

# def save_chat(session_id, user_msg, assistant_msg):
#     conn = sqlite3.connect("chat_history.db")
#     cursor = conn.cursor()
#     timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     cursor.execute("INSERT INTO chats (session_id, timestamp, user_msg, assistant_msg) VALUES (?, ?, ?, ?)",
#                    (session_id, timestamp, user_msg, assistant_msg))
#     conn.commit()
#     conn.close()

# def get_chats_by_session(session_id):
#     conn = sqlite3.connect("chat_history.db")
#     cursor = conn.cursor()
#     cursor.execute("SELECT timestamp, user_msg, assistant_msg FROM chats WHERE session_id=? ORDER BY timestamp", (session_id,))
#     rows = cursor.fetchall()
#     conn.close()
#     return rows

# def search_chats(session_id, keyword):
#     conn = sqlite3.connect("chat_history.db")
#     cursor = conn.cursor()
#     cursor.execute("""
#         SELECT timestamp, user_msg, assistant_msg FROM chats
#         WHERE session_id=? AND (user_msg LIKE ? OR assistant_msg LIKE ?)
#     """, (session_id, f"%{keyword}%", f"%{keyword}%"))
#     rows = cursor.fetchall()
#     conn.close()
#     return rows

# def delete_session(session_id):
#     conn = sqlite3.connect("chat_history.db")
#     cursor = conn.cursor()
#     cursor.execute("DELETE FROM chats WHERE session_id=?", (session_id,))
#     cursor.execute("DELETE FROM sessions WHERE id=?", (session_id,))
#     conn.commit()
#     conn.close()
#     shutil.rmtree(f"vectorstore/session_{session_id}", ignore_errors=True)

# # PDF Processing
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             if page.extract_text():
#                 text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     return splitter.split_text(text)

# def get_vector_store(text_chunks, session_id):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     if not text_chunks:
#         return
#     store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     store.save_local(f"vectorstore/session_{session_id}", index_name="index")

# def load_vector_store(session_id):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     return FAISS.load_local(f"vectorstore/session_{session_id}", embeddings, index_name="index", allow_dangerous_deserialization=True)

# def get_conversational_chain():
#     prompt_template = """
#     Answer the question using the provided context. Be detailed. If the answer is not in the context, respond with: "Answer is not available in the context."

#     Context:
#     {context}

#     Question: {question}

#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# def respond_to_user(session_id, user_question):
#     vector_db = load_vector_store(session_id)
#     docs = vector_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     save_chat(session_id, user_question, response["output_text"])
#     return response["output_text"]

# # Streamlit UI
# st.set_page_config("📚 Chat with PDF - Gemini")
# st.title("📚 Chat with PDF using Gemini")

# if "session_id" not in st.session_state:
#     st.session_state.session_id = create_session()

# with st.sidebar:
#     st.subheader("🧠 Chat Sessions")
#     sessions = get_all_sessions()
#     selected = st.selectbox("Select Session", sessions, format_func=lambda x: x[1])
#     if selected:
#         st.session_state.session_id = selected[0]

#     if st.button("➕ Start New Chat"):
#         st.session_state.session_id = create_session()

#     if st.button("❌ Delete Current Session"):
#         delete_session(st.session_state.session_id)
#         st.success("Session deleted")
#         st.experimental_rerun()

#     with st.expander("🔍 Search Chats"):
#         keyword = st.text_input("Enter keyword")
#         if keyword:
#             results = search_chats(st.session_state.session_id, keyword)
#             for ts, user, bot in results:
#                 st.markdown(f"**[{ts}]** 👤: {user}\n\n🤖: {bot}")

#     st.markdown("---")
#     st.subheader("📤 Upload PDFs")
#     pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
#     if st.button("📚 Process PDFs") and pdf_docs:
#         with st.spinner("Processing PDFs..."):
#             raw_text = get_pdf_text(pdf_docs)
#             chunks = get_text_chunks(raw_text)
#             get_vector_store(chunks, st.session_state.session_id)
#             st.success("Vector store created ✅")

# # Chat History
# st.markdown("---")
# st.subheader(f"💬 Chat - Session #{st.session_state.session_id}")
# chat_history = get_chats_by_session(st.session_state.session_id)
# for ts, user, bot in chat_history:
#     with st.chat_message("user"):
#         st.markdown(f"**[{ts}]**\n{user}")
#     with st.chat_message("assistant"):
#         st.markdown(f"**[{ts}]**\n{bot}")

# # Input
# if prompt := st.chat_input("Ask a question from the PDF"):
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     with st.spinner("Gemini thinking..."):
#         reply = respond_to_user(st.session_state.session_id, prompt)
#     with st.chat_message("assistant"):
#         st.markdown(reply)






















import streamlit as st
import os
import shutil
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import sqlite3

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

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

# PDF Processing
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if not text_chunks:
        return
    store = FAISS.from_texts(text_chunks, embedding=embeddings)
    store.save_local("vectorstore", index_name="index")

# def load_vector_store():
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     return FAISS.load_local("vectorstore", embeddings, index_name="index", allow_dangerous_deserialization=True)

def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_file = os.path.join("vectorstore", "index.faiss")
    if not os.path.exists(faiss_file):
        st.error("❗ Vector store not found. Please upload and process PDFs first.")
        st.stop()
    return FAISS.load_local("vectorstore", embeddings, index_name="index", allow_dangerous_deserialization=True)


# def get_conversational_chain():
#     system_prompt = (
#         "You are a helpful assistant that can answer questions based on PDF documents uploaded by the user. "
#         "If no PDF is uploaded, kindly explain that your answers will be limited and suggest the user to upload one for better help."
#     )
#     prompt_template = """
#     Answer the question using the provided context. Be detailed. If the answer is not in the context, respond with: "Answer is not available in the context."

#     Context:
#     {context}

#     Question: {question}

#     Answer:
# #     """
#     model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def get_conversational_chain():
    prompt_template = """
     Answer the questions as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in the provided context 
    just say, "answer is not available in the context, don't provide the wrong answer\n\n
    context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    
    """

    # model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)
    model = ChatGoogleGenerativeAI(model="models/gemini-pro", temperature=0.3)


    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain




# def respond_to_user(user_question):
#     vector_db = load_vector_store()
#     docs = vector_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     save_chat(user_question, response["output_text"])
#     return response["output_text"]

def respond_to_user(user_question):
    faiss_file = os.path.join("vectorstore", "index.faiss")
    chain = get_conversational_chain()

    if os.path.exists(faiss_file):
        vector_db = load_vector_store()
        docs = vector_db.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        answer = response["output_text"]
    else:
        # Direct Gemini response
        # raw_output = chain.llm_chain.llm.invoke(user_question)
        # answer = str(raw_output)  # 🔥 convert to string
        raw_output = chain.llm_chain.llm.invoke(user_question)
        answer = raw_output.text()  # ✅ this gives just "Hi there! How can I help you today?"


    save_chat(str(user_question), str(answer))  # 🔥 safe insert
    return answer
    



# Streamlit UI
st.set_page_config("📚 Chat with PDF - Gemini")
st.title("💀 Chat with PDF using Gemini")

with st.sidebar:
    st.subheader("📤 Upload PDFs")
    pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
    if st.button("📚 Process PDFs") and pdf_docs:
        with st.spinner("Processing PDFs..."):
            raw_text = get_pdf_text(pdf_docs)
            chunks = get_text_chunks(raw_text)
            get_vector_store(chunks)
            st.success("Vector store created ✅")

    if st.button("🧹 Clear Chat History"):
        delete_all_chats()
        st.success("History cleared")
        st.experimental_rerun()

# Chat History
st.markdown("---")
st.subheader("💬 Chat")
chat_history = get_chats()
for user, bot in chat_history:
    with st.chat_message("user"):
        st.markdown(user)
    with st.chat_message("assistant"):
        st.markdown(bot)

# Input
if prompt := st.chat_input("Ask a question from the PDF"):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner("Gemini thinking..."):
        reply = respond_to_user(prompt)
    with st.chat_message("assistant"):
        st.markdown(reply)





































