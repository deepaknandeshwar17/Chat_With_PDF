# 📄 Chat With PDF – Lightweight RAG Chatbot using Groq LLaMA3

Interact with your PDFs like never before! This project allows you to upload one or more PDFs and ask questions about their contents. Powered by **Retrieval-Augmented Generation (RAG)** using **Google Embeddings**, **FAISS**, and the blazing-fast **Groq LLaMA3-70B** model.

---

## 🚀 Features

- 📤 Upload and process multiple PDFs
- 🧠 Ask natural language questions based on document content
- 🗃️ Uses **GoogleGenerativeAIEmbeddings** and **FAISS** for vector search
- ⚡ Ultra-fast, detailed answers using **Groq’s LLaMA3-70B**
- 💬 Persistent chat history with **SQLite**
- 🧹 Option to clear history and vector DB
- 🌐 Streamlit-based clean and interactive UI

---

## 🛠️ Tech Stack

| Component         | Tech Used                              |
|------------------|-----------------------------------------|
| Frontend         | [Streamlit](https://streamlit.io)       |
| LLM               | Groq `llama3-70b-8192` via `ChatGroq`   |
| Embeddings       | Google `models/embedding-001`           |
| Vector Store     | FAISS                                   |
| PDF Parsing      | PyPDF2                                  |
| Chat History     | SQLite                                  |
| Prompt Handling  | LangChain's `load_qa_chain`             |

---


## ⚙️ Installation

```bash
git clone https://github.com/deepaknandeshwar17/Chat_With_PDF.git
cd Chat_With_PDF
pip install -r requirements.txt
