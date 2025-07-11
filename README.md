🧠 PDF Chatbot with Groq LLaMA3, LangChain & Streamlit
A simple yet powerful app that lets you upload PDFs and ask questions, powered by:
🦙 Groq LLaMA3 (via LangChain)
🔍 FAISS Vector Store for semantic search
🧠 Google GenAI Embeddings
🗃️ SQLite to store chat history
🌐 Streamlit frontend


🚀 Features
✅ Upload multiple PDFs
✅ Chunk and embed text using Google GenAI
✅ Store embeddings in FAISS
✅ Ask questions using a chatbot interface
✅ Context-aware answers via Groq LLaMA3
✅ Chat history saved in SQLite
✅ Option to clear chat/vector store anytime


🔧 Tech Stack
Streamlit
LangChain
Groq LLaMA3
Google Generative AI Embeddings
FAISS
SQLite
PyPDF2


📁 Project Structure
├── main.py                   # Streamlit app
├── chat_history.db           # SQLite DB
├── vectorstore/              # FAISS index folder
├── .env                      # API keys
├── requirements.txt
└── README.md


🔑 Setup & Run
Clone the repo
git clone https://github.com/yourname/pdf-chatbot.git
cd pdf-chatbot

Install dependencies
pip install -r requirements.txt

Add API Keys
Create a .env file:
GROQ_API_KEY=your_groq_api_key

Run the app
streamlit run main.py


🧼 Optional: Clear Chat & Vector Store
Use the sidebar buttons in the app to:
Clear chat history
Remove stored FAISS index


📌 To-Do / Improvements
🔒 Add user authentication
🧠 Support multiple models (Groq, OpenAI, Gemini)
📄 Summarize PDFs before asking
💬 Live chatbot deployment



🙌 Credits
Built using tools from the awesome open-source ecosystem.
Special thanks to LangChain, Groq, and Streamlit.
