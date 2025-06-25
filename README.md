# 🤖 RAG Chatbot – Document-Aware AI Assistant

A **Retrieval-Augmented Generation (RAG)** chatbot built with **LangChain**, **Streamlit**, **Sentence Transformers**, and **HuggingFace Transformers**.  
This intelligent assistant can **read your uploaded PDF documents** and answer questions strictly based on those documents — if it's not in the source, it says:  
> *"I don't know."*

---

## 🔍 How It Works

This chatbot follows a **classic RAG architecture**, which enhances a base language model with document retrieval to ground its answers in reality.

### 🧠 Architecture Overview

1. **Document Ingestion**
   - PDFs are parsed using `PyPDFLoader`
   - Text is split into manageable chunks

2. **Vector Store Creation**
   - Chunks are embedded using `sentence-transformers`
   - Stored in a lightweight vector DB (ChromaDB)

3. **Retrieval + QA Chain**
   - User query → semantic similarity search
   - Top-k relevant docs retrieved
   - Passed to an LLM (T5 or Mistral) for final answer

4. **Interface**
   - Clean, responsive chatbot UI using Streamlit

---

## 🛠️ Technologies Used

| Layer          | Tool                             |
|----------------|----------------------------------|
| LLM            | HuggingFace Transformers (T5)    |
| Embedding      | Sentence Transformers            |
| Vector Store   | ChromaDB                         |
| RAG Framework  | LangChain (v0.2+)                |
| Frontend       | Streamlit                        |
| File Parsing   | LangChain PyPDFLoader            |

---

## 📁 Recommended Project Structure

```bash
my_rag_chatbot/
│
├── rag_chatbot.py           # Main Streamlit app
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignored files for Git
├── README.md                # This documentation
└── pdfs/                    # Folder with uploaded PDFs
    ├── document1.pdf
    └── document2.pdf
