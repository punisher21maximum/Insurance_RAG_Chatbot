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
```

<img width="1425" alt="Screenshot 2025-06-25 at 07 54 39" src="https://github.com/user-attachments/assets/f6c3bcde-af3c-4cc4-90b4-8a58533f07c6" />
<img width="1420" alt="Screenshot 2025-06-25 at 07 54 58" src="https://github.com/user-attachments/assets/9138528e-a0d9-4504-8c8a-8a1343c9d540" />
<img width="1438" alt="Screenshot 2025-06-25 at 07 55 17" src="https://github.com/user-attachments/assets/dd0855a8-154e-43ab-be5a-e8aaa5f208da" />
<img width="1431" alt="Screenshot 2025-06-25 at 07 55 36" src="https://github.com/user-attachments/assets/d2fbd887-7269-45ab-a027-e8ffff994c79" />

<img width="1421" alt="Screenshot 2025-06-25 at 07 56 16" src="https://github.com/user-attachments/assets/73fa054c-4329-49c3-a190-b09c31683baa" />


