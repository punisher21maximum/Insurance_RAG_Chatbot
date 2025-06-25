import os
import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

PDF_FOLDER = "pdfs"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-small"

WEB_URLS = [
    "https://www.angelone.in/support/add-and-withdraw-funds/add-funds",
    "https://www.angelone.in/support/angel-one-recommendations/charges-and-frequency",
    "https://www.angelone.in/support/charges-and-cashbacks/dp-charges",
    "https://www.angelone.in/support/charts/chart-not-loading",
    "https://www.angelone.in/support/complaince/trading-surveillance",
    "https://www.angelone.in/support/fixed-deposits/account-verification",
    "https://www.angelone.in/support/ipo-ofs/ipo",
    "https://www.angelone.in/support/loans/active-loans",
    "https://www.angelone.in/support/margin-pledging-and-margin-trading-facility/available-margin-to-trade",
    "https://www.angelone.in/support/mutual-funds/orders",
    "https://www.angelone.in/support/portfolio-and-corporate-actions/bonus-issue",
    "https://www.angelone.in/support/reports-and-statements/client-master-list",
    "https://www.angelone.in/support/your-account/family-declaration",
    "https://www.angelone.in/support/your-orders/watchlist",
]

def load_web_docs():
    docs = []
    for url in WEB_URLS:
        try:
            res = requests.get(url, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")
            content = soup.get_text(separator=" ", strip=True)
            docs.append(Document(page_content=content, metadata={"source": url}))
        except Exception as e:
            st.warning(f"Failed to load {url}: {e}")
    return docs

@st.cache_resource
def load_docs():
    pages = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_FOLDER, filename))
            pages.extend(loader.load())
    pages.extend(load_web_docs())  # Add web pages
    return pages

@st.cache_resource
def load_vectorstore(_docs):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.from_documents(_docs, embeddings)

@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME).to("cpu")
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=128)
    return HuggingFacePipeline(pipeline=pipe)

@st.cache_resource
def setup_qa_chain():
    docs = load_docs()
    vectordb = load_vectorstore(docs)
    llm = load_llm()
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False
    )

st.title("🧠 Simple RAG Chatbot")
query = st.text_input("Ask something based on the PDF documents and webpages")

if query:
    chain = setup_qa_chain()
    result = chain.run(query)
    if result.lower().strip() in ["", "i don't know", "i don’t know"]:
        st.warning("I don't know based on the provided documents.")
    else:
        st.success(result)
