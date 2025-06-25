import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

PDF_FOLDER = "pdfs"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-small"

@st.cache_resource
def load_docs():
    pages = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_FOLDER, filename))
            pages.extend(loader.load())
    return pages

@st.cache_resource
def load_vectorstore(_docs):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return Chroma.from_documents(_docs, embeddings)

@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
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
query = st.text_input("Ask something based on the PDF documents")

if query:
    chain = setup_qa_chain()
    result = chain.run(query)
    if result.lower().strip() in ["", "i don't know", "i don’t know"]:
        st.warning("I don't know based on the provided documents.")
    else:
        st.success(result)
