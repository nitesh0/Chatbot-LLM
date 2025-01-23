import streamlit as st 
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
load_dotenv()
## load the GROQ API key
groq_api_key = os.getenv("GROQ_API_KEY")

# llm = ChatGroq(groq_api_key=groq_api_key, model_name ="Llama3-8b-8192")
repo_id = "alterf/tinymodel"
token = os.getenv("HUGGINGFACE_API_KEY")
llm = HuggingFaceEndpoint(repo_id =repo_id,max_length=150, temperature=0.7, token=token)


prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question:{input}
    """
)

def Embeddings():
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
    return hf

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = Embeddings()
        st.session_state.loader = PyPDFDirectoryLoader("test") ## data ingestion
        st.session_state.docs = st.session_state.loader.load() #document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        
user_prompt = st.text_input("Enter your question")   

if st.button("Document embedding"):
    create_vector_embeddings()
    st.write("Vector datase")
    
if user_prompt:
    document_chain =create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = RetrievalQA.from_chain_type(retriever,document_chain)
    response = retriever_chain.invoke({'input':user_prompt})
    st.write("Response:", response)
