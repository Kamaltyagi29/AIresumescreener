import sys, os
sys.dont_write_bytecode = True

import time
from dotenv import load_dotenv
import subprocess

import pandas as pd
import streamlit as st
from interactive import convert_pdf
from streamlit_modal import Modal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

from llm_agent import ChatBot
from interactive.convert_pdf import extract_resumes_to_csv
from ingest_data import ingest
from retriever import SelfQueryRetriever
import chatbot_verbosity as chatbot_verbosity

import os
os.environ["GEMINI_API_KEY"] = "AIzaSyAKu8SX1AusIVmR7DjZnJhjf7JXfs4fc_g"

load_dotenv()

PDF_STORAGE_PATH = "/home/kamaltyagi14/Resume-Screening-RAG-Pipeline/demo/DATA/data/supplementary-data/pdf-resumes"
DATA_PATH = "/home/kamaltyagi14/Resume-Screening-RAG-Pipeline/demo/DATA/data/supplementary-data/pdf-resumes.csv"
FAISS_PATH = os.getenv("FAISS_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

st.set_page_config(page_title="Resume Screening AI")
st.title("Resume Screening AI")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Welcome to Resume Screening AI!")]

def initialize_csv():
    """Ensure the CSV file exists, has headers, and is pre-filled with PDF resume data if empty."""
    if not os.path.exists(DATA_PATH) or os.stat(DATA_PATH).st_size == 0:
        df = pd.DataFrame(columns=["ID", "Resume_name", "Resume"])
        df.to_csv(DATA_PATH, index=False)
        pdf_files = [f for f in os.listdir(PDF_STORAGE_PATH) if f.endswith(".pdf")]
        if pdf_files:
            for pdf_file in pdf_files:
                file_path = os.path.join(PDF_STORAGE_PATH, pdf_file)
                extract_resumes_to_csv(file_path, DATA_PATH)

if "df" not in st.session_state:
    initialize_csv()
    st.session_state.df = pd.read_csv(DATA_PATH)


if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})

if "rag_pipeline" not in st.session_state:
    vectordb = FAISS.load_local(
        FAISS_PATH, st.session_state.embedding_model, 
        distance_strategy=DistanceStrategy.COSINE, 
        allow_dangerous_deserialization=True
    )
    st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)

if "resume_list" not in st.session_state:
    st.session_state.resume_list = []

def upload_file():
    modal = Modal(key="upload_error", title="File Error", max_width=500)
    uploaded_file = st.session_state.uploaded_file

    if uploaded_file is not None:
        file_type = uploaded_file.type
        if file_type == "text/csv":
            try:
                df_load = pd.read_csv(uploaded_file)

                if "Resume" not in df_load.columns or "ID" not in df_load.columns:
                    with modal.container():
                        st.error("Uploaded CSV must contain 'Resume' and 'ID' columns.")
                    return

            except Exception as error:
                with modal.container():
                    st.error(f"Error in uploaded CSV: {error}")
                return

        elif file_type == "application/pdf":
            try:
                file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
                if os.path.exists(file_path):
                    with modal.container():
                        st.warning("Resume already present.")
                    return
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Process only the newly added PDF
                extract_resumes_to_csv(file_path, DATA_PATH)

                # Load only the newly added resume
                df_load = pd.read_csv(DATA_PATH).iloc[[-1]]  # Last row (newly added)
                new_vectordb = ingest(df_load, "Resume", st.session_state.embedding_model)

                if st.session_state.rag_pipeline is None:
                    st.session_state.rag_pipeline = SelfQueryRetriever(new_vectordb, df_load)
                else:
                    st.session_state.rag_pipeline.vectorstore.merge_from(new_vectordb)

            except Exception as error:
                with modal.container():
                    st.error(f"Error processing PDF: {error}")
                return

        else:
            with modal.container():
                st.error("Unsupported file format. Please upload CSV or PDF.")
            return

        with st.spinner("Indexing uploaded data..."):
                st.session_state.df = pd.concat([st.session_state.df, df_load], ignore_index=True)

        st.success("File uploaded successfully!")

def delete_resume(file_name):
    file_path = os.path.join(PDF_STORAGE_PATH, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        df = pd.read_csv(DATA_PATH)
        df = df[df["Resume_name"] != file_name]
        df.to_csv(DATA_PATH, index=False)
        st.session_state.df = df
        st.success(f"{file_name} has been deleted from storage and CSV.")

def clear_message():
    st.session_state.resume_list = []
    st.session_state.chat_history = [AIMessage(content="Welcome to Resume Screening AI!")]

user_query = st.chat_input("Type your message here...")

# Sidebar
with st.sidebar:
    st.header("Control Panel")
    # st.selectbox("RAG Mode", ["Generic RAG", "RAG Fusion"], placeholder="Generic RAG", key="rag_selection")
    st.file_uploader(
        "Upload resumes (CSV or PDF format)",
        type=["csv", "pdf"], 
        key="uploaded_file", 
        on_change=upload_file
    )
    df_resumes = pd.read_csv(DATA_PATH)
    resume_mapping = {row["Resume_name"]: row["ID"] for _, row in df_resumes.iterrows()}
    resume_files = os.listdir("/home/kamaltyagi14/Resume-Screening-RAG-Pipeline/demo/DATA/data/supplementary-data/pdf-resumes")
    resume_options = [f"{resume_mapping[file]} - {file}" if file in resume_mapping else file for file in resume_files]
    file_to_download = st.selectbox("Select resume to download", resume_options, index=None, placeholder="Choose a file...")
    if file_to_download:
        selected_file = file_to_download.split(" - ", 1)[-1]
        with open(os.path.join(PDF_STORAGE_PATH, selected_file), "rb") as pdf_file:
            st.download_button(label=f"Download {selected_file}", data=pdf_file, file_name=selected_file, mime="application/pdf")
    file_to_delete = st.selectbox("Select resume to delete", resume_files, index=None, placeholder="Choose a file...")
    if file_to_delete:
        if st.button("Delete Selected Resume"):
            delete_resume(file_to_delete)
    st.button("Clear conversation", on_click=clear_message)

for message in st.session_state.chat_history:
    if isinstance(message, (AIMessage, HumanMessage)):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
    else:
        with st.chat_message("AI"):
            message[0].render(*message[1:])

retriever = st.session_state.rag_pipeline
st.session_state.api_key = os.getenv("GEMINI_API_KEY")
st.session_state.gpt_selection = "gemini-1.5-pro-latest"
llm = ChatBot(
    # api_key=st.session_state.api_key,
    # model=st.session_state.gpt_selection
)
if user_query is not None and user_query != "":
    with st.chat_message("Human"):
        st.markdown(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("AI"):
        start_time = time.time()
        with st.spinner("Generating response..."):
          document_list = retriever.retrieve_docs(user_query,llm,"Generic RAG")
          query_type = retriever.meta_data["query_type"]
          st.session_state.resume_list = document_list
          stream_message = llm.generate_message_stream(user_query, document_list, [], query_type)
        end_time = time.time()
        response = st.write_stream(stream_message)
        retriever_message = chatbot_verbosity
        retriever_message.render(document_list, retriever.meta_data, end_time - start_time)
        st.session_state.chat_history.append(AIMessage(content=response))
        st.session_state.chat_history.append((retriever_message, document_list, retriever.meta_data, end_time-start_time))


