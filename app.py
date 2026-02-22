import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


# --- 1. UI EXPERIENCE SETUP (2 Points) ---
st.set_page_config(page_title="Thai Constitution Chatbot", layout="wide")
st.title("📜 Thai Constitution RAG Chatbot")
st.markdown("ระบบแชทบอทอัจฉริยะที่ตอบคำถามโดยอ้างอิงจาก **รัฐธรรมนูญไทย**")

# --- 2. SIDEBAR CONFIGURATION (No Hardcoded API Key) ---
with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("1. Enter Groq API Key", type="password")
    uploaded_file = st.file_uploader("2. Upload Thai Constitution PDF", type="pdf")
    
    if st.button("Clear Chat History"):
        if "memory" in st.session_state:
            st.session_state.memory.clear()
            st.session_state.messages = []
            st.rerun()

# --- 3. CORE CHATBOT ENGINE ---
@st.cache_resource
def setup_knowledge_base(file_data):
    # Save uploaded file temporarily
    with open("temp_doc.pdf", "wb") as f:
        f.write(file_data.getbuffer())
    
    # [4 Points] Functional RAG: Load & Split
    loader = PyPDFLoader("temp_doc.pdf")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Embeddings & Vector DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# --- 4. MAIN INTERFACE LOGIC ---
if groq_api_key and uploaded_file:
    # Initialize RAG Retriever
    retriever = setup_knowledge_base(uploaded_file)

    # [2 Points] Proper Model ID: llama-3.1-8b-instant
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        groq_api_key=groq_api_key,
        temperature=0
    )

    # [2 Points] Memory Setup
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )

    # [4 Points] Functional RAG: Combine all into Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory
    )

    # Display Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("พิมพ์คำถามเกี่ยวกับรัฐธรรมนูญที่นี่..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นหาคำตอบจากรัฐธรรมนูญ..."):
                # ดึงคำตอบจาก RAG
                response = qa_chain.invoke({"question": prompt})
                full_response = response["answer"]
                st.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.warning("⚠️ กรุณาใส่ API Key และอัปโหลดไฟล์ PDF ในแถบด้านข้างเพื่อเริ่มการสนทนา")
