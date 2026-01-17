import streamlit as st
import os
from src.ingestion import process_and_store_pdf
from src.retrieval import get_retriever
from src.generator import create_rag_chain

# Page Config
st.set_page_config(page_title="Groq-Powered RAG Chat", layout="centered")
st.title("📄 AI Document Assistant")
st.write("Upload a PDF and ask questions using the ultra-fast Groq Llama 3!")

# 1. Sidebar for PDF Upload
with st.sidebar:
    st.header("Upload Center")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file:
        # Create a 'data' directory if it doesn't exist
        if not os.path.exists("data"):
            os.makedirs("data")
            
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Indexing your document..."):
            # Call your ingestion logic
            process_and_store_pdf(file_path)
        st.success("Indexing Complete!")

# 2. Chat Interface
if uploaded_file:
    # Initialize the RAG Chain
    retriever = get_retriever()
    if retriever:
        rag_chain = create_rag_chain(retriever)
        
        user_query = st.chat_input("Ask something about the document...")
        
        if user_query:
            # Display user message
            with st.chat_message("user"):
                st.write(user_query)
                
            # Generate and display AI response
            with st.chat_message("assistant"):
                with st.spinner("Groq is thinking..."):
                    # Invoke the modular chain
                    response = rag_chain.invoke({"input": user_query})
                    st.write(response["answer"])
    else:
        st.warning("Please upload a file to start chatting.")
else:
    st.info("Welcome! Please upload a PDF in the sidebar to begin.")