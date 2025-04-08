"""
Streamlit application for the LightRAG PDF Chatbot.
"""
import os
import streamlit as st
import asyncio
import nest_asyncio
from utils.load_documents import load_documents
from utils.lightrag_setup import setup_lightrag
from lightrag import QueryParam

# Apply nest_asyncio to solve event loop issues in Streamlit
nest_asyncio.apply()

async def insert_documents(rag, doc_paths):
    """Insert documents into LightRAG.
    
    Args:
        rag: LightRAG instance
        doc_paths: List of document paths
        
    Returns:
        int: Number of processed documents
    """
    for doc_path in doc_paths:
        with open(doc_path, "rb") as f:
            content = f.read()
            await rag.insert(content, source_id=doc_path)
    
    return len(doc_paths)

async def process_query(rag, query):
    """Process a query using LightRAG.
    
    Args:
        rag: LightRAG instance
        query: User query string
        
    Returns:
        str: Response with citations
    """
    # Retrieve context from LightRAG
    response = await rag.query(
        query=query,
        param=QueryParam(mode="hybrid", top_k=5)
    )
    
    # Format the response with citations
    if not response:
        return "I couldn't find relevant information to answer your question."
    
    answer = response
    
    # Check if response is a dictionary with sources
    if isinstance(response, dict) and "sources" in response:
        sources = response.get("sources", [])
        if sources:
            citations = "\n\nSources:\n" + "\n".join([f"- {src}" for src in sources])
            return response.get("answer", "") + citations
    
    return answer

def main():
    """Main function for the Streamlit application."""
    # Set up Streamlit page
    st.set_page_config(page_title="LightRAG PDF Chatbot", layout="wide")
    
    # Initialize session state
    if "lightrag" not in st.session_state:
        st.session_state.lightrag = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Set up UI
    st.title("LightRAG PDF Chatbot")
    
    # Sidebar for document upload and processing
    with st.sidebar:
        st.header("Document Processing")
        
        # Document upload
        uploaded_files = st.file_uploader(
            "Upload Documents", 
            accept_multiple_files=True,
            type=["pdf", "doc", "docx", "ppt", "pptx", "csv", "txt"]
        )
        
        # Process button
        if uploaded_files and st.button("Process Documents"):
            # Ensure temp directory exists
            os.makedirs("./temp_docs", exist_ok=True)
            
            # Save uploaded files
            doc_paths = []
            for file in uploaded_files:
                file_path = os.path.join("./temp_docs", file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                doc_paths.append(file_path)
            
            # Initialize LightRAG
            with st.spinner("Setting up LightRAG..."):
                st.session_state.lightrag = asyncio.run(setup_lightrag("./lightrag_data"))
            
            # Process documents
            with st.spinner(f"Processing {len(doc_paths)} documents..."):
                num_processed = asyncio.run(insert_documents(st.session_state.lightrag, doc_paths))
                st.success(f"Processed {num_processed} documents successfully!")
    
    # Main area for chat interface
    st.header("Chat")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if query := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(query)
        
        # Generate response
        if st.session_state.lightrag:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = asyncio.run(process_query(st.session_state.lightrag, query))
                    st.write(response)
            
            # Add assistant message to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            with st.chat_message("assistant"):
                st.write("Please upload and process documents first.")
                st.session_state.chat_history.append({"role": "assistant", "content": "Please upload and process documents first."})

if __name__ == "__main__":
    main()
