from pocketflow import Node, AsyncNode
from utils.call_llm import call_llm
from utils.load_documents import load_documents
from utils.gemini_client import call_gemini
from utils.lightrag_setup import setup_lightrag
import asyncio
import os

class LoadDocumentsNode(Node):
    """Node to scan directory for document files and populate the document list."""
    
    def prep(self, shared):
        """Get the documents directory path from shared memory."""
        return shared.get("config", {}).get("temp_docs_path", "./temp_docs")
    
    def exec(self, docs_path):
        """Find all supported document files in the directory."""
        # Ensure the directory exists
        if not os.path.exists(docs_path):
            os.makedirs(docs_path, exist_ok=True)
            
        # Load documents from the directory
        return load_documents(docs_path)
    
    def post(self, shared, prep_res, exec_res):
        """Store the list of documents in shared memory."""
        shared["documents"] = exec_res
        return "default"

class ProcessDocumentsNode(AsyncNode):
    """Node to process documents with LightRAG and build knowledge graph."""
    
    async def prep(self, shared):
        """Get documents and config from shared memory."""
        return {
            "documents": shared.get("documents", []),
            "config": shared.get("config", {})
        }
    
    async def exec(self, prep_data):
        """Initialize LightRAG and process each document."""
        documents = prep_data["documents"]
        config = prep_data["config"]
        
        if not documents:
            return None
            
        # Get LightRAG data path from config
        lightrag_data_path = config.get("lightrag_data_path", "./lightrag_data")
        
        # Initialize LightRAG with built-in knowledge graph persistence
        rag = await setup_lightrag(lightrag_data_path)
        
        # We're using LightRAG's built-in knowledge graph persistence, so no need to manually save the graph
        
        # Process each document
        for doc_path in documents:
            try:
                with open(doc_path, "rb") as f:
                    content = f.read()
                    await rag.insert(content, source_id=doc_path)
            except Exception as e:
                print(f"Error processing document {doc_path}: {e}")
        
        return rag
    
    async def post(self, shared, prep_res, exec_res):
        """Store the LightRAG instance in shared memory."""
        shared["rag_instance"] = exec_res
        return "default"

class QueryProcessingNode(Node):
    """Node to process user query and retrieve relevant context."""
    
    def prep(self, shared):
        """Get query and LightRAG instance from shared memory."""
        return {
            "query": shared.get("query"),
            "rag_instance": shared.get("rag_instance")
        }
    
    def exec(self, prep_data):
        """Use LightRAG to retrieve context from vector and graph stores."""
        query = prep_data["query"]
        rag_instance = prep_data["rag_instance"]
        
        if not query or not rag_instance:
            return None
            
        # Use asyncio to run the async retrieval in a sync context
        async def retrieve_context():
            return await rag_instance.retrieve(query)
            
        # Run the async function and return the result
        return asyncio.run(retrieve_context())
    
    def post(self, shared, prep_res, exec_res):
        """Store the retrieved context in shared memory."""
        shared["retrieved_context"] = exec_res
        return "default"

class AnswerGenerationNode(Node):
    """Node to generate answer using Gemini LLM with retrieved context."""
    
    def prep(self, shared):
        """Get query and retrieved context from shared memory."""
        return {
            "query": shared.get("query"),
            "context": shared.get("retrieved_context")
        }
    
    def exec(self, prep_data):
        """Call the Gemini LLM with query and context."""
        query = prep_data["query"]
        context = prep_data["context"]
        
        if not query:
            return "No question was provided."
            
        if not context:
            return "I don't have enough context to answer that question."
        
        # Extract text and source information from context
        context_text = "\n".join([item["text"] for item in context])
        sources = [item["source_id"] for item in context if "source_id" in item]
        
        # Call Gemini with the query and context
        answer = call_gemini(query, context_text)
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    def post(self, shared, prep_res, exec_res):
        """Store the answer and source citations in shared memory."""
        if isinstance(exec_res, dict):
            shared["answer"] = exec_res.get("answer", "")
            shared["source_citations"] = exec_res.get("sources", [])
        else:
            shared["answer"] = exec_res
            shared["source_citations"] = []
            
        return "default"