"""
Utility to set up and initialize LightRAG with proper configuration.
"""
import os
import asyncio
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define embedding function using sentence-transformers
async def embedding_func(texts: list[str]) -> np.ndarray:
    """Embedding function using sentence-transformers.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        numpy.ndarray: Embeddings for the texts
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings
    except ImportError:
        print("Please install sentence-transformers: uv add sentence-transformers")
        raise

# Define LLM function using Gemini
async def llm_model_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs) -> str:
    """LLM function using Gemini.
    
    Args:
        prompt: The prompt to send to the LLM
        system_prompt: Optional system prompt
        history_messages: List of previous messages
        keyword_extraction: Whether this is for keyword extraction
        
    Returns:
        str: LLM response
    """
    try:
        from openai import OpenAI
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1"
        )
        
        # Combine prompts: system prompt, history, and user prompt
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if history_messages:
            messages.extend(history_messages)
        
        messages.append({"role": "user", "content": prompt})
        
        # Call the Gemini model
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=messages,
            max_tokens=500,
            temperature=0.1
        )
        
        return response.choices[0].message.content
    except ImportError:
        print("Please install openai: uv add openai")
        raise

async def setup_lightrag(data_path):
    """Set up and initialize LightRAG with built-in knowledge graph persistence.
    
    Args:
        data_path (str): Path to store LightRAG data
        
    Returns:
        LightRAG: Initialized LightRAG instance
    """
    # Create LightRAG instance with mixed approach (vector + knowledge graph)
    rag = LightRAG(
        working_dir=data_path,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,  # Dimension for all-MiniLM-L6-v2
            max_token_size=8192,
            func=embedding_func,
        ),
    )
    
    # Initialize storages
    await rag.initialize_storages()
    
    # Initialize pipeline status
    from lightrag.kg.shared_storage import initialize_pipeline_status
    await initialize_pipeline_status()
    
    return rag

if __name__ == "__main__":
    # Test the function
    async def test_setup():
        try:
            rag = await setup_lightrag("./lightrag_data")
            print("LightRAG setup successful")
            return rag
        except Exception as e:
            print(f"Error setting up LightRAG: {e}")
            return None
    
    # Run the test
    asyncio.run(test_setup())
