"""
Utility to interact with Google Gemini LLM using OpenAI's compatibility layer.
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_gemini_client():
    """Initialize and return a Gemini client using OpenAI compatibility.
    
    Returns:
        OpenAI: Configured client for Gemini
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
        
    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1"
    )
    
    return client

def call_gemini(prompt, context=None):
    """Call Gemini LLM with a prompt and optional context.
    
    Args:
        prompt (str): The query or prompt to send to the LLM
        context (str, optional): Additional context to include with the prompt
        
    Returns:
        str: The LLM response
    """
    client = get_gemini_client()
    
    # Prepare the messages with context if provided
    if context:
        messages = [
            {"role": "system", "content": f"Use the following context to answer the question: {context}"},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [
            {"role": "user", "content": prompt}
        ]
    
    # Call the Gemini model
    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=messages
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    # Test the function
    print(call_gemini("Explain how AI works in one paragraph."))
