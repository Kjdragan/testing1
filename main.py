from flow import qa_flow
import os

def main():
    """Main function to run the LightRAG PDF Chatbot."""
    # Initialize shared memory with configuration
    shared = {
        "config": {
            "lightrag_data_path": "./lightrag_data",
            "temp_docs_path": "./temp_docs",
            "models": {
                "embedding": "gemini-embedding-exp-03-07",
                "llm": "gemini-2.0-flash"
            }
        },
        "documents": [],
        "rag_instance": None,
        "query": None,
        "retrieved_context": None,
        "answer": None,
        "source_citations": []
    }

    # Set a sample query for testing
    shared["query"] = "What are the key features of LightRAG?"

    # Run the flow
    qa_flow.run(shared)
    
    # Print results
    print("\nQuery:", shared["query"])
    print("\nAnswer:", shared["answer"])
    
    if shared["source_citations"]:
        print("\nSources:")
        for source in shared["source_citations"]:
            print(f"- {source}")

if __name__ == "__main__":
    main()