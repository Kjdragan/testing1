"""
Utility to find and load document files from a directory.
"""
import os

def load_documents(directory_path):
    """Find all PDF, DOC, PPT, and CSV files in a directory.
    
    Args:
        directory_path (str): Path to the directory to scan
        
    Returns:
        list: List of absolute paths to found documents
    """
    supported_extensions = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.csv']
    documents = []
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                documents.append(os.path.join(root, file))
    
    return documents

if __name__ == "__main__":
    # Test the function
    docs_path = "./temp_docs"
    found_docs = load_documents(docs_path)
    print(f"Found {len(found_docs)} documents: {found_docs}")
