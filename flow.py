from pocketflow import Flow
from nodes import LoadDocumentsNode, ProcessDocumentsNode, QueryProcessingNode, AnswerGenerationNode

def create_qa_flow():
    """Create and return a question-answering flow for the LightRAG PDF Chatbot."""
    # Create nodes
    load_documents_node = LoadDocumentsNode()
    process_documents_node = ProcessDocumentsNode()
    query_processing_node = QueryProcessingNode()
    answer_generation_node = AnswerGenerationNode()
    
    # Connect nodes in sequence
    load_documents_node >> process_documents_node >> query_processing_node >> answer_generation_node
    
    # Create flow starting with input node
    return Flow(start=load_documents_node)

# Create a global instance of the flow for easy access
qa_flow = create_qa_flow()