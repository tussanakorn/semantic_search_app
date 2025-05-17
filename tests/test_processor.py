import os
import sys
from api.document_processor import DocumentProcessor

# Initialize the document processor
processor = DocumentProcessor()

# Sample documents for testing
test_docs = [
    "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
    "Natural Language Processing is a field of AI that focuses on the interaction between computers and humans through natural language.",
    "Deep learning is a subset of machine learning that uses neural networks with many layers.",
    "Vector databases are specialized systems designed to store and query embeddings or vector representations of data.",
    "Semantic search is a data searching technique where a search query aims to understand the contextual meaning of the search terms."
]

# Add documents
processor.add_documents(test_docs)

# Create embeddings and build index
processor.create_embeddings()
processor.build_faiss_index()

# Test search with different queries
test_queries = [
    "What is machine learning?",
    "Explain deep neural networks"
]

# Run test searches
print("--- SEARCH RESULTS ---")
for query in test_queries:
    print(f"\nQuery: '{query}'")
    results = processor.search(query, top_k=2)
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result['score']:.4f} - {result['text'][:80]}...")