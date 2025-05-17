import os
import pandas as pd
from api.document_processor import DocumentProcessor

# Create test dataset
test_data = [
    {"id": 1, "text": "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed."},
    {"id": 2, "text": "Natural Language Processing is a field of AI that focuses on the interaction between computers and humans through natural language."},
    {"id": 3, "text": "Deep learning is a subset of machine learning that uses neural networks with many layers."},
    {"id": 4, "text": "Vector databases are specialized systems designed to store and query embeddings."},
    {"id": 5, "text": "Semantic search is a data searching technique that understands the contextual meaning of search terms."}
]

# Create a DataFrame and save to CSV
os.makedirs('test_data', exist_ok=True)
pd.DataFrame(test_data).to_csv('test_data/test_docs.csv', index=False)
print("Created test CSV file")

# Initialize processor
processor = DocumentProcessor()

# Process documents from CSV
processor.process_documents_from_csv('test_data/test_docs.csv', 'text', 'id')
print(f"Loaded {len(processor.documents)} documents")

# Create embeddings and build index
processor.create_embeddings()
processor.build_faiss_index()
print("Built search index")

# Save index
processor.save_index('test_data')
print("Saved index to disk")

# Load index in a new processor
new_processor = DocumentProcessor()
new_processor.load_index('test_data')
print(f"Loaded index with {len(new_processor.documents)} documents")

# Test search functionality
query = "How does semantic search work?"
results = new_processor.search(query, top_k=3)
print(f"\nQuery: '{query}'")
for i, result in enumerate(results):
    print(f"  {i+1}. Score: {result['score']:.4f} - {result['text']}")