import os
from api.document_processor import DocumentProcessor

# Initialize the document processor
processor = DocumentProcessor()

# Load documents from sample CSV
processor.process_documents_from_csv('data/sample_docs.csv', 'text', 'id')
print(f"Loaded {len(processor.documents)} documents from CSV")

# Display the first few documents
print("\nSample documents:")
for i, doc in enumerate(processor.documents[:3]):
    print(f"{i+1}. ID: {doc['id']} - {doc['text'][:80]}...")

# Create embeddings and build index
processor.create_embeddings()
processor.build_faiss_index()

# Test a search query
query = "Tell me about vector databases"
print(f"\nQuery: '{query}'")
results = processor.search(query, top_k=3)
for i, result in enumerate(results):
    print(f"  {i+1}. Score: {result['score']:.4f} - {result['text'][:80]}...")

# Test index saving and loading
test_dir = "./test_data"
os.makedirs(test_dir, exist_ok=True)

print("\nSaving index...")
processor.save_index(test_dir)

print("Loading index...")
new_processor = DocumentProcessor()
new_processor.load_index(test_dir)

print(f"Loaded {len(new_processor.documents)} documents from saved index")

# Test search with loaded index
print(f"\nQuery with loaded index: '{query}'")
results = new_processor.search(query, top_k=3)
for i, result in enumerate(results):
    print(f"  {i+1}. Score: {result['score']:.4f} - {result['text'][:80]}...")