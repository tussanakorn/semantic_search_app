import random
import string
from api.document_processor import DocumentProcessor
import time

# Create a larger dataset
def generate_random_text(length=200):
    return ' '.join(''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10))) 
                    for _ in range(length // 5))

# Generate a larger test dataset
large_docs = [generate_random_text() for _ in range(1000)]

# Time the operations
processor = DocumentProcessor()

# Add documents
start_time = time.time()
processor.add_documents(large_docs)
print(f"Added {len(large_docs)} documents in {time.time() - start_time:.2f} seconds")

# Create embeddings
start_time = time.time()
processor.create_embeddings()
print(f"Created embeddings in {time.time() - start_time:.2f} seconds")

# Build index
start_time = time.time()
processor.build_faiss_index()
print(f"Built FAISS index in {time.time() - start_time:.2f} seconds")

# Test search performance
start_time = time.time()
processor.search("test query", top_k=10)
print(f"Search completed in {time.time() - start_time:.2f} seconds")