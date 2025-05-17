import os
import psutil
import random
from api.document_processor import DocumentProcessor

# Get current process
process = psutil.Process(os.getpid())

# Helper function to print memory usage
def print_memory_usage():
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")

# Starting memory usage
print("Initial memory usage:")
print_memory_usage()

# Initialize processor
processor = DocumentProcessor()

# Generate a larger test dataset (adjust size as needed)
large_docs = [f"Test document {i} with some random content" for i in range(5000)]

# Memory after adding documents
processor.add_documents(large_docs)
print("\nMemory after adding documents:")
print_memory_usage()

# Memory after creating embeddings
processor.create_embeddings()
print("\nMemory after creating embeddings:")
print_memory_usage()

# Memory after building index
processor.build_faiss_index()
print("\nMemory after building FAISS index:")
print_memory_usage()