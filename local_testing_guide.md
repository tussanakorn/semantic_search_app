# Local Testing Guide for Semantic Search System

Before deploying your semantic search system to AWS EC2 with Docker, it's best to test everything locally. This ensures your code works as expected and helps identify issues early in the development process.

## Prerequisites

Make sure you have the following installed on your local machine:

- Python 3.10 or later
- pip (Python package manager)
- Virtual environment tool (venv or conda)

## Step 1: Set Up a Virtual Environment

First, create and activate a virtual environment to isolate your dependencies:

```bash
# Create a virtual environment
python -m venv semantic_search_env

# Activate it (Windows)
semantic_search_env\Scripts\activate

# Activate it (macOS/Linux)
source semantic_search_env/bin/activate
```

## Step 2: Install Dependencies

Install all required packages:

```bash
# Navigate to your project directory
cd semantic_search_app

# Install dependencies
pip install -r requirements.txt
```

## Step 3: Create a Test Script

Create a file called `test_processor.py` in the project directory:

```python
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
    "Explain deep neural networks",
    "How does semantic search work?",
    "Vector databases for embeddings"
]

# Run test searches
print("--- SEARCH RESULTS ---")
for query in test_queries:
    print(f"\nQuery: '{query}'")
    results = processor.search(query, top_k=2)
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result['score']:.4f} - {result['text'][:80]}...")
```

## Step 4: Test the Document Processor

Run the test script to verify that the document processor works correctly:

```bash
python test_processor.py
```

This should output search results for each test query. Ensure that the results make sense semantically.

## Step 5: Test the CSV Loading Functionality

Create a test script called `test_csv_loading.py`:

```python
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
```

Run this test script:

```bash
python test_csv_loading.py
```

## Step 6: Test the FastAPI Application

Now, let's test the FastAPI application locally:

1. Run the API server:

```bash
cd api
uvicorn main:app --reload
```

2. In a new terminal, test the API endpoints using curl:

```bash
# Test the health endpoint
curl http://localhost:8000/health

# Test the search endpoint
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"what is semantic search?", "top_k":3}'
```

3. Alternatively, access the interactive Swagger UI in your browser:
   - Open http://localhost:8000/docs
   - Try out the search endpoint with different queries

## Step 7: Test the Index Building Script

Test the index building script with your sample data:

```bash
python scripts/build_index.py \
  --csv data/sample_docs.csv \
  --text-column text \
  --id-column id \
  --data-dir ./test_data
```

Verify that the index files are created in the specified directory.

## Step 8: Test Error Handling

Create test cases that might cause errors to ensure your system handles them gracefully:

1. Try searching before building an index
2. Try loading a non-existent index
3. Provide a CSV file with missing columns
4. Use invalid model names

## Step 9: Test Performance with Larger Datasets

If you plan to use the system with larger datasets, create a test with a bigger sample:

```python
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
```

## Step 10: Verify Memory Usage

For larger datasets, it's crucial to monitor memory usage:

```python
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
```

## Step 11: End-to-End Test

Create an end-to-end test that covers the complete workflow:

```python
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
```

## Step 12: Validate Docker Setup Locally

Before deploying to EC2, test your Docker setup locally:

1. Build Docker images:

```bash
docker build -t semantic-search-api -f Dockerfile.api .
docker build -t semantic-search-indexer -f Dockerfile.indexer .
```

2. Run the index builder container:

```bash
docker run --rm -v $(pwd)/data:/app/data semantic-search-indexer \
  --csv /app/data/sample_docs.csv \
  --text-column text \
  --id-column id \
  --data-dir /app/data
```

3. Start the API container:

```bash
docker run -p 8000:8000 -v $(pwd)/data:/app/data semantic-search-api
```

4. Test the API:

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"what is semantic search?", "top_k":3}'
```

5. Test using Docker Compose:

```bash
docker-compose up -d
curl -X POST "http://localhost/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"what is semantic search?", "top_k":3}'
```

## Conclusion

By thoroughly testing your semantic search system locally, you can identify and fix issues before deploying to the cloud. This process helps ensure a smoother deployment and reduces the risk of unexpected problems in production.

Once all tests pass successfully, you can confidently proceed with your EC2 deployment as outlined in the main guide.
