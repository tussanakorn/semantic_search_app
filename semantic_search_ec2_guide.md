# LLM Semantic Search Deployment on EC2 with Docker

This guide walks through deploying a containerized semantic search system on an AWS EC2 instance using sentence-transformers for embeddings and FAISS for efficient vector search.

## Overview

Semantic search uses the meaning of text rather than just keywords. Our implementation will:

1. Convert documents into vector embeddings using a pre-trained model
2. Index these vectors using FAISS for efficient similarity search
3. Embed user queries and find the most semantically similar documents
4. Containerize the system with Docker
5. Deploy the containers on an EC2 instance using Docker Compose

## Architecture

```
                  ┌────────────────────────────────────────────────┐
                  │             EC2 Instance                       │
                  │                                                │
┌──────────┐      │  ┌──────────────────────────────────────────┐  │      ┌───────────┐
│  Client  │      │  │           Docker Environment             │  │      │ Document  │
│ (Users)  │──────┼──►                                          │  │      │  Corpus   │
└──────────┘      │  │  ┌─────────────┐      ┌──────────────┐   │  │      └─────┬─────┘
                  │  │  │   NGINX     │      │  API Service  │   │  │            │
                  │  │  │  Container  │──────► (FastAPI in   │   │  │            │
                  │  │  │             │      │   Container)  │   │  │            │
                  │  │  └─────────────┘      └──────┬───────┘   │  │            │
                  │  │                              │           │  │            │
                  │  │                       ┌──────▼───────┐   │  │            │
                  │  │                       │   Vector     │◄──┼──┼────────────┘
                  │  │                       │  Database    │   │  │
                  │  │                       │  Container   │   │  │
                  │  │                       │  (FAISS)     │   │  │
                  │  │                       └──────────────┘   │  │
                  │  │                                          │  │
                  │  └──────────────────────────────────────────┘  │
                  │                                                │
                  └────────────────────────────────────────────────┘
```

## Prerequisites

- AWS account with EC2 access
- Basic knowledge of Python, Docker, and AWS
- SSH client for connecting to EC2
- Documents for your corpus

## Step 1: Launch an EC2 Instance

1. Log into AWS Console and navigate to EC2
2. Launch a new instance with these specifications:
   - Amazon Linux 2023 AMI
   - t2.large instance type (minimum recommended for ML workloads)
   - At least 30GB storage
   - Create or use existing key pair for SSH access
   - Configure security group to allow:
     - SSH (port 22) from your IP
     - HTTP (port 80) and/or HTTPS (port 443)

3. Connect to your instance:
```bash
ssh -i /path/to/your-key.pem ec2-user@your-instance-public-dns
```

## Step 2: Install Docker and Docker Compose

Update the system and install Docker:

```bash
# Update system
sudo yum update -y

# Install Docker
sudo yum install -y docker
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Log out and log back in for group changes to take effect
exit
```

Reconnect to your instance and verify Docker installation:

```bash
ssh -i /path/to/your-key.pem ec2-user@your-instance-public-dns
docker --version
docker-compose --version
```

## Step 3: Create the Project Structure

Create a project directory and necessary files:

```bash
mkdir -p semantic_search_app/{api,data,scripts}
cd semantic_search_app
```

### 3.1: Create Python Files

#### 3.1.1: Document Processor (`api/document_processor.py`)

```python
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import time

class DocumentProcessor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.document_embeddings = None
        self.faiss_index = None
        
    def add_documents(self, documents, document_ids=None):
        """
        Add documents to the corpus
        documents: list of strings
        document_ids: optional list of IDs corresponding to documents
        """
        if document_ids is None:
            document_ids = [i for i in range(len(self.documents), len(self.documents) + len(documents))]
        
        self.documents.extend([{'id': doc_id, 'text': doc} for doc_id, doc in zip(document_ids, documents)])
        print(f"Added {len(documents)} documents to corpus. Total: {len(self.documents)}")
    
    def process_documents_from_csv(self, filepath, text_column, id_column=None):
        """Load documents from a CSV file"""
        df = pd.read_csv(filepath)
        texts = df[text_column].tolist()
        
        ids = None
        if id_column and id_column in df.columns:
            ids = df[id_column].tolist()
            
        self.add_documents(texts, ids)
        
    def create_embeddings(self):
        """Create embeddings for all documents in the corpus"""
        start_time = time.time()
        texts = [doc['text'] for doc in self.documents]
        self.document_embeddings = self.model.encode(texts)
        print(f"Created embeddings in {time.time() - start_time:.2f} seconds")
        
    def build_faiss_index(self):
        """Build FAISS index from document embeddings"""
        if self.document_embeddings is None:
            raise ValueError("No embeddings available. Run create_embeddings() first.")
            
        vector_dimension = self.document_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(vector_dimension)
        
        # Convert to float32 as FAISS requires this format
        embeddings_float32 = np.array(self.document_embeddings).astype('float32')
        self.faiss_index.add(embeddings_float32)
        print(f"FAISS index built with {self.faiss_index.ntotal} vectors of dimension {vector_dimension}")
        
    def search(self, query, top_k=5):
        """
        Search for most similar documents to the query
        Returns list of dictionaries with document info and similarity score
        """
        query_embedding = self.model.encode([query])
        query_embedding_float32 = np.array(query_embedding).astype('float32')
        
        # Search the FAISS index
        distances, indices = self.faiss_index.search(query_embedding_float32, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):  # Ensure the index is valid
                doc = self.documents[idx]
                results.append({
                    'document_id': doc['id'],
                    'text': doc['text'],
                    'score': 1 - distances[0][i]/100  # Convert distance to similarity score
                })
        
        return results
    
    def save_index(self, directory='/app/data'):
        """Save FAISS index and documents to disk"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save the FAISS index
        faiss.write_index(self.faiss_index, f"{directory}/document_index.faiss")
        
        # Save the documents and embeddings
        with open(f"{directory}/documents.pkl", 'wb') as f:
            pickle.dump(self.documents, f)
            
        with open(f"{directory}/embeddings.pkl", 'wb') as f:
            pickle.dump(self.document_embeddings, f)
            
        print(f"Index and documents saved to {directory}")
    
    def load_index(self, directory='/app/data'):
        """Load FAISS index and documents from disk"""
        # Load the FAISS index
        self.faiss_index = faiss.read_index(f"{directory}/document_index.faiss")
        
        # Load the documents and embeddings
        with open(f"{directory}/documents.pkl", 'rb') as f:
            self.documents = pickle.load(f)
            
        with open(f"{directory}/embeddings.pkl", 'rb') as f:
            self.document_embeddings = pickle.load(f)
            
        print(f"Loaded index with {len(self.documents)} documents")
```

#### 3.1.2: FastAPI Service (`api/main.py`)

```python
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from document_processor import DocumentProcessor
import os

app = FastAPI(title="Semantic Search API")

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize document processor
doc_processor = DocumentProcessor()

# Data directory path from environment variable or default
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")

# Try to load existing index
@app.on_event("startup")
async def startup_load_index():
    try:
        doc_processor.load_index(DATA_DIR)
        print(f"Loaded existing index from {DATA_DIR}")
    except Exception as e:
        print(f"No existing index found in {DATA_DIR}: {e}")
        print("Please build an index before using the search functionality")

class SearchQuery(BaseModel):
    query: str
    top_k: Optional[int] = 5

class SearchResult(BaseModel):
    document_id: str
    text: str
    score: float

class SearchResponse(BaseModel):
    results: List[SearchResult]

@app.post("/search", response_model=SearchResponse)
def search(search_query: SearchQuery):
    if doc_processor.faiss_index is None:
        raise HTTPException(status_code=500, detail="Search index not initialized")
    
    results = doc_processor.search(search_query.query, search_query.top_k)
    return {"results": results}

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "documents_indexed": len(doc_processor.documents) if doc_processor.documents else 0,
        "data_directory": DATA_DIR
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

#### 3.1.3: Index Building Script (`scripts/build_index.py`)

```python
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from api.document_processor import DocumentProcessor
import argparse

def main():
    parser = argparse.ArgumentParser(description='Build FAISS index from documents')
    parser.add_argument('--csv', type=str, help='Path to CSV file containing documents')
    parser.add_argument('--text-column', type=str, default='text', help='Column name containing document text')
    parser.add_argument('--id-column', type=str, help='Optional column name containing document IDs')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2', help='SentenceTransformer model name')
    parser.add_argument('--data-dir', type=str, default='/app/data', help='Directory to save index files')
    
    args = parser.parse_args()
    
    # Initialize document processor with specified model
    processor = DocumentProcessor(model_name=args.model)
    
    # Load documents from CSV
    if args.csv and os.path.exists(args.csv):
        print(f"Processing CSV file: {args.csv}")
        processor.process_documents_from_csv(args.csv, args.text_column, args.id_column)
    else:
        print(f"CSV file not found: {args.csv}")
        return
        
    # Create embeddings and build index
    processor.create_embeddings()
    processor.build_faiss_index()
    
    # Save index to disk
    processor.save_index(args.data_dir)
    
    print(f"Index built successfully with {len(processor.documents)} documents")

if __name__ == "__main__":
    main()
```

#### 3.1.4: Sample Data

```bash
cat > data/sample_docs.csv << EOF
id,title,text
1,"Introduction to Machine Learning","Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed. It focuses on developing algorithms that can learn from and make predictions on data."
2,"Natural Language Processing","NLP is a field of AI that focuses on the interaction between computers and humans through natural language. It helps machines understand, interpret and manipulate human language."
3,"Deep Learning Basics","Deep learning is a subset of machine learning that uses neural networks with many layers. These deep neural networks can automatically learn hierarchical features from data."
4,"Vector Databases","Vector databases are specialized systems designed to store and query embeddings or vector representations of data. They're optimized for similarity search operations."
5,"Semantic Search Explained","Semantic search is a data searching technique where a search query aims to not only find keywords but understand the contextual meaning of the search terms to improve accuracy."
6,"AWS EC2 Instances","Amazon Elastic Compute Cloud (EC2) provides scalable computing capacity in the cloud. It allows users to rent virtual computers on which to run their own applications."
7,"FAISS for Vector Search","Facebook AI Similarity Search (FAISS) is a library that allows developers to quickly search for embeddings of multimedia documents that are similar to each other."
8,"Text Embeddings","Text embeddings are numerical representations of text in a continuous vector space where semantically similar texts are mapped to nearby points."
9,"FastAPI Development","FastAPI is a modern, fast web framework for building APIs with Python based on standard Python type hints. It's designed to be easy to use and high performance."
10,"Sentence Transformers","Sentence Transformers is a Python framework for state-of-the-art sentence, paragraph and image embeddings. The models are based on transformer networks like BERT."
EOF
```

## Step 4: Create Docker Configuration Files

Now we'll create the Docker configuration files for our containers.

### 4.1: API Service Dockerfile

Create a Dockerfile for the API service:

```bash
cat > Dockerfile.api << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY api/ /app/
COPY data/ /app/data/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
```

### 4.2: Requirements File

Create a requirements.txt file with all the dependencies:

```bash
cat > requirements.txt << 'EOF'
fastapi==0.103.1
uvicorn==0.23.2
sentence-transformers==2.2.2
faiss-cpu==1.7.4
pandas==2.1.0
numpy==1.25.2
scikit-learn==1.3.0
EOF
```

### 4.3: NGINX Configuration File

Create an NGINX configuration file for the reverse proxy:

```bash
cat > nginx.conf << 'EOF'
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                     '$status $body_bytes_sent "$http_referer" '
                     '"$http_user_agent" "$http_x_forwarded_for"';
    
    access_log /var/log/nginx/access.log main;
    
    sendfile on;
    keepalive_timeout 65;
    
    upstream api {
        server api:8000;
    }
    
    server {
        listen 80;
        server_name _;
        
        location / {
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
EOF
```

### 4.4: Docker Compose File

Create a Docker Compose file to orchestrate all services:

```bash
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    container_name: semantic-search-api
    volumes:
      - ./data:/app/data
    environment:
      - DATA_DIR=/app/data
    restart: unless-stopped
    networks:
      - semantic-search-network

  nginx:
    image: nginx:1.25-alpine
    container_name: semantic-search-nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - api
    restart: unless-stopped
    networks:
      - semantic-search-network

networks:
  semantic-search-network:
    driver: bridge
EOF
```

### 4.5: Index Builder Dockerfile

Create a Dockerfile for the index builder:

```bash
cat > Dockerfile.indexer << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY api/ /app/api/
COPY scripts/ /app/scripts/
COPY data/ /app/data/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the entrypoint to the index builder script
ENTRYPOINT ["python", "scripts/build_index.py"]
EOF
```

## Step 5: Deploy and Run the Application

### 5.1: Build the Docker Images and Start Services

Build and start all containers using Docker Compose:

```bash
docker-compose up -d
```

### 5.2: Build the Initial Index

Run the index builder container to generate the vector index:

```bash
docker build -t semantic-search-indexer -f Dockerfile.indexer .
docker run --rm -v $(pwd)/data:/app/data semantic-search-indexer \
  --csv /app/data/sample_docs.csv \
  --text-column text \
  --id-column id \
  --data-dir /app/data
```

### 5.3: Verify the Application is Running

Check that all containers are running:

```bash
docker-compose ps
```

Test the API endpoint:

```bash
curl -X POST "http://localhost/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"what is semantic search?", "top_k":3}'
```

## Step 6: Managing and Maintaining the Deployment

### 6.1: Updating the Document Corpus

To update your document corpus and rebuild the index:

1. Add new documents to your CSV file in the data directory
2. Rebuild the index:

```bash
docker run --rm -v $(pwd)/data:/app/data semantic-search-indexer \
  --csv /app/data/your_updated_docs.csv \
  --text-column text \
  --id-column id \
  --data-dir /app/data
```

3. Restart the API container to load the new index:

```bash
docker-compose restart api
```

### 6.2: Updating the Application Code

To update your application code:

1. Modify the source code files as needed
2. Rebuild and restart the containers:

```bash
docker-compose down
docker-compose build
docker-compose up -d
```

### 6.3: Backup and Restore

To back up your data:

```bash
tar -czvf semantic_search_backup.tar.gz data/
```

To restore from a backup:

```bash
tar -xzvf semantic_search_backup.tar.gz
docker-compose restart api
```

## Production Considerations

For a production deployment, consider these enhancements:

1. **Security:**
   - Use HTTPS with a valid SSL certificate
   - Add API authentication (JWT, OAuth, etc.)
   - Restrict CORS to specific origins
   
2. **Scaling:**
   - Use Docker Swarm or Kubernetes for multi-node deployments
   - Implement load balancing for the API service
   - Use larger EC2 instances for bigger corpora
   
3. **Monitoring:**
   - Add logging with CloudWatch or similar
   - Set up alerts for errors or performance issues
   - Implement metrics tracking (response time, usage, etc.)
   
4. **Data Management:**
   - Implement periodic index rebuilds 
   - Set up backup and restore procedures
   - Consider using S3 for document storage

5. **Advanced FAISS Usage:**
   - Use IVF indexes for larger datasets
   - Implement quantization for memory efficiency
   - Consider GPU-enabled instances with faiss-gpu

## Customization Options

### Using Different Embedding Models

You can use different SentenceTransformer models by specifying the model name when building the index:

```bash
docker run --rm -v $(pwd)/data:/app/data semantic-search-indexer \
  --csv /app/data/your_docs.csv \
  --text-column text \
  --id-column id \
  --model all-mpnet-base-v2 \
  --data-dir /app/data
```

### Improving Search Quality

1. **Document Chunking:** Split large documents into smaller chunks for more precise search
2. **Query Expansion:** Enhance queries with synonyms or related terms
3. **Re-ranking:** Use a more powerful model to re-rank the top results

### Scaling to Larger Document Collections

For larger collections (100K+ documents):

1. Use IVF indexes in FAISS for better performance
2. Implement document filtering by metadata
3. Consider a distributed vector database like Milvus or Weaviate

## Conclusion

You now have a containerized semantic search system running on EC2 with Docker Compose! This setup provides a solid foundation that you can customize and expand based on your specific needs. The Docker-based deployment makes it easier to manage, update, and scale the application as your requirements grow.