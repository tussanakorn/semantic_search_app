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