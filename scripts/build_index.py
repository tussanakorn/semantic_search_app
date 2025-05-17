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