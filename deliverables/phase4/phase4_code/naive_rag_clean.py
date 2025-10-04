#!/usr/bin/env python3
"""
Clean and Simple RAG System
No config files - everything is direct parameters
"""

import os
import logging
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional


class SimpleRAG:
    """
    Simple RAG system with direct parameters - no config files needed
    """
    
    def __init__(self, model_name: str):
        """
        Initialize RAG system
        
        Args:
            model_name: SentenceTransformer model name (e.g., "BAAI/bge-small-en-v1.5")
        """
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        self.training_data = []
        self.embeddings = None
        self.index = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(f"RAG System initialized with model: {model_name}")
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension of current model"""
        if hasattr(self.embedding_model, "get_sentence_embedding_dimension"):
            return int(self.embedding_model.get_sentence_embedding_dimension())
        # Fallback: encode a test sentence
        test_embedding = self.embedding_model.encode(["test"])
        return int(test_embedding.shape[1])
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        return self.embedding_model.encode(texts)
    
    def build_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """Build FAISS index from embeddings"""
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        return self.index
    
    def load_training_data(self, csv_file: str, passage_column: str = "passage") -> List[str]:
        """
        Load training data from CSV
        
        Args:
            csv_file: Path to CSV file
            passage_column: Name of column containing passages
            
        Returns:
            List of passages
        """
        df = pd.read_csv(csv_file)
        self.training_data = df[passage_column].tolist()
        logging.info(f"Loaded {len(self.training_data)} training passages")
        return self.training_data
    
    def build_and_save_database(self, 
                                training_csv: str,
                                output_dir: str,
                                passage_column: str = "passage",
                                tag: str = None) -> Tuple[str, str]:
        """
        Build embeddings and save vector database
        
        Args:
            training_csv: Path to training CSV file
            output_dir: Directory to save vector database
            passage_column: Name of column containing passages
            tag: Optional tag for filenames
            
        Returns:
            (index_path, metadata_path)
        """
        # Load training data
        self.load_training_data(training_csv, passage_column)
        
        # Create embeddings
        logging.info("Creating embeddings...")
        self.embeddings = self.encode_texts(self.training_data)
        
        # Build index
        logging.info("Building FAISS index...")
        self.build_index(self.embeddings)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filenames
        dim = self.get_embedding_dimension()
        model_slug = self.model_name.replace("/", "_").replace("-", "_")
        
        if tag:
            index_name = f"faiss_index_d{dim}_{model_slug}_{tag}.bin"
            meta_name = f"metadata_d{dim}_{model_slug}_{tag}.pkl"
        else:
            index_name = f"faiss_index_d{dim}_{model_slug}.bin"
            meta_name = f"metadata_d{dim}_{model_slug}.pkl"
        
        index_path = os.path.join(output_dir, index_name)
        meta_path = os.path.join(output_dir, meta_name)
        
        # Save index
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata = {
            "training_data": self.training_data,
            "embeddings": self.embeddings,
            "model_name": self.model_name,
            "embedding_dimension": dim,
            "created_at": datetime.now().isoformat()
        }
        
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)
        
        logging.info(f"Vector database saved:")
        logging.info(f"  Index: {index_path}")
        logging.info(f"  Metadata: {meta_path}")
        
        return index_path, meta_path
    
    def search(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Search for similar passages
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            Search results with passages and scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_and_save_database() first.")
        
        # Encode query
        query_embedding = self.encode_texts([query])
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            results.append({
                "rank": i + 1,
                "passage": self.training_data[int(idx)],
                "distance": float(dist),
                "similarity": float(1 / (1 + dist))  # Convert distance to similarity
            })
        
        return {
            "query": query,
            "results": results,
            "total_found": len(results)
        }


class SimpleVectorDB:
    """
    Simple vector database loader - no config files needed
    """
    
    def __init__(self, model_name: str, db_dir: str, index_file: str = None, metadata_file: str = None):
        """
        Initialize vector database
        
        Args:
            model_name: Model name for embedding
            db_dir: Directory containing vector database
            index_file: FAISS index filename (auto-detect if None)
            metadata_file: Metadata filename (auto-detect if None)
        """
        self.model_name = model_name
        self.db_dir = db_dir
        self.embedding_model = SentenceTransformer(model_name)
        
        # Auto-detect files if not provided
        if index_file is None or metadata_file is None:
            files = os.listdir(db_dir)
            index_files = [f for f in files if f.startswith("faiss_index") and f.endswith(".bin")]
            meta_files = [f for f in files if f.startswith("metadata") and f.endswith(".pkl")]
            
            if not index_files or not meta_files:
                raise FileNotFoundError(f"No vector database files found in {db_dir}")
            
            self.index_file = index_file or index_files[0]
            self.metadata_file = metadata_file or meta_files[0]
        else:
            self.index_file = index_file
            self.metadata_file = metadata_file
        
        # Load database
        self.load_database()
    
    def load_database(self):
        """Load vector database from disk"""
        index_path = os.path.join(self.db_dir, self.index_file)
        meta_path = os.path.join(self.db_dir, self.metadata_file)
        
        # Load index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        
        self.training_data = metadata["training_data"]
        self.embeddings = metadata["embeddings"]
        
        logging.info(f"Loaded vector database from {self.db_dir}")
        logging.info(f"  Training data: {len(self.training_data)} passages")
        logging.info(f"  Embedding dimension: {self.embeddings.shape[1]}")
    
    def search(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Search for similar passages
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            Search results
        """
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            results.append({
                "rank": i + 1,
                "passage": self.training_data[int(idx)],
                "distance": float(dist),
                "similarity": float(1 / (1 + dist))
            })
        
        return {
            "query": query,
            "results": results,
            "total_found": len(results)
        }


def build_rag_database(model_name: str, 
                      training_csv: str, 
                      output_dir: str,
                      tag: str = None) -> Tuple[str, str]:
    """
    Simple function to build RAG database
    
    Args:
        model_name: SentenceTransformer model name
        training_csv: Path to training CSV
        output_dir: Output directory
        tag: Optional tag for filenames
        
    Returns:
        (index_path, metadata_path)
    """
    rag = SimpleRAG(model_name)
    return rag.build_and_save_database(training_csv, output_dir, tag=tag)


def load_rag_database(model_name: str, db_dir: str) -> SimpleVectorDB:
    """
    Simple function to load RAG database
    
    Args:
        model_name: Model name for embeddings
        db_dir: Database directory
        
    Returns:
        SimpleVectorDB instance
    """
    return SimpleVectorDB(model_name, db_dir)


def main():
    """Example usage"""
    print("🚀 Simple RAG System Demo")
    print("=" * 40)
    
    # Example 1: Build database
    print("\n1. Building vector database...")
    index_path, meta_path = build_rag_database(
        model_name="BAAI/bge-small-en-v1.5",
        training_csv="data/training.csv",
        output_dir="data/vector_database_simple",
        tag="demo"
    )
    print(f"✅ Built database: {index_path}")
    
    # Example 2: Load and search
    print("\n2. Loading database and searching...")
    db = load_rag_database("BAAI/bge-small-en-v1.5", "data/vector_database_simple")
    
    # Search example
    results = db.search("What is the capital of France?", k=3)
    print(f"Query: {results['query']}")
    print(f"Found {results['total_found']} results:")
    for result in results['results']:
        print(f"  {result['rank']}. Similarity: {result['similarity']:.3f}")
        print(f"     {result['passage'][:100]}...")


if __name__ == "__main__":
    main()
