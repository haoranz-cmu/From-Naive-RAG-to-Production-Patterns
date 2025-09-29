"""
RAG (Retrieval-Augmented Generation) System

This module implements a simple RAG system using sentence transformers and FAISS for 
semantic search on a corpus of passages.
"""

import os
import logging
import yaml
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class NaiveRAG:
    """
    Retrieval-Augmented Generation (RAG) System class that handles:
    - Loading training and test data
    - Creating embeddings using sentence transformers
    - Building a FAISS index for semantic search
    - Retrieving relevant passages for queries
    - Evaluating system performance
    """
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', config_file: str = None):
        """
        Initialize the RAG system
        
        Args:
            model_name: Name of the sentence transformer model to use
            config_file: Path to configuration file (optional)
        """
        # Load configuration if provided
        self.config = self.load_config(config_file) if config_file else self.get_default_config()
        
        self.model_name = self.config['model']['name']
        self.model = None
        self.training_data = []
        self.test_data = {}
        self.embeddings = None
        self.index = None
        self.log_file = self.setup_logging()
        
        logging.info(f"RAG System initialized with model: {self.model_name}")

    # ============================
    # Config & Logging (existing)
    # ============================
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            logging.info(f"Configuration loaded from {config_file}")
            return config
        except Exception as e:
            logging.warning(f"Failed to load config from {config_file}: {e}. Using defaults.")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Minimal fallback configuration (used only if no config file is provided).
        
        Returns:
            Default configuration dictionary
        """
        return {
            'model': {
                'name': 'sentence-transformers/all-MiniLM-L6-v2',
                'device': 'auto',
                'batch_size': 32
            },
            'data': {
                'training_file': 'data/training.csv',
                'test_file': 'data/test.csv',
                'passage_column': 'passage',
                'question_column': 'question',
                'answer_column': 'answer'
            },
            'retrieval': {
                'top_k': 1,
                'similarity_threshold': 0.0,
                'max_passage_length': 1000
            },
            'evaluation': {
                'num_samples': 10,
                'accuracy_threshold': 0.5,
                'verbose': True
            },
            'logging': {
                'level': 'INFO',
                'log_dir': 'logs',
                'console_output': True,
                'file_output': True
            },
            'system': {
                'max_memory_usage': '4GB',
                'num_workers': 4,
                'cache_embeddings': True
            }
        }
    
    def setup_logging(self) -> str:
        """
        Setup logging configuration
        
        Returns:
            Path to the log file
        """
        # Create log directory if it doesn't exist
        log_dir = "../logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"rag_system_{timestamp}.log")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # Also print to console
            ]
        )
        
        logging.info(f"Logging initialized. Log file: {log_file}")
        return log_file

    # ============================
    # Data I/O
    # ============================
    def _validate_dataframe_columns(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """Validate that required columns exist in a dataframe."""
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def load_training_data(self, file_path: str) -> List[str]:
        """
        Load training data from CSV file
        
        Args:
            file_path: Path to the CSV file containing training data
            
        Returns:
            List of passages
        """
        logging.info(f"Loading training data from {file_path}")
        df = pd.read_csv(file_path)
        self._validate_dataframe_columns(df, ["passage"])
        self.training_data = df["passage"].tolist()
        return self.training_data
    
    def load_test_data(self, file_path: str) -> Dict[str, str]:
        """
        Load test data from CSV file
        
        Args:
            file_path: Path to the CSV file containing test data
            
        Returns:
            Dictionary mapping questions to answers
        """
        logging.info(f"Loading test data from {file_path}")
        df = pd.read_csv(file_path)
        self._validate_dataframe_columns(df, ["question", "answer"])
        self.test_data = dict(zip(df["question"].tolist(), df["answer"].tolist()))
        return self.test_data
    
    def initialize_model(self) -> SentenceTransformer:
        """
        Initialize the sentence transformer model
        
        Returns:
            Initialized SentenceTransformer model
        """
        logging.info(f"Initializing sentence transformer model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        return self.model
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for given texts
        
        Args:
            texts: List of texts to encode
            
        Returns:
            NumPy array of embeddings
        """
        if self.model is None:
            self.initialize_model()
            
        logging.info(f"Generating embeddings for {len(texts)} texts")
        return self.model.encode(texts)
    
    def create_index(self, embeddings: Optional[np.ndarray] = None) -> faiss.IndexFlatL2:
        """
        Create FAISS index from embeddings
        
        Args:
            embeddings: NumPy array of embeddings (optional)
            
        Returns:
            FAISS index
        """
        if embeddings is not None:
            self.embeddings = embeddings
        elif self.embeddings is None and self.training_data:
            self.embeddings = self.get_embeddings(self.training_data)
        
        logging.info(f"Creating FAISS index with {self.embeddings.shape[0]} embeddings of dimension {self.embeddings.shape[1]}")
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        return self.index
    
    # ============================
    # Retrieval Internals
    # ============================
    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string to embedding (2D array shape: (1, dim))."""
        return self.get_embeddings([query])

    def _search(self, query_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search the FAISS index and return (distances, indices)."""
        distances, indices = self.index.search(query_vec, k)
        return distances, indices

    def _postprocess_indices(
        self,
        distances: np.ndarray,
        indices: np.ndarray,
        similarity_threshold: float,
    ) -> Tuple[List[int], List[float]]:
        """Apply optional similarity threshold over distances and return (kept_indices, kept_distances)."""
        if similarity_threshold <= 0.0:
            return list(indices[0]), list(distances[0])

        kept_idx: List[int] = []
        kept_dist: List[float] = []
        for i, dist in enumerate(distances[0]):
            # L2 distance based threshold: smaller is more similar
            if dist <= (1.0 - similarity_threshold):
                kept_idx.append(indices[0][i])
                kept_dist.append(dist)
        return kept_idx, kept_dist

    def search_similar_passages(self, query: str, k: int = None) -> List[str]:
        """
        Search for similar passages using FAISS index
        
        Args:
            query: Query text
            k: Number of results to return (uses config if None)
            
        Returns:
            List of retrieved passages
        """
        if self.index is None:
            raise ValueError("Index not created. Call create_index() first.")
        
        if k is None:
            k = self.config['retrieval']['top_k']
        
        logging.info(f"Searching for passages similar to query: '{query[:50]}...'")
        query_embedding = self._encode_query(query)
        distances, indices = self._search(query_embedding, k)
        
        # Apply similarity threshold if configured
        similarity_threshold = self.config['retrieval']['similarity_threshold']
        kept_idx, _ = self._postprocess_indices(distances, indices, similarity_threshold)
        return [self.training_data[i] for i in kept_idx]

    # ============================
    # Public API
    # ============================
    def prepare(self, training_file: str, test_file: Optional[str] = None) -> None:
        """Prepare the RAG system: load data, init model, build index."""
        self.load_training_data(training_file)
        if test_file:
            self.load_test_data(test_file)

        if self.model is None:
            self.initialize_model()
        self.embeddings = self.get_embeddings(self.training_data)
        self.create_index()

    def answer(self, query: str, k: int = 1, include_scores: bool = False) -> Dict[str, Any]:
        """Retrieve passages for a query and return a structured result."""
        if self.index is None:
            raise ValueError("Index not created. Call prepare() first.")

        k_eff = k or self.config['retrieval']['top_k']
        query_vec = self._encode_query(query)
        distances, indices = self._search(query_vec, k_eff)
        kept_idx, kept_dist = self._postprocess_indices(
            distances,
            indices,
            self.config['retrieval']['similarity_threshold'],
        )
        passages = [self.training_data[i] for i in kept_idx]
        result: Dict[str, Any] = {
            "query": query,
            "passages": passages,
            "meta": {"k": k_eff, "threshold": self.config['retrieval']['similarity_threshold']},
        }
        if include_scores:
            result["scores"] = kept_dist
        return result
    
    def evaluate_retrieval_accuracy(self, num_samples: int = None) -> Tuple[float, int, int]:
        """
        Evaluate retrieval accuracy on a subset of test data
        
        Args:
            num_samples: Number of samples to evaluate (uses config if None)
            
        Returns:
            Tuple of (accuracy, number correct, total samples)
        """
        if num_samples is None:
            num_samples = self.config['evaluation']['num_samples']
            
        logging.info(f"Evaluating retrieval accuracy on {num_samples} samples")
        test_subset = dict(list(self.test_data.items())[:num_samples])
        correct = 0
        
        for question, answer in test_subset.items():
            results = self.search_similar_passages(question)
            retrieved_text = " ".join(results).lower()
            
            if answer.lower() in retrieved_text:
                correct += 1
        
        accuracy = correct / len(test_subset)
        logging.info(f"Evaluation results: {correct}/{len(test_subset)} correct, accuracy: {accuracy:.1%}")
        return accuracy, correct, len(test_subset)

    def evaluate(self, num_samples: Optional[int] = None) -> Dict[str, Any]:
        """Wrapper that returns a dict for easier downstream consumption."""
        accuracy, correct, total = self.evaluate_retrieval_accuracy(num_samples=num_samples)
        return {"accuracy": accuracy, "correct": correct, "total": total}
    
    def test_sample_questions(self, num_samples: int = 3) -> None:
        """
        Test the RAG system with sample questions
        
        Args:
            num_samples: Number of sample questions to test
        """
        logging.info(f"Testing system with {num_samples} sample questions")
        print("=== RAG System Test ===")
        print(f"Training passages: {len(self.training_data)}")
        print(f"Test questions: {len(self.test_data)}")
        print("="*50)
        
        for i, (question, answer) in enumerate(list(self.test_data.items())[:num_samples]):
            print(f"\nTest {i+1}:")
            print(f"Question: {question}")
            print(f"Expected Answer: {answer}")
            
            results = self.search_similar_passages(question, k=1)
            print("Retrieved passages:")
            for j, passage in enumerate(results, 1):
                print(f"  {j}. {passage[:100]}...")
            
            print("-" * 50)
    
    def print_system_summary(self) -> None:
        """Print system summary information"""
        if self.embeddings is None:
            raise ValueError("Embeddings not created yet.")
            
        logging.info("Printing system summary...")
        print("=== System Summary ===")
        print(f"Model: {self.model_name}")
        print(f"Training passages: {len(self.training_data)}")
        print(f"Test questions: {len(self.test_data)}")
        print(f"Embedding dimension: {self.embeddings.shape[1]}")
        print("✓ Semantic search with FAISS")
        print("✓ Top-1 passage retrieval")
        print("• No answer generation (retrieval only)")
        logging.info("System summary printed")

    def status(self) -> Dict[str, Any]:
        """Return current system status for quick inspection."""
        return {
            "model": self.model_name,
            "num_training": len(self.training_data),
            "num_test": len(self.test_data),
            "embedding_dim": None if self.embeddings is None else int(self.embeddings.shape[1]),
            "index_ready": self.index is not None,
        }
    
    def run(self, training_file: str, test_file: str) -> None:
        """
        Run the complete RAG system pipeline
        
        Args:
            training_file: Path to the training data CSV file
            test_file: Path to the test data CSV file
        """
        logging.info("Starting RAG system execution...")
        
        try:
            # Prepare pipeline
            self.prepare(training_file=training_file, test_file=test_file)
            
            # Print system summary
            self.print_system_summary()
            print("\n" + "="*50)

            # Test sample questions
            print("\n=== Sample Questions Test ===")
            self.test_sample_questions()
            
            # Evaluate accuracy
            print("\n=== Evaluation ===")
            eval_res = self.evaluate()
            print(f"Accuracy: {eval_res['accuracy']:.1%} ({eval_res['correct']}/{eval_res['total']})")
            
            logging.info("RAG system execution completed successfully")
            logging.info(f"Log file saved at: {self.log_file}")
            
        except Exception as e:
            logging.error(f"Error during execution: {str(e)}")
            raise



# Run the main function
if __name__ == "__main__":
    import os
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RAG System with Configuration Support')
    parser.add_argument('--config', type=str, default='../config_rag.yaml', 
                       help='Path to configuration file (YAML)')
    parser.add_argument('--training-file', type=str, 
                       help='Override training file path')
    parser.add_argument('--test-file', type=str, 
                       help='Override test file path')
    args = parser.parse_args()
    
    # Resolve project root and config path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    # Initialize system
    if os.path.exists(config_path):
        print(f"Using configuration file: {config_path}")
        rag_system = NaiveRAG(config_file=config_path)
    else:
        print("No configuration file found, using minimal defaults")
        rag_system = NaiveRAG()
    
    # Resolve data paths from config (allow CLI override)
    cfg_train = rag_system.config['data']['training_file']
    cfg_test = rag_system.config['data']['test_file']
    training_file = args.training_file or (cfg_train if os.path.isabs(cfg_train) else os.path.join(project_root, cfg_train))
    test_file = args.test_file or (cfg_test if os.path.isabs(cfg_test) else os.path.join(project_root, cfg_test))
    
    print(f"Project root: {project_root}")
    print(f"Training file path: {training_file}")
    print(f"Test file path: {test_file}")
    print(f"Training file exists: {os.path.exists(training_file)}")
    print(f"Test file exists: {os.path.exists(test_file)}")
    
    rag_system.run(
        training_file=training_file,
        test_file=test_file
    )
