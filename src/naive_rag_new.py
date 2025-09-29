import os

import logging
import yaml
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import math
from typing import Union


class NaiveRAG:
    """
    Retrieval-Augmented Generation (RAG) System class that handles:
    - Loading training and test data
    - Creating embeddings using sentence transformers
    - Building a FAISS index for semantic search
    - Retrieving relevant passages for queries
    - Evaluating system performance
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialize the RAG system
        
        Args:
            model_name: Name of the sentence transformer model to use
            config_file: Path to configuration file (optional)
        """
        # Load configuration if provided
        self.config = self.load_config(config_file) if config_file else self.get_default_config()
        
        self.model_name = self.config['model']['name']
        self.embedding_model = SentenceTransformer(self.model_name)
        self.training_data = []
        self.test_data = {}
        self.embeddings = None
        self.index = None
        self.log_file = self.setup_logging()
        
        
        logging.info(f"RAG System initialized with model: {self.model_name}")

    def load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get_default_config(self) -> Dict[str, Any]:
        return {
            'model': {'name': 'sentence-transformers/all-MiniLM-L6-v2'},
            'data': {
                'training_file': 'data/training.csv',
                'test_file': 'data/test.csv',
                'vector_database_path': 'data/vector_database'
            }
        }

    def setup_logging(self) -> str:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return ''
    
   
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """

        Args:
            texts (List[str]): _description_

        Returns:
            np.ndarray: _description_
        """
        return self.embedding_model.encode(texts)
   
    def faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """
        Create a FAISS index from the embeddings
        """
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        return self.index


    def load_training_data(self, file_path: str) -> List[str]:
            """
            Load training data from CSV file
            
            Args:
                file_path: Path to the CSV file containing training data
                
            Returns:
                List of passages
            """
            
            df = pd.read_csv(file_path)
            self.training_data = df["passage"].tolist()
            self.embeddings = self.get_embeddings(self.training_data)
            self.index = self.faiss_index(self.embeddings)
            return self.training_data
    
    def save_vector_data(self, file_path: str) -> None:
        """
        Save the vector data to a file
        """
        os.makedirs(file_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(file_path, "faiss_index.bin"))
        with open(os.path.join(file_path, "metadata.pkl"), "wb") as f:
            pickle.dump({
                "training_data": self.training_data,
                "embeddings": self.embeddings
            }, f)
        print(f"Vector database saved to {file_path}")
    
    def save_index(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss_index.bin"))

    def load_index(self, path: str) -> bool:
        index_path = os.path.join(path, "faiss_index.bin")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            return True
        return False
    
    def establish_vector_database(self):
        self.load_training_data(self.config["data"]["training_file"])
        self.save_vector_data(self.config["data"]["vector_database_path"])
    
    
   


class VectorDatabase:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.vector_database_path = self.config["data"].get("vector_database_path", "data/vector_database")
        self.vector_database_file = self.config["data"].get("vector_database_file", "faiss_index.bin")
        self.metadata_file = self.config["data"].get("metadata_file", "metadata.pkl")
        self.embedding_model = SentenceTransformer(self.config["model"]["name"])
        result = self.load_vector_database(self.vector_database_path)
        if result is not None:
            self.training_data, self.embeddings, self.index = result
        else:
            self.training_data = []
            self.embeddings = None
            self.index = None
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """

        Args:
            texts (List[str]): _description_

        Returns:
            np.ndarray: _description_
        """
        return self.embedding_model.encode(texts)
   
    def load_vector_database(self, file_path: str) -> None:
        self.vector_database_path = file_path
        if os.path.exists(self.vector_database_path):
            with open(os.path.join(self.vector_database_path, "metadata.pkl"), "rb") as f:
                metadata = pickle.load(f)
            self.training_data = metadata["training_data"]
            self.embeddings = metadata["embeddings"]
            self.index = faiss.read_index(os.path.join(self.vector_database_path, "faiss_index.bin"))
            return self.training_data, self.embeddings, self.index
        else:
            return None
    
    
    def query_vector_database(self, query: str, 
                              k: int = 3,
                              include_scores: bool = True,
                              threshold: float | None = None
                              ) -> Dict[str, Any]:
        """
        Query the vector database for similar passages
        
        Args:
            query: The query text
            k: Number of results to return
            include_scores: Whether to include distance/similarity scores
            threshold: Optional distance threshold (lower distance = more similar)
        
        Returns:
            {"query": str, "passages": [str], "scores": [float], "meta": {"k": int, "threshold": float|None}}
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load_vector_database() first.")
        # 获取查询的嵌入向量
        query_embedding = self.get_embeddings([query])
        
        # 搜索向量数据库，获取k个最相似的结果
        distances, indices = self.index.search(query_embedding, k)
        
        # 将结果转换为一维数组
        distances = distances[0]
        indices = indices[0]
        
        # 如果设置了阈值，过滤结果
        if threshold is not None:
            valid_idx = distances < threshold
            distances = distances[valid_idx]
            indices = indices[valid_idx]
        
        passages = [self.training_data[int(idx)] for idx in indices]
        result: Dict[str, Any] = {
            "query": query,
            "passages": passages,
            "meta": {"k": int(k), "threshold": None if threshold is None else float(threshold)}
        }
        if include_scores:
            result["scores"] = [float(d) for d in distances]
        return result

class performanceEvaluator:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.vector_database = VectorDatabase(config_path)
        self.default_test_file_path = self.config["data"]["test_file"]
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_test_data(self) -> Tuple[List[str], List[str]]:
        df = pd.read_csv(self.default_test_file_path)
        questions = df["question"].tolist()
        answers = df["answer"].tolist()
        return questions, answers
    
    def accuracy(self, retrieved_passages: List[str], true_answer: str) -> int:
        return 1 if true_answer in " ".join(retrieved_passages) else 0
    
    def single_sample_evaluation(self, query: str, true_answer: str) -> Tuple[List[str], str]:
        res = self.vector_database.query_vector_database(query, k=3, include_scores=False)
        return res["passages"], true_answer
    
    def multiple_sample_evaluation(self, queries: List[str], answers: List[str]) -> float:
        accurate_count = 0
        total_count = len(queries)
        for q, a in zip(queries, answers):
            res = self.vector_database.query_vector_database(q, k=3, include_scores=False)
            accurate_count += self.accuracy(res["passages"], a)
        return accurate_count / total_count
        
if __name__ == "__main__":
    config_file = "config_rag.yaml"
    
    
    rag = NaiveRAG(config_file)
    rag.establish_vector_database()
    
    evaluator = performanceEvaluator(config_file)
    queries, answers = evaluator.load_test_data()
    accuracy = evaluator.multiple_sample_evaluation(queries, answers)
    print(f"Accuracy: {accuracy}")