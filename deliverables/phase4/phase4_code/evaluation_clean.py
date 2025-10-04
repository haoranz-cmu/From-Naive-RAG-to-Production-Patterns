#!/usr/bin/env python3
"""
Clean and Simple Evaluation System
No config files - everything is direct parameters
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import re
from datetime import datetime

from naive_rag_clean import SimpleVectorDB
from llm_pipeline import LLM_Pipeline


class SimpleEvaluator:
    """
    Simple evaluation system with direct parameters
    """
    
    def __init__(self, model_name: str, db_dir: str, test_csv: str):
        """
        Initialize evaluator
        
        Args:
            model_name: Model name for embeddings
            db_dir: Vector database directory
            test_csv: Path to test CSV file
        """
        self.model_name = model_name
        self.db_dir = db_dir
        self.test_csv = test_csv
        
        # Load vector database
        self.db = SimpleVectorDB(model_name, db_dir)
        
        # Load test data
        self.test_data = pd.read_csv(test_csv)
        print(f"Loaded {len(self.test_data)} test questions")
    
    def calculate_f1_score(self, prediction: str, reference: str) -> float:
        """Calculate F1 score between prediction and reference"""
        def normalize_text(text):
            text = text.lower().strip()
            text = re.sub(r'[^\w\s]', '', text)
            return text
        
        pred_tokens = normalize_text(prediction).split()
        ref_tokens = normalize_text(reference).split()
        
        if not ref_tokens:
            return 1.0 if not pred_tokens else 0.0
        
        # Calculate precision and recall
        common_tokens = set(pred_tokens) & set(ref_tokens)
        
        if not pred_tokens:
            precision = 0.0
        else:
            precision = len(common_tokens) / len(pred_tokens)
        
        recall = len(common_tokens) / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def calculate_exact_match(self, prediction: str, reference: str) -> float:
        """Calculate exact match score"""
        pred_normalized = prediction.lower().strip()
        ref_normalized = reference.lower().strip()
        
        if pred_normalized == ref_normalized:
            return 1.0
        elif ref_normalized in pred_normalized:
            return 1.0
        else:
            return 0.0
    
    def evaluate_retrieval_only(self, k: int = 5, num_samples: int = 10, random_state: int = 42) -> Dict[str, float]:
        """
        Evaluate retrieval-only performance (no LLM)
        
        Args:
            k: Number of documents to retrieve
            num_samples: Number of test samples
            random_state: Random seed for reproducible results
            
        Returns:
            Dictionary with accuracy metrics
        """
        # Sample test data with fixed seed
        sample_data = self.test_data.sample(n=min(num_samples, len(self.test_data)), random_state=random_state)
        
        correct = 0
        total = len(sample_data)
        
        for _, row in sample_data.iterrows():
            question = row['question']
            true_answer = row['answer']
            
            # Search for relevant passages
            results = self.db.search(question, k=k)
            retrieved_passages = [r['passage'] for r in results['results']]
            
            # Check if true answer is in any retrieved passage
            found = any(true_answer.lower() in passage.lower() for passage in retrieved_passages)
            if found:
                correct += 1
        
        accuracy = correct / total
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "k": k
        }
    
    def evaluate_with_llm(self, 
                         strategy: str = "basic",
                         k: int = 5, 
                         num_samples: int = 5,
                         llm_config: str = "config_llm.yaml",
                         random_state: int = 42) -> Dict[str, Any]:
        """
        Evaluate with LLM pipeline
        
        Args:
            strategy: Prompting strategy (basic, few_shot, persona)
            k: Number of documents to retrieve
            num_samples: Number of test samples
            llm_config: LLM configuration file
            random_state: Random seed for reproducible results
            
        Returns:
            Evaluation results
        """
        # Sample test data with fixed seed
        sample_data = self.test_data.sample(n=min(num_samples, len(self.test_data)), random_state=random_state)
        
        # Initialize LLM pipeline
        try:
            pipeline = LLM_Pipeline(
                rag_config_file="config_rag.yaml",  # Still need this for LLM pipeline
                llm_config_file=llm_config
            )
            
            # Override vector database
            pipeline.vd = self.db
            
        except Exception as e:
            print(f"Error initializing LLM pipeline: {e}")
            return {"error": str(e)}
        
        predictions = []
        references = []
        f1_scores = []
        exact_matches = []
        
        for idx, row in sample_data.iterrows():
            question = row['question']
            true_answer = row['answer']
            
            try:
                # Generate answer using LLM
                answer = pipeline.generate_answer(
                    query=question,
                    strategy=strategy,
                    k=k,
                    temperature=0.1,
                    max_tokens=150
                )
                
                predictions.append(answer)
                references.append(true_answer)
                
                # Calculate metrics
                f1 = self.calculate_f1_score(answer, true_answer)
                em = self.calculate_exact_match(answer, true_answer)
                
                f1_scores.append(f1)
                exact_matches.append(em)
                
                print(f"Q{idx+1}: {question[:50]}...")
                print(f"  True: {true_answer}")
                print(f"  Pred: {answer[:50]}...")
                print(f"  F1: {f1:.3f}, EM: {em}")
                print()
                
            except Exception as e:
                print(f"Error with question {idx+1}: {e}")
                predictions.append("")
                references.append(true_answer)
                f1_scores.append(0.0)
                exact_matches.append(0.0)
        
        # Calculate overall metrics
        avg_f1 = np.mean(f1_scores)
        avg_em = np.mean(exact_matches)
        
        return {
            "strategy": strategy,
            "k": k,
            "num_samples": len(predictions),
            "avg_f1": avg_f1,
            "avg_exact_match": avg_em,
            "predictions": predictions,
            "references": references,
            "f1_scores": f1_scores,
            "exact_matches": exact_matches
        }
    
    def run_comparison(self, 
                      strategies: List[str] = None,
                      k_values: List[int] = None,
                      num_samples: int = 5) -> pd.DataFrame:
        """
        Run comparison across different strategies and k values
        
        Args:
            strategies: List of strategies to test
            k_values: List of k values to test
            num_samples: Number of test samples
            
        Returns:
            DataFrame with results
        """
        if strategies is None:
            strategies = ["basic", "few_shot", "persona"]
        if k_values is None:
            k_values = [3, 5, 10]
        
        results = []
        
        for strategy in strategies:
            for k in k_values:
                print(f"\n🧪 Testing {strategy} strategy with k={k}")
                
                result = self.evaluate_with_llm(
                    strategy=strategy,
                    k=k,
                    num_samples=num_samples
                )
                
                if "error" not in result:
                    results.append({
                        "strategy": strategy,
                        "k": k,
                        "f1_score": result["avg_f1"],
                        "exact_match": result["avg_exact_match"],
                        "num_samples": result["num_samples"]
                    })
                    print(f"  F1: {result['avg_f1']:.4f}, EM: {result['avg_exact_match']:.4f}")
                else:
                    print(f"  Error: {result['error']}")
        
        return pd.DataFrame(results)


def main():
    """Example usage"""
    print("🚀 Simple Evaluation Demo")
    print("=" * 40)
    
    # Example: Evaluate retrieval-only performance
    evaluator = SimpleEvaluator(
        model_name="BAAI/bge-small-en-v1.5",
        db_dir="data/vector_database_simple",
        test_csv="data/test.csv"
    )
    
    # Test retrieval-only
    print("\n1. Testing retrieval-only performance...")
    retrieval_results = evaluator.evaluate_retrieval_only(k=5, num_samples=10)
    print(f"Retrieval Accuracy: {retrieval_results['accuracy']:.4f}")
    print(f"Correct: {retrieval_results['correct']}/{retrieval_results['total']}")
    
    # Test with LLM (if available)
    print("\n2. Testing with LLM...")
    try:
        llm_results = evaluator.evaluate_with_llm(
            strategy="basic",
            k=3,
            num_samples=3
        )
        print(f"F1 Score: {llm_results['avg_f1']:.4f}")
        print(f"Exact Match: {llm_results['avg_exact_match']:.4f}")
    except Exception as e:
        print(f"LLM evaluation failed: {e}")


if __name__ == "__main__":
    main()
