#!/usr/bin/env python3
"""
Evaluation System for Prompting Strategies
System for evaluating different prompting strategies
"""

import pandas as pd
import sys
import os
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime
import re

# Import necessary modules
import llm_pipeline as llm

class EvaluationSystem:
    """
    Evaluation system class for comparing the effectiveness of different prompting strategies
    """
    
    def __init__(self, test_data_path: str = "../data/test.csv"):
        """
        Initialize evaluation system
        
        Args:
            test_data_path: Path to test data
        """
        self.test_data_path = test_data_path
        
        # Check current working directory and adjust config file paths
        import os
        if os.path.exists("../config_rag.yaml"):
            # Running from src directory
            rag_config = "../config_rag.yaml"
            llm_config = "../config_llm.yaml"
        else:
            # Running from root directory
            rag_config = "config_rag.yaml"
            llm_config = "config_llm.yaml"
            
        print(f"🔧 Using config files: {rag_config}, {llm_config}")
        self.pipeline = llm.LLM_Pipeline(
            rag_config_file=rag_config,
            llm_config_file=llm_config
        )
        print(f"🔧 VectorDatabase index loaded: {self.pipeline.vd.index is not None}")
        print(f"🔧 VectorDatabase training data: {len(self.pipeline.vd.training_data) if self.pipeline.vd.training_data else 0}")
        print(f"🔧 VectorDatabase path: {self.pipeline.vd.vector_database_path}")
        print(f"🔧 Path exists: {os.path.exists(self.pipeline.vd.vector_database_path)}")
        self.results = {}
        
        # Prompting strategies to test
        self.strategies = ["basic", "few_shot", "persona"]
        
    def load_test_data(self, num_samples: int = 5) -> pd.DataFrame:
        """Load test data"""
        df = pd.read_csv(self.test_data_path)
        # Random sampling of specified number of samples
        return df.sample(n=min(num_samples, len(df)), random_state=50)
    
    def calculate_f1_score(self, prediction: str, reference: str) -> float:
        """
        Calculate F1 score
        
        Args:
            prediction: Predicted answer
            reference: True answer
            
        Returns:
            F1 score
        """
        def normalize_text(text):
            """Normalize text"""
            text = text.lower().strip()
            # Remove punctuation
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
        """
        Calculate exact match score
        
        Args:
            prediction: Predicted answer
            reference: True answer
            
        Returns:
            Exact match score (0.0 or 1.0)
        """
        pred_normalized = prediction.lower().strip()
        ref_normalized = reference.lower().strip()
        
        # Check for exact match or substring match
        if pred_normalized == ref_normalized:
            return 1.0
        elif ref_normalized in pred_normalized:
            return 1.0
        else:
            return 0.0
    
    def calculate_squad_metrics(self, predictions: List[str], 
                               references: List[str]) -> Dict[str, float]:
        """
        Calculate Squad-style metrics
        
        Args:
            predictions: List of predicted answers
            references: List of true answers
            
        Returns:
            Dictionary containing F1 and EM scores
        """
        f1_scores = []
        exact_matches = []
        
        for pred, ref in zip(predictions, references):
            f1 = self.calculate_f1_score(pred, ref)
            em = self.calculate_exact_match(pred, ref)
            
            f1_scores.append(f1)
            exact_matches.append(em)
        
        return {
            "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            "exact_match": sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
        }
    
    def evaluate_strategy(self, strategy: str, test_data: pd.DataFrame, k: int = 1) -> Dict[str, Any]:
        """
        Evaluate a single prompting strategy
        
        Args:
            strategy: Strategy name
            test_data: Test data
            k: Number of documents to retrieve (default: 1)
            
        Returns:
            Evaluation results dictionary
        """
        print(f"\n🔄 Evaluating {strategy} strategy with k={k}...")
        
        predictions = []
        references = []
        contexts = []
        detailed_results = []
        
        for idx, row in test_data.iterrows():
            question = row['question']
            true_answer = row['answer']
            
            try:
                # Use custom k for retrieval
                context = self.pipeline.generate_context(question, k=k)
                contexts.append(context)
                
                # Generate answer
                answer = self.pipeline.generate_answer(
                    query=question,
                    strategy=strategy,
                    k=k,  # Use custom k
                    temperature=0.1,
                    max_tokens=150
                )
                
                predictions.append(answer)
                references.append(true_answer)
                
                # Calculate metrics for individual question
                f1 = self.calculate_f1_score(answer, true_answer)
                em = self.calculate_exact_match(answer, true_answer)
                
                # Get complete prompt
                prompt = self.pipeline.prompting.generate_prompt(strategy, question, context)
                
                detailed_results.append({
                    "question": question,
                    "true_answer": true_answer,
                    "predicted_answer": answer,
                    "context": context,
                    "prompt": prompt,
                    "f1_score": f1,
                    "exact_match": em
                })
                
                print(f"  ✅ Question {idx+1}/{len(test_data)}: {question[:50]}...")
                print(f"     True: {true_answer} | Predicted: {answer[:50]}...")
                print(f"     F1: {f1:.3f} | EM: {em}")
                
            except Exception as e:
                print(f"  ❌ Error with question {idx+1}: {e}")
                predictions.append("")
                references.append(true_answer)
                contexts.append("")
                detailed_results.append({
                    "question": question,
                    "true_answer": true_answer,
                    "predicted_answer": "",
                    "context": "",
                    "f1_score": 0.0,
                    "exact_match": 0.0
                })
        
        # Calculate overall metrics
        metrics = self.calculate_squad_metrics(predictions, references)
        
        # Calculate accuracy (simple matching)
        exact_matches = sum(1 for pred, ref in zip(predictions, references) 
                          if ref.lower() in pred.lower())
        accuracy = exact_matches / len(predictions)
        
        return {
            "strategy": strategy,
            "f1_score": metrics["f1"],
            "exact_match": metrics["exact_match"],
            "accuracy": accuracy,
            "total_questions": len(predictions),
            "successful_answers": len([p for p in predictions if p.strip()]),
            "detailed_results": detailed_results
        }
    
    def run_evaluation(self, k: int = 1) -> Dict[str, Any]:
        """
        Run complete evaluation
        
        Args:
            k: Number of documents to retrieve (default: 1)
            
        Returns:
            Evaluation results for all strategies
        """
        print("🚀 Starting Evaluation of Prompting Strategies")
        print(f"📊 Using k={k} for retrieval")
        print("=" * 60)
        
        # Load test data
        test_data = self.load_test_data(num_samples=5)
        print(f"📊 Loaded {len(test_data)} test questions")
        
        # Display test questions
        print("\n📋 Test Questions:")
        for idx, row in test_data.iterrows():
            print(f"  {idx+1}. {row['question']} (Answer: {row['answer']})")
        
        # Evaluate each strategy
        all_results = {}
        
        for strategy in self.strategies:
            try:
                result = self.evaluate_strategy(strategy, test_data, k=k)
                all_results[strategy] = result
                
                print(f"\n📈 {strategy.upper()} Results:")
                print(f"  F1-Score: {result['f1_score']:.4f}")
                print(f"  Exact Match: {result['exact_match']:.4f}")
                print(f"  Accuracy: {result['accuracy']:.4f}")
                print(f"  Successful Answers: {result['successful_answers']}/{result['total_questions']}")
                
            except Exception as e:
                print(f"❌ Error evaluating {strategy}: {e}")
                all_results[strategy] = None
        
        self.results = all_results
        return all_results
    
    def find_best_strategy(self) -> Tuple[str, Dict[str, Any]]:
        """
        Find best strategy
        
        Returns:
            (Best strategy name, Best strategy results)
        """
        if not self.results:
            return None, None
        
        # Sort by F1 score
        valid_results = {k: v for k, v in self.results.items() if v is not None}
        
        if not valid_results:
            return None, None
        
        best_strategy = max(valid_results.keys(), 
                          key=lambda x: valid_results[x]['f1_score'])
        
        return best_strategy, valid_results[best_strategy]
    
    def generate_report(self) -> str:
        """
        Generate evaluation report
        
        Returns:
            Evaluation report string
        """
        if not self.results:
            return "No results available."
        
        report = []
        report.append("# Prompting Strategies Evaluation Report")
        report.append("=" * 50)
        report.append("")
        
        # Results summary
        report.append("## Results Summary")
        report.append("")
        report.append("| Strategy | F1-Score | Exact Match | Accuracy |")
        report.append("|----------|----------|-------------|----------|")
        
        for strategy, result in self.results.items():
            if result:
                report.append(f"| {strategy} | {result['f1_score']:.4f} | {result['exact_match']:.4f} | {result['accuracy']:.4f} |")
        
        # Find best strategy
        best_strategy, best_result = self.find_best_strategy()
        
        if best_strategy:
            report.append("")
            report.append(f"## Best Strategy: {best_strategy.upper()}")
            report.append("")
            report.append(f"**F1-Score**: {best_result['f1_score']:.4f}")
            report.append(f"**Exact Match**: {best_result['exact_match']:.4f}")
            report.append(f"**Accuracy**: {best_result['accuracy']:.4f}")
            report.append("")
            
            # Hypothesis analysis
            report.append("## Hypothesis Analysis")
            report.append("")
            
            if best_strategy == "basic":
                report.append("**Why Basic Strategy Performed Best:**")
                report.append("- Simple and direct prompting format")
                report.append("- Minimal cognitive overhead for the model")
                report.append("- Clear task specification without distractions")
                report.append("- Effective for factual question-answering tasks")
            elif best_strategy == "persona":
                report.append("**Why Persona Strategy Performed Best:**")
                report.append("- Expert persona provides authoritative context")
                report.append("- Structured response format")
                report.append("- Professional tone improves answer quality")
            elif best_strategy == "few_shot":
                report.append("**Why Few-shot Strategy Performed Best:**")
                report.append("- Examples provide clear format guidance")
                report.append("- Demonstrates expected answer style")
                report.append("- Helps model understand task requirements")
            
            report.append("")
            report.append("## Detailed Results")
            report.append("")
            
            # Detailed results
            for strategy, result in self.results.items():
                if result:
                    report.append(f"### {strategy.upper()} Strategy")
                    report.append("")
                    for i, detail in enumerate(result['detailed_results']):
                        report.append(f"**Question {i+1}**: {detail['question']}")
                        report.append(f"- True Answer: {detail['true_answer']}")
                        report.append(f"- Predicted: {detail['predicted_answer']}")
                        report.append(f"- F1: {detail['f1_score']:.3f} | EM: {detail['exact_match']}")
                        report.append("")
                        report.append("**Retrieved Context:**")
                        report.append(f"```")
                        report.append(f"{detail['context'][:300]}...")
                        report.append(f"```")
                        report.append("")
                 
        return "\n".join(report)


def main():
    """Main function"""
    # Create evaluation system
    evaluator = EvaluationSystem()
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Find best strategy
    best_strategy, best_result = evaluator.find_best_strategy()
    
    if best_strategy:
        print(f"\n🏆 BEST STRATEGY: {best_strategy.upper()}")
        print(f"   F1-Score: {best_result['f1_score']:.4f}")
        print(f"   Exact Match: {best_result['exact_match']:.4f}")
        print(f"   Accuracy: {best_result['accuracy']:.4f}")
    
    # Generate report
    report = evaluator.generate_report()
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"evaluation_report_{timestamp}.md"
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 Report saved to {report_filename}")
    
    return results


if __name__ == "__main__":
    main()
