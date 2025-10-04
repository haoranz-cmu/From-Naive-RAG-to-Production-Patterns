
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple
import logging
from dotenv import load_dotenv
try:
    from .langchain_native_pipeline import build_native_rag_pipeline
    from .enhanced_rag_features import build_enhanced_rag_pipeline
except ImportError:
    from langchain_native_pipeline import build_native_rag_pipeline
    from enhanced_rag_features import build_enhanced_rag_pipeline

# Load .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
GEMINI_MODEL = "gemini-2.0-flash-lite"


from RAGASCalculator import RAGASCalculator

class RAGASEvaluator:
    """
    RAGAS Evaluator - Evaluates RAG system performance
    """

    def __init__(self, test_data: pd.DataFrame, seed: int = 42,
                 num_samples: int = 5, strategy: str = "basic",
                 k: int = 5, top_k: int = 1):
        """
        Initialize RAGAS evaluator

        Args:
            test_data: Test data DataFrame
            seed: Random seed
            num_samples: Number of samples
            strategy: Prompt strategy
            k: Number of retrieved documents
            top_k: Number of returned documents
        """
        self.test_data = test_data
        self.ragas_calculator = RAGASCalculator()
        self.seed = seed
        self.num_samples = num_samples
        self.strategy = strategy
        self.k = k
        self.top_k = top_k

        # Initialize RAG systems
        self.naive_rag = None
        self.enhanced_rag = None

        logging.info(f"RAGAS Evaluator initialized with {num_samples} samples")

    def load_test_data(self) -> Tuple[List[str], List[str]]:
        """
        Load test data

        Returns:
            (questions, ground_truth_answers)
        """
        sampled_data = self.test_data.sample(
            n=min(self.num_samples, len(self.test_data)),
            random_state=self.seed
        )

        self.questions = sampled_data["question"].tolist()
        self.ground_truth_answers = sampled_data["answer"].tolist()
        
        # Save questions and ground truth answers to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sampled_data[["question", "answer"]].to_csv(f"data/test_sampled_{timestamp}.csv", index=False)
        

        logging.info(f"Loaded {len(self.questions)} test questions")
        return self.questions, self.ground_truth_answers

    def initialize_rag_systems(self):
        """Initialize RAG systems"""
        try:
            # Initialize basic RAG system
            logging.info("Initializing basic RAG system...")
            self.naive_rag = build_native_rag_pipeline(
                embedding_model="BAAI/bge-small-en-v1.5",
                training_csv="data/training.csv",
                output_dir="data/native_rag",
                llm_model=GEMINI_MODEL,
                tag="native_rag"
            )
            self.naive_rag.create_advanced_chain(self.strategy)
            logging.info("Basic RAG system initialized")

            # Initialize enhanced RAG system
            logging.info("Initializing enhanced RAG system...")
            self.enhanced_rag = build_enhanced_rag_pipeline(
                embedding_model="BAAI/bge-small-en-v1.5",
                training_csv="data/training.csv",
                output_dir="data/enhanced_rag",
                llm_model=GEMINI_MODEL,
                reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                tag="enhanced_rag"
            )
            self.enhanced_rag.create_enhanced_qa_chain(self.strategy)
            logging.info("Enhanced RAG system initialized")

        except Exception as e:
            logging.error(f"Failed to initialize RAG systems: {e}")
            raise

    def generate_answers(self, question: str, rag_system) -> Tuple[str, List[str]]:
        """
        Generate answers and retrieve context

        Args:
            question: Question
            rag_system: RAG system

        Returns:
            (answer, retrieved_contexts)
        """
        try:
            # Generate answer
            if hasattr(rag_system, 'query'):
                answer = rag_system.query(question)
            elif hasattr(rag_system, 'enhanced_query'):
                answer = rag_system.enhanced_query(question, k=self.k, rerank_k=self.top_k)
            else:
                answer = "Error: No query method found"

            # Retrieve context
            if hasattr(rag_system, 'search_documents'):
                docs = rag_system.search_documents(question, k=self.k)
                retrieved_contexts = [doc.page_content for doc in docs]
            elif hasattr(rag_system, 'enhanced_search'):
                # For enhanced system, use enhanced search
                reranked_results = rag_system.enhanced_search(
                    question, k=self.k, rerank_k=self.top_k
                )
                retrieved_contexts = [doc.page_content for doc, score in reranked_results]
            else:
                retrieved_contexts = []

            return answer, retrieved_contexts

        except Exception as e:
            logging.error(f"Failed to generate answer for question: {question}, error: {e}")
            return "Error generating answer", []

    def evaluate_rag_system(self, rag_system, system_name: str) -> Dict[str, float]:
        """
        Evaluate a single RAG system

        Args:
            rag_system: RAG system
            system_name: System name

        Returns:
            Evaluation results dictionary
        """
        logging.info(f"Evaluating {system_name} system...")

        generated_answers = []
        retrieved_contexts = []

        # Generate answers and retrieve context
        for question in self.questions:
            answer, contexts = self.generate_answers(question, rag_system)
            generated_answers.append(answer)
            retrieved_contexts.append(contexts)

        # Calculate RAGAS metrics - using async calls
        import asyncio
        
        async def calculate_metrics():
            faithfulness_score = await self.ragas_calculator.evaluate_faithfulness(
                generated_answers, 
                retrieved_contexts, 
                self.questions
            )

            precision_score, recall_score = await self.ragas_calculator.evaluate_context_precision(
                self.questions, 
                retrieved_contexts, 
                generated_answers, 
                self.ground_truth_answers
            )

            relevance_score = await self.ragas_calculator.evaluate_answer_relevance(
                self.questions, 
                generated_answers, 
                retrieved_contexts
            )
            
            return faithfulness_score, precision_score, recall_score, relevance_score
        
        # Run async calculation
        faithfulness_score, precision_score, recall_score, relevance_score = asyncio.run(calculate_metrics())

        results = {
            "faithfulness": faithfulness_score,
            "context_precision": precision_score,
            "context_recall": recall_score,
            "answer_relevance": relevance_score
        }
        self.save_results(generated_answers, 
                          retrieved_contexts, 
                          self.questions, 
                          self.ground_truth_answers, 
                          results, 
                          system_name)
        logging.info(f"{system_name} evaluation completed")
        return results

    def run_evaluation(self) -> Dict[str, Dict[str, float]]:
        """
        Run complete evaluation

        Returns:
            Evaluation results dictionary
        """
        logging.info("Starting RAGAS evaluation...")

        # Load test data
        self.load_test_data()

        # Initialize RAG systems
        self.initialize_rag_systems()

        # Evaluate basic RAG system
        naive_results = self.evaluate_rag_system(self.naive_rag, "Basic RAG")

        # Evaluate enhanced RAG system
        enhanced_results = self.evaluate_rag_system(self.enhanced_rag, "Enhanced RAG")

        # Return results
        results = {
            "basic_rag": naive_results,
            "enhanced_rag": enhanced_results
        }

        logging.info("RAGAS evaluation completed")
        return results

    
    
    def save_results(self, 
                     generated_answers: List[str], 
                     retrieved_contexts: List[str], 
                     questions: List[str], 
                     ground_truth_answers: List[str],
                     results: Dict[str, Dict[str, float]],
                     rag_system: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ragas_evaluation_results_{timestamp}.csv"
        
        df = pd.DataFrame({
            "questions": questions,
            "generated_answers": generated_answers,
            "retrieved_contexts": retrieved_contexts,
            "ground_truth_answers": ground_truth_answers,
            "rag_system": rag_system,
            "faithfulness": results["faithfulness"],
            "context_precision": results["context_precision"],
            "context_recall": results["context_recall"],
            "answer_relevance": results["answer_relevance"]

        })
        df.to_csv(filename, index=False)
        print(f"✅ Results saved to: {filename}")
        return filename

def main():
    """Main function - Run RAGAS evaluation"""
    print("🔬 RAGAS Evaluation Demo")
    print("=" * 50)
    
    try:
      
        
        # Load test data
        test_data = pd.read_csv("data/test.csv")
        print(f"✅ Loaded {len(test_data)} test samples")
        
        # Create evaluator
        evaluator = RAGASEvaluator(
            test_data=test_data,
            num_samples=20,  # Evaluate 20 samples (reduced time)
            strategy="basic",
            k=5,
            top_k=3
        )
        
        # Run evaluation
        print("\n🚀 Starting RAGAS evaluation...")
        results = evaluator.run_evaluation()
        

      
        # Display key results
        print("\n📊 Key Results:")
        print("=" * 30)
        for system, metrics in results.items():
            print(f"\n{system.upper().replace('_', ' ')}:")
            for metric, score in metrics.items():
                print(f"  {metric}: {score:.3f}")
        
        # Calculate improvements
        basic = results['basic_rag']
        enhanced = results['enhanced_rag']
        
        print("\n📈 Performance Improvement:")
        print("=" * 30)
        for metric in ['faithfulness', 'context_precision', 'context_recall', 'answer_relevance']:
            improvement = enhanced[metric] - basic[metric]
            print(f"{metric}: {improvement:+.3f}")
        
        print("\n🎯 Summary:")
        print("=" * 20)
        if enhanced['faithfulness'] > basic['faithfulness']:
            print("✅ Enhanced RAG shows improvement in faithfulness")
        else:
            print("❌ Enhanced RAG shows no improvement in faithfulness")
            
        if enhanced['answer_relevance'] > basic['answer_relevance']:
            print("✅ Enhanced RAG shows improvement in answer relevance")
        else:
            print("❌ Enhanced RAG shows no improvement in answer relevance")
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Please ensure data/test.csv exists")
    except Exception as e:
        print(f"❌ RAGAS evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
