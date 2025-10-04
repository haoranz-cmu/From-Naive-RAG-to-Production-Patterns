#!/usr/bin/env python3
"""
Enhanced RAG Features Implementation
Implementing two advanced RAG features: Query Rewriting and Reranking
"""

import logging
import pandas as pd
from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Import base RAG system
from langchain_native_pipeline import NativeLangChainRAG, OpenRouterLLM, GeminiLLM, GEMINI_AVAILABLE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class QueryRewriter:
    """
    Query Rewriter - Intelligently optimizes queries while avoiding over-complexification
    """
    
    def __init__(self, llm_model: str = "gpt-4o-mini", strategy: str = "balanced"):
        """
        Initialize query rewriter
        
        Args:
            llm_model: LLM model name
            strategy: Rewriting strategy ('conservative', 'balanced', 'aggressive')
        """
        # Select LLM based on model name
        if "gemini" in llm_model.lower():
            if not GEMINI_AVAILABLE:
                raise ImportError("Gemini not available. Install with: pip install google-generativeai")
            self.llm = GeminiLLM(model_name=llm_model)
        else:
            self.llm = OpenRouterLLM(model_name=llm_model)
        self.strategy = strategy
        
        # Define prompt templates for different strategies
        self.prompts = {
            "conservative": PromptTemplate(
                input_variables=["original_query"],
                template="""You are a query optimization expert. Given a question, provide a slightly improved version that maintains the original intent.

Original Query: {original_query}

Please provide a concise, improved version that:
1. Keeps the core question intact
2. Adds only essential synonyms if helpful
3. Maintains the same level of specificity
4. Does NOT expand the scope significantly

Improved Query:"""
            ),
            
            "balanced": PromptTemplate(
                input_variables=["original_query"],
                template="""You are a query enhancement expert. Given a question, provide a moderately enhanced version.

Original Query: {original_query}

Please enhance this query by:
1. Adding 1-2 relevant synonyms or related terms
2. Keeping the original structure and intent
3. Making it slightly more specific if it's too vague
4. NOT dramatically changing the scope

Enhanced Query:"""
            ),
            
            "aggressive": PromptTemplate(
                input_variables=["original_query"],
                template="""You are a query expansion expert. Given a question, expand it into a comprehensive query.

Original Query: {original_query}

Please rewrite this query to:
1. Include related terms and synonyms
2. Add relevant context
3. Make it more specific if needed
4. Expand the search scope moderately

Rewritten Query:"""
            )
        }
        
        self.rewrite_prompt = self.prompts[strategy]
        logging.info(f"Query Rewriter initialized with {strategy} strategy")
    
    def analyze_query_complexity(self, query: str) -> str:
        """
        Analyze query complexity and automatically select rewriting strategy
        
        Args:
            query: Original query
            
        Returns:
            Suggested strategy ('conservative', 'balanced', 'aggressive')
        """
        # Simple heuristic rules
        query_lower = query.lower()
        
        # Simple question indicators
        simple_indicators = [
            "what is", "who is", "when did", "where is", "how many",
            "does", "is", "are", "can", "will", "do", "did"
        ]
        
        # Complex question indicators
        complex_indicators = [
            "explain", "describe", "analyze", "compare", "discuss",
            "why", "how does", "what are the", "what causes"
        ]
        
        # Calculate complexity score
        simple_score = sum(1 for indicator in simple_indicators if indicator in query_lower)
        complex_score = sum(1 for indicator in complex_indicators if indicator in query_lower)
        
        # Select strategy based on query length and keywords
        if len(query.split()) <= 5 and simple_score > 0:
            return "conservative"
        elif len(query.split()) <= 10 and complex_score == 0:
            return "balanced"
        else:
            return "aggressive"
    
    def rewrite_query(self, original_query: str, auto_strategy: bool = True) -> str:
        """
        Rewrite query
        
        Args:
            original_query: Original query
            auto_strategy: Whether to automatically select strategy
            
        Returns:
            Rewritten query
        """
        try:
            # Auto-select strategy
            if auto_strategy:
                suggested_strategy = self.analyze_query_complexity(original_query)
                if suggested_strategy != self.strategy:
                    logging.info(f"Auto-switching strategy from {self.strategy} to {suggested_strategy}")
                    self.rewrite_prompt = self.prompts[suggested_strategy]
            
            # Generate rewriting prompt
            prompt = self.rewrite_prompt.format(original_query=original_query)
            
            # Use LLM to rewrite query
            rewritten_query = self.llm._call(prompt)
            
            # Validate rewriting results
            if self._is_overly_complex(original_query, rewritten_query):
                logging.warning("Query became overly complex, using conservative approach")
                conservative_prompt = self.prompts["conservative"].format(original_query=original_query)
                rewritten_query = self.llm._call(conservative_prompt)
            
            logging.info(f"Query rewritten: '{original_query}' -> '{rewritten_query}'")
            return rewritten_query.strip()
            
        except Exception as e:
            logging.warning(f"Query rewriting failed: {e}. Using original query.")
            return original_query
    
    def _is_overly_complex(self, original: str, rewritten: str) -> bool:
        """
        Check if rewritten query is overly complex
        
        Args:
            original: Original query
            rewritten: Rewritten query
            
        Returns:
            Whether it's overly complex
        """
        # Simple complexity check
        original_words = len(original.split())
        rewritten_words = len(rewritten.split())
        
        # If rewritten query is more than 3x longer than original, consider it overly complex
        if rewritten_words > original_words * 3:
            return True
        
        # Check if it contains too many irrelevant keywords
        complex_terms = ["comprehensive", "detailed", "thorough", "extensive", "complete"]
        if sum(1 for term in complex_terms if term in rewritten.lower()) > 2:
            return True
        
        return False
    
    def batch_rewrite_queries(self, queries: List[str]) -> List[str]:
        """
        Batch rewrite queries
        
        Args:
            queries: List of queries
            
        Returns:
            List of rewritten queries
        """
        rewritten_queries = []
        for query in queries:
            rewritten = self.rewrite_query(query)
            rewritten_queries.append(rewritten)
        
        return rewritten_queries


class DocumentReranker:
    """
    Document Reranker - Uses cross-encoder to reorder retrieval results
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker
        
        Args:
            model_name: Cross-encoder model name
        """
        self.reranker = CrossEncoder(model_name)
        logging.info(f"Document Reranker initialized with model: {model_name}")
    
    def rerank_documents(self, query: str, documents: List[Document], top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Rerank documents
        
        Args:
            query: Query text
            documents: List of documents
            top_k: Number of documents to return
            
        Returns:
            List of reranked documents and scores
        """
        if not documents:
            return []
        
        # Prepare query-document pairs
        query_doc_pairs = [(query, doc.page_content) for doc in documents]
        
        # Calculate relevance scores
        scores = self.reranker.predict(query_doc_pairs)
        
        # Create document-score pairs
        doc_score_pairs = list(zip(documents, scores))
        
        # Sort by score in descending order
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        reranked_results = doc_score_pairs[:top_k]
        
        logging.info(f"Reranked {len(documents)} documents, returning top {len(reranked_results)}")
        return reranked_results


class EnhancedRAGSystem:
    """
    Enhanced RAG System - Integrates query rewriting and reranking features
    """
    
    def __init__(self,
                 embedding_model: str = "BAAI/bge-small-en-v1.5",
                 llm_model: str = "deepseek/deepseek-chat-v3.1:free",
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize enhanced RAG system
        
        Args:
            embedding_model: Embedding model name
            llm_model: LLM model name
            reranker_model: Reranking model name
        """
        # Base RAG system
        self.llm_model = llm_model
        
        self.rag_system = NativeLangChainRAG(embedding_model, llm_model)
        
        # Enhanced features
        self.query_rewriter = QueryRewriter(self.llm_model)
        self.document_reranker = DocumentReranker(reranker_model)
        
        logging.info("Enhanced RAG System initialized")
    
    def load_documents(self, csv_file: str, passage_column: str = "passage") -> List[Document]:
        """Load documents"""
        return self.rag_system.load_documents(csv_file, passage_column)
    
    def build_vectorstore(self, documents: List[Document]) -> Any:
        """Build vector store"""
        return self.rag_system.build_vectorstore(documents)
    
    def save_vectorstore(self, output_dir: str, tag: str = None) -> str:
        """Save vector store"""
        return self.rag_system.save_vectorstore(output_dir, tag)
    
    def load_vectorstore(self, vectorstore_path: str):
        """Load vector store"""
        self.rag_system.load_vectorstore(vectorstore_path)
    
    def enhanced_search(self, query: str, k: int = 10, rerank_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Enhanced search - Includes query rewriting and reranking
        
        Args:
            query: Original query
            k: Initial retrieval document count
            rerank_k: Number of documents to return after reranking
            
        Returns:
            List of reranked documents and scores
        """
        # Step 1: Query rewriting
        rewritten_query = self.query_rewriter.rewrite_query(query)
        
        # Step 2: Initial retrieval (using rewritten query)
        initial_docs = self.rag_system.search_documents(rewritten_query, k=k)
        
        # Step 3: Reranking
        reranked_results = self.document_reranker.rerank_documents(
            query, initial_docs, top_k=rerank_k
        )
        
        logging.info(f"Enhanced search completed: {len(reranked_results)} results")
        return reranked_results
    
    def create_enhanced_qa_chain(self, strategy: str = "basic"):
        """
        Create enhanced QA chain
        
        Args:
            strategy: Prompt strategy
        """
        # Use original prompt strategy
        self.rag_system.create_advanced_chain(strategy)
    
    def enhanced_query(self, question: str, k: int = 10, rerank_k: int = 5, use_rewriting: bool = True) -> str:
        """
        Enhanced query - Uses optimized query rewriting and reranking
        
        Args:
            question: User question
            k: Initial retrieval document count
            rerank_k: Number of documents to return after reranking
            use_rewriting: Whether to use query rewriting
            
        Returns:
            Generated answer
        """
        try:
            # Step 1: Intelligent query rewriting (optional)
            if use_rewriting:
                rewritten_query = self.query_rewriter.rewrite_query(question, auto_strategy=True)
                logging.info(f"Original query: {question}")
                logging.info(f"Rewritten query: {rewritten_query}")
                
                # If rewritten query is too complex, use original query
                if self.query_rewriter._is_overly_complex(question, rewritten_query):
                    logging.warning("Rewritten query too complex, using original query")
                    search_query = question
                else:
                    search_query = rewritten_query
            else:
                search_query = question
                logging.info(f"Using original query without rewriting: {question}")
            
            # Step 2: Enhanced search
            reranked_results = self.enhanced_search(search_query, k=k, rerank_k=rerank_k)
            
            # Check retrieval result quality
            if not reranked_results or len(reranked_results) == 0:
                logging.warning("No documents retrieved, falling back to basic RAG")
                return self.rag_system.query(question)
            
            # Extract document content
            context_docs = [doc for doc, score in reranked_results]
            
            # Create temporary retriever
            from langchain_core.retrievers import BaseRetriever
            from typing import List
            
            class CustomRetriever(BaseRetriever):
                documents: List[Document] = []
                
                def __init__(self, documents):
                    super().__init__()
                    self.documents = documents
                
                def _get_relevant_documents(self, query: str) -> List[Document]:
                    return self.documents
            
            # Set temporary retriever
            original_retriever = self.rag_system.retriever
            self.rag_system.retriever = CustomRetriever(context_docs)
            
            try:
                # Use enhanced documents for querying
                result = self.rag_system.retrieval_chain.invoke({"input": question})
                answer = result["answer"]
                
                logging.info(f"Enhanced query completed for: {question}")
                return answer
                
            finally:
                # Restore original retriever
                self.rag_system.retriever = original_retriever
                
        except Exception as e:
            logging.error(f"Enhanced query failed: {e}")
            # Fallback to basic RAG
            return self.rag_system.query(question)


def build_enhanced_rag_pipeline(embedding_model: str,
                              training_csv: str,
                              output_dir: str,
                              llm_model: str = "deepseek/deepseek-chat-v3.1:free",
                              reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                              tag: str = None) -> EnhancedRAGSystem:
    """
    Build enhanced RAG pipeline
    
    Args:
        embedding_model: Embedding model name
        training_csv: Training CSV file path
        output_dir: Output directory
        llm_model: LLM model name
        reranker_model: Reranking model name
        tag: Optional tag
        
    Returns:
        Configured enhanced RAG system
    """
    # Create enhanced RAG system
    enhanced_rag = EnhancedRAGSystem(embedding_model, llm_model, reranker_model)
    
    # Load documents
    documents = enhanced_rag.load_documents(training_csv)
    
    # Build vector store
    enhanced_rag.build_vectorstore(documents)
    
    # Save vector store
    enhanced_rag.save_vectorstore(output_dir, tag)
    
    # Create enhanced QA chain
    enhanced_rag.create_enhanced_qa_chain("basic")
    
    return enhanced_rag


def load_enhanced_rag_pipeline(embedding_model: str,
                             vectorstore_path: str,
                             llm_model: str = "deepseek/deepseek-chat-v3.1:free",
                             reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> EnhancedRAGSystem:
    """
    Load enhanced RAG pipeline
    
    Args:
        embedding_model: Embedding model name
        vectorstore_path: Vector store path
        llm_model: LLM model name
        reranker_model: Reranking model name
        
    Returns:
        Loaded enhanced RAG system
    """
    # Create enhanced RAG system
    enhanced_rag = EnhancedRAGSystem(embedding_model, llm_model, reranker_model)
    
    # Load vector store
    enhanced_rag.load_vectorstore(vectorstore_path)
    
    # Create enhanced QA chain
    enhanced_rag.create_enhanced_qa_chain("basic")
    
    return enhanced_rag


def evaluate_enhanced_features(test_csv: str, enhanced_rag: EnhancedRAGSystem) -> Dict[str, Any]:
    """
    Evaluate the effectiveness of enhanced features
    
    Args:
        test_csv: Test CSV file path
        enhanced_rag: Enhanced RAG system
        
    Returns:
        Evaluation results
    """
    # Load test data
    df = pd.read_csv(test_csv)
    test_questions = df['question'].tolist()[:10]  # Test first 10 questions
    
    results = {
        "total_questions": len(test_questions),
        "query_rewriting_examples": [],
        "reranking_examples": [],
        "enhanced_answers": []
    }
    
    for i, question in enumerate(test_questions):
        print(f"\nProcessing question {i+1}/{len(test_questions)}: {question}")
        
        try:
            # 1. Query rewriting examples
            rewritten = enhanced_rag.query_rewriter.rewrite_query(question)
            results["query_rewriting_examples"].append({
                "original": question,
                "rewritten": rewritten
            })
            
            # 2. Reranking examples
            reranked_results = enhanced_rag.enhanced_search(question, k=10, rerank_k=3)
            results["reranking_examples"].append({
                "query": question,
                "top_documents": [
                    {
                        "content": doc.page_content[:100] + "...",
                        "score": float(score)
                    }
                    for doc, score in reranked_results
                ]
            })
            
            # 3. Enhanced answer generation
            enhanced_answer = enhanced_rag.enhanced_query(question, k=10, rerank_k=5)
            results["enhanced_answers"].append({
                "question": question,
                "answer": enhanced_answer
            })
            
        except Exception as e:
            print(f"Error processing question {question}: {e}")
            continue
    
    return results


def main():
    """Demonstrate enhanced RAG features"""
    print("🚀 Enhanced RAG Features Demo")
    print("=" * 50)
    
    try:
        # Build enhanced RAG system
        print("\n1. Building enhanced RAG pipeline...")
        enhanced_rag = build_enhanced_rag_pipeline(
            embedding_model="BAAI/bge-small-en-v1.5",
            training_csv="data/training.csv",
            output_dir="data/enhanced_rag",
            llm_model="gpt-4o-mini",
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            tag="enhanced"
        )
        print("✅ Enhanced RAG pipeline built successfully!")
        
        # Test enhanced features
        print("\n2. Testing enhanced features...")
        
        # Test query rewriting
        print("\n📝 Query Rewriting Examples:")
        test_queries = [
            "Was Abraham Lincoln the sixteenth President?",
            "Did Lincoln sign the National Banking Act?",
            "What did The Legal Tender Act establish?"
        ]
        
        for query in test_queries:
            rewritten = enhanced_rag.query_rewriter.rewrite_query(query)
            print(f"Original: {query}")
            print(f"Rewritten: {rewritten}")
            print("-" * 50)
        
        # Test reranking
        print("\n🔄 Reranking Examples:")
        for query in test_queries[:2]:
            print(f"\nQuery: {query}")
            reranked_results = enhanced_rag.enhanced_search(query, k=5, rerank_k=3)
            for i, (doc, score) in enumerate(reranked_results):
                print(f"  {i+1}. Score: {score:.3f}")
                print(f"     Content: {doc.page_content[:100]}...")
        
        # Evaluate enhanced features
        print("\n📊 Evaluating enhanced features...")
        evaluation_results = evaluate_enhanced_features("data/test.csv", enhanced_rag)
        
        print(f"\n✅ Enhanced RAG features demo completed!")
        print(f"Processed {evaluation_results['total_questions']} test questions")
        print(f"Query rewriting examples: {len(evaluation_results['query_rewriting_examples'])}")
        print(f"Reranking examples: {len(evaluation_results['reranking_examples'])}")
        print(f"Enhanced answers: {len(evaluation_results['enhanced_answers'])}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
