#!/usr/bin/env python3
"""
Native LangChain Pipeline Implementation (Clean Version)
Implementing RAG system using LangChain native pipeline and chain operations
"""

import os
import logging
import pandas as pd
from typing import List, Optional, Any, Dict

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field

# OpenRouter integration
import requests
from dotenv import load_dotenv

# Gemini integration
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Google Generative AI not available. Install with: pip install google-generativeai")


load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class GeminiLLM(LLM):
    """
    Gemini LLM wrapper, compatible with LangChain
    """
    
    model_name: str = Field(default="gemini-2.5-flash", description="Gemini model name")
    api_key: Optional[str] = Field(default=None, description="Gemini API key")
    temperature: float = Field(default=0.7, description="Generation temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum token count")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI not available. Install with: pip install google-generativeai")
        
        # Get API key
        if not self.api_key:
            self.api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
    
    @property
    def _llm_type(self) -> str:
        return "gemini"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Call Gemini model to generate content
        """
        try:
            # Set generation parameters
            generation_config = {
                "temperature": self.temperature,
            }
            
            if self.max_tokens:
                generation_config["max_output_tokens"] = self.max_tokens
            
            # Create model instance
            model = genai.GenerativeModel(self.model_name)
            
            # Generate content
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return response.text
            
        except Exception as e:
            raise Exception(f"Error calling Gemini API: {e}")
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }




class OpenRouterLLM(LLM):
    """
    Custom LangChain LLM wrapper for OpenRouter
    Wraps OpenRouter API as LangChain LLM
    """

    model_name: str = "deepseek/deepseek-chat-v3.1:free"
    api_key: str = None
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1000

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. "
                           "Set OPENROUTER_API_KEY environment variable.")

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs) -> str:
        """Call OpenRouter API"""
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_tokens": self.max_tokens,
                    "stream": False
                }
            )

            response_data = response.json()
            if "choices" in response_data and response_data["choices"]:
                return response_data["choices"][0]["message"]["content"].strip()
            else:
                raise ValueError(f"Unexpected response structure: {response_data}")

        except Exception as e:
            raise ValueError(f"Error in OpenRouter response generation: {e}")


class OpenAILLM(LLM):
    """
    Custom LangChain LLM wrapper for OpenAI
    """
    model_name: str = "gpt-4o-mini"
    api_key: str = None
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 1000
    
    def __init__(self, model_name: str = "gpt-4o-mini", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")

        
    @property
    def _llm_type(self) -> str:
        return "openai"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs) -> str:
        """Call OpenAI API"""
        try:
            response = requests.post(
                url="https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_tokens": self.max_tokens,
                    "stream": False
                }
            )
            response_data = response.json()
            if "choices" in response_data and response_data["choices"]:
                return response_data["choices"][0]["message"]["content"].strip()
            else:
                raise ValueError(f"Unexpected response structure: {response_data}")
        except Exception as e:
            raise ValueError(f"Error in OpenAI response generation: {e}")               


class NativeLangChainRAG:
    """
    RAG system implemented using LangChain native pipeline
    """

    def __init__(self,
                 embedding_model: str = "BAAI/bge-small-en-v1.5",
                 llm_model: str = "gpt-4o-mini"):
        """
        Initialize native LangChain RAG system

        Args:
            embedding_model: Embedding model name
            llm_model: LLM model name
        """
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # Initialize embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Initialize LLM
        # Select LLM based on model name
        if "openai" in llm_model.lower():
            self.llm = OpenAILLM(model_name=llm_model)
        elif "gemini" in llm_model.lower():
            self.llm = GeminiLLM(model_name=llm_model)
        else:
            self.llm = OpenRouterLLM(model_name=llm_model)


        # Vector store and retriever
        self.vectorstore = None
        self.retriever = None

        # Pipeline chain
        self.retrieval_chain = None

        logging.info("Native LangChain RAG initialized")
        logging.info(f"  Embedding model: {embedding_model}")
        logging.info(f"  LLM model: {llm_model}")

    def load_documents(self, csv_file: str,
                      passage_column: str = "passage") -> List[Document]:
        """
        Load documents from CSV and convert to LangChain Document objects

        Args:
            csv_file: CSV file path
            passage_column: Column name containing text

        Returns:
            List of Document objects
        """
        df = pd.read_csv(csv_file)
        passages = df[passage_column].tolist()

        documents = [
            Document(
                page_content=passage,
                metadata={"source": f"passage_{i}", "index": i}
            )
            for i, passage in enumerate(passages)
        ]

        logging.info(f"Loaded {len(documents)} documents from {csv_file}")
        return documents

    def build_vectorstore(self, documents: List[Document]) -> FAISS:
        """
        Build vector store

        Args:
            documents: List of Document objects

        Returns:
            FAISS vector store
        """
        logging.info("Building vectorstore with LangChain...")
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)

        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        logging.info("Vectorstore and retriever created successfully")
        return self.vectorstore

    def save_vectorstore(self, output_dir: str, tag: str = None):
        """
        Save vector store to disk

        Args:
            output_dir: Output directory
            tag: Optional tag
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not built. Call build_vectorstore() first.")

        os.makedirs(output_dir, exist_ok=True)

        # Generate filename
        model_slug = self.embedding_model.replace("/", "_").replace("-", "_")
        if tag:
            vectorstore_name = f"native_vectorstore_{model_slug}_{tag}"
        else:
            vectorstore_name = f"native_vectorstore_{model_slug}"

        vectorstore_path = os.path.join(output_dir, vectorstore_name)
        self.vectorstore.save_local(vectorstore_path)

        logging.info(f"Vectorstore saved to: {vectorstore_path}")
        return vectorstore_path

    def load_vectorstore(self, vectorstore_path: str):
        """
        Load vector store from disk

        Args:
            vectorstore_path: Vector store path
        """
        self.vectorstore = FAISS.load_local(vectorstore_path, self.embeddings, allow_dangerous_deserialization=True)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        logging.info(f"Vectorstore loaded from: {vectorstore_path}")

    def create_qa_chain(self, prompt_template: str = None):
        """
        Create QA chain

        Args:
            prompt_template: Custom prompt template
        """
        if self.retriever is None:
            raise ValueError("Retriever not initialized. Call build_vectorstore() first.")

        # Default prompt template
        if prompt_template is None:
            prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""

        # Create prompt template - Note: create_retrieval_chain uses "input" as question variable
        prompt = PromptTemplate(
            template=prompt_template.replace("{question}", "{input}"),
            input_variables=["context", "input"]
        )

        # Create document chain
        document_chain = create_stuff_documents_chain(self.llm, prompt)

        # Create retrieval chain
        self.retrieval_chain = create_retrieval_chain(self.retriever, document_chain)

        logging.info("QA chain created successfully")

    def create_advanced_chain(self, strategy: str = "basic"):
        """
        Create advanced chain supporting different strategies

        Args:
            strategy: Strategy name
        """
        if self.retriever is None:
            raise ValueError("Retriever not initialized. Call build_vectorstore() first.")

        # Use original prompt strategy templates
        templates = {
            "basic": """Based on the following context, please answer the question.
        Please provide only a short, direct answer without any explanation or additional information.

Context: {context}

Question: {question}

Answer:""",

            "cot": """Based on the following context, please answer the question step by step.

Context: {context}

Question: {question}

Let me think through this step by step:
1. First, I need to understand what the question is asking
2. Then, I'll look for relevant information in the context
3. Finally, I'll provide a comprehensive answer
Please provide only a short, direct answer without any explanation or additional information.
Answer:""",

            "persona": """You are an expert assistant with deep knowledge in the subject matter.
Please provide only a short, direct answer without any explanation or additional information.

Context: {context}

Question: {question}

As an expert, I can tell you that:""",

            "instruction": """Task: Answer the question using only the information provided in the context.

Instructions:
- Use only the information from the context
- If the context doesn't contain enough information, say so
- Be precise and concise
- Cite relevant parts of the context when possible
Please provide only a short, direct answer without any explanation or additional information.
Context: {context}

Question: {question}

Answer:""",

            "few_shot": """Here are some examples of how to answer questions based on context:

Example 1:
Context: "The capital of France is Paris."
Question: "What is the capital of France?"
Answer: "Paris."

Example 2:
Context: "Python is a programming language."
Question: "What is Python?"
Answer: "programming language."

Now, based on the following context, answer the question:

Context: {context}

Question: {question}

Please provide only a short, direct answer without any explanation or additional information.
Answer:"""
        }

        if strategy not in templates:
            raise ValueError(f"Unknown strategy: {strategy}. "
                           f"Available: {list(templates.keys())}")

        # Create prompt template - Note: create_retrieval_chain uses "input" as question variable
        prompt = PromptTemplate(
            template=templates[strategy].replace("{question}", "{input}"),
            input_variables=["context", "input"]
        )

        # Create document chain
        document_chain = create_stuff_documents_chain(self.llm, prompt)

        # Create retrieval chain
        self.retrieval_chain = create_retrieval_chain(self.retriever, document_chain)

        logging.info(f"Advanced chain created with strategy: {strategy}")

    def query(self, question: str) -> str:
        """
        Query using chain operations

        Args:
            question: User question

        Returns:
            Generated answer
        """
        if self.retrieval_chain is None:
            raise ValueError("Chain not created. Call create_qa_chain() or create_advanced_chain() first.")

        # Use LangChain chain operations
        result = self.retrieval_chain.invoke({"input": question})
        return result["answer"]

    def search_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        Search relevant documents

        Args:
            query: Search query
            k: Number of documents to return

        Returns:
            List of relevant documents
        """
        if self.retriever is None:
            raise ValueError("Retriever not initialized. Call build_vectorstore() first.")

        # Update retriever parameters
        self.retriever.search_kwargs = {"k": k}

        # Search documents
        docs = self.retriever.get_relevant_documents(query)
        return docs

    def get_available_strategies(self) -> List[str]:
        """Get list of available strategies"""
        return ["basic", "cot", "persona", "instruction", "few_shot"]


def build_native_rag_pipeline(embedding_model: str,
                            training_csv: str,
                            output_dir: str,
                            llm_model: str = "openai/gpt-3.5-turbo",
                            tag: str = None) -> NativeLangChainRAG:
    """
    Build native LangChain RAG pipeline

    Args:
        embedding_model: Embedding model name
        training_csv: Training CSV file path
        output_dir: Output directory
        llm_model: LLM model name
        tag: Optional tag

    Returns:
        Configured RAG pipeline
    """
    # Create RAG system
    rag = NativeLangChainRAG(embedding_model, llm_model)

    # Load documents
    documents = rag.load_documents(training_csv)

    # Build vector store
    rag.build_vectorstore(documents)

    # Save vector store
    rag.save_vectorstore(output_dir, tag)

    # Create default QA chain
    rag.create_qa_chain()

    return rag


def load_native_rag_pipeline(embedding_model: str,
                        vectorstore_path: str,
                        llm_model: str = "openai/gpt-3.5-turbo") -> NativeLangChainRAG:
    """
    Load native LangChain RAG pipeline

    Args:
        embedding_model: Embedding model name
        vectorstore_path: Vector store path
        llm_model: LLM model name

    Returns:
        Loaded RAG pipeline
    """
    # Create RAG system
    rag = NativeLangChainRAG(embedding_model, llm_model)

    # Load vector store
    rag.load_vectorstore(vectorstore_path)

    # Create default QA chain
    rag.create_qa_chain()

    return rag




