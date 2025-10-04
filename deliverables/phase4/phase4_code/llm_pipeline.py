from naive_rag_clean import SimpleVectorDB
from prompt_strategy import PromptingStrategy
import dotenv
import requests
import json
import yaml
import os
from typing import Dict, Any, List

dotenv.load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


class Chatbot:
    """
    This class is used to create a chatbot.
    """

    def __init__(self, config_file: str = "config_llm.yaml"):
        self.config = self.load_config(config_file)
        self.model = self.config["models"]["default"]

    def load_config(self, config_file: str) -> Dict[str, Any]:
        with open(config_file, "r") as f:
            return yaml.safe_load(f)

    def generate_response(self, content: str,
                          temperature: float = 0.7,
                          top_p: float = 0.9,
                          max_tokens: int = 1000) -> str:
        """
        Generate a response for the given content with customizable parameters.

        Args:
            content: The input text to generate a response for
            temperature: Controls randomness (0.0-2.0). Lower is more deterministic, higher is more random.
            top_p: Controls diversity via nucleus sampling (0.0-1.0)
            max_tokens: Maximum number of tokens to generate

        Returns:
            The generated response text
        """
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "stream": False
                }
            )

            response_data = response.json()
            if "choices" in response_data and response_data["choices"]:
                choice = response_data["choices"][0]
                response_content = choice["message"]["content"].strip()
                return response_content
            else:
                raise ValueError(f"Unexpected response structure: {response_data}")
        except Exception as e:
            raise ValueError(f"Error in chatbot response generation: {e}")


class LLM_Pipeline:
    """
    This class is used to create a pipeline for the LLM.
    """

    def __init__(self, rag_config_file: str = "config_rag.yaml",
                 llm_config_file: str = "config_llm.yaml"):
        # 向量数据库将在运行时设置
        self.vd = None
        self.llm = Chatbot(llm_config_file)
        self.prompting = PromptingStrategy()

    def generate_context(self, query: str, k: int = 3) -> str:
        """
        Generate a context for the given query.

        Args:
            query: The user's question
            k: Number of passages to retrieve

        Returns:
            Retrieved context as a single string
        """
        if self.vd is None:
            raise ValueError("Vector database not set. Please set self.vd before calling this method.")
        
        # 使用SimpleVectorDB的search方法
        res = self.vd.search(query, k=k)
        return " ".join([r['passage'] for r in res['results']])

    def generate_answer(self, query: str,
                        strategy: str = "basic",
                        k: int = 3,
                        temperature: float = 0.7,
                        top_p: float = 0.9,
                        max_tokens: int = 1000) -> str:
        """
        Generate an answer using RAG + LLM pipeline.

        Args:
            query: The user's question
            strategy: Prompting strategy to use
            k: Number of passages to retrieve
            temperature: LLM temperature parameter
            top_p: LLM top_p parameter
            max_tokens: Maximum tokens to generate

        Returns:
            Generated answer
        """
        # Step 1: Retrieve context using RAG
        context = self.generate_context(query, k=k)

        # Step 2: Generate prompt using selected strategy
        prompt = self.prompting.generate_prompt(strategy, query, context)

        # Step 3: Generate answer using LLM
        answer = self.llm.generate_response(
            content=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )

        return answer

    def get_available_strategies(self) -> List[str]:
        """Get list of available prompting strategies."""
        return self.prompting.get_available_strategies()


def test_prompting_strategies():
    """
    Test function to demonstrate different prompting strategies.
    """
    # Initialize the pipeline
    pipeline = LLM_Pipeline()

    # Test query
    query = "What is the capital of France?"

    print("Available prompting strategies:")
    for strategy in pipeline.get_available_strategies():
        print(f"- {strategy}")

    print("\n" + "="*50)

    # Test each strategy
    for strategy in pipeline.get_available_strategies():
        print(f"\nTesting {strategy} strategy:")
        print("-" * 30)

        try:
            # Generate context
            context = pipeline.generate_context(query, k=3)
            print(f"Context: {context[:200]}...")

            # Generate prompt
            prompt = pipeline.prompting.generate_prompt(strategy, query, context)
            print(f"Prompt: {prompt[:200]}...")

            # Note: Uncomment the following lines to actually call the LLM
            # answer = pipeline.generate_answer(query, strategy=strategy)
            # print(f"Answer: {answer}")

        except Exception as e:
            print(f"Error with {strategy}: {e}")


