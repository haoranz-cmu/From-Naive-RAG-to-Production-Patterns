
from typing import List

class PromptingStrategy:
    """
    This class handles different prompting strategies for RAG systems.
    """

    def __init__(self):
        self.strategies = {
            "basic": self._basic_prompt,
            "cot": self._chain_of_thought_prompt,
            "persona": self._persona_prompt,
            "instruction": self._instruction_prompt,
            "few_shot": self._few_shot_prompt
        }

    def _basic_prompt(self, query: str, context: str) -> str:
        """Basic prompting strategy."""
        return f"""Based on the following context, please answer the question.
        Please provide only a short, direct answer without any explanation or additional information.

Context: {context}

Question: {query}

Answer:"""

    def _chain_of_thought_prompt(self, query: str, context: str) -> str:
        """Chain of thought prompting strategy."""
        return f"""Based on the following context, please answer the question step by step.

Context: {context}

Question: {query}

Let me think through this step by step:
1. First, I need to understand what the question is asking
2. Then, I'll look for relevant information in the context
3. Finally, I'll provide a comprehensive answer
Please provide only a short, direct answer without any explanation or additional information.
Answer:"""

    def _persona_prompt(self, query: str, context: str) -> str:
        """Persona-based prompting strategy."""
        return f"""You are an expert assistant with deep knowledge in the subject matter.
Please provide only a short, direct answer without any explanation or additional information.

Context: {context}

Question: {query}

As an expert, I can tell you that:"""

    def _instruction_prompt(self, query: str, context: str) -> str:
        """Instruction-based prompting strategy."""
        return f"""Task: Answer the question using only the information provided in the context.

Instructions:
- Use only the information from the context
- If the context doesn't contain enough information, say so
- Be precise and concise
- Cite relevant parts of the context when possible
Please provide only a short, direct answer without any explanation or additional information.
Context: {context}

Question: {query}

Answer:"""

    def _few_shot_prompt(self, query: str, context: str) -> str:
        """Few-shot prompting strategy with examples."""
        return f"""Here are some examples of how to answer questions based on context:

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

Question: {query}

Please provide only a short, direct answer without any explanation or additional information.
Answer:"""

    def generate_prompt(self, strategy: str, query: str, context: str) -> str:
        """
        Generate a prompt using the specified strategy.

        Args:
            strategy: The prompting strategy to use
            query: The user's question
            context: The retrieved context from RAG

        Returns:
            The formatted prompt
        """
        if strategy not in self.strategies:
            available = list(self.strategies.keys())
            raise ValueError(f"Unknown strategy: {strategy}. "
                             f"Available strategies: {available}")

        return self.strategies[strategy](query, context)

    def get_available_strategies(self) -> List[str]:
        """Get list of available prompting strategies."""
        return list(self.strategies.keys())


