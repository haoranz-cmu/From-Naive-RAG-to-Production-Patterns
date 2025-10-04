import os
import asyncio
from typing import List
from ragas.metrics import Faithfulness
from ragas.metrics import LLMContextPrecisionWithoutReference
from ragas.metrics import LLMContextRecall
from ragas.metrics import ResponseRelevancy
from ragas.dataset_schema import SingleTurnSample
import numpy as np
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class RAGASCalculator:
    """
    RAGAS Evaluation Calculator
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        openai_key = os.environ.get("OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            model_name=model_name, 
            api_key=openai_key,
            temperature=0.1,  # Lower temperature for more consistent output
            model_kwargs={
                "response_format": {"type": "json_object"}  # Force JSON output
            }
        )
    
    async def evaluate_faithfulness(self, 
                              response: List[str],
                              retrieved_contexts:  List[List[str]],
                              questions: List[str]) -> float:
        faithfulness_score = []
        faithfulness = Faithfulness(llm=self.llm)
        
        for i in range(len(questions)):
            data = {
                'user_input': questions[i],
                'response': response[i],
                'retrieved_contexts': retrieved_contexts[i]
            }
            
            obj = SingleTurnSample(user_input=data['user_input'], 
                                   response=data['response'], 
                                   retrieved_contexts=data['retrieved_contexts'])
            score = await faithfulness.single_turn_ascore(obj)
            faithfulness_score.append(score)
            
        return np.mean(faithfulness_score)

    async def evaluate_context_precision(self, 
                                   questions: List[str],
                                   retrieved_contexts: List[List[str]],
                                   response: List[str],
                                   ground_truth_answers: List[str]):
        """
        Evaluate context precision and recall

        Args:
            questions: List of questions
            retrieved_contexts: List of retrieved contexts
            ground_truth_answers: List of ground truth answers

        Returns:
            (precision_score, recall_score)
        """
        context_precision_score = []
        context_recall_score = []
        context_precision = LLMContextPrecisionWithoutReference(llm=self.llm)
        context_recall = LLMContextRecall(llm=self.llm)
        
        for i in range(len(questions)):
            data = {
                'user_input': questions[i],
                'response': response[i],
                'retrieved_contexts': retrieved_contexts[i]
            }
            obj = SingleTurnSample(user_input=data['user_input'], 
                                   response=data['response'], 
                                   retrieved_contexts=data['retrieved_contexts'])
            precision_score = await context_precision.single_turn_ascore(obj)
            context_precision_score.append(precision_score)
            
            data['reference'] = ground_truth_answers[i]
            obj = SingleTurnSample(user_input=data['user_input'], 
                                   response=data['response'], 
                                   retrieved_contexts=data['retrieved_contexts'],
                                   reference=data['reference'])
            recall_score = await context_recall.single_turn_ascore(obj)
            context_recall_score.append(recall_score)
        return np.mean(context_precision_score), np.mean(context_recall_score)

    async def evaluate_answer_relevance(self, 
                                  questions: List[str],
                                  responses: List[str],
                                  retrieved_contexts: List[List[str]]):
        """
        Evaluate answer relevance

        Args:
            questions: List of questions
            responses: List of generated answers

        Returns:
            relevance_score
        """
        relevance_score = []
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        relevance = ResponseRelevancy(llm=self.llm, embeddings=embeddings)
        
        for i in range(len(questions)):
            data = {
                'user_input': questions[i],
                'response': responses[i],
                'retrieved_contexts': retrieved_contexts[i]
            }
            obj = SingleTurnSample(user_input=data['user_input'], 
                                   response=data['response'], 
                                   retrieved_contexts=data['retrieved_contexts'])
            score = await relevance.single_turn_ascore(obj)
            relevance_score.append(score)
        return np.mean(relevance_score)

async def main():
    calculator = RAGASCalculator()
    result = await calculator.evaluate_faithfulness(
        response=["The answer is 10"],
        retrieved_contexts=[["The answer is 10"]],
        questions=["What is the answer?"]
    )
    print(f"Faithfulness score: {result}")

if __name__ == "__main__":
    asyncio.run(main())