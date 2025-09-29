from config_manager import LLMConfigManager
from naive_rag_new import NaiveRAG, VectorDatabase
import dotenv
import requests
import json

dotenv.load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


class Chatbot:
    """
    This class is used to create a chatbot.
    """
    def __init__(self):
        
    def generate_response(self, content: str) -> str:
        
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}"
                },
                data=json.dumps({
                    "model": "openai/gpt-4o", # Optional
                    "messages": [
                    {
                        "role": "user",
                        "content": f"{content}"
                    }
                    ]
                })
            )

            response_data = response.json()
            if "choices" in response_data and response_data["choices"]:
                
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
   
    def __init__(self):  #default constructor
        rag_config_file_path = "config_rag.yaml"
        self.vd = VectorDatabase(rag_config_file_path)
        
    def generate_context(self, query: str) -> str:
        """
        Generate a context for the given query.
        """
        res = self.vd.query_vector_database(query, k=3, include_scores=False)
        return res["passages"]
    
    
    
        