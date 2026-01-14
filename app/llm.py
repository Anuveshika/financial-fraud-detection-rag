from langchain_community.llms import HuggingFaceHub
from app.config import LLM_REPO_ID

def load_llm():
    return HuggingFaceHub(
        repo_id=LLM_REPO_ID,
        model_kwargs={
            "temperature": 0.2,
            "max_new_tokens": 512
        }
    )
