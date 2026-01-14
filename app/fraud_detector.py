from app.rag_pipeline import run_rag

class FraudDetector:

    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm

    def analyze(self, transaction_text: str) -> str:
        return run_rag(transaction_text, self.vector_store, self.llm)
