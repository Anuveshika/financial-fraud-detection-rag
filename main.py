# main.py

from app.embeddings import get_embedding_model
from app.vector_store import load_vector_store
from app.llm import load_llm
from app.fraud_detector import FraudDetector
from app.config import VECTOR_DB_PATH

embedding_model = get_embedding_model()
vector_store = load_vector_store(VECTOR_DB_PATH, embedding_model)
llm = load_llm()

detector = FraudDetector(vector_store, llm)

query = "Transaction of $15,000 from offshore account"
result = detector.analyze(query)

print("\nFraud Analysis Result:\n")
print(result)
