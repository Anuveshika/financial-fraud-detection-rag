from app.data_loader import load_transactions
from app.embeddings import get_embedding_model
from app.vector_store import create_vector_store
from app.config import VECTOR_DB_PATH

docs = load_transactions("data/transactions.csv")
embedding_model = get_embedding_model()

vector_store = create_vector_store(docs, embedding_model)
vector_store.save_local(VECTOR_DB_PATH)

print("Vector index built successfully")
