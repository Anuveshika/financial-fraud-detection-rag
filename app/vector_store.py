from langchain.vectorstores import FAISS

def create_vector_store(documents, embedding_model):
    return FAISS.from_texts(documents, embedding_model)

def load_vector_store(path, embedding_model):
    return FAISS.load_local(path, embedding_model)
