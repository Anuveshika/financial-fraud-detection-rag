from app.prompt import FRAUD_PROMPT
from app.config import TOP_K_RESULTS

def run_rag(query, vector_store, llm):
    docs = vector_store.similarity_search(query, k=TOP_K_RESULTS)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = FRAUD_PROMPT.format(
        context=context,
        question=query
    )

    return llm(prompt)
