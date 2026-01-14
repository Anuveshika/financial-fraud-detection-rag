# app/mock_llm.py

class MockLLM:
    def __call__(self, prompt: str) -> str:
        return (
            "Mock Fraud Analysis:\n"
            "Transaction analyzed successfully.\n"
            "No real LLM was called."
        )
