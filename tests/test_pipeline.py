import unittest
import os

from app.data_loader import load_transactions
from app.embeddings import get_embedding_model
from app.vector_store import create_vector_store
from app.llm import load_llm
from app.fraud_detector import FraudDetector
from app.mock_llm import MockLLM


class TestFraudDetectionPipeline(unittest.TestCase):
    """
    Integration tests for LLM + RAG fraud detection pipeline.

    Note:
    - These tests validate pipeline stability, not exact LLM output.
    - LLM responses are non-deterministic by design.
    """

    @classmethod
    def setUpClass(cls):
        # ---------- Data loading ----------
        cls.data_path = "data/transactions.csv"
        assert os.path.exists(cls.data_path), "transactions.csv not found"

        cls.documents = load_transactions(cls.data_path)
        assert len(cls.documents) >= 100, "Expected at least 100 transactions"

        # ---------- Embeddings ----------
        cls.embedding_model = get_embedding_model()

        # ---------- Vector store ----------
        cls.vector_store = create_vector_store(
            cls.documents,
            cls.embedding_model
        )

        # ---------- LLM ----------
        #cls.llm = load_llm()
        cls.llm = MockLLM()

        # ---------- Detector ----------
        cls.detector = FraudDetector(
            vector_store=cls.vector_store,
            llm=cls.llm
        )

    def test_data_loaded_correctly(self):
        """Ensure transaction text is properly constructed."""
        sample = self.documents[0]
        self.assertIsInstance(sample, str)
        self.assertIn("Transaction ID", sample)
        self.assertIn("Amount", sample)

    def test_vector_store_similarity_search(self):
        """Vector store should return relevant documents."""
        query = "Large offshore transaction with shell companies"
        results = self.vector_store.similarity_search(query, k=3)

        self.assertGreater(len(results), 0)
        self.assertTrue(hasattr(results[0], "page_content"))

    def test_llm_returns_response(self):
        """LLM should generate a non-empty response."""
        query = "High value transaction from sanctioned country"
        response = self.detector.analyze(query)

        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response.strip()), 10)

    def test_fraud_like_query_does_not_fail(self):
        """Fraud-heavy query should not crash pipeline."""
        query = "Suspicious offshore transfer of $50,000 through shell entities"
        response = self.detector.analyze(query)

        self.assertIsInstance(response, str)

    def test_normal_transaction_query(self):
        """Normal transaction query should still return explanation."""
        query = "Regular food delivery transaction in Bangalore"
        response = self.detector.analyze(query)

        self.assertIsInstance(response, str)

    def test_empty_query_handling(self):
        """Empty input should be handled gracefully."""
        response = self.detector.analyze("")
        self.assertIsInstance(response, str)


if __name__ == "__main__":
    unittest.main()
