# Financial Fraud Detection using LLM + RAG

## Overview

This project implements an **explainable Financial Fraud Detection system** using **Retrieval-Augmented Generation (RAG)** with Large Language Models (LLMs). Instead of treating fraud detection as a black-box classification problem, the system retrieves relevant historical transaction patterns and allows an LLM to reason over them, producing **human-readable, audit-friendly explanations**.

The project is designed to be:

* Production-ready
* Interview- and resume-optimized
* Cost-aware (no mandatory paid APIs)
* Testable and modular

---

## Problem Statement

Traditional fraud detection systems:

* Rely heavily on black-box ML models
* Provide limited or no explanations
* Are difficult to audit and justify in regulated environments (banking, fintech, AML)

This project solves that by combining:

* Semantic retrieval of historical fraud patterns (RAG)
* LLM-based reasoning
* Explainable outputs suitable for compliance teams

---

## High-Level Architecture (HLD)
<img width="3333" height="1655" alt="image" src="https://github.com/user-attachments/assets/2c4f982a-cb24-4a3a-a2c0-bc81c01cdb7e" />



---

## Low-Level Design (LLD)

### 1. Data Ingestion

* Reads transaction data from CSV
* Handles real-world issues:

  * BOM encoding
  * Excel corruption
  * Wrong delimiters
* Validates schema automatically

### 2. Text Construction

Each transaction is converted into a structured text format:

```
Transaction ID: TXN001
Amount: 12000
Location: Offshore-Cayman
Merchant: Unknown
Description: Large transfer to offshore account...
```

### 3. Embeddings Layer

* Uses `sentence-transformers/all-MiniLM-L6-v2`
* Converts text into dense vectors
* Runs fully locally (no API key required)

### 4. Vector Store

* Stores embeddings for similarity search
* Supports:

  * FAISS (high-performance, production)
  * Chroma (pure Python, Windows-safe)

### 5. Retrieval-Augmented Generation (RAG)

* Retrieves top-K similar transactions
* Injects retrieved context into prompt
* Prevents hallucination by grounding LLM responses

### 6. LLM Layer

* Abstracted behind a loader
* Supports:

  * HuggingFaceHub (Zephyr)
  * Local / mocked LLMs (for tests)

### 7. Fraud Detector API

* Single entry point for analysis
* Clean interface:

```python
FraudDetector.analyze(transaction_text)
```

---

## Testing Strategy

### Testing Approach Used

* Pipeline-level integration tests
* Mock LLM for deterministic testing
* Validates:

  * Data loading
  * Vector search
  * End-to-end execution

This ensures CI-safe, fast, and reliable tests.

---

## How to Run the Project

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python main.py
```

---

## How to Run Tests

```bash
python -m unittest tests/test_pipeline.py
```

---

## Key Engineering Decisions

* **RAG over Fine-Tuning**: Dynamic knowledge updates without retraining
* **Mocked LLM in Tests**: Deterministic, CI-safe testing
* **Defensive CSV Parsing**: Real-world data robustness
* **Model-Agnostic Design**: Easy to swap LLMs

---
