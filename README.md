Proposal Indicator Extraction

This repository contains a Jupyter notebook and supporting module to extract monitoring & evaluation indicators from NGO/humanitarian proposals.

What was added:
- extraction_utils.py: deterministic parsing helpers, chunk scoring, deduplication, caching skeleton and LLM client skeleton.
- requirements.txt: pinned dependencies recommended by the plan.
- tests/test_extraction_utils.py: pytest tests for deterministic parts (chunk scoring, dedup, classification).
- proposal_indicator_analysis.ipynb: a minimal starter notebook (skeleton) to run the pipeline interactively.

Running tests:
1. Create a virtual environment and install dependencies: pip install -r requirements.txt
2. From repository root run: pytest -q

LLM provider:
- Two providers are supported and selected in the notebook config cell (or via extraction_utils.configure_llm):
  - openai (default): gpt-4o-mini via OPENAI_API_KEY in .env. The cached extraction responses in outputs/api_cache were produced with this model, so re-running extraction is free.
  - ollama (commented out in the notebook): a local model through Ollama's OpenAI-compatible endpoint. Install Ollama from https://ollama.com, then run `ollama pull llama3.2`. The server must be running (`ollama serve`, or the desktop app) at http://localhost:11434.
  - azure (commented out in the notebook): Azure OpenAI. Put AZURE_OPENAI_ENDPOINT (https://<resource>.openai.azure.com/), AZURE_OPENAI_API_KEY and optionally AZURE_OPENAI_API_VERSION (default 2024-10-21) in .env. Set MODEL_NAME to the *deployment name* shown in the Azure portal, not the base model name.
- LLM responses are cached in outputs/api_cache per chunk and per model, so switching models never reuses another model's answers.

Embeddings:
- Similarity matching and the Phase 4 shortlist use all-MiniLM-L6-v2 through fastembed (ONNX Runtime, no PyTorch), so they work on Intel Macs where PyTorch is stuck at 2.2.2. sentence-transformers is used automatically as a fallback if it is installed and fastembed is not.

Notes:
- The notebook and module are designed so the LLM call path can be mocked for testing. Do not call the live API in unit tests.
