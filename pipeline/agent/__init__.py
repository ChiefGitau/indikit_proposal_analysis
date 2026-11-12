"""LLM agent module (paper Section 3.4).

Phase 1 provides the prompt template (``prompts``), the answer schema
(``models``) and the LLM plumbing moved from ``extraction_utils`` (``answer``,
``novel_pass``). Retrieval and the actual answering loop arrive in Phase 3.
"""
