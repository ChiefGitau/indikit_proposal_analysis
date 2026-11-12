"""ESGReveal-style indicator pipeline for ZOA project documents.

Three modules, following Zou et al. (2023, arXiv:2312.17264):

- ``pipeline.metadata``   - the metadata module (IndiKit entities + extensions)   [Phase 1]
- ``pipeline.preprocess`` - the report preprocessing module (parsers, KBs)        [Phase 2]
- ``pipeline.agent``      - the LLM agent module (retrieval, prompt, answer)      [Phase 3]
- ``pipeline.eval``       - gold-set format and metrics                          [Phase 1 scaffold]

See ``docs/plan.md`` for the design and ``docs/phase_1.md`` for the Phase 1 scope.
"""

__all__ = ["config"]
