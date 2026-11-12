import pytest
from extraction_utils import (
    DocumentChunk,
    normalize_text_for_dedupe,
    score_chunk_for_indicators,
    is_table_like_text,
    deduplicate_indicators,
    Indicator,
    classify_match_status,
    THRESHOLDS,
)


def test_normalize_text_for_dedupe_basic():
    s = "  Percentage of girls (aged 5-9) achieving 80%+  "
    n = normalize_text_for_dedupe(s)
    assert "percentage of girls" in n
    assert "80" in n


def test_score_chunk_for_indicators_text_keywords():
    c = DocumentChunk(
        doc_id="d1",
        source_file="p.docx",
        chunk_type="text",
        page_or_section="1",
        content="This section describes the indicators and means of verification. Baseline: 10. Target: 50.",
        char_count=120,
    )
    score = score_chunk_for_indicators(c)
    assert score > 0.4


def test_score_chunk_table_boost_and_short_penalty():
    table_text = "Indicator | Baseline | Target\nEnrollment rate | 10 | 20"
    c = DocumentChunk(
        doc_id="d2",
        source_file="p.pdf",
        chunk_type="table",
        page_or_section="1",
        content=table_text,
        char_count=30,
    )
    score = score_chunk_for_indicators(c)
    # table gets base boost but short length reduces; ensure >0
    assert 0 < score <= 1


def test_is_table_like_text():
    t1 = "Col1\tCol2\nval1\tval2"
    t2 = "Header | A | B\nrow1 | 1 | 2"
    t3 = "This is plain text. No table here."
    assert is_table_like_text(t1)
    assert is_table_like_text(t2)
    assert not is_table_like_text(t3)


def test_deduplicate_indicators_simple():
    inds = [
        Indicator(indicator_text="% of households with access to water", level="output", source_confidence="high"),
        Indicator(indicator_text="Percentage of households with access to water", level="output", source_confidence="high"),
        Indicator(indicator_text="Number of trainings delivered", level="output", source_confidence="medium"),
    ]
    unique, merged = deduplicate_indicators(inds)
    # expect one merge between first two
    assert len(unique) == 2
    assert any(m[2] >= 90 for m in merged)


def test_split_indikit_wording():
    from extraction_utils import _split_indikit_wording

    block = "English\nFrench\nNumber of people with access to \r\nsafe water\nNombre de personnes ayant accès à l'eau potable"
    en, fr = _split_indikit_wording(block)
    assert en == "Number of people with access to safe water"
    assert fr.startswith("Nombre de personnes")
    # no header lines: first line is treated as the English text
    assert _split_indikit_wording("Just one line") == ("Just one line", "")
    assert _split_indikit_wording("") == ("", "")
    assert _split_indikit_wording(None) == ("", "")


def test_build_reference_library_from_zoa(tmp_path):
    import pandas as pd
    from extraction_utils import build_reference_library_from_zoa, load_reference_library

    zoa = pd.DataFrame(
        [
            {
                "url": "https://zoa.indikit.net/a",
                "slug": "1-a",
                "title": "Water Access",
                "sector": "Water",
                "section:Indicator’s Wording": "English\nFrench\n% of households with access to safe water\n% de ménages ayant accès à l'eau potable",
                "section:Disaggregate by": "Disaggregate\nby sex.",
            },
            {
                "url": "https://zoa.indikit.net/b",
                "slug": "2-b",
                "title": "Resources Saved",
                "sector": "Environment",
                "section:Indicator’s Wording": None,
                "section:Disaggregate by": None,
            },
        ]
    )
    src = tmp_path / "indicators.csv"
    out = tmp_path / "reference_library.csv"
    zoa.to_csv(src, index=False)

    ref = build_reference_library_from_zoa(str(src), str(out))
    assert list(ref["indicator_id"]) == ["1-a", "2-b"]
    assert ref.loc[0, "indicator_text"] == "% of households with access to safe water"
    assert ref.loc[0, "indicator_text_fr"].startswith("% de ménages")
    assert ref.loc[0, "text_source"] == "wording"
    assert ref.loc[0, "disaggregation"] == "Disaggregate by sex."
    # missing wording falls back to the title
    assert ref.loc[1, "indicator_text"] == "Resources Saved"
    assert ref.loc[1, "text_source"] == "title"

    # the written file is accepted by the matcher's loader and is not flagged as dummy
    loaded, dummy = load_reference_library(str(out))
    assert not dummy
    assert set(["indicator_id", "indicator_text", "sector", "level"]).issubset(loaded.columns)


def test_indicator_model_coerces_llm_output():
    # Real cached LLM responses return nulls for required enums and numbers for text fields;
    # these must be normalised rather than dropped.
    ind = Indicator(indicator_text="  Number of households\n trained ", level=None, target=60, baseline=None, source_confidence=None)
    assert ind.indicator_text == "Number of households trained"
    assert ind.level == "unclear"
    assert ind.target == "60"
    assert ind.source_confidence == "low"

    assert Indicator(indicator_text="x", level="INTERMEDIATE OUTCOME", source_confidence="High").level == "outcome"
    assert Indicator(indicator_text="x", level="Output 2.1").level == "output"
    assert Indicator(indicator_text="x", level="refugees").level == "unclear"
    assert Indicator(indicator_text="x", level="Result 2 - Indicator 1").level == "unclear"
    assert Indicator(indicator_text="x", disaggregation=["sex", "age"]).disaggregation == "sex; age"
    assert Indicator(indicator_text="x", baseline="N/A").baseline is None

    with pytest.raises(Exception):
        Indicator(indicator_text="   ", level="output")


def test_shortlist_reference_candidates_lexical_fallback():
    import pandas as pd
    from extraction_utils import shortlist_reference_candidates

    ref = pd.DataFrame(
        {
            "indicator_id": ["a", "b", "c"],
            "indicator_text": [
                "% of households with access to safe drinking water",
                "Number of farmers adopting climate-smart practices",
                "% of children under five vaccinated",
            ],
            "sector": ["Water", "Agriculture", "Health"],
        }
    )
    out = shortlist_reference_candidates(["Percentage of households with safe drinking water access"], ref, k=2, embedding_model=None)
    assert len(out) == 1 and len(out[0]) == 2
    assert out[0][0]["indicator_id"] == "a"
    assert out[0][0]["method"] == "lexical"
    assert 0 <= out[0][0]["score"] <= 1


def test_link_indicator_to_reference_uses_cache_and_validates(monkeypatch, tmp_path):
    import extraction_utils as eu

    candidates = [
        {"indicator_id": "a", "indicator_text": "% of households with access to safe drinking water", "sector": "Water", "score": 0.9, "method": "embedding"},
        {"indicator_id": "b", "indicator_text": "Number of farmers trained", "sector": "Agriculture", "score": 0.4, "method": "embedding"},
    ]
    calls = []

    def fake_call(prompt, model_name, **kwargs):
        calls.append(prompt)
        return '{"reference_id": "made-up-id", "status": "present", "confidence": 1.7, "reason": "same thing"}'

    monkeypatch.setattr(eu, "_call_openai_with_retries", fake_call)
    monkeypatch.setattr(eu, "FORCE_REFRESH", False)

    r1 = eu.link_indicator_to_reference("% of HH with safe water", candidates, model_name="test-model", cache_dir=str(tmp_path))
    assert r1["status"] == "Present"
    # invented id is rejected and replaced by the top candidate; confidence is clamped
    assert r1["reference_id"] == "a"
    assert r1["reference_text"].startswith("% of households")
    assert r1["confidence"] == 1.0
    assert r1["cached"] is False

    r2 = eu.link_indicator_to_reference("% of HH with safe water", candidates, model_name="test-model", cache_dir=str(tmp_path))
    assert r2["cached"] is True
    assert len(calls) == 1  # second call served from cache

    def fake_missing(prompt, model_name, **kwargs):
        return '{"reference_id": "a", "status": "Missing", "confidence": 0.8, "reason": "no"}'

    monkeypatch.setattr(eu, "_call_openai_with_retries", fake_missing)
    r3 = eu.link_indicator_to_reference("Something else entirely", candidates, model_name="test-model", cache_dir=str(tmp_path))
    assert r3["status"] == "Missing" and r3["reference_id"] is None

    def fake_error(prompt, model_name, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(eu, "_call_openai_with_retries", fake_error)
    r4 = eu.link_indicator_to_reference("Another one", candidates, model_name="test-model", cache_dir=str(tmp_path))
    assert r4["status"] == "Error" and "boom" in r4["reason"]


def test_configure_llm_and_client_targets(monkeypatch):
    import extraction_utils as eu

    cfg = eu.configure_llm(provider="ollama", model_name="llama3.2", base_url="http://localhost:11434/v1")
    assert cfg == {"provider": "ollama", "model": "llama3.2", "base_url": "http://localhost:11434/v1"}
    client = eu._make_client()
    assert str(client.base_url).startswith("http://localhost:11434/v1")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = eu.configure_llm(provider="openai", model_name="gpt-4o-mini")
    assert cfg["provider"] == "openai" and cfg["model"] == "gpt-4o-mini"
    assert "api.openai.com" in str(eu._make_client().base_url)

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        eu._make_client()
    with pytest.raises(ValueError):
        eu.configure_llm(provider="anthropic")

    # Azure: endpoint + key required, MODEL_NAME is the deployment name
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    cfg = eu.configure_llm(provider="azure", model_name="my-gpt4o-mini-deployment", azure_endpoint="https://example.openai.azure.com/", azure_api_version="2024-10-21")
    assert cfg["provider"] == "azure" and cfg["base_url"] == "https://example.openai.azure.com/"
    ok, msg = eu.llm_available()
    assert not ok and "AZURE_OPENAI_API_KEY" in msg
    with pytest.raises(RuntimeError):
        eu._make_client()
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-test-key")
    ok, msg = eu.llm_available()
    assert ok and "my-gpt4o-mini-deployment" in msg
    client = eu._make_client()
    assert type(client).__name__ == "AzureOpenAI"
    assert "example.openai.azure.com" in str(client.base_url)

    # restore module default for other tests
    eu.configure_llm(provider="ollama", model_name="llama3.2")


def test_extraction_cache_is_per_model(monkeypatch, tmp_path):
    import extraction_utils as eu

    monkeypatch.setattr(eu, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(eu, "FORCE_REFRESH", False)
    chunk = DocumentChunk(doc_id="d", source_file="d.docx", chunk_type="text", page_or_section="1", content="Indicator: x", char_count=12)

    eu.cache_response_for_chunk(chunk, {"indicators": [{"indicator_text": "from llama"}]}, "v1", "llama3.2")
    assert eu.load_cached_response(chunk, "v1", "llama3.2")["indicators"][0]["indicator_text"] == "from llama"
    # a different model must not see llama's answer
    assert eu.load_cached_response(chunk, "v1", "gpt-4o-mini") is None

    # legacy entries (no model in key) are only served to the model that produced them
    eu.cache_response_for_chunk(chunk, {"indicators": [{"indicator_text": "legacy gpt"}]}, "v1", None)
    assert eu.load_cached_response(chunk, "v1", eu.LEGACY_CACHE_MODEL)["indicators"][0]["indicator_text"] == "legacy gpt"
    assert eu.load_cached_response(chunk, "v1", "llama3.2")["indicators"][0]["indicator_text"] == "from llama"


def test_llm_client_handles_bare_list_and_uses_json_mode(monkeypatch, tmp_path):
    import extraction_utils as eu

    monkeypatch.setattr(eu, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(eu, "FORCE_REFRESH", False)
    seen = {}

    def fake_call(prompt, model_name, **kwargs):
        seen["model"] = model_name
        seen["response_format"] = kwargs.get("response_format")
        # small local models sometimes wrap the list under a different key
        return '{"results": [{"indicator_text": "Number of wells built", "level": "output", "target": 12}]}'

    monkeypatch.setattr(eu, "_call_openai_with_retries", fake_call)
    chunk = DocumentChunk(doc_id="d", source_file="d.docx", chunk_type="text", page_or_section="1", content="wells", char_count=5)
    out = eu.LLMClient(model_name="llama3.2").extract_indicators_from_chunk(chunk)
    assert seen["model"] == "llama3.2"
    assert seen["response_format"] == {"type": "json_object"}
    assert len(out) == 1 and out[0].indicator_text == "Number of wells built" and out[0].target == "12"
    assert out[0].doc_id == "d"


def test_classify_match_status_thresholds():
    t = THRESHOLDS
    # exact
    assert classify_match_status(0.96, 92, t) == "exact"
    # close
    assert classify_match_status(0.85, 50, t) == "close"
    # related
    assert classify_match_status(0.66, 10, t) == "related"
    # novel
    assert classify_match_status(0.5, 10, t) == "novel"


def _reset_embedding_state():
    import extraction_utils as eu
    eu._EMBEDDING_MODEL_CACHE.clear()
    eu._EMBEDDING_BACKEND_USED = None


def test_compute_embeddings_prefers_fastembed_and_normalises(monkeypatch):
    import sys, types
    import numpy as np
    import extraction_utils as eu

    _reset_embedding_state()
    seen = {}

    class FakeTextEmbedding:
        def __init__(self, name):
            seen["model"] = name

        def embed(self, texts):
            for i, _ in enumerate(texts):
                yield np.array([3.0 * (i + 1), 4.0 * (i + 1)])

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTextEmbedding))
    emb = eu._compute_embeddings(["a", "b"], "all-MiniLM-L6-v2")
    assert seen["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert emb.shape == (2, 2)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0)
    assert eu._EMBEDDING_BACKEND_USED == "fastembed"
    _reset_embedding_state()


def test_compute_embeddings_falls_back_to_sentence_transformers(monkeypatch):
    import sys, types
    import numpy as np
    import extraction_utils as eu

    _reset_embedding_state()

    class BrokenTextEmbedding:
        def __init__(self, name):
            raise ImportError("onnxruntime missing")

    class FakeSentenceTransformer:
        def __init__(self, name):
            pass

        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0] for _ in texts])

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=BrokenTextEmbedding))
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer))
    emb = eu._compute_embeddings(["x"], "all-MiniLM-L6-v2")
    assert emb.shape == (1, 2)
    assert eu._EMBEDDING_BACKEND_USED == "sentence-transformers"
    _reset_embedding_state()


def test_compute_embeddings_reports_all_backend_failures(monkeypatch):
    import sys, types
    import extraction_utils as eu

    _reset_embedding_state()

    class Broken:
        def __init__(self, name):
            raise ImportError("not installed")

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=Broken))
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=Broken))
    with pytest.raises(RuntimeError) as excinfo:
        eu._compute_embeddings(["x"], "all-MiniLM-L6-v2")
    assert "fastembed" in str(excinfo.value) and "sentence-transformers" in str(excinfo.value)
    _reset_embedding_state()


def test_llm_client_cache_only_mode(monkeypatch, tmp_path):
    import extraction_utils as eu

    monkeypatch.setattr(eu, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(eu, "FORCE_REFRESH", False)
    chunk = DocumentChunk(doc_id="d.pdf", source_file="d.pdf", chunk_type="text", page_or_section="1", content="indicator: % of households", char_count=25)
    client = eu.LLMClient(model_name="test-model", cache_dir=str(tmp_path))

    def boom(*args, **kwargs):
        raise AssertionError("API must not be called in cache_only mode")

    monkeypatch.setattr(eu, "_call_openai_with_retries", boom)
    with pytest.raises(eu.CacheMiss):
        client.extract_indicators_from_chunk(chunk, cache_only=True)

    eu.cache_response_for_chunk(chunk, {"indicators": [{"indicator_text": "% of households with latrines", "level": "output"}]}, "v1", "test-model")
    out = client.extract_indicators_from_chunk(chunk, cache_only=True)
    assert len(out) == 1 and out[0].doc_id == "d.pdf"
