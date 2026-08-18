import pytest

from src.search_engine import TravelSearchEngine


@pytest.mark.integration
def test_synthesize_response_real():
    """
    Real integration test for synthesize_response().

    Uses:
      - Azure OpenAI Chat deployment
      - Azure OpenAI embeddings
      - Azure AI Search
      - GovernanceGate
      - MLflow

    No mocks are used.
    """

    query = "What is the baggage allowance for Air India?"

    print("=" * 70)
    print("REAL RAG RESPONSE SYNTHESIS EVIDENCE")
    print("=" * 70)
    print(f"Query: {query}")
    print()

    engine = TravelSearchEngine()

    # Real retrieval from Azure AI Search
    docs, processed_query = engine.search_by_text(
        query,
        k=5,
    )

    assert processed_query == query
    assert len(docs) > 0, "Azure AI Search returned no documents."

    print(f"Retrieved documents: {len(docs)}")
    print()

    # Real Azure OpenAI generation
    answer = engine.synthesize_response(
        docs,
        query,
    )

    assert answer is not None
    assert isinstance(answer, str)
    assert answer.strip(), "LLM returned an empty response."

    print("GENERATED RESPONSE")
    print("-" * 70)
    print(answer)
    print()

    metrics = engine._last_generation_metrics

    print("GENERATION METRICS")
    print("-" * 70)
    print(
        f"Generation latency : "
        f"{metrics.get('generation_latency_seconds', 0):.3f}s"
    )
    print(
        f"Input tokens       : "
        f"{metrics.get('input_tokens', 0)}"
    )
    print(
        f"Output tokens      : "
        f"{metrics.get('output_tokens', 0)}"
    )
    print(
        f"Total tokens       : "
        f"{metrics.get('total_tokens', 0)}"
    )
    print()

    assert metrics["generation_latency_seconds"] >= 0
    assert metrics["input_tokens"] >= 0
    assert metrics["output_tokens"] >= 0
    assert metrics["total_tokens"] >= 0

    print("=" * 70)
    print("RESULT: REAL RAG RESPONSE SYNTHESIS SUCCESSFUL")
    print("=" * 70)
