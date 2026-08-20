"""
REAL STREAMLIT UI EVIDENCE
RUBRIC:
1. Page config and layout implemented - 2 marks
2. Search integrated correctly - 2 marks
3. Results and sources displayed - 1 mark
4. UI/UX design and examples - 1 mark
"""

import ast
from pathlib import Path


APP = Path("app.py")

if not APP.exists():
    APP = Path("src/app.py")

if not APP.exists():
    APP = Path("streamlit_app.py")


def read_app():
    assert APP.exists(), f"Streamlit application not found: {APP}"
    return APP.read_text()


def test_page_config_and_layout():
    print("=" * 70)
    print("REAL STREAMLIT UI - PAGE CONFIG AND LAYOUT")
    print("=" * 70)

    source = read_app()
    tree = ast.parse(source)

    assert "st.set_page_config" in source
    assert 'page_title="WanderNest Travels - AI Assistant"' in source
    assert 'layout="wide"' in source

    assert 'st.title("WanderNest Travels - AI Assistant")' in source
    assert "st.sidebar" in source
    assert "st.columns(3)" in source

    print(f"Application : {APP}")
    print("Page title  : WanderNest Travels - AI Assistant")
    print("Layout      : wide")
    print("Sidebar     : implemented")
    print("Main title  : implemented")
    print("3-column UI : implemented")

    print("-" * 70)
    print("RESULT: PAGE CONFIG AND LAYOUT SUCCESSFUL")
    print("=" * 70)


def test_search_integration():
    print("=" * 70)
    print("REAL STREAMLIT UI - SEARCH INTEGRATION")
    print("=" * 70)

    source = read_app()

    assert "TravelSearchEngine" in source
    assert "def get_engine" in source
    assert "return TravelSearchEngine()" in source
    assert "engine.answer_query(" in source
    assert "query_text" in source
    assert "k=5" in source

    print("Search engine       : TravelSearchEngine")
    print("Engine initialization: implemented")
    print("Search method       : engine.answer_query()")
    print("Query parameter     : query_text")
    print("Retrieval count     : k=5")

    print("-" * 70)
    print("RESULT: SEARCH INTEGRATION SUCCESSFUL")
    print("=" * 70)


def test_results_and_sources_displayed():
    print("=" * 70)
    print("REAL STREAMLIT UI - RESULTS AND SOURCES")
    print("=" * 70)

    source = read_app()

    assert "AI Response" in source
    assert "View Source Documents" in source
    assert "metadata.get(" in source
    assert "source" in source
    assert "category" in source
    assert "doc.page_content" in source

    print("AI response display : implemented")
    print("Source documents    : implemented")
    print("Source metadata     : implemented")
    print("Category metadata   : implemented")
    print("Document content    : implemented")

    print("-" * 70)
    print("RESULT: RESULTS AND SOURCES DISPLAY SUCCESSFUL")
    print("=" * 70)


def test_ui_ux_and_examples():
    print("=" * 70)
    print("REAL STREAMLIT UI - UI/UX AND EXAMPLES")
    print("=" * 70)

    source = read_app()

    assert "Baggage Rules" in source
    assert "Visa Info" in source
    assert "Cancellation Policy" in source

    assert "placeholder=" in source
    assert "st.spinner(" in source
    assert "st.warning(" in source
    assert "st.error(" in source
    assert "st.metric(" in source
    assert "query_count" in source

    print("Quick example 1    : Baggage Rules")
    print("Quick example 2    : Visa Info")
    print("Quick example 3    : Cancellation Policy")
    print("Input placeholder   : implemented")
    print("Loading indicator   : implemented")
    print("Error handling     : implemented")
    print("Usage statistics   : implemented")

    print("-" * 70)
    print("RESULT: UI/UX AND EXAMPLES SUCCESSFUL")
    print("=" * 70)


if __name__ == "__main__":
    print("Streamlit application:", APP)
