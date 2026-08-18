"""
Streamlit UI Application
RUBRIC: Streamlit UI Application (6 marks total)
- Page config and layout implemented (2 marks)
- Search integrated correctly (2 marks)
- Results and sources displayed (1 mark)
- UI/UX design and examples (1 mark)

TASK: Create Streamlit web interface for travel chatbot
"""

import sys
import os
import time

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import streamlit as st

from src.search_engine import TravelSearchEngine
from src.config import Config
import src.monitoring  # Enable MLflow/Azure Monitor


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="WanderNest Travels - AI Assistant",
    layout="wide",
)

st.title("WanderNest Travels - AI Assistant")

st.markdown(
    "Get instant answers about flights, hotels, policies, "
    "and travel requirements."
)


# ============================================================
# Initialize Engine
# ============================================================

@st.cache_resource
def get_engine():
    """
    Initialize and cache the search engine.
    """
    try:
        return TravelSearchEngine()

    except Exception as e:
        st.error(
            f"Failed to initialize search engine: {e}"
        )
        return None


# ============================================================
# Session State Initialization
# ============================================================

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

if "quick_search" not in st.session_state:
    st.session_state.quick_search = False

if "query_count" not in st.session_state:
    st.session_state.query_count = 0


# ============================================================
# Quick Action Callbacks
# ============================================================

def set_quick_query(query: str):
    """
    Populate the chatbot text box with a predefined query
    and request an automatic search on the next rerun.

    Streamlit executes callbacks before rerunning the app,
    allowing the text_input widget to receive the updated
    session-state value when it is rendered.
    """
    st.session_state.query_input = query
    st.session_state.quick_search = True


# ============================================================
# Display Results
# ============================================================

def display_results(
    results,
    query_text,
    generated_response,
):
    """
    Display search results and AI response.

    Shows:
    1. Result count
    2. AI-generated response
    3. Source documents
    """

    unique_sources = {
        doc.metadata.get(
            "source",
            "Unknown",
        )
        for doc in results
    }

    st.success(
        f"Retrieved {len(results)} relevant passages "
        f"from {len(unique_sources)} source documents."
    )

    # --------------------------------------------------------
    # AI Response
    # --------------------------------------------------------

    st.subheader("AI Response")

    with st.container():
        st.markdown(generated_response)

    st.divider()

    # --------------------------------------------------------
    # Source Documents
    # --------------------------------------------------------

    if results:

        with st.expander("📚 View Source Documents"):

            for i, doc in enumerate(results):

                with st.container():

                    st.markdown(
                        f"**{i + 1}. Source: "
                        f"{doc.metadata.get('source', 'Unknown')}**"
                    )

                    st.markdown(
                        f"*Category: "
                        f"{doc.metadata.get('category', 'N/A')}*"
                    )

                    content = doc.page_content

                    if len(content) > 400:
                        content = content[:400] + "..."

                    st.write(content)

                    st.divider()

    else:
        st.warning(
            "No relevant documents found."
        )


# ============================================================
# Get Engine Instance
# ============================================================

engine = get_engine()


# ============================================================
# Cache Clear Option
# ============================================================

if st.sidebar.button("Clear Cache"):

    st.cache_resource.clear()

    st.rerun()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("WanderNest")

    st.info(
        """
        **Wanderlust Travels AI Assistant**

        This chatbot helps you with:
        - ✈️ Flight policies & routes
        - 🎫 Baggage rules
        - 📋 Visa requirements
        - 🏨 Hotel information
        - 🎟️ Booking & cancellation policies

        Powered by Azure AI & RAG
        """
    )

    st.divider()

    st.header("📊 Statistics")

    st.metric(
        "Total Queries",
        st.session_state.query_count,
    )


# ============================================================
# Main Search Interface
# ============================================================

st.markdown(
    "### 🔍 Ask Your Travel Questions"
)


# ============================================================
# Quick Action Buttons
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.button(
        "✈️ Baggage Rules",
        key="baggage_rules_button",
        on_click=set_quick_query,
        args=(
            "What are the baggage allowance rules "
            "for international flights?",
        ),
        use_container_width=True,
    )


with col2:

    st.button(
        "📋 Visa Info",
        key="visa_info_button",
        on_click=set_quick_query,
        args=(
            "Do I need a visa to travel from India to UK?",
        ),
        use_container_width=True,
    )


with col3:

    st.button(
        "🎫 Cancellation Policy",
        key="cancellation_policy_button",
        on_click=set_quick_query,
        args=(
            "What is the cancellation policy "
            "for Air India flights?",
        ),
        use_container_width=True,
    )


st.divider()


# ============================================================
# Chatbot Text Input
# ============================================================
#
# IMPORTANT:
# The text box is ALWAYS rendered.
#
# The previous implementation conditionally rendered the
# text_input only when example_query did not exist. Therefore,
# clicking a quick-action button caused the text box to vanish.
#
# Using a persistent widget key fixes that behavior.
# ============================================================

query_text = st.text_input(
    "Enter your travel question",
    key="query_input",
    placeholder=(
        "e.g., 'What are the baggage rules for BLR to LON?'"
    ),
    label_visibility="collapsed",
)


# ============================================================
# Search Button
# ============================================================

search_button = st.button(
    "🔍 Search",
    use_container_width=True,
    type="primary",
)


# ============================================================
# Determine Whether Search Should Execute
# ============================================================
#
# A quick-action callback sets quick_search=True.
#
# On the resulting rerun:
#   - query_input contains the predefined question
#   - the text box remains visible
#   - quick_search causes the query to execute automatically
#
# A normal Search button works independently.
# ============================================================

should_search = (
    search_button
    or st.session_state.quick_search
)


# ============================================================
# Search Logic
# ============================================================

if should_search and engine and query_text:

    # Clear the quick-search trigger immediately so that
    # subsequent normal reruns do not repeat the search.
    st.session_state.quick_search = False

    st.session_state.query_count += 1

    st.markdown("---")

    with st.spinner(
        "🔍 Searching travel knowledge base..."
    ):

        start_time = time.time()

        try:

            results, processed_query, generated_response = (
                engine.answer_query(
                    query_text,
                    k=5,
                )
            )

            latency = (
                time.time() - start_time
            )

            st.info(
                f"✅ Search completed in {latency:.2f}s"
            )

            display_results(
                results,
                processed_query,
                generated_response,
            )

        except Exception as e:

            # Make sure a failed quick search does not remain
            # armed for another rerun.
            st.session_state.quick_search = False

            st.error(
                f"❌ Error: {str(e)}"
            )

            st.info(
                "⚠️ Please try rephrasing your question "
                "or contact support."
            )


elif should_search and not query_text:

    # Clear quick-search state if there was no usable query.
    st.session_state.quick_search = False

    st.warning(
        "⚠️ Please enter a travel question."
    )
