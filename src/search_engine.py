"""
Travel Search Engine with RAG

RUBRIC: Search Engine Implementation (13 marks total)
- TravelSearchEngine initialized correctly (4 marks)
- search_by_text performs similarity search (3 marks)
- synthesize_response generates grounded answers (4 marks)
- Governance checks integrated (2 marks)

TASK: Implement RAG search engine with governance integration
"""

import time
from contextlib import contextmanager

import mlflow
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from src.config import Config
from governance.governance_gate import GovernanceGate
from src.vector_store import get_vector_store


class TravelSearchEngine:
    """RAG-powered search engine for travel queries."""

    def __init__(self):
        """
        Initialize search engine components.

        Initializes:
        1. Governance gate
        2. Azure Chat OpenAI LLM
        3. Azure OpenAI Embeddings
        4. Azure AI Search vector store
        """

        # ---------------------------------------------------------
        # MLflow configuration
        # ---------------------------------------------------------
        self.mlflow_enabled = self._configure_mlflow()

        # ---------------------------------------------------------
        # Governance
        # ---------------------------------------------------------
        self.governance_gate = GovernanceGate()

        # ---------------------------------------------------------
        # Azure OpenAI Chat LLM
        # ---------------------------------------------------------
        print(
            f"DEBUG: Initializing AzureChatOpenAI "
            f"with endpoint {Config.AZURE_OPENAI_ENDPOINT} "
            f"and deployment {Config.AZURE_OPENAI_DEPLOYMENT_NAME}"
        )

        self.llm = AzureChatOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            deployment_name=Config.AZURE_OPENAI_DEPLOYMENT_NAME,
            temperature=0.1,
        )

        print("DEBUG: AzureChatOpenAI initialized successfully.")

        # ---------------------------------------------------------
        # Azure OpenAI Embeddings
        # ---------------------------------------------------------
        print(
            f"DEBUG: Initializing AzureOpenAIEmbeddings "
            f"with endpoint {Config.AZURE_OPENAI_ENDPOINT} "
            f"and deployment {Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT}"
        )

        self.embeddings = AzureOpenAIEmbeddings(
            api_key=Config.AZURE_OPENAI_API_KEY,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            azure_deployment=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            api_version=Config.AZURE_OPENAI_API_VERSION,
        )

        print("DEBUG: AzureOpenAIEmbeddings initialized successfully.")

        # ---------------------------------------------------------
        # Vector Store
        # ---------------------------------------------------------
        print("DEBUG: Initializing vector store...")

        self.vector_store = get_vector_store(self.embeddings)

        print("DEBUG: Vector store initialized successfully.")

        # ---------------------------------------------------------
        # Latest generation metrics
        #
        # These values are populated by synthesize_response()
        # and exposed to answer_query() for trace-level telemetry.
        # ---------------------------------------------------------
        self._last_generation_metrics = {
            "generation_latency_seconds": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    def _configure_mlflow(self):
        """Configure MLflow to use Azure ML as the tracking backend."""

        try:
            mlflow.set_tracking_uri(
                Config.MLFLOW_TRACKING_URI
            )

            mlflow.set_experiment(
                Config.MLFLOW_EXPERIMENT_NAME
            )

            print(
                "DEBUG: MLflow configured successfully."
            )

            print(
                f"DEBUG: Tracking URI: "
                f"{mlflow.get_tracking_uri()}"
            )

            return True

        except Exception as e:

            print(
                f"⚠️ MLflow configuration failed: {e}"
            )

            return False

    # =============================================================
    # MLflow Helper
    # =============================================================

    @contextmanager
    def _mlflow_run(self, run_name):
        """
        Start and manage an MLflow run.

        MLflow failures do not prevent the RAG application
        from continuing to operate.
        """

        mlflow_active = False

        try:
            mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)

            mlflow.start_run(run_name=run_name)

            mlflow_active = True

            print(
                f"DEBUG: MLflow run started: {run_name}"
            )

        except Exception as e:
            print(
                f"⚠️ MLflow disabled for {run_name}: {e}"
            )

        try:
            yield mlflow_active

        finally:
            if mlflow_active:
                try:
                    mlflow.end_run()

                    print(
                        f"DEBUG: MLflow run completed: {run_name}"
                    )

                except Exception as e:
                    print(
                        f"⚠️ Could not close MLflow run: {e}"
                    )

    # =============================================================
    # Text Search
    # =============================================================

    def search_by_text(
        self,
        query_text: str,
        k: int = 5,
    ):
        """
        Search for travel information using a text query.

        Steps:
        1. Validate search parameters.
        2. Validate input using governance gate.
        3. Perform similarity search.
        4. Capture retrieval latency.
        5. Capture source statistics.
        6. Log retrieval metrics to MLflow.
        7. Return documents and query.
        """

        print(
            f"DEBUG: Text Query: {query_text}"
        )

        # ---------------------------------------------------------
        # Validate k
        # ---------------------------------------------------------
        if k <= 0 or k > 20:
            raise ValueError(
                "k must be between 1 and 20"
            )

        # ---------------------------------------------------------
        # MLflow run
        # ---------------------------------------------------------
        with self._mlflow_run(
            "search_by_text"
        ) as mlflow_active:

            # -----------------------------------------------------
            # Governance: Input validation
            # -----------------------------------------------------
            gov_check = (
                self.governance_gate.validate_input(
                    query_text
                )
            )

            if not gov_check.get(
                "passed",
                False,
            ):

                print(
                    "⚠️ Query blocked by governance gate."
                )

                if mlflow_active:

                    mlflow.log_param(
                        "governance_check_failed",
                        True,
                    )

                    mlflow.log_param(
                        "violations",
                        ",".join(
                            gov_check.get(
                                "violations",
                                [],
                            )
                        ),
                    )

                return (
                    [],
                    "Query blocked by security checks.",
                )

            # -----------------------------------------------------
            # MLflow parameters
            # -----------------------------------------------------
            if mlflow_active:

                mlflow.log_param(
                    "k",
                    k,
                )

                mlflow.log_param(
                    "query_text",
                    query_text,
                )

                mlflow.log_param(
                    "governance_check_failed",
                    False,
                )

            # -----------------------------------------------------
            # Similarity Search
            # -----------------------------------------------------
            print(
                f"DEBUG: Performing similarity search "
                f"with k={k}"
            )

            retrieval_start = time.perf_counter()

            try:

                docs = (
                    self.vector_store.similarity_search(
                        query_text,
                        k=k,
                    )
                )

            except Exception as e:

                print(
                    f"⚠️ Retrieval failed: {e}"
                )

                if mlflow_active:

                    mlflow.log_param(
                        "retrieval_error",
                        True,
                    )

                    mlflow.log_param(
                        "retrieval_error_type",
                        type(e).__name__,
                    )

                    mlflow.log_text(
                        str(e),
                        "retrieval_error.txt",
                    )

                raise

            retrieval_latency = (
                time.perf_counter()
                - retrieval_start
            )

            print(
                f"DEBUG: Similarity search returned "
                f"{len(docs)} documents."
            )

            # -----------------------------------------------------
            # Source statistics
            # -----------------------------------------------------
            unique_sources = len({
                doc.metadata.get(
                    "source",
                    "Unknown",
                )
                for doc in docs
            })

            print(
                f"DEBUG: Retrieval latency: "
                f"{retrieval_latency:.3f}s"
            )

            print(
                f"DEBUG: Unique source documents: "
                f"{unique_sources}"
            )

            # -----------------------------------------------------
            # MLflow metrics
            # -----------------------------------------------------
            if mlflow_active:

                mlflow.log_metric(
                    "results_count",
                    len(docs),
                )

                mlflow.log_metric(
                    "unique_sources",
                    unique_sources,
                )

                mlflow.log_metric(
                    "retrieval_latency_seconds",
                    retrieval_latency,
                )

            return (
                docs,
                query_text,
            )

    # =============================================================
    # Response Synthesis
    # =============================================================

    def synthesize_response(
        self,
        docs,
        user_query,
    ):
        """
        Generate a conversational response based on retrieved
        documents.

        Steps:
        1. Start MLflow run.
        2. Handle empty retrieval results.
        3. Validate user input.
        4. Build knowledge-base context.
        5. Create grounded prompt.
        6. Generate response using LLM.
        7. Capture latency and token usage.
        8. Validate output using governance gate.
        9. Log response and metrics.
        10. Return final response.
        """

        # ---------------------------------------------------------
        # Reset generation metrics
        # ---------------------------------------------------------
        self._last_generation_metrics = {
            "generation_latency_seconds": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        # ---------------------------------------------------------
        # MLflow
        # ---------------------------------------------------------
        with self._mlflow_run(
            "synthesize_response"
        ) as mlflow_active:

            # -----------------------------------------------------
            # Handle no documents
            # -----------------------------------------------------
            if not docs:

                response = (
                    "I couldn’t find any travel policy "
                    "information relevant to your question. "
                    "Please try a more specific query."
                )

                if mlflow_active:

                    mlflow.log_param(
                        "documents_found",
                        0,
                    )

                    mlflow.log_metric(
                        "generation_latency_seconds",
                        0.0,
                    )

                    mlflow.log_metric(
                        "input_tokens",
                        0,
                    )

                    mlflow.log_metric(
                        "output_tokens",
                        0,
                    )

                    mlflow.log_metric(
                        "total_tokens",
                        0,
                    )

                    mlflow.log_text(
                        response,
                        "final_response.txt",
                    )

                return response

            # -----------------------------------------------------
            # Governance: Input validation
            # -----------------------------------------------------
            gov_input_check = (
                self.governance_gate.validate_input(
                    user_query
                )
            )

            if not gov_input_check.get(
                "passed",
                False,
            ):

                print(
                    "⚠️ User query blocked by governance gate."
                )

                if mlflow_active:

                    mlflow.log_param(
                        "input_governance_check_failed",
                        True,
                    )

                    mlflow.log_param(
                        "input_violations",
                        ",".join(
                            gov_input_check.get(
                                "violations",
                                [],
                            )
                        ),
                    )

                return (
                    "Query blocked by security checks."
                )

            # -----------------------------------------------------
            # Build Knowledge Base Context
            # -----------------------------------------------------
            context_parts = []

            for i, doc in enumerate(docs):

                source = doc.metadata.get(
                    "source",
                    "Unknown",
                )

                content = doc.page_content

                context_parts.append(
                    f"""SOURCE {i + 1}: {source}

CONTENT:
{content}"""
                )

            context = "\n\n".join(
                context_parts
            )

            # -----------------------------------------------------
            # Build Grounded Prompt
            # -----------------------------------------------------
            prompt = f"""
You are a helpful travel assistant for Wanderlust Travels,
an online travel agency.

Use ONLY the information provided in the Knowledge Base below
to answer the customer's question.

==================== KNOWLEDGE BASE ====================

{context}

==================== CUSTOMER QUESTION ====================

{user_query}

==================== INSTRUCTIONS ====================

- Provide a clear, helpful, and accurate answer.
- Ground your answer ONLY in the Knowledge Base.
- Do not invent or assume facts that are not supported by
  the Knowledge Base.
- If the Knowledge Base does not contain sufficient information
  to answer the question, clearly say that the available
  information is insufficient.
- When providing factual information, identify the relevant
  source document when appropriate.
- Do not use outside knowledge to fill gaps in the Knowledge Base.
"""

            # -----------------------------------------------------
            # MLflow parameters
            # -----------------------------------------------------
            if mlflow_active:

                mlflow.log_param(
                    "documents_found",
                    len(docs),
                )

                mlflow.log_param(
                    "user_query",
                    user_query,
                )

                mlflow.log_param(
                    "input_governance_check_failed",
                    False,
                )

            # -----------------------------------------------------
            # Generate LLM Response
            # -----------------------------------------------------
            print(
                "DEBUG: Generating response using Azure OpenAI..."
            )

            generation_start = (
                time.perf_counter()
            )

            try:

                response = self.llm.invoke(
                    prompt
                )

            except Exception as e:

                generation_latency = (
                    time.perf_counter()
                    - generation_start
                )

                self._last_generation_metrics[
                    "generation_latency_seconds"
                ] = generation_latency

                print(
                    f"⚠️ LLM generation failed: {e}"
                )

                if mlflow_active:

                    mlflow.log_param(
                        "generation_error",
                        True,
                    )

                    mlflow.log_param(
                        "generation_error_type",
                        type(e).__name__,
                    )

                    mlflow.log_metric(
                        "generation_latency_seconds",
                        generation_latency,
                    )

                    mlflow.log_text(
                        str(e),
                        "generation_error.txt",
                    )

                raise

            generation_latency = (
                time.perf_counter()
                - generation_start
            )

            generated_response = (
                response.content
                if hasattr(
                    response,
                    "content",
                )
                else str(response)
            )

            print(
                "DEBUG: LLM response generated successfully."
            )

            print(
                f"DEBUG: LLM generation latency: "
                f"{generation_latency:.3f}s"
            )

            # -----------------------------------------------------
            # Token Usage
            # -----------------------------------------------------
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0

            usage_metadata = getattr(
                response,
                "usage_metadata",
                None,
            )

            if usage_metadata:

                input_tokens = usage_metadata.get(
                    "input_tokens",
                    0,
                )

                output_tokens = usage_metadata.get(
                    "output_tokens",
                    0,
                )

                total_tokens = usage_metadata.get(
                    "total_tokens",
                    input_tokens + output_tokens,
                )

            else:

                response_metadata = getattr(
                    response,
                    "response_metadata",
                    {},
                )

                token_usage = (
                    response_metadata.get(
                        "token_usage",
                        {},
                    )
                )

                input_tokens = token_usage.get(
                    "prompt_tokens",
                    0,
                )

                output_tokens = token_usage.get(
                    "completion_tokens",
                    0,
                )

                total_tokens = token_usage.get(
                    "total_tokens",
                    input_tokens + output_tokens,
                )

            print(
                f"DEBUG: Token usage - "
                f"input={input_tokens}, "
                f"output={output_tokens}, "
                f"total={total_tokens}"
            )

            # -----------------------------------------------------
            # Save metrics for answer_query()
            # -----------------------------------------------------
            self._last_generation_metrics = {
                "generation_latency_seconds": (
                    generation_latency
                ),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }

            # -----------------------------------------------------
            # MLflow generation metrics
            # -----------------------------------------------------
            if mlflow_active:

                mlflow.log_param(
                    "generation_error",
                    False,
                )

                mlflow.log_metric(
                    "generation_latency_seconds",
                    generation_latency,
                )

                mlflow.log_metric(
                    "input_tokens",
                    input_tokens,
                )

                mlflow.log_metric(
                    "output_tokens",
                    output_tokens,
                )

                mlflow.log_metric(
                    "total_tokens",
                    total_tokens,
                )

            # -----------------------------------------------------
            # Governance: Output validation
            # -----------------------------------------------------
            gov_output_check = (
                self.governance_gate.validate_output(
                    generated_response
                )
            )

            if not gov_output_check.get(
                "passed",
                False,
            ):

                print(
                    "⚠️ Generated response blocked "
                    "by governance gate."
                )

                if mlflow_active:

                    mlflow.log_param(
                        "output_governance_check_failed",
                        True,
                    )

                    mlflow.log_param(
                        "output_violations",
                        ",".join(
                            gov_output_check.get(
                                "violations",
                                [],
                            )
                        ),
                    )

                return (
                    "I generated a response but it didn't "
                    "pass safety checks. Please rephrase "
                    "your question."
                )

            # -----------------------------------------------------
            # Log successful response
            # -----------------------------------------------------
            if mlflow_active:

                mlflow.log_param(
                    "output_governance_check_failed",
                    False,
                )

                mlflow.log_text(
                    generated_response,
                    "final_response.txt",
                )

            return generated_response

    # =============================================================
    # Complete Chat Query
    # =============================================================

    # =============================================================
    # Complete Chat Query
    # =============================================================

    def answer_query(
        self,
        query_text: str,
        k: int = 5,
    ):
        """
        Execute one complete travel chatbot interaction.

        MLflow telemetry is handled by the existing
        search_by_text() and synthesize_response() runs.

        This method intentionally does NOT use mlflow.start_span()
        because Azure ML MLflow tracking is working for runs,
        parameters, metrics, and artifacts, while the MLflow
        trace endpoint currently returns HTTP 404.

        Captures:
        - Query
        - Retrieval latency
        - Retrieved document count
        - Unique source count
        - Generation latency
        - Input tokens
        - Output tokens
        - Total tokens
        - Final answer
        """

        print("DEBUG: Starting answer_query()")

        # ---------------------------------------------------------
        # Retrieval
        #
        # search_by_text() already creates its own MLflow run
        # and logs retrieval metrics.
        # ---------------------------------------------------------

        print("DEBUG: Starting retrieval...")

        retrieval_start = time.perf_counter()

        try:

            docs, processed_query = self.search_by_text(
                query_text,
                k=k,
            )

        except Exception as e:

            retrieval_latency = (
                time.perf_counter()
                - retrieval_start
            )

            print(
                f"⚠️ Retrieval failed after "
                f"{retrieval_latency:.3f}s: {e}"
            )

            raise

        retrieval_latency = (
            time.perf_counter()
            - retrieval_start
        )

        unique_sources = len({
            doc.metadata.get(
                "source",
                "Unknown",
            )
            for doc in docs
        })

        print(
            f"DEBUG: Retrieval completed in "
            f"{retrieval_latency:.3f}s"
        )

        print(
            f"DEBUG: Documents found: "
            f"{len(docs)}"
        )

        print(
            f"DEBUG: Unique sources: "
            f"{unique_sources}"
        )

        # ---------------------------------------------------------
        # Generation
        #
        # synthesize_response() already creates its own MLflow
        # run and logs generation metrics.
        # ---------------------------------------------------------

        print("DEBUG: Starting response generation...")

        try:

            answer = self.synthesize_response(
                docs,
                query_text,
            )

        except Exception as e:

            print(
                f"⚠️ Response generation failed: {e}"
            )

            raise

        # ---------------------------------------------------------
        # Generation metrics
        # ---------------------------------------------------------

        generation_metrics = (
            self._last_generation_metrics
        )

        input_tokens = (
            generation_metrics.get(
                "input_tokens",
                0,
            )
        )

        output_tokens = (
            generation_metrics.get(
                "output_tokens",
                0,
            )
        )

        total_tokens = (
            generation_metrics.get(
                "total_tokens",
                0,
            )
        )

        generation_latency = (
            generation_metrics.get(
                "generation_latency_seconds",
                0.0,
            )
        )

        # ---------------------------------------------------------
        # Final debug information
        # ---------------------------------------------------------

        print(
            "DEBUG: Generation completed successfully."
        )

        print(
            f"DEBUG: Generation latency: "
            f"{generation_latency:.3f}s"
        )

        print(
            f"DEBUG: Token usage - "
            f"input={input_tokens}, "
            f"output={output_tokens}, "
            f"total={total_tokens}"
        )

        print(
            "DEBUG: answer_query() completed successfully."
        )

        # ---------------------------------------------------------
        # Return complete RAG result
        # ---------------------------------------------------------

        return (
            docs,
            processed_query,
            answer,
        )