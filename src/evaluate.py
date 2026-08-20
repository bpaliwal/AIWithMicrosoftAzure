"""
Ragas Evaluation for Travel Chatbot

RUBRIC: Evaluation Framework (RAGAS) (8 marks total)
- RAGAS evaluation implemented (3 marks)
- Golden dataset created (2 marks)
- All four metrics computed (2 marks)
- Results saved with pass/fail logic (1 mark)

Ragas version: 0.4.3

This implementation uses the Ragas 0.4 Collections API:
    - Faithfulness
    - AnswerRelevancy
    - ContextPrecision
    - ContextRecall

The project uses Azure OpenAI for both:
    - Ragas evaluation LLM
    - Ragas evaluation embeddings
"""

import asyncio
import json
import logging
import sys
import types
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# RAGAS 0.4.3 compatibility shim
# ---------------------------------------------------------------------------
#
# Ragas 0.4.3 may import the legacy VertexAI module path during package
# initialization:
#
#     langchain_community.chat_models.vertexai
#
# Newer langchain-community versions may no longer provide that module.
#
# This project does not use VertexAI. The stub below allows Ragas to import
# without requiring the optional VertexAI integration.
# ---------------------------------------------------------------------------

try:
    import importlib.util

    vertexai_module = importlib.util.find_spec(
        "langchain_community.chat_models.vertexai"
    )

    if vertexai_module is None:
        vertexai_stub = types.ModuleType(
            "langchain_community.chat_models.vertexai"
        )

        class ChatVertexAI:
            """Compatibility placeholder for Ragas 0.4.3."""

            pass

        vertexai_stub.ChatVertexAI = ChatVertexAI

        sys.modules[
            "langchain_community.chat_models.vertexai"
        ] = vertexai_stub

except Exception:
    pass


# ---------------------------------------------------------------------------
# Ragas imports
# ---------------------------------------------------------------------------

from openai import AsyncAzureOpenAI

from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from src.config import Config
from src.search_engine import TravelSearchEngine


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluation")


# ===========================================================================
# TravelChatbotEvaluator
# ===========================================================================

class TravelChatbotEvaluator:
    """Evaluates Travel Chatbot using Ragas 0.4.3."""

    def __init__(self):
        # ---------------------------------------------------------------
        # Initialize search engine
        # ---------------------------------------------------------------
        self.engine = TravelSearchEngine()

        # ---------------------------------------------------------------
        # Golden dataset
        # ---------------------------------------------------------------
        self.golden_dataset_path = (
            Path("data") / "golden_dataset.json"
        )

        # ---------------------------------------------------------------
        # Evaluation output
        # ---------------------------------------------------------------
        self.output_dir = Path("reports")

        # ---------------------------------------------------------------
        # Ragas models
        # ---------------------------------------------------------------
        self.ragas_llm = None
        self.ragas_embeddings = None

        # ---------------------------------------------------------------
        # Ragas metrics
        # ---------------------------------------------------------------
        self.metrics = {}

    # ===================================================================
    # Ragas model initialization
    # ===================================================================

    def _initialize_ragas_models(self):
        """
        Initialize Ragas 0.4.3 evaluator models using Azure OpenAI.

        IMPORTANT:
        This deliberately does NOT use:

            AsyncOpenAI()

        because that requires OPENAI_API_KEY.

        Instead we use:

            AsyncAzureOpenAI(...)

        with the Azure credentials already defined in Config.
        """

        logger.info("Initializing Ragas evaluator models...")

        # ---------------------------------------------------------------
        # Validate required Azure configuration
        # ---------------------------------------------------------------

        required_config = {
            "AZURE_OPENAI_API_KEY": Config.AZURE_OPENAI_API_KEY,
            "AZURE_OPENAI_ENDPOINT": Config.AZURE_OPENAI_ENDPOINT,
            "AZURE_OPENAI_API_VERSION": Config.AZURE_OPENAI_API_VERSION,
            "AZURE_OPENAI_DEPLOYMENT_NAME": (
                Config.AZURE_OPENAI_DEPLOYMENT_NAME
            ),
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": (
                Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
            ),
        }

        missing = [
            name
            for name, value in required_config.items()
            if not value
        ]

        if missing:
            raise EnvironmentError(
                "Missing Azure OpenAI configuration: "
                + ", ".join(missing)
            )

        logger.info(
            "Azure OpenAI endpoint: %s",
            Config.AZURE_OPENAI_ENDPOINT,
        )

        logger.info(
            "Ragas LLM deployment: %s",
            Config.AZURE_OPENAI_DEPLOYMENT_NAME,
        )

        logger.info(
            "Ragas embedding deployment: %s",
            Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        )

        # ---------------------------------------------------------------
        # Create Azure OpenAI client
        # ---------------------------------------------------------------
        #
        # This is the critical fix.
        #
        # DO NOT use:
        #
        #     AsyncOpenAI()
        #
        # because that looks for OPENAI_API_KEY.
        #
        # ---------------------------------------------------------------

        azure_client = AsyncAzureOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            azure_deployment=Config.AZURE_OPENAI_DEPLOYMENT_NAME,
        )

        logger.info(
            "Azure AsyncOpenAI client initialized successfully."
        )

        # ---------------------------------------------------------------
        # Ragas LLM
        # ---------------------------------------------------------------

        self.ragas_llm = llm_factory(
            model=Config.AZURE_OPENAI_DEPLOYMENT_NAME,
            provider="openai",
            client=azure_client,
        )

        # -------------------------------------------------------------------
        # GPT-5.x compatibility for Ragas 0.4.x
        # -------------------------------------------------------------------
        # Ragas' InstructorLLM parameter mapper incorrectly treats versions
        # such as "gpt-5.4-mini" as non-reasoning models because it attempts
        # int("5.4"). Azure GPT-5.x requires:
        #
        #     max_completion_tokens
        #
        # instead of:
        #
        #     max_tokens
        #
        # and only supports temperature=1.0.
        #
        # Short-circuit Ragas' mapper for our Azure GPT-5.x deployment rather
        # than changing the actual Azure/OpenAI model.
        # -------------------------------------------------------------------

        _original_map_provider_params = self.ragas_llm._map_provider_params

        def _azure_gpt5_map_provider_params():
            params = _original_map_provider_params()

            model_name = self.ragas_llm.model.lower()

            if model_name.startswith("gpt-5"):
                if "max_tokens" in params:
                    params["max_completion_tokens"] = params.pop("max_tokens")

                params["temperature"] = 1.0
                params.pop("top_p", None)

            return params

        self.ragas_llm._map_provider_params = _azure_gpt5_map_provider_params

        logger.info(
            "Ragas LLM initialized successfully with GPT-5.x parameter compatibility."
        )

        # ---------------------------------------------------------------
        # Ragas embeddings
        # ---------------------------------------------------------------
        #
        # Use the SAME Azure client but specify the embedding deployment.
        #
        # Ragas 0.4.x embedding_factory() accepts the OpenAI-compatible
        # client and model name.
        # ---------------------------------------------------------------

        embedding_client = AsyncAzureOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            azure_deployment=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        )

        self.ragas_embeddings = embedding_factory(
            "openai",
            model=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            client=embedding_client,
        )
        
        logger.info(
            "Ragas embeddings initialized successfully."
        )

        logger.info(
            "Ragas evaluator models initialized successfully."
        )

    # ===================================================================
    # Golden dataset
    # ===================================================================

    def load_golden_dataset(self) -> List[Dict]:
        """
        Load golden dataset for evaluation.

        If the dataset does not exist, create a sample dataset.
        """

        if not self.golden_dataset_path.exists():

            logger.warning(
                "Golden dataset not found at %s",
                self.golden_dataset_path,
            )

            logger.info(
                "Creating sample golden dataset..."
            )

            return self._create_sample_dataset()

        with open(
            self.golden_dataset_path,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        logger.info(
            "Golden dataset loaded from %s",
            self.golden_dataset_path,
        )

        return data

    # ===================================================================
    # Create sample golden dataset
    # ===================================================================

    def _create_sample_dataset(self) -> List[Dict]:
        """
        Create sample golden dataset if it does not exist.

        Each record contains:
            question
            ground_truth
        """

        sample_data = [

            {
                "question": (
                    "What are the baggage allowance rules "
                    "for international flights?"
                ),
                "ground_truth": (
                    "Passengers usually receive one checked bag and one "
                    "cabin bag for international flights, but exact "
                    "allowances depend on the fare class and route. "
                    "Always verify your ticket rules before travel."
                ),
            },

            {
                "question": (
                    "What is Air India's cancellation policy?"
                ),
                "ground_truth": (
                    "Air India cancellation rules vary by fare type and "
                    "route. Refund eligibility depends on whether the "
                    "ticket is refundable, partially refundable, or "
                    "non-refundable. Check the fare conditions before "
                    "canceling."
                ),
            },

            {
                "question": (
                    "Do I need a visa to travel from India to UK?"
                ),
                "ground_truth": (
                    "Visa requirements depend on nationality and travel "
                    "purpose. Indian citizens generally need a visa to "
                    "travel to the UK, and requirements can vary by trip "
                    "type and passport status."
                ),
            },

            {
                "question": (
                    "What are the refund policies for flight cancellations?"
                ),
                "ground_truth": (
                    "Refund policies depend on the airline fare conditions, "
                    "cancellation timing, and reason for cancellation. "
                    "Flexible fares are often refundable, while basic or "
                    "discounted fares may be non-refundable or partially "
                    "refundable."
                ),
            },

            {
                "question": (
                    "What documents do I need for international travel?"
                ),
                "ground_truth": (
                    "Travel documents usually include a valid passport, "
                    "visa if required, and any onward or return travel "
                    "documentation. Airlines may also request proof of "
                    "accommodation or travel insurance depending on the "
                    "destination."
                ),
            },

        ]

        self.golden_dataset_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            self.golden_dataset_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                sample_data,
                f,
                indent=2,
            )

        logger.info(
            "Sample golden dataset saved to %s",
            self.golden_dataset_path,
        )

        return sample_data

    # ===================================================================
    # Generate chatbot responses
    # ===================================================================

    def generate_responses(
        self,
        questions: List[str],
    ) -> Tuple[List[str], List[List[str]]]:
        """
        Generate chatbot responses.

        For each question:
            1. Retrieve relevant documents.
            2. Generate answer.
            3. Collect retrieved contexts.

        Returns:
            answers
            contexts
        """

        answers = []
        contexts = []

        for question in questions:

            logger.info(
                "Generating answer for: %s",
                question,
            )

            try:

                # -------------------------------------------------------
                # Retrieve documents
                # -------------------------------------------------------

                docs, _ = self.engine.search_by_text(
                    question,
                    k=5,
                )

                # -------------------------------------------------------
                # Generate answer
                # -------------------------------------------------------

                answer = self.engine.synthesize_response(
                    docs,
                    question,
                )

                # -------------------------------------------------------
                # Extract context text
                # -------------------------------------------------------

                context_texts = [
                    doc.page_content
                    for doc in docs
                ]

                answers.append(answer)
                contexts.append(context_texts)

            except Exception as e:

                logger.error(
                    "Error generating answer for '%s': %s",
                    question,
                    e,
                )

                answers.append(
                    "I’m unable to generate a verified answer "
                    "for this question right now."
                )

                contexts.append([])

        return answers, contexts

    # ===================================================================
    # Score one metric for one question
    # ===================================================================

    async def _score_single_question(
        self,
        metric_name: str,
        metric,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
    ) -> float:
        """
        Score one question using one Ragas 0.4.3 collection metric.

        Ragas 0.4.x:
            await metric.ascore(...)

        returns:
            MetricResult

        Numeric score:
            result.value
        """

        try:

            # -----------------------------------------------------------
            # Faithfulness
            # -----------------------------------------------------------

            if metric_name == "faithfulness":

                result = await metric.ascore(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=contexts,
                )

            # -----------------------------------------------------------
            # Answer Relevancy
            # -----------------------------------------------------------

            elif metric_name == "answer_relevancy":

                result = await metric.ascore(
                    user_input=question,
                    response=answer,
                )

            # -----------------------------------------------------------
            # Context Precision
            # -----------------------------------------------------------

            elif metric_name == "context_precision":

                result = await metric.ascore(
                    user_input=question,
                    reference=ground_truth,
                    retrieved_contexts=contexts,
                )

            # -----------------------------------------------------------
            # Context Recall
            # -----------------------------------------------------------

            elif metric_name == "context_recall":

                result = await metric.ascore(
                    user_input=question,
                    reference=ground_truth,
                    retrieved_contexts=contexts,
                )

            else:

                raise ValueError(
                    f"Unknown metric: {metric_name}"
                )

            score = float(result.value)

            logger.debug(
                "%s score: %.4f",
                metric_name,
                score,
            )

            return score

        except Exception as e:

            logger.error(
                "Failed to calculate %s for question '%s': %s",
                metric_name,
                question,
                e,
            )

            return 0.0

    # ===================================================================
    # Run Ragas evaluation
    # ===================================================================

    async def run_ragas_evaluation(self):
        """
        Run complete Ragas 0.4.3 evaluation.

        Calculates:

            1. Faithfulness
            2. Answer Relevancy
            3. Context Precision
            4. Context Recall

        Saves:

            reports/evaluation_summary.json
            reports/evaluation_detailed.csv
        """

        logger.info("=" * 70)
        logger.info("Starting Ragas 0.4.3 Evaluation...")
        logger.info("=" * 70)

        # ---------------------------------------------------------------
        # Load golden dataset
        # ---------------------------------------------------------------

        golden_data = self.load_golden_dataset()

        if not golden_data:

            logger.error(
                "No evaluation data available."
            )

            return None

        logger.info(
            "Loaded %d test cases",
            len(golden_data),
        )

        # ---------------------------------------------------------------
        # Extract questions and references
        # ---------------------------------------------------------------

        questions = [
            item["question"]
            for item in golden_data
        ]

        ground_truths = [
            item["ground_truth"]
            for item in golden_data
        ]

        # ---------------------------------------------------------------
        # Generate chatbot responses
        # ---------------------------------------------------------------

        logger.info(
            "Generating responses..."
        )

        answers, contexts = self.generate_responses(
            questions
        )

        # ---------------------------------------------------------------
        # Initialize Azure Ragas models
        # ---------------------------------------------------------------

        try:

            self._initialize_ragas_models()

        except Exception as e:

            logger.error(
                "Unable to initialize Ragas models: %s",
                e,
            )

            logger.error(
                "Check Azure OpenAI configuration in .env."
            )

            return None

        # ---------------------------------------------------------------
        # Create Ragas metrics
        # ---------------------------------------------------------------

        logger.info(
            "Initializing Ragas metrics..."
        )

        self.metrics = {

            "faithfulness": Faithfulness(
                llm=self.ragas_llm,
            ),

            "answer_relevancy": AnswerRelevancy(
                llm=self.ragas_llm,
                embeddings=self.ragas_embeddings,
            ),

            "context_precision": ContextPrecision(
                llm=self.ragas_llm,
            ),

            "context_recall": ContextRecall(
                llm=self.ragas_llm,
            ),

        }

        logger.info(
            "All four Ragas metrics initialized."
        )

        # ---------------------------------------------------------------
        # Evaluate every test case
        # ---------------------------------------------------------------

        detailed_results = []

        for index, (
            question,
            answer,
            retrieved_contexts,
            ground_truth,
        ) in enumerate(
            zip(
                questions,
                answers,
                contexts,
                ground_truths,
            ),
            start=1,
        ):

            logger.info(
                "Evaluating test case %d/%d",
                index,
                len(questions),
            )

            logger.info(
                "Question: %s",
                question,
            )

            row = {

                "question": question,

                "answer": answer,

                "contexts": retrieved_contexts,

                "ground_truth": ground_truth,

            }

            # -----------------------------------------------------------
            # Calculate all four metrics
            # -----------------------------------------------------------

            for metric_name, metric in self.metrics.items():

                score = await self._score_single_question(
                    metric_name=metric_name,
                    metric=metric,
                    question=question,
                    answer=answer,
                    contexts=retrieved_contexts,
                    ground_truth=ground_truth,
                )

                row[metric_name] = score

                logger.info(
                    "  %-20s %.4f",
                    metric_name,
                    score,
                )

            detailed_results.append(row)

        # ---------------------------------------------------------------
        # Calculate aggregate scores
        # ---------------------------------------------------------------

        metric_names = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]

        metric_scores = {}

        for metric_name in metric_names:

            scores = [
                float(row[metric_name])
                for row in detailed_results
            ]

            if scores:

                metric_scores[metric_name] = (
                    sum(scores) / len(scores)
                )

            else:

                metric_scores[metric_name] = 0.0

        # ---------------------------------------------------------------
        # Display results
        # ---------------------------------------------------------------

        logger.info("")
        logger.info("=" * 70)
        logger.info("RAGAS EVALUATION RESULTS")
        logger.info("=" * 70)

        logger.info(
            "  Faithfulness:       %.4f",
            metric_scores["faithfulness"],
        )

        logger.info(
            "  Answer Relevancy:   %.4f",
            metric_scores["answer_relevancy"],
        )

        logger.info(
            "  Context Precision:  %.4f",
            metric_scores["context_precision"],
        )

        logger.info(
            "  Context Recall:     %.4f",
            metric_scores["context_recall"],
        )

        logger.info("=" * 70)

        # ---------------------------------------------------------------
        # Save results
        # ---------------------------------------------------------------

        dataset_dict = {

            "question": questions,

            "answer": answers,

            "contexts": contexts,

            "ground_truth": ground_truths,

        }

        self._save_results(
            metric_scores,
            dataset_dict,
            detailed_results,
        )

        return metric_scores

    # ===================================================================
    # Save evaluation results
    # ===================================================================

    def _save_results(
        self,
        results: dict,
        dataset_dict: dict,
        detailed_results: List[Dict],
    ):
        """
        Save evaluation results.

        Files:

            reports/evaluation_summary.json
            reports/evaluation_detailed.csv
        """

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ---------------------------------------------------------------
        # Pass/fail thresholds
        # ---------------------------------------------------------------

        min_faithfulness = 0.70
        min_relevancy = 0.70

        faithfulness_score = float(
            results.get(
                "faithfulness",
                0,
            )
            or 0
        )

        answer_relevancy_score = float(
            results.get(
                "answer_relevancy",
                0,
            )
            or 0
        )

        passed = (
            faithfulness_score >= min_faithfulness
            and
            answer_relevancy_score >= min_relevancy
        )

        # ---------------------------------------------------------------
        # Summary
        # ---------------------------------------------------------------

        summary = {

            "faithfulness": faithfulness_score,

            "answer_relevancy": answer_relevancy_score,

            "context_precision": float(
                results.get(
                    "context_precision",
                    0,
                )
                or 0
            ),

            "context_recall": float(
                results.get(
                    "context_recall",
                    0,
                )
                or 0
            ),

            "total_test_cases": len(
                dataset_dict["question"]
            ),

            "thresholds": {

                "faithfulness": min_faithfulness,

                "answer_relevancy": min_relevancy,

            },

            "passed": passed,

        }

        summary_path = (
            self.output_dir /
            "evaluation_summary.json"
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                summary,
                f,
                indent=2,
            )

        logger.info(
            "Evaluation summary saved to %s",
            summary_path,
        )

        # ---------------------------------------------------------------
        # Detailed CSV
        # ---------------------------------------------------------------

        detailed_rows = []

        for row in detailed_results:

            detailed_rows.append({

                "question": row["question"],

                "answer": row["answer"],

                "contexts": " | ".join(
                    row["contexts"]
                ),

                "ground_truth": row["ground_truth"],

                "faithfulness": row[
                    "faithfulness"
                ],

                "answer_relevancy": row[
                    "answer_relevancy"
                ],

                "context_precision": row[
                    "context_precision"
                ],

                "context_recall": row[
                    "context_recall"
                ],

            })

        detailed_df = pd.DataFrame(
            detailed_rows
        )

        detailed_path = (
            self.output_dir /
            "evaluation_detailed.csv"
        )

        detailed_df.to_csv(
            detailed_path,
            index=False,
        )

        logger.info(
            "Detailed results saved to %s",
            detailed_path,
        )

        # ---------------------------------------------------------------
        # Final pass/fail
        # ---------------------------------------------------------------

        if passed:

            logger.info(
                "✅ EVALUATION PASSED"
            )

        else:

            logger.warning(
                "⚠️ EVALUATION BELOW THRESHOLDS"
            )

        return summary

    # ===================================================================
    # Synchronous wrapper
    # ===================================================================

    def run(self):
        """
        Run evaluation synchronously.
        """

        try:

            return asyncio.run(
                self.run_ragas_evaluation()
            )

        except RuntimeError as e:

            logger.warning(
                "asyncio.run() could not be used: %s",
                e,
            )

            loop = asyncio.new_event_loop()

            try:

                asyncio.set_event_loop(loop)

                return loop.run_until_complete(
                    self.run_ragas_evaluation()
                )

            finally:

                loop.close()


# ===========================================================================
# Main evaluation function
# ===========================================================================

def run_evaluation():
    """
    Main evaluation function.

    Pass criteria:

        Faithfulness >= 0.70
        Answer Relevancy >= 0.70

    All four metrics are calculated and saved.
    """

    evaluator = TravelChatbotEvaluator()

    results = evaluator.run()

    if results:

        min_faithfulness = 0.70
        min_relevancy = 0.70

        faithfulness_score = float(
            results.get(
                "faithfulness",
                0,
            )
            or 0
        )

        answer_relevancy_score = float(
            results.get(
                "answer_relevancy",
                0,
            )
            or 0
        )

        passed = (
            faithfulness_score >= min_faithfulness
            and
            answer_relevancy_score >= min_relevancy
        )

        if passed:

            logger.info(
                "✅ EVALUATION PASSED"
            )

            logger.info(
                "Faithfulness %.4f >= %.2f",
                faithfulness_score,
                min_faithfulness,
            )

            logger.info(
                "Answer Relevancy %.4f >= %.2f",
                answer_relevancy_score,
                min_relevancy,
            )

            return 0

        else:

            logger.warning(
                "⚠️ EVALUATION BELOW THRESHOLDS"
            )

            logger.warning(
                "Faithfulness: %.4f (minimum %.2f)",
                faithfulness_score,
                min_faithfulness,
            )

            logger.warning(
                "Answer Relevancy: %.4f (minimum %.2f)",
                answer_relevancy_score,
                min_relevancy,
            )

            return 1

    logger.error(
        "❌ EVALUATION FAILED"
    )

    return 1


# ===========================================================================
# Command-line entry point
# ===========================================================================

if __name__ == "__main__":

    exit_code = run_evaluation()

    sys.exit(exit_code)