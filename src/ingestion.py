"""
Document Ingestion Pipeline
RUBRIC: Document Ingestion Pipeline (8 marks total)
- Ingestion script loads all documents (2 marks)
- Documents are chunked properly (2 marks)
- Batch indexing implemented (2 marks)
- Index verification performed (2 marks)

TASK: Ingest and index documents to Azure AI Search
"""
import time
from tqdm import tqdm

from src.search_engine import TravelSearchEngine
from src.data_loader import TravelDataLoader
from src.config import Config

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents.indexes import SearchIndexClient

import mlflow
import os


def recreate_search_index():
    """
    Delete the existing Azure AI Search index, if it exists.

    The LangChain AzureSearch vector store will create the index again
    when it is initialized. This makes ingestion idempotent and ensures
    every ingestion starts with a clean index.
    """
    endpoint = Config.AZURE_SEARCH_ENDPOINT
    key = Config.AZURE_SEARCH_KEY
    index_name = Config.AZURE_SEARCH_INDEX_NAME

    if not endpoint or not key or not index_name:
        raise ValueError(
            "AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, "
            "and AZURE_SEARCH_INDEX_NAME must be configured."
        )

    print(f"\n🗑️  Recreating Azure AI Search index: {index_name}")

    client = SearchIndexClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )

    try:
        client.delete_index(index_name)
        print(f"   ✅ Deleted existing index: {index_name}")

    except ResourceNotFoundError:
        print(f"   ℹ️  Index does not exist yet: {index_name}")

    print("   ✅ Index cleanup complete")


def ingest_travel_documents():
    """
    Ingests travel documents into Azure AI Search vector store

    HINT: This function should:
    1. Initialize data loader and search engine
    2. Load all documents
    3. Split into chunks
    4. Batch index to Azure Search (batch_size=50)
    5. Verify with test query
    """
    print("\n🚀 Starting Travel Document Ingestion")
    print("=" * 70)

    # HINT: Initialize components
    loader = TravelDataLoader()

    try:
        # Always start with a clean Azure AI Search index.
        recreate_search_index()

        engine = TravelSearchEngine()

    except Exception as e:
        print(f"❌ Failed to initialize search engine: {e}")
        return

    # ====================
    # MLflow Setup (fail-safe)
    # ====================
    mlflow_active = False
    if Config.MLFLOW_TRACKING_URI:
        try:
            mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
            mlflow.start_run(run_name="document_ingestion")
            mlflow_active = True
        except Exception as e:
            print(f"⚠️  MLflow disabled: {e}")

    try:
        # HINT: Load documents
        documents = loader.load_all_travel_documents()

        if not documents:
            print("\n⚠️  No documents found in data directory")
            print("\nExpected structure:")
            print("  data/")
            print("    ├── *.pdf   (policies, FAQs, rules)")
            print("    └── *.csv   (routes or tabular data)")
            return

        # HINT: Split into chunks
        chunks = loader.split_documents(documents)

        print(f"\n📊 Ingestion Summary:")
        print(f"   Total chunks to index: {len(chunks)}")

        if mlflow_active:
            mlflow.log_param("total_chunks", len(chunks))
            mlflow.log_param(
                "chunk_size",
                loader.text_splitter._chunk_size
            )
            mlflow.log_param(
                "chunk_overlap",
                loader.text_splitter._chunk_overlap
            )

        # ====================
        # Batch Ingestion
        # ====================
        print("\n📥 Indexing documents to Azure AI Search...")
        batch_size = 50
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        ingested_count = 0
        failed_count = 0

        # HINT: Loop through chunks in batches
        for i in tqdm(
            range(0, len(chunks), batch_size),
            desc="Indexing",
            total=total_batches
        ):
            batch = chunks[i:i + batch_size]

            try:
                # HINT: Add documents to vector store
                engine.vector_store.add_documents(batch)
                ingested_count += len(batch)
                time.sleep(0.1)

            except Exception as e:
                print(
                    f"\n❌ Error indexing batch "
                    f"{i // batch_size + 1}: {e}"
                )
                failed_count += len(batch)

        print(f"\n✅ Ingestion Complete!")
        print(f"   Successfully indexed: {ingested_count} chunks")

        if failed_count > 0:
            print(f"   Failed: {failed_count} chunks")

        if mlflow_active:
            mlflow.log_metric("ingested_count", ingested_count)
            mlflow.log_metric("failed_count", failed_count)

        # ====================
        # Verification
        # ====================
        print("\n🔍 Verifying index...")

        test_query = (
            "What are the baggage allowance rules "
            "for international flights?"
        )

        results, _ = engine.search_by_text(
            test_query,
            k=int(os.getenv("RETRIEVAL_K", "5"))
        )

        if results:
            print("✅ Index verification successful!")
            print(f"   Test query: '{test_query}'")
            print(f"   Retrieved: {len(results)} documents")
        else:
            print("⚠️  Warning: Test query returned no results")

    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")

    finally:
        if mlflow_active:
            mlflow.end_run()

    print("\n" + "=" * 70)
    print("🎉 Ingestion pipeline completed!\n")


if __name__ == "__main__":
    ingest_travel_documents()
