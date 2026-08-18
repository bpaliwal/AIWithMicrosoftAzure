"""
Configuration tests for the WanderNest LLMOps pipeline.

Covers:
- Azure OpenAI configuration
- Azure AI Search configuration
- Azure Content Safety configuration
- Azure Blob Storage configuration
- Vector store configuration
- MLflow configuration
- Configuration validation
"""

import pytest

from src.config import Config


# ============================================================
# Azure OpenAI
# ============================================================

def test_azure_openai_config_loaded():
    """Azure OpenAI credentials and deployment settings are loaded."""
    assert Config.AZURE_OPENAI_API_KEY
    assert Config.AZURE_OPENAI_ENDPOINT
    assert Config.AZURE_OPENAI_API_VERSION
    assert Config.AZURE_OPENAI_DEPLOYMENT_NAME
    assert Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT


# ============================================================
# Azure AI Search
# ============================================================

def test_azure_search_config_loaded():
    """Azure AI Search configuration is loaded."""
    assert Config.AZURE_SEARCH_ENDPOINT
    assert Config.AZURE_SEARCH_KEY
    assert Config.AZURE_SEARCH_INDEX_NAME


# ============================================================
# Azure Content Safety
# ============================================================

def test_content_safety_config_loaded():
    """Azure Content Safety endpoint and key are loaded."""
    assert Config.AZURE_CONTENT_SAFETY_ENDPOINT
    assert Config.AZURE_CONTENT_SAFETY_KEY


# ============================================================
# Azure Storage
# ============================================================

def test_azure_storage_config_loaded():
    """Azure Blob Storage configuration is loaded."""
    assert Config.AZURE_STORAGE_CONNECTION_STRING
    assert Config.AZURE_STORAGE_CONTAINER_NAME


# ============================================================
# Vector Store
# ============================================================

def test_vector_store_type():
    """Vector store type is one of the supported options."""
    assert Config.VECTOR_STORE_TYPE in ["chroma", "azure_search"]


# ============================================================
# MLflow
# ============================================================

def test_mlflow_config():
    """MLflow configuration is loaded."""
    assert Config.MLFLOW_TRACKING_URI
    assert Config.MLFLOW_EXPERIMENT_NAME


# ============================================================
# Configuration Validation
# ============================================================

def test_config_validation():
    """Required configuration passes validation."""
    assert Config.validate() is True
    