"""
Configuration Management
========================

Configuration is loaded from environment variables.

Supported environments
----------------------

1. Local development (.venv)
   - Requires a .env file at the project root.
   - .env is loaded automatically.

2. Docker
   - .env is NOT required inside the container.
   - Environment variables must be supplied at container runtime.
   - Example:
       docker run --env-file .env -p 8501:8501 wander-nest-ai:latest

3. Azure
   - .env is NOT required.
   - Azure environment variables are used directly.
   - Recommended production approach:
       Azure App Service / Container Apps configuration
       backed by Azure Key Vault references.

The application never stores secrets in this source file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse


# ============================================================
# Environment Detection
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def is_running_in_docker() -> bool:
    """
    Detect whether the application is running inside Docker.

    Docker normally creates /.dockerenv inside the container.
    We also check the cgroup information as a fallback.
    """

    if Path("/.dockerenv").exists():
        return True

    try:
        cgroup = Path("/proc/1/cgroup")

        if cgroup.exists():
            content = cgroup.read_text(errors="ignore").lower()

            if any(
                marker in content
                for marker in ("docker", "containerd", "kubepods")
            ):
                return True

    except Exception:
        pass

    return False


RUNNING_IN_DOCKER = is_running_in_docker()


# ============================================================
# Load Local .env
# ============================================================

if ENV_FILE.exists():

    # Local development:
    # Load .env into the process environment.
    load_dotenv(
        dotenv_path=ENV_FILE,
        override=False,
    )

elif not RUNNING_IN_DOCKER:

    # Outside Docker, .env is expected.
    #
    # This protects local development from accidentally running
    # with an incomplete configuration.
    raise FileNotFoundError(
        f"""
.env file not found at expected location:

    {ENV_FILE}

For local development, create a .env file at the project root.

For Docker/Azure deployments, .env is not required.
Environment variables should be supplied by the deployment
environment instead.
"""
    )

else:

    # Docker / Azure:
    # Do not require .env.
    #
    # Environment variables are expected to have been injected
    # by Docker or the Azure hosting environment.
    print(
        "DEBUG: Running in Docker/container environment. "
        "Using environment variables; .env is not required."
    )


# ============================================================
# Helper Functions
# ============================================================

def get_env(name: str, default=None):
    """
    Read an environment variable.

    Empty strings are treated as missing.
    """

    value = os.getenv(name)

    if value is None:
        return default

    value = value.strip()

    if value == "":
        return default

    return value


# ============================================================
# Configuration
# ============================================================

class Config:
    """Configuration for WanderNest Travel Chatbot."""

    # ========================================================
    # Azure OpenAI
    # ========================================================

    AZURE_OPENAI_API_KEY = get_env(
        "AZURE_OPENAI_API_KEY"
    )

    AZURE_OPENAI_ENDPOINT = get_env(
        "AZURE_OPENAI_ENDPOINT"
    )

    AZURE_OPENAI_API_VERSION = get_env(
        "AZURE_OPENAI_API_VERSION"
    )

    AZURE_OPENAI_DEPLOYMENT_NAME = get_env(
        "AZURE_OPENAI_DEPLOYMENT_NAME",
        "gpt-5.4-mini",
    )

    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = get_env(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        "text-embedding-3-small",
    )

    # ========================================================
    # Azure AI Search
    # ========================================================

    AZURE_SEARCH_ENDPOINT = get_env(
        "AZURE_SEARCH_ENDPOINT"
    )

    AZURE_SEARCH_KEY = get_env(
        "AZURE_SEARCH_KEY"
    )

    AZURE_SEARCH_INDEX_NAME = get_env(
        "AZURE_SEARCH_INDEX_NAME",
        "travel-kb-index",
    )

    AZURE_SEARCH_API_VERSION = get_env(
        "AZURE_SEARCH_API_VERSION",
        "2021-04-30-Preview",
    )

    # Production vector store
    VECTOR_STORE_TYPE = get_env(
        "VECTOR_STORE_TYPE_PROD",
        "azure_search",
    )

    # ========================================================
    # Azure Blob Storage
    # ========================================================

    AZURE_STORAGE_CONNECTION_STRING = get_env(
        "AZURE_STORAGE_CONNECTION_STRING"
    )

    AZURE_STORAGE_CONTAINER_NAME = get_env(
        "AZURE_STORAGE_CONTAINER_NAME",
        "travel-documents",
    )

    # ========================================================
    # Blob Storage switch
    # ========================================================

    BLOB_STORAGE = (
        get_env("BLOB_STORAGE", "False").lower()
        == "true"
    )

    # ========================================================
    # Azure Content Safety
    # ========================================================

    AZURE_CONTENT_SAFETY_ENDPOINT = get_env(
        "AZURE_CONTENT_SAFETY_ENDPOINT"
    )

    AZURE_CONTENT_SAFETY_KEY = get_env(
        "AZURE_CONTENT_SAFETY_KEY"
    )

    # ========================================================
    # Azure Monitor / Application Insights
    # ========================================================

    APPLICATIONINSIGHTS_CONNECTION_STRING = get_env(
        "APPLICATIONINSIGHTS_CONNECTION_STRING"
    )

    # ========================================================
    # MLflow
    # ========================================================

    MLFLOW_TRACKING_URI = get_env(
        "MLFLOW_TRACKING_URI"
    )

    MLFLOW_EXPERIMENT_NAME = get_env(
        "MLFLOW_EXPERIMENT_NAME",
        "wanderlust-travel-chatbot",
    )

    # ========================================================
    # Ingestion
    # ========================================================

    INGESTION_LIMIT = int(
        get_env("INGESTION_LIMIT", "0")
    )

    # ========================================================
    # Environment Information
    # ========================================================

    RUNNING_IN_DOCKER = RUNNING_IN_DOCKER

    # Optional explicit deployment environment.
    #
    # Examples:
    #   ENVIRONMENT=local
    #   ENVIRONMENT=docker
    #   ENVIRONMENT=azure
    #
    ENVIRONMENT = get_env(
        "ENVIRONMENT",
        "docker" if RUNNING_IN_DOCKER else "local",
    )

    # ========================================================
    # Validation
    # ========================================================

    @staticmethod
    def validate():
        """
        Validate required environment variables.

        Raises EnvironmentError if required configuration
        is missing.
        """

        required = {
            "AZURE_OPENAI_API_KEY":
                Config.AZURE_OPENAI_API_KEY,

            "AZURE_OPENAI_DEPLOYMENT_NAME":
                Config.AZURE_OPENAI_DEPLOYMENT_NAME,

            "AZURE_OPENAI_ENDPOINT":
                Config.AZURE_OPENAI_ENDPOINT,

            "AZURE_OPENAI_API_VERSION":
                Config.AZURE_OPENAI_API_VERSION,

            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT":
                Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,

            "AZURE_SEARCH_ENDPOINT":
                Config.AZURE_SEARCH_ENDPOINT,

            "AZURE_SEARCH_KEY":
                Config.AZURE_SEARCH_KEY,

            "AZURE_SEARCH_INDEX_NAME":
                Config.AZURE_SEARCH_INDEX_NAME,
        }

        missing = [
            name
            for name, value in required.items()
            if not value
        ]

        if missing:
            raise EnvironmentError(
                "Missing required config environment variables: "
                + ", ".join(missing)
            )

        return True

    # ========================================================
    # Connection Validation
    # ========================================================

    @staticmethod
    def validate_connections(
        timeout: int = 3,
        parallel: bool = True,
    ) -> dict:
        """
        Lightweight connectivity checks for configured endpoints.

        Returns:

        {
            "AZURE_OPENAI_ENDPOINT": {
                "ok": True,
                "msg": "...",
                "status_code": 200
            },
            ...
        }
        """

        results = {}

        def probe(name, func):

            try:

                ok, msg, code = func()

                results[name] = {
                    "ok": ok,
                    "msg": msg,
                    "status_code": code,
                }

            except Exception as exc:

                results[name] = {
                    "ok": False,
                    "msg": str(exc),
                    "status_code": None,
                }

        # ====================================================
        # Azure OpenAI
        # ====================================================

        def probe_openai():

            endpoint = Config.AZURE_OPENAI_ENDPOINT
            key = Config.AZURE_OPENAI_API_KEY
            api_ver = Config.AZURE_OPENAI_API_VERSION

            if not endpoint:
                return (
                    False,
                    "no endpoint configured",
                    None,
                )

            url = (
                endpoint.rstrip("/")
                + f"/openai/deployments?api-version={api_ver}"
            )

            headers = {}

            if key:
                headers["api-key"] = key

            try:

                response = requests.get(
                    url,
                    headers=headers,
                    timeout=timeout,
                )

                return (
                    response.status_code in (200, 401, 403),
                    response.reason,
                    response.status_code,
                )

            except Exception as exc:

                return False, str(exc), None

        # ====================================================
        # Azure AI Search
        # ====================================================

        def probe_search():

            endpoint = Config.AZURE_SEARCH_ENDPOINT
            key = Config.AZURE_SEARCH_KEY

            if not endpoint:
                return (
                    False,
                    "no endpoint configured",
                    None,
                )

            url = (
                endpoint.rstrip("/")
                + "/indexes"
                + "?api-version=2021-04-30-Preview"
            )

            headers = {}

            if key:
                headers["api-key"] = key

            try:

                response = requests.get(
                    url,
                    headers=headers,
                    timeout=timeout,
                )

                return (
                    response.status_code in (200, 401, 403),
                    response.reason,
                    response.status_code,
                )

            except Exception as exc:

                return False, str(exc), None

        # ====================================================
        # Azure Blob Storage
        # ====================================================

        def probe_storage():

            connection_string = (
                Config.AZURE_STORAGE_CONNECTION_STRING
            )

            if not connection_string:

                return (
                    False,
                    "no connection string",
                    None,
                )

            try:

                from azure.storage.blob import (
                    BlobServiceClient
                )

                client = (
                    BlobServiceClient
                    .from_connection_string(
                        connection_string
                    )
                )

                containers = list(
                    client.list_containers()
                )

                return (
                    True,
                    f"found {len(containers)} containers",
                    200,
                )

            except ImportError:

                return (
                    False,
                    "azure-storage-blob not installed",
                    None,
                )

            except Exception as exc:

                return False, str(exc), None

        # ====================================================
        # Azure Content Safety
        # ====================================================

        def probe_content_safety():

            endpoint = (
                Config.AZURE_CONTENT_SAFETY_ENDPOINT
            )

            key = Config.AZURE_CONTENT_SAFETY_KEY

            if not endpoint:

                return (
                    False,
                    "no endpoint configured",
                    None,
                )

            headers = {}

            if key:
                headers[
                    "Ocp-Apim-Subscription-Key"
                ] = key

            try:

                response = requests.get(
                    endpoint,
                    headers=headers,
                    timeout=timeout,
                )

                return (
                    response.status_code
                    in (200, 401, 403),
                    response.reason,
                    response.status_code,
                )

            except Exception as exc:

                return False, str(exc), None

        # ====================================================
        # MLflow
        # ====================================================

        def probe_mlflow():

            uri = Config.MLFLOW_TRACKING_URI

            if not uri:

                return (
                    False,
                    "no mlflow uri configured",
                    None,
                )

            parsed = urlparse(uri)

            if not parsed.scheme:
                uri = "http://" + uri

            try:

                response = requests.get(
                    uri,
                    timeout=timeout,
                )

                return (
                    response.status_code < 500,
                    response.reason,
                    response.status_code,
                )

            except Exception as exc:

                return False, str(exc), None

        # ====================================================
        # Application Insights
        # ====================================================

        def probe_appinsights():

            connection_string = (
                Config.APPLICATIONINSIGHTS_CONNECTION_STRING
            )

            if not connection_string:

                return (
                    False,
                    "no connection string",
                    None,
                )

            if (
                "InstrumentationKey="
                in connection_string
                or "ikey="
                in connection_string.lower()
            ):

                return (
                    True,
                    "connection string looks valid",
                    None,
                )

            return (
                False,
                "invalid format",
                None,
            )

        # ====================================================
        # Execute Probes
        # ====================================================

        probes = {

            "AZURE_OPENAI_ENDPOINT":
                probe_openai,

            "AZURE_SEARCH_ENDPOINT":
                probe_search,

            "AZURE_STORAGE_CONNECTION_STRING":
                probe_storage,

            "AZURE_CONTENT_SAFETY_ENDPOINT":
                probe_content_safety,

            "MLFLOW_TRACKING_URI":
                probe_mlflow,

            "APPLICATIONINSIGHTS_CONNECTION_STRING":
                probe_appinsights,
        }

        if parallel:

            with ThreadPoolExecutor(
                max_workers=min(6, len(probes))
            ) as executor:

                futures = {
                    executor.submit(
                        probe,
                        name,
                        function,
                    ): name
                    for name, function
                    in probes.items()
                }

                for future in as_completed(futures):
                    # Results are written by probe().
                    pass

        else:

            for name, function in probes.items():
                probe(name, function)

        return results