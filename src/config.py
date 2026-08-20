"""
Configuration Management
RUBRIC: Environment Setup & Configuration (8 marks total)
- Azure OpenAI credentials configured correctly (1 mark)
- Azure AI Search credentials set up properly (1 mark)
- config.py implemented with validation (3 marks)
- All required packages installed and imported without errors (3 marks)

TASK: Load all configuration from environment variables
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse

# HINT: Load environment variables from .env file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

if not ENV_FILE.exists():
    raise FileNotFoundError(
        f".env file not found at expected location: {ENV_FILE}"
    )

load_dotenv(dotenv_path=ENV_FILE, override=True)

class Config:
    """Configuration for Wanderlust Travel Chatbot"""
    
    # ====================
    # Azure OpenAI Configuration
    # ====================
    # HINT: Load Azure OpenAI credentials from environment
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") 
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") 
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.4-mini")  
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small") 

    # ====================
    # Azure AI Search Configuration (Only vector store - no ChromaDB)
    # ====================
    # HINT: Load Azure AI Search credentials
    AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT") 
    AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY") 
    AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "travel-kb-index")  # HINT: "AZURE_SEARCH_INDEX_NAME", "travel-kb-index"
    AZURE_SEARCH_API_VERSION = os.getenv("AZURE_SEARCH_API_VERSION", "default")  # HINT: "AZURE_SEARCH_API_VERSION", "2021-04-30-Preview"
    
    # Vector store selection: 'chroma' for local, 'azure_search' for production
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE_PROD", "azure_search")
    
    # ====================
    # Azure Storage (Optional)
    # ====================
    AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")  
    AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "travel-documents")  # HINT: "AZURE_STORAGE_CONTAINER_NAME", "travel-documents"
    
    # ====================
    # Azure Content Safety (Optional)
    # ====================
    AZURE_CONTENT_SAFETY_ENDPOINT = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")  
    AZURE_CONTENT_SAFETY_KEY = os.getenv("AZURE_CONTENT_SAFETY_KEY") 
    
    # ====================
    # Azure Monitor (Optional)
    # ====================
    APPLICATIONINSIGHTS_CONNECTION_STRING = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING") 
    
    # ====================
    # MLflow Configuration
    # ====================
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")  # HINT: "MLFLOW_TRACKING_URI"
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "wanderlust-travel-chatbot")  # HINT: "MLFLOW_EXPERIMENT_NAME", "wanderlust-travel-chatbot"
    
    # ====================
    # Ingestion Settings
    # ====================
    # HINT: Convert to integer, 0 means no limit
    INGESTION_LIMIT = int(os.getenv("INGESTION_LIMIT", "0"))  # HINT: "INGESTION_LIMIT", "0"

    @staticmethod
    def validate():
        """Validate that required environment variables are set.

        Raises EnvironmentError if any required variable is missing.
        """
        required = {
            "AZURE_OPENAI_API_KEY": Config.AZURE_OPENAI_API_KEY,
            "AZURE_OPENAI_DEPLOYMENT_NAME": Config.AZURE_OPENAI_DEPLOYMENT_NAME,
            "AZURE_OPENAI_ENDPOINT": Config.AZURE_OPENAI_ENDPOINT,
            "AZURE_OPENAI_API_VERSION": Config.AZURE_OPENAI_API_VERSION,
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            "AZURE_SEARCH_ENDPOINT": Config.AZURE_SEARCH_ENDPOINT,
            "AZURE_SEARCH_KEY": Config.AZURE_SEARCH_KEY,
            "AZURE_SEARCH_INDEX_NAME": Config.AZURE_SEARCH_INDEX_NAME,
        }            

        missing = [name for name, val in required.items() if not val]
        if missing:
            raise EnvironmentError(f"Missing required config environment variables: {', '.join(missing)}")
        return True

    @staticmethod
    def validate_connections(timeout: int = 3, parallel: bool = True) -> dict:
        """Lightweight connectivity checks for configured endpoints.

        Returns a dict mapping config names to {ok: bool, msg: str, status_code: int|None}.
        This is opt-in and uses short timeouts to avoid startup delays.
        """
        results = {}

        def probe(name, func):
            try:
                ok, msg, code = func()
                results[name] = {"ok": ok, "msg": msg, "status_code": code}
            except Exception as e:
                results[name] = {"ok": False, "msg": str(e), "status_code": None}

        def probe_openai():
            endpoint = Config.AZURE_OPENAI_ENDPOINT
            key = Config.AZURE_OPENAI_API_KEY
            api_ver = Config.AZURE_OPENAI_API_VERSION
            if not endpoint:
                return False, "no endpoint configured", None
            url = endpoint.rstrip("/") + f"/openai/deployments?api-version={api_ver}"
            headers = {"api-key": key} if key else {}
            try:
                r = requests.get(url, headers=headers, timeout=timeout)
                return (r.status_code in (200, 401, 403), f"{r.reason}", r.status_code)
            except Exception as e:
                return False, str(e), None

        def probe_search():
            endpoint = Config.AZURE_SEARCH_ENDPOINT
            key = Config.AZURE_SEARCH_KEY
            if not endpoint:
                return False, "no endpoint configured", None
            url = endpoint.rstrip("/") + "/indexes?api-version=2021-04-30-Preview"
            headers = {"api-key": key} if key else {}
            try:
                r = requests.get(url, headers=headers, timeout=timeout)
                return (r.status_code in (200, 401, 403), f"{r.reason}", r.status_code)
            except Exception as e:
                return False, str(e), None

        def probe_storage():
            conn = Config.AZURE_STORAGE_CONNECTION_STRING
            if not conn:
                return False, "no connection string", None
            try:
                from azure.storage.blob import BlobServiceClient
                client = BlobServiceClient.from_connection_string(conn)
                # try to list containers with a very short timeout if supported
                containers = list(client.list_containers())
                return True, f"found {len(containers)} containers", 200
            except ImportError:
                return False, "azure-storage-blob not installed", None
            except Exception as e:
                return False, str(e), None

        def probe_content_safety():
            endpoint = Config.AZURE_CONTENT_SAFETY_ENDPOINT
            key = Config.AZURE_CONTENT_SAFETY_KEY
            if not endpoint:
                return False, "no endpoint configured", None
            headers = {"Ocp-Apim-Subscription-Key": key} if key else {}
            try:
                r = requests.get(endpoint, headers=headers, timeout=timeout)
                return (r.status_code in (200, 401, 403)), f"{r.reason}", r.status_code
            except Exception as e:
                return False, str(e), None

        def probe_mlflow():
            uri = Config.MLFLOW_TRACKING_URI
            if not uri:
                return False, "no mlflow uri configured", None
            parsed = urlparse(uri)
            if not parsed.scheme:
                uri2 = "http://" + uri
            else:
                uri2 = uri
            try:
                r = requests.get(uri2, timeout=timeout)
                return (r.status_code < 500), f"{r.reason}", r.status_code
            except Exception as e:
                return False, str(e), None

        def probe_appinsights():
            conn = Config.APPLICATIONINSIGHTS_CONNECTION_STRING
            if not conn:
                return False, "no connection string", None
            # Basic format validation only
            if "InstrumentationKey=" in conn or "ikey=" in conn.lower():
                return True, "connection string looks valid", None
            return False, "invalid format", None

        probes = {
            "AZURE_OPENAI_ENDPOINT": probe_openai,
            "AZURE_SEARCH_ENDPOINT": probe_search,
            "AZURE_STORAGE_CONNECTION_STRING": probe_storage,
            "AZURE_CONTENT_SAFETY_ENDPOINT": probe_content_safety,
            "MLFLOW_TRACKING_URI": probe_mlflow,
            "APPLICATIONINSIGHTS_CONNECTION_STRING": probe_appinsights,
        }

        if parallel:
            with ThreadPoolExecutor(max_workers=min(6, len(probes))) as ex:
                futures = {ex.submit(lambda k=k: probe(k, probes[k])): k for k in probes}
                # wait for completion
                for fut in as_completed(futures):
                    pass
        else:
            for name, fn in probes.items():
                probe(name, fn)

        return results