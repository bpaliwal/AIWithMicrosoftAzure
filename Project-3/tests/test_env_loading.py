from pathlib import Path
import os
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

print("=" * 60)
print("ENVIRONMENT CONFIGURATION TEST")
print("=" * 60)

print("Project root :", PROJECT_ROOT)
print(".env path    :", ENV_FILE)
print(".env exists  :", ENV_FILE.exists())

load_dotenv(dotenv_path=ENV_FILE, override=True)

variables = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "AZURE_SEARCH_ENDPOINT",
    "AZURE_SEARCH_KEY",
    "AZURE_SEARCH_INDEX_NAME",
    "AZURE_CONTENT_SAFETY_ENDPOINT",
    "AZURE_CONTENT_SAFETY_KEY",
    "AZURE_STORAGE_CONNECTION_STRING",
    "AZURE_STORAGE_CONTAINER_NAME",
]

for name in variables:
    value = os.getenv(name)

    if value:
        if "KEY" in name or "CONNECTION_STRING" in name:
            display = f"LOADED ({len(value)} characters)"
        else:
            display = value
    else:
        display = "MISSING"

    print(f"{name:40} : {display}")

print("=" * 60)