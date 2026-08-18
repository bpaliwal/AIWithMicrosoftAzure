from src.config import Config

print("DEBUG: checking environment")
print("AZURE_SEARCH_ENDPOINT:", Config.AZURE_SEARCH_ENDPOINT)
print("AZURE_SEARCH_KEY present:", bool(Config.AZURE_SEARCH_KEY))
print("AZURE_SEARCH_INDEX_NAME:", Config.AZURE_SEARCH_INDEX_NAME)
print("AZURE_SEARCH_API_VERSION:", Config.AZURE_SEARCH_API_VERSION if hasattr(Config, 'AZURE_SEARCH_API_VERSION') else 'not-set')

from src.vector_store import get_vector_store

# minimal fake embedding so the object can initialize without OpenAI calls
fake = lambda x: [0.1, 0.2, 0.3]
vs = get_vector_store(fake)
print("DEBUG: vector_store created:", type(vs).__name__)
