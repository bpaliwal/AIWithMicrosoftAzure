from src.config import Config
from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(
    api_key=Config.AZURE_OPENAI_API_KEY,
    azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
    azure_deployment=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    api_version=Config.AZURE_OPENAI_API_VERSION,
)

print("AzureOpenAIEmbeddings object created")

result = embeddings.embed_query("What are the baggage rules for international flights?")

print("Embedding generated successfully")
print("Embedding dimensions:", len(result))
print("First 5 values:", result[:5])