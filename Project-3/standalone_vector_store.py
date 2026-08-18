from src.config import Config
from langchain_openai import AzureOpenAIEmbeddings
from src.vector_store import get_vector_store

print("Creating Azure OpenAI embeddings...")

embeddings = AzureOpenAIEmbeddings(
    api_key=Config.AZURE_OPENAI_API_KEY,
    azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
    azure_deployment=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    api_version=Config.AZURE_OPENAI_API_VERSION,
)

print("Embeddings created successfully")

print("Creating LangChain AzureSearch vector store...")

vector_store = get_vector_store(embeddings)


print("AzureSearch vector store created successfully")
print("Index:", Config.AZURE_SEARCH_INDEX_NAME)

"""

print("Testing similarity search...")

docs = vector_store.similarity_search(
    "What are the baggage rules for international flights?",
    k=3,
)

print("Search completed successfully")
print("Documents returned:", len(docs))

for i, doc in enumerate(docs, 1):
    print(f"\n--- Document {i} ---")
    print("Content:", doc.page_content[:500])
    print("Metadata:", doc.metadata)

"""