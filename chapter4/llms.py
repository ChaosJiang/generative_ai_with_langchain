from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings

from config import set_environment

set_environment()

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", temperature=0, max_tokens=None, max_retries=2
)

store = LocalFileStore("./cache/")
underlying_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)
EMBEDDINGS = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=underlying_embeddings.model
)
