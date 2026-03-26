from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    OLLAMA_BASE_URL: str
    OLLAMA_GENERATION_MODEL: str
    OLLAMA_EMBEDDING_MODEL: str
    QDRANT_URL: str
    QDRANT_API_KEY: str
    QDRANT_COLLECTION: str

    CHUNK_SIZE_TOKENS: int
    CHUNK_OVERLAP_TOKENS: int
    TOPIC_DEFAULT: str
    LANGUAGE_DEFAULT: str
    EMBEDDING_BATCH_SIZE: int = 64
    QDRANT_UPSERT_BATCH_SIZE: int = 128

    model_config = SettingsConfigDict(env_file=".env")



settings = Settings()
