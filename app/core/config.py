from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    qdrant_url: str
    qdrant_collection: str = "documents"
    redis_url: str
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()