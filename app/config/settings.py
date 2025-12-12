"""Application Settings Configuration"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings"""

    # Application
    APP_NAME: str = "SpeedJobs Backend API"
    APP_VERSION: str = "2.0.0"
    APP_DESCRIPTION: str = "채용공고 평가, 분석 및 RAG 검색 API"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # Database
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "3306"))
    DB_USER: str = os.getenv("DB_USER", "admin")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "admin")
    DB_NAME: str = os.getenv("DB_NAME", "speedjobs")

    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Tavily (Web Search)
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # Qdrant (VectorDB)
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://speedjobs-vectordb.skala-practice.svc.cluster.local:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "speedjobs_vectors")
    
    # Embedding
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "1024"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))


settings = Settings()
