"""
Configuration management for the Autonomous Knowledge Graph.
Handles environment variables, Firebase credentials, and system-wide settings.
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum

from loguru import logger
from pydantic import BaseSettings, Field, validator


class Environment(str, Enum):
    """Environment types for the system."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class VectorStoreType(str, Enum):
    """Supported vector store implementations."""
    FAISS = "faiss"
    ANNOY = "annoy"
    IN_MEMORY = "in_memory"


class Settings(BaseSettings):
    """Main settings class using Pydantic for validation."""
    
    # Environment
    env: Environment = Environment.DEVELOPMENT
    debug: bool = True
    
    # Firebase Configuration (CRITICAL - as per constraints)
    firebase_project_id: str = Field(..., env="FIREBASE_PROJECT_ID")
    firebase_credentials_path: Optional[Path] = Field(None, env="FIREBASE_CREDENTIALS_PATH")
    
    # Service Account JSON (alternative to path)
    firebase_service_account_json: Optional[Dict[str, Any]] = Field(None, env="FIREBASE_SERVICE_ACCOUNT_JSON")
    
    # Vector Store Configuration
    vector_store_type: VectorStoreType = VectorStoreType.FAISS
    vector_dimension: int = 768  # Standard BERT dimension
    faiss_index_path: Path = Path("./data/vector_store/faiss_index")
    annoy_index_path: Path = Path("./data/vector_store/annoy_index")
    
    # NLP Configuration
    spacy_model: str = "en_core_web_lg"
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    
    # Performance Settings
    batch_size: int = 32
    cache_size: int = 10000
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Logging
    log_level: str = "INFO"
    log_rotation: str = "10 MB"
    log_retention: str = "30 days"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    @validator("firebase_credentials_path", pre=True)
    def validate_credentials_path(cls, v: Optional[str]) -> Optional[Path]:
        """Validate and convert credentials path."""
        if v is None:
            return None
        path = Path(v)
        if not path.exists():
            logger.warning(f"Firebase credentials path does not exist: {path}")
            # In production, this should fail. In development, we might use emulator.
            if os.getenv("USE_FIREBASE_EMULATOR", "").lower() != "true":
                raise ValueError(f"Firebase credentials file not found: {path}")
        return path
    
    @validator("firebase_service_account_json", pre=True)
    def parse_service_account_json(cls, v: Optional[str]) -> Optional[Dict[str, Any]]:
        """Parse service account JSON from environment variable."""
        if v is None or v == "":
            return None
        import json
        try:
            return json.loads(v)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in service account: {e}")


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings instance with singleton pattern."""
    global _settings
    
    if _settings is None:
        try:
            _settings = Settings()
            logger.info(f"Settings loaded for environment: {_settings.env}")
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            # Provide helpful error message
            if "FIREBASE_PROJECT_ID" in str(e):
                logger.error("""
                Firebase configuration missing. You need to:
                1. Create Firebase project at: https://console.firebase.google.com/
                2. Generate service account key from Project Settings > Service Accounts
                3. Set FIREBASE_PROJECT_ID and FIREBASE_CREDENTIALS_PATH environment variables
                
                OR use the Firebase CLI: firebase login && firebase init
                """)
            raise
    
    return _settings


def setup_logging() -> None:
    """Configure Loguru logging based on settings."""
    logger.remove()  # Remove default handler
    
    settings = get_settings()
    
    # Configure