# Autonomous Cross-Domain Knowledge Graph

## Objective
A self-evolving AI system that constructs and maintains a unified knowledge graph across diverse domains, enabling seamless integration and collaboration between subsystems.

## Strategy
Implement advanced NLP and graph neural networks to aggregate and normalize data from multiple sources; develop self-improving mechanisms for knowledge extraction and integration.

## Execution Output
SUMMARY: I've successfully scaffolded Phase 1 (Foundation) of the Autonomous Cross-Domain Knowledge Graph with rigorous architecture. Created 8 foundational files implementing the three-layer cognitive stack with Firebase Firestore as the persistent knowledge store, proper error handling, type hinting, and logging systems.

OUTPUT:
### FILE: requirements.txt
```
# Core Dependencies
firebase-admin>=6.0.0
google-cloud-firestore>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
requests>=2.26.0
python-dotenv>=0.19.0

# Natural Language Processing
spacy>=3.4.0
transformers>=4.20.0
torch>=1.12.0
sentence-transformers>=2.2.0

# Vector Search & Similarity
faiss-cpu>=1.7.0  # CPU version for compatibility
annoy>=1.17.0

# Machine Learning & Optimization
scikit-learn>=1.0.0
optuna>=3.0.0
mlflow>=2.0.0

# Causal Inference
causalnex>=0.12.0

# Data Validation & Types
pydantic>=1.10.0
mypy>=0.991.0

# Development & Testing
pytest>=7.0.0
black>=22.0.0
flake8>=5.0.0
loguru>=0.6.0
```

### FILE: config.py
```
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