from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    # ============================================================================
    # PATH CONFIGURATION (L2 Branch)
    # ============================================================================
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    
    # Processed Data (L2 Features)
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    DATA_PATH: str = "data/processed/l2_features_1min_final.parquet"
    
    # Live Data
    LIVE_DATA_DIR: Path = DATA_DIR / "live"
    LIVE_DATA_PATH: Path = LIVE_DATA_DIR / "l2_live_1min.parquet"
    
    # Raw Data (LakeAPI Downloads)
    RAW_L2_DIR: Path = DATA_DIR / "L2" / "raw"
    
    # Logs
    PREDICTION_HORIZON: int = 60
    LOGS_DIR: Path = BASE_DIR / "logs"
    
    # --- Live Configuration ---
    BYBIT_WS_URL: str = "wss://stream.bybit.com/v5/public/linear"
    
    # Environment
    ENV: str = "development"

    # --- L2 Hyperparameters (Updated by Optuna) ---
    SEQ_LEN: int = 720
    BATCH_SIZE: int = 32
    D_MODEL: int = 256
    NHEAD: int = 8
    LEARNING_RATE: float = 7.73220264611388e-05
    DROPOUT: float = 0.3102876208762215
    
    # --- Feature Configuration ---
    NUM_FEATURES: int = 9
    
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Ensure directories exist
settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
settings.LIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.RAW_L2_DIR.mkdir(parents=True, exist_ok=True)
settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)