from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import List

class Settings(BaseSettings):
    # ============================================================================
    # PATH CONFIGURATION (Caminhos de Diretórios)
    # ============================================================================
    # Caminho base do projeto
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    
    # Diretório principal de dados
    DATA_DIR: Path = BASE_DIR / "data"
    
    # [CAMINHOS DINÂMICOS]
    # Local onde ficam os arquivos Parquet de trades e klines brutos (histórico)
    RAW_HISTORICAL_DIR: Path = DATA_DIR / "raw" / "historical"
    # Local temporário para dados de streaming em tempo real
    RAW_STREAM_DIR: Path = DATA_DIR / "raw" / "stream"
    # Local para dados processados (Tensores salvos, se necessário)
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    # Diretório onde os logs de execução serão salvos
    LOGS_DIR: Path = BASE_DIR / "logs"
    
    # Ambiente de execução (development/production)
    ENV: str = "development"

    # ============================================================================
    # DATA CONFIGURATION (Configuração de Dados)
    # ============================================================================
    # Lista de meses para usar no Treinamento (Exemplo representativo)
    TRAIN_MONTHS: List[str] = ["2020-01", "2024-10"]
    # Lista de meses para usar na Validação (Exemplo representativo)
    VAL_MONTHS: List[str] = ["2024-11", "2026-01"]
    
    # [INTERVALOS DE DATA]
    # Data de início exata para carregar o Dataset de Treino
    TRAIN_START_DATE: str = "2020-01-01"
    # Data de fim exata para o Dataset de Treino
    TRAIN_END_DATE: str = "2024-10-31"
    # Data de início exata para carregar o Dataset de Validação
    VAL_START_DATE: str = "2024-11-01"
    # Data de fim exata para o Dataset de Validação
    VAL_END_DATE: str = "2026-01-31"

    # ============================================================================
    # TRAINING HYPERPARAMETERS (Hiperparâmetros de Treinamento)
    # ============================================================================
    # [IMPORTANTE] Tamanho da sequência de entrada (Lookback).
    # O modelo olha para trás SEQ_LEN steps.
    # Cálculo: 96 steps * 15 minutos = 1440 minutos = 24 Horas de contexto (Eagle View).
    SEQ_LEN: int = 96
    
    # Tamanho do batch físico (Limitado pela VRAM da GPU)
    # Ajustado para 3 conforme feedback de consumo.
    BATCH_SIZE: int = 3
    
    # Passos de acumulação de gradiente para simular batch maior.
    # Batch Efetivo = 3 * 5 = 15 amostras por update de pesos.
    ACCUMULATION_STEPS: int = 5
    
    # Número de épocas de treinamento (passadas completas pelos dados)
    EPOCHS: int = 15
    
    # Taxa de aprendizado inicial do otimizador AdamW
    LEARNING_RATE: float = 1e-4
    
    # Regularização L2 para evitar overfitting
    WEIGHT_DECAY: float = 1e-5
    
    # Pesos das classes na Loss Function para tratar desbalanceamento.
    # [Neutro, Stop, Long, Super Long].
    # Penaliza erros nas classes de alta convicção.
    CLASS_WEIGHTS: List[float] = [1.0, 10.0, 10.0, 20.0]

    # ============================================================================
    # MODEL CONFIGURATION (Arquitetura do Modelo VIViT)
    # ============================================================================
    # Número de canais de entrada no Tensor (Ex: Bids, Asks, OFI, Price, OFI_W, Price_W)
    INPUT_CHANNELS: int = 6
    # Profundidade do Order Book (níveis de preço)
    PRICE_LEVELS: int = 128
    # Dimensão interna do modelo Transformer
    D_MODEL: int = 128
    # Número de classes de saída (0: Neutro, 1: Stop, 2: Long, 3: Super Long)
    NUM_CLASSES: int = 4

    # ============================================================================
    # LABELING CONFIGURATION (Rótulos Hierárquicos)
    # ============================================================================
    # Janela de tempo para calcular o retorno futuro (Horizonte de Previsão)
    LABEL_WINDOW_HOURS: int = 2
    # Retorno mínimo necessário para classificar como Long (Target)
    # 0.008 = 0.8%
    LABEL_TARGET_PCT: float = 0.008
    # Retorno máximo contrário aceitável (Stop Loss)
    # 0.0075 = 0.75%
    LABEL_STOP_PCT: float = 0.0075

    # ============================================================================
    # SIMULATION CONFIGURATION (Reconstrução do Order Book)
    # ============================================================================
    # Janela de tempo ou frequência de snapshots do Order Book.
    # "15m" = Snapshots a cada 15 minutos.
    SIM_WINDOW: str = "15m"

    # ============================================================================
    # EXCHANGE / STREAM CONFIG (Opcional - Futuro)
    # ============================================================================
    # Par de negociação
    SYMBOL: str = "BTCUSDT"
    # URL do WebSocket da Binance Futures
    WS_URL: str = "wss://fstream.binance.com/ws"
    # Intervalo para salvar dados do stream em disco (segundos)
    STREAM_FLUSH_INTERVAL_SECONDS: int = 60
    # Tamanho do buffer em memória para dados de streaming antes de flush
    STREAM_BUFFER_SIZE_MB: int = 50
    # Data de início para o coletor histórico
    HISTORICAL_START_DATE: str = "2017-08-01"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Instancia o objeto settings para uso global
settings = Settings()

# Garante que os diretórios existem ao importar este arquivo
settings.RAW_HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
settings.RAW_STREAM_DIR.mkdir(parents=True, exist_ok=True)
settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)