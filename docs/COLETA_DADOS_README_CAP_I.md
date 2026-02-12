# 📘 SAIMP: Data Collection Module

> **Status**: Em Execução (Coleta de Dados Ativa) 🟢  
> **Versão**: 1.0 (MVP)

## 1. Visão Geral
Este documento serve como manual técnico e operacional para o módulo de ingestão de dados do **SAIMP**. O sistema foi projetado para capturar e armazenar dados de alta frequência do mercado de criptomoedas (Binance Futures - BTCUSDT) com foco em **baixa latência** e **eficiência de armazenamento**.

### Filosofia "GeoAI"
Tratamos o mercado não apenas como uma série temporal, mas como um terreno topográfico em evolução:
- **Order Book (Liq. Passiva)**: Paredes, suportes e resistências (Montanhas/Vales).
- **Execuções (Liq. Agressiva)**: Fluxo de ordens que consome a liquidez (Erosão).

## 2. Arquitetura de Dados

### Stack Tecnológica
- **Data Engine**: `Polars` (Rust-backed, High Performance).
- **Storage**: `Apache Parquet` (Compressão `zstd`, Colunar).
- **Async I/O**: `asyncio`, `aiohttp`, `websockets`.
- **Config**: `pydantic-settings`.

### Árvore de Diretórios de Dados
```bash
data/
└── raw/
    ├── historical/           # Dados Históricos (Backtesting)
    │   ├── klines_YYYY-MM.parquet
    │   └── aggTrades_YYYY-MM.parquet
    └── stream/               # Dados em Tempo Real (Live Trading/Training)
        ├── depth/YYYY-MM-DD/ # Snapshots do Order Book
        └── trades/YYYY-MM-DD/# Execuções em Tempo Real
```

## 3. Manual de Operação

Para manter a coleta ativa, dois processos distintos devem rodar em paralelo.

### 🔴 Terminal 1: O Gravador (Stream)
**Função**: Capturar o "agora". Conecta ao WebSocket da Binance e grava o Order Book e Trades.
* **Frequência de Flush**: A cada 15 minutos ou 50MB de buffer.
* **Resiliência**: Reconexão automática com backoff exponencial.

```powershell
# Executar no Terminal 1
python -m src.collectors.stream
```

### 🚜 Terminal 2: A Escavadeira (Historical)
**Função**: Capturar o "passado". Baixa dados históricos mensais da Binance Vision.
* **Performance**: Download paralelo (3 meses p/ vez) e processamento em memória (sem unzip em disco).
* **Dados**: `aggTrades` (Tick-by-tick) e `klines` (1m).

```powershell
# Executar no Terminal 2
python -m src.collectors.historical
```

## 4. Dicionário de Dados

### A. Histórico (`data/raw/historical/`)
Dados oficiais da Binance Vision, consolidados mensalmente.

| Tipo | Nome do Arquivo | Conteúdo Principal | Uso |
|:---|:---|:---|:---|
| **Klines** | `klines_YYYY-MM.parquet` | OHLCV (1m), Volume, Taker Buy Vol | Contexto Macro, Tendência |
| **Trades** | `aggTrades_YYYY-MM.parquet` | Preço, Qtd, Tempo, IsBuyerMaker | Análise de Fluxo, OFI, Delta |

### B. Streaming (`data/raw/stream/`)
Dados proprietários gravados em tempo real. Essenciais para treinar a IA a "ler a fita".

| Tipo | Conteúdo | Estrutura | Uso |
|:---|:---|:---|:---|
| **Depth** | Order Book (Top 20 levels) | **bids**: `[[price, qty], ...]`<br>**asks**: `[[price, qty], ...]` | Identificar Liquidez, Spoofing |
| **Trade** | Execuções em Tempo Real | Igual ao aggTrades histórico | Sincronizar erosão da liquidez |

## 5. Auditoria de Dados
Para garantir que o download foi completado sem buracos, execute o script de auditoria:

```powershell
python src/audit/check_completeness.py
```

Para verificar se o Stream está capturando dados corretamente:
```powershell
python src/audit/check_stream.py
```
*Saída esperada*: Detalhes do último arquivo `.parquet` gerado (Tamanho, Colunas, Amostra).

## 6. Protocolo de Recuperação (Disaster Recovery)

### O que fazer se o PC desligar ou a internet cair?
1. **Não entre em pânico**. O dado até o último flush (15m atrás) está salvo.
2. **Reinicie imediatamente** o script `stream.py`.
3. **Mapeie o Gap**: O período offline será um "buraco" nos dados de *depth*.
    - *Impacto*: A IA perderá o contexto de curto prazo.
    - *Solução futura*: O pipeline de treino ignorará janelas com gaps > 15min.

### Backfill (Preenchimento)
Se o `stream.py` ficar dias desligado, você pode baixar os dias perdidos usando o `historical.py` (quando a Binance disponibilizar os dados mensais/diários), mas perderá a granularidade fina do Order Book (Depth) desse período.

## 7. Roadmap & Próximos Passos 🚀

- [x] **Fase 1: Coleta de Dados** (Infraestrutura Pronta)
- [ ] **Fase 2: Processamento (ETL)** 
    - Unificar Stream + Histórico.
    - Calcular Features (OFI, VPIN, Microstructure noise).
    - Gerar Tensores (Imagens espectrais do Book).
- [ ] **Fase 3: Treinamento** (ViViT Model)
- [ ] **Fase 4: Produção** (Live Inference)
