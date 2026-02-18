# QuantGod Cloud Infrastructure ☁️

Este diretório contém a infraestrutura necessária para rodar o processamento de dados L2 (Orderbook) em instâncias de CPU na nuvem (ex: RunPod).

## 🚀 Guia Rápido

### 1. Preparação do Ambiente
O script `setup_cloud.sh` automatiza a instalação de dependências do sistema, criação de ambiente virtual e diretórios necessários.

```bash
cd cloud
chmod +x setup_cloud.sh
./setup_cloud.sh
```

**O que ele faz:**
- Instala `rclone`, `pip` e `venv`.
- Cria o ambiente virtual `.venv`.
- Instala as dependências de `requirements.txt`.
- Cria as pastas `data/L2/pre_processed`, `data/L2/labelled`, `data/artifacts` e `logs` na raiz do projeto.

---

### 2. Configuração (`configs/cloud_config.yaml`)
Toda a lógica do pipeline é controlada por este arquivo.

#### Parâmetros de Caminho (`paths`)
- `rclone_mount`: Onde o Google Drive está montado via rclone. Padrão: `/workspace/gdrive/My Drive/...`.
- `processed_output`: Onde os arquivos `.parquet` finais serão salvos (Ex: `data/L2/pre_processed`).
- `scaler_path`: Local para salvar/carregar o `scaler.pkl` (Ex: `data/artifacts/scaler.pkl`).

#### Parâmetros de ETL (`etl`)
- `sampling_interval_ms`: Frequência de amostragem dos ticks (Ex: `1000` para 1 segundo).
- `resampling_interval`: Janela de agregação OHLCV (Ex: `1min`).
- `orderbook_levels`: **Hard Cut**. Define quantos níveis de Bid/Ask serão mantidos (Ex: `200`).
- `compression`: Formato de compressão do Parquet (Recomendado: `snappy`).

#### Funcionalidades (`features`)
- `apply_zscore`: Se `true`, o pipeline aplicará normalização Z-Score e persistirá o scaler.

---

### 3. Execução do Pipeline
Após configurar o `.yaml` e rodar o `.sh`:

```bash
source .venv/bin/activate
python -m cloud.orchestration.run_pipeline
```

## 🛠️ Detalhes Técnicos

### Streaming de Dados
O pipeline foi desenhado para **Zero-Copy Disk Usage**. Ele lê os arquivos JSON/CSV diretamente do buffer de memória do ZIP montado pelo rclone, evitando escritas desnecessárias no SSD do RunPod e economizando RAM.

### Validação Automática
Ao final de cada processamento, o módulo `validate.py` verifica automaticamente:
- Presença de NaNs ou Infinitos.
- Integridade da ordem cronológica.
- Gaps temporais anormais.

---

## 📋 Requisitos
Certifique-se de configurar o `rclone` (`rclone config`) antes de iniciar o processo para que o mount esteja acessível.
