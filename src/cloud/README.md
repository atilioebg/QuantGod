# QuantGod Cloud Infrastructure ☁️

Este diretório contém o pipeline modular de Processamento e Treinamento projetado para rodar em VMs de alta performance (RunPod/GCP/AWS).

## 🚀 Guia de Execução Passo a Passo

Siga esta ordem para reproduzir todo o ciclo de vida do modelo, desde os dados brutos até o modelo treinado.

### 1. Pré-processamento (ETL)
Extrai dados brutos do Google Drive (via rclone), reconstrói o Orderbook (200 níveis), limpa e calcula features essenciais.
- **Comando**:
  ```powershell
  python -m src.cloud.pre_processamento.orchestration.run_pipeline
  ```
- **O que faz**: Lê ZIPs do Drive montado -> Gera Parquets em `data/L2/pre_processed`.
- **Validação**: `pytest tests/test_cloud_etl_output.py`

### 2. Rotulagem (Labelling)
Aplica a lógica econômica de alvos (Buy, Sell, Neutral) nos dados processados usando thresholds assimétricos.
- **Comando**:
  ```powershell
  python src/cloud/labelling/run_labelling.py
  ```
- **O que faz**: Lê `data/L2/pre_processed` -> Salva Parquets rotulados em `data/L2/labelled`.
- **Validação**: `pytest tests/test_labelling_output.py`

### 3. Otimização de Hiperparâmetros (Optuna)
Utiliza o framework **Optuna** para encontrar a melhor arquitetura do Transformer, maximizando o F1-Score Ponderado.
- **Comando**:
  ```powershell
  python src/cloud/otimizacao/run_optuna.py
  ```
- **Output**: Salva os melhores parâmetros em `src/cloud/otimizacao/best_params.json` e o estudo em `optuna_study.db`.

#### 📊 Monitoramento em Tempo Real (Optuna Dashboard)
Você pode acompanhar a evolução da otimização, gráficos de importância de parâmetros e curvas de aprendizado via dashboard web.
1. Em um novo terminal, execute:
   ```powershell
   optuna-dashboard sqlite:///optuna_study.db
   ```
2. Abra o navegador em: `http://127.0.0.1:8080/`

### 4. Treinamento Final (Fine-Tuning)
Treina o modelo `QuantGodModel` final utilizando os melhores hiperparâmetros encontrados na etapa anterior.
- **Comando**:
  ```powershell
  python src/cloud/treino/run_training.py
  ```
- **Output**: Salva o modelo treinado em `data/models/quantgod_cloud_model.pth`.

---

## 📂 Logs e Monitoramento
Todo o processo gera logs detalhados para auditoria em `logs/`:
- `logs/etl/`: Progresso do processamento de arquivos.
- `logs/labelling/`: Distribuição de classes (Buy/Sell/Neutral) por arquivo.
- `logs/optimization/`: Métricas de cada trial (Loss, F1, Acurácia).
- `logs/training/`: Evolução de Loss e F1 por época.

---

## 🛠️ Requisitos
- Python 3.10+
- Dependências: `pip install -r requirements.txt`
- Rclone configurado e montado (G: ou Z:) para acesso aos dados brutos.
