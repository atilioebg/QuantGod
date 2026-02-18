# QuantGod Cloud Infrastructure ☁️

Este diretório contém o pipeline modular de Processamento e Treinamento projetado para rodar em VMs de alta performance (RunPod/GCP/AWS).

## 🚀 Módulos do Pipeline

O pipeline é dividido em 4 etapas independentes, cada uma com sua própria configuração YAML:

### 1. Pré-processamento (`cloud/pre_processamento`)
Extrai dados brutos do Google Drive (via rclone), reconstrói o Orderbook (200 níveis), limpa e calcula features essenciais.
- **Execução**: `python -m src.cloud.pre_processamento.orchestration.run_pipeline`
- **Otimização**: Suporte a processamento paralelo multi-core.

### 2. Rotulagem (`cloud/labelling`)
Aplica a lógica econômica de alvos (Buy, Sell, Neutral) nos dados processados.
- **Execução**: `python src/cloud/labelling/run_labelling.py`
- **Ajuste**: Thresholds configuráveis via `labelling_config.yaml`.

### 3. Otimização (`cloud/otimizacao`)
Utiliza **Optuna** para encontrar os melhores hiperparâmetros do Transformer.
- **Execução**: `python src/cloud/otimizacao/run_optuna.py`
- **Output**: Salva `best_params.json` para uso no treino final.

### 4. Treinamento (`cloud/treino`)
Treino final do modelo `QuantGodModel` usando os melhores parâmetros.
- **Execução**: `python src/cloud/treino/run_training.py`
- **Output**: Modelo treinado `.pt`.

---

## 🧪 Validação e Testes

Para garantir que a migração não corrompa os dados, use:
```powershell
pytest tests/test_cloud_etl_output.py
```

Para inspeção visual das labels:
```powershell
python tests/visualize_labels.py
```

---

## 🛠️ Requisitos
- Python 3.10+
- Polars, PyTorch, Optuna, PyYAML, Pandas, tqdm
- Rclone configurado para o Google Drive
