# 🧠 SAIMP: AI-Powered Crypto Trading

> **Status**: Produção (Fase 5 Concluída) ✅  
> **Versão**: 5.1 (Full Pipeline)

O **SAIMP** é um sistema de trading quantitativo que utiliza **Deep Learning (Vision Transformers)** para analisar a microestrutura do mercado de criptomoedas (Order Book e Fluxo de Ordens) e prever movimentos de curto prazo.

## 📚 Documentação Oficial (Manuais)

A inteligência do projeto está dividida em 5 Capítulos Técnicos. **Leia na ordem:**

| Cap | Módulo | Descrição | Arquivo |
|:---|:---|:---|:---|
| **I** | **Coleta** | Download de dados históricos e conexão WebSocket. | [COLETA_DADOS_README_CAP_I.md](COLETA_DADOS_README_CAP_I.md) |
| **II** | **Estruturação** | Engenharia de Features, Simulação de Book e Tensores 4D. | [ESTRUTURACAO_DADOS_README_CAP_II.md](ESTRUTURACAO_DADOS_README_CAP_II.md) |
| **III** | **Labeling** | Metodologia Triple Barrier para criação de alvos de risco. | [LABELLING_DADOS_README_CAP_III.md](LABELLING_DADOS_README_CAP_III.md) |
| **IV** | **Cérebro (IA)** | Arquitetura ViViT (Video Vision Transformer) e Treinamento. | [MODEL_DADOS_README_CAP_IV.md](MODEL_DADOS_README_CAP_IV.md) |
| **V** | **QA & Testes** | Protocolos de Teste, Auditoria e Validação. | [TESTS_README_CAP_V.md](TESTS_README_CAP_V.md) |

---

## 🚀 Guia Rápido de Execução

### 1. Coleta de Dados
Coloque os robôs para trabalhar:
```powershell
# Histórico (Passado)
python -m src.collectors.historical

# Tempo Real (Presente)
python -m src.collectors.stream
```

### 2. Auditoria (Health Check)
Verifique se os dados estão saudáveis:
```powershell
# Verificar integridade do histórico (2023-Hoje)
python src/audit/check_completeness.py

# Verificar se o stream está vivo
python src/audit/check_stream.py
```

### 3. Validação do Sistema (Smoke Test)
Teste se o pipeline inteiro (Dados -> IA) está funcionando:
```powershell
python tests/integration/test_integration.py
```
*Saída esperada: "CHECK-MATE!"*

### 4. Treinamento da IA
Treine o modelo com os dados auditados (Modo Local Lite otimizado para 2GB VRAM):
```powershell
python -m src.training.train
```

### 5. Testes Unitários
Para desenvolvedores:
```powershell
pytest tests/
```

---
> *SAIMP - Onde a Microestrutura encontra o Deep Learning.*
