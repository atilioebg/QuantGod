# 🏷️ SAIMP: Labeling & Dataset Module (CAPÍTULO III)

> **Status**: Concluído (Fase 3) ✅  
> **Versão**: 3.0 (Supervised Learning Engine)
> **Dependência**: Requer `ESTRUTURACAO_DADOS_README_CAP_II.md` (Tensors)

## 1. Visão Geral
Este documento detalha o "Professor" do **SAIMP**. O Capítulo II nos deu os "olhos" (Tensores 4D), mas olhos sem instrução não aprendem nada. Aqui, definimos o **Gabarito (Ground Truth)**.

O objetivo é ensinar a IA a responder não apenas "O preço vai subir?", mas sim "Vale a pena o risco?".

### Filosofia do Rotulagem (Labeling)
Em vez de prever o preço futuro fixo (Regressão), usamos o **Triple Barrier Method** (Método das Três Barreiras).
Isso simula a realidade de um trader:
1. Você abre uma posição.
2. Você define um Alvo de Lucro (Take Profit).
3. Você define um Limite de Perda (Stop Loss).
4. Você define um Tempo Limite (Time Horizon).

O primeiro evento que ocorrer define o Rótulo (Label).

## 2. Metodologia: O Método das Três Barreiras

| Barreira | Tipo | Evento | Label (Classe) | Significado para a IA |
|:---|:---|:---|:---|:---|
| **Superior** | Horizontal | Preço toca `High * (1 + Alvo)` | **2 (Long/Buy)** | "Compre aqui, o lucro é provável antes do stop." |
| **Inferior** | Horizontal | Preço toca `Low * (1 - Stop)` | **1 (Short/Stop)** | "Não compre (ou Venda). O risco de stop é alto." |
| **Vertical** | Tempo | Nenhuma barreira tocada em `N` horas | **0 (Neutral/Hold)** | "O mercado está de lado. Não vale o risco/taxas." |

### Parâmetros Padrão
* **Janela Temporal**: 24 Horas (Swing Trade).
* **Alvo (Profit)**: +3.5%
* **Stop (Loss)**: -1.5%
* **Relação Risco/Retorno**: > 2:1

## 3. Arquitetura de Treinamento

### A. O Gerador de Rótulos (`src/processing/labeling.py`)
Módulo de alta performance escrito em `Polars`.
* **Vetorização**: Não usamos loops Python lentos. Usamos `rolling_max` e `rolling_min` para olhar o futuro de milhões de candles em milissegundos.
* **Path Dependency**: O algoritmo verifica se o *Low* tocou o Stop antes do *High* tocar o Alvo na mesma janela. (Prioridade ao Risco: Se ambos tocam, assumimos Stop para ser conservador).

### B. O Dataset PyTorch (`src/training/dataset.py`)
Classe `SAIMPDataset` compatível com `torch.utils.data.DataLoader`.
* **Lazy Loading**: Não carrega 1TB de tensores na RAM. Lê apenas o necessário para o batch atual.
* **Sincronização**: Usa o `timestamp` para alinhar o Tensor 4D (Input) com o Label Calculado (Target).
* **On-the-Fly Generation**: Reconstrói a imagem do Order Book a partir dos dados compactados (Parquet) em tempo real durante o treino.

### C. Auditoria Visual dos Rótulos (`src/visualization/verify_labels.py`)
Antes de treinar a IA, precisamos garantir que o professor (Gabarito) não está ensinando errado.

O script `verify_labels.py` desenha a "Autópsia" de cada rótulo gerado:

```powershell
python src/visualization/verify_labels.py
```
Uma janela gráfica será aberta. Pressione `ENTER` no terminal para avançar frame a frame.

#### 🔍 O Que Você Está Vendo (Interpretação)
O gráfico mostra o futuro de 24 horas a partir do momento do snapshot.

1.  **Linha Amarela (Entrada)**: Preço exato no momento zero ($t=0$).
2.  **Linha Branca (A Realidade)**: Caminho percorrido pelo preço nas próximas 24h.
3.  **Linha Verde (O Sonho)**: Alvo de Lucro (+3.5%). Se a linha branca tocar aqui primeiro -> Rótulo **COMPRA**.
4.  **Linha Vermelha (O Pesadelo)**: Stop Loss (-1.5%). Se a linha branca tocar aqui primeiro -> Rótulo os **VENDA**.

#### ⚖️ O Veredito (Cor do Fundo)
A cor de fundo indica a classificação final do algoritmo para aquele momento:

*   **Fundo CINZA (Neutro)**: O preço "sambou" mas não atingiu nem o alvo verde nem o stop vermelho no tempo limite.
    *   *Lição para a IA*: "Não faça nada. Evite taxas em mercado lateral."
*   **Fundo VERDE (Compra)**: O preço atingiu o alvo antes de ser stopado.
    *   *Lição para a IA*: "Sinal forte de alta. Compre!"
*   **Fundo VERMELHO (Venda)**: O preço foi stopado antes de atingir o alvo.
    *   *Lição para a IA*: "Sinal de perigo. Venda ou fique fora."


## 4. Como Utilizar (Exemplo de Pipeline)

```python
import polars as pl
from src.processing.labeling import generate_labels
from src.training.dataset import SAIMPDataset

# 1. Carregar Velas (O Futuro)
klines = pl.read_parquet("data/raw/historical/klines_2024.parquet")

# 2. Gerar o Gabarito (Labels)
labels_df = generate_labels(
    klines, 
    window_hours=24, 
    target_pct=0.035, 
    stop_pct=0.015
)
# Saída: DataFrame [timestamp, label]

# 3. Listar Arquivos de Tensores (O Passado)
# (Assumindo que você rodou a simulação e salvou)
tensor_files = ["data/processed/simulation_2024.parquet"]

# 4. Criar o Dataset pronta para GPU
dataset = SAIMPDataset(tensor_files, labels_df)

# X = Tensor (4, 128, 128), Y = Label (0, 1, 2)
X, Y = dataset[0] 
```

## 5. Próximos Passos (Fase 4 - The Brain) 🧠

Agora temos:
1. **Inputs (X)**: Tensores 4D ricos em microestrutura.
2. **Targets (Y)**: Labels realistas baseados em risco de trading.

O Palco está montado para o **Deep Learning**:

- [ ] **ViViT (Video Vision Transformer)**: Implementar a rede neural que processa sequências de vídeo (nosso Tensor 4D ao longo do tempo).
- [ ] **Training Loop**: O script que fará a mágica acontecer (Backpropagation).
- [ ] **Validation Strategy**: Walk-forward validation para evitar overfitting.

---
> *SAIMP - Ensinando a máquina não a prever o futuro, mas a gerenciar o risco.*
