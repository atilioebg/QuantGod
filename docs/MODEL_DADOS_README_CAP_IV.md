# 🤖 SAIMP: The AI Brain Module (CAPÍTULO IV)

> **Status**: Concluído (Fase 4) ✅  
> **Versão**: 4.0 (ViViT: Video Vision Transformer)
> **Dependência**: Requer `LABELLING_DADOS_README_CAP_III.md` (Targets)

## 1. Visão Geral
Este documento detalha o "Córtex" do **SAIMP**. Aqui, os Tensores 4D (Input) encontram os Rótulos de Risco (Target) dentro de uma arquitetura neural híbrida de última geração.

O objetivo não é apenas processar dados, mas **entender a narrativa do mercado**. Para isso, combinamos duas superpotências da IA:
1.  **Visão Computacional (CNN)**: Para "ver" a estrutura do Order Book a cada instante.
2.  **Processamento Sequencial (Transformers)**: Para "lembrar" a evolução do fluxo ao longo do tempo.

---

## 2. A Arquitetura: SAIMPViViT

O modelo `SAIMPViViT` (`src/models/vivit.py`) é uma adaptação de arquiteturas de classificação de vídeo para o mercado financeiro.

### A. O Olho: Spatial Feature Extractor (CNN 1D)
Antes de entender o tempo, precisamos entender o espaço (Preço e Volume).
*   **Input**: Um único snapshot do tensor `(4 canais, 128 níveis)`.
*   **Camadas**:
    *   3 Blocos de Convolução 1D (`Conv1d -> BatchNorm -> ReLU -> MaxPool`).
    *   Reduz a dimensão de altura (128 níveis) para um vetor latente denso (`d_model=128`).
*   **Função**: Aprende padrões visuais como "Paredes de Compra", "Absorção", "Spread Vazio".

### B. A Memória: Temporal Transformer Encoder
Depois de extrair as características de cada frame, precisamos conectar os pontos.
*   **Input**: Uma sequência de vetores latentes `(Tempo=96, Features=128)`.
*   **Positional Encoding**: Adiciona informação de ordem temporal (quem veio antes de quem).
*   **Encoder Layer**: Mecanismo de **Self-Attention** que permite à rede relacionar um evento no início da janela (ex: agressão forte) com o resultado final (ex: rompimento).
*   **Output**: O estado oculto do último timestep, contendo o resumo de toda a sequência.

### C. A Decisão: Classification Head
*   **Camadas**: MLP (Linear -> ReLU -> Dropout -> Linear).
*   **Output**: Logits para 3 classes (`Neutral`, `Sell`, `Buy`).

---

## 3. O Processo de Treinamento (`train.py`)

O script `src/training/train.py` é a academia onde o modelo exercita seus neurônios.

### Pipeline de Dados (On-the-Fly ETL)
Os datasets de **Treino** e **Validação** são criados **dinamicamente (On-the-Fly)** durante o treinamento, e **não antes**.

Isso é feito através da classe `StreamingDataset` que implementamos. Aqui está o fluxo exato do que acontece no código:

#### 1. Definição Logica (Metadados)
No início do `train.py`, apenas definimos **quais dias** pertencem a cada conjunto. Nada é carregado na memória neste momento.
```python
# train.py
train_dataset = StreamingDataset(TRAIN_MONTHS)
train_dataset.set_date_range("2026-01-01", "2026-01-21") # Define o intervalo de data

val_dataset = StreamingDataset(VAL_MONTHS)
val_dataset.set_date_range("2026-01-22", "2026-01-31")   # Define o intervalo de data
```

#### 2. Geração Just-in-Time (Durante o Loop)
Quando o `DataLoader` pede dados (no loop `for batch_idx, (data, target) in enumerate(train_loader)`), o `StreamingDataset` entra em ação:

1.  **Carrega um Chunk (Dia)**: Lê o arquivo Parquet original (`aggTrades` e `klines`) apenas para o dia atual.
2.  **Processa em Memória**:
    *   Roda `build_simulated_book` (recria o Order Book).
    *   Roda `generate_labels` (cria os alvos Buy/Sell/Hold).
    *   Faz o Join dos dois.
3.  **Cria Tensores**: Converte os dados processados para tensores PyTorch (`build_tensor_4d`).
4.  **Entrega (Yield)**: Entrega as sequências uma a uma para o `DataLoader`.
5.  **Descarta**: Assim que o dia termina, ele **apaga** tudo da memória (`gc.collect`) e carrega o próximo dia.

#### Por que fizemos assim?
*   **Vantagem**: **Economia Extrema de RAM**. Você consegue treinar com terabytes de dados usando apenas ~2GB de RAM, pois só carrega um dia por vez.
*   **Desvantagem**: **Uso Intenso de CPU**. A CPU precisa processar (simular book, gerar labels) enquanto a GPU treina.

Se criássemos os datasets **antes** (salvando em disco como tensores prontos), o treino seria mais rápido (menos CPU), mas ocuparia muito espaço em disco e exigiria um pré-processamento longo. A abordagem atual prioriza a capacidade de rodar em hardware modesto.

### Hiperparâmetros (Configuração Padrão)
| Parâmetro | Valor | Descrição |
|:---|:---|:---|
| `seq_len` | 96 | Janela de observação (ex: 96 snapshots de 15m = 24h). |
| `input_channels` | 4 | Bids, Asks, OFI, Activity. |
| `price_levels` | 128 | Altura da imagem do book. |
| `d_model` | 128 | Tamanho do vetor latente (embedding). |
| `batch_size` | 32 | Amostras por passo de treino. |
| `learning_rate` | 1e-4 | Taxa de aprendizado (AdamW). |

### Função de Perda (Loss Function)
Usamos **CrossEntropyLoss** com **Class Weights**.
*   **Problema**: O mercado fica "Neutro" (Class 0) na maior parte do tempo.
*   **Solução**: Penalizamos mais o erro nas classes raras (Compra/Venda).
    *   Peso Neutro: 1.0
    *   Peso Buy/Sell: 2.0

---

## 4. Validação e Métricas

Como saber se a IA não está apenas "decorando" o passado?

### Split Temporal (Walk-Forward)
Jamais misturamos o futuro com o passado.
*   **Treino**: Jan-Set (Dados Antigos).
*   **Validação**: Out-Dez (Dados Recentes).

### Métricas Chave
1.  **Loss (Perda)**: Deve diminuir consistentemente no treino e validação. Se subir na validação, é *Overfitting*.
2.  **Acurácia**: % de acertos totais. (Cuidado: num mercado lateral, chutar "Neutro" dá alta acurácia mas lucro zero).
3.  **Precision/Recall (Futuro)**: Focaremos em precisão de entradas (evitar falsos positivos).

---

## 5. Como Treinar

```powershell
# 1. Certifique-se de ter dados históricos em data/raw/historical
# 2. Execute o script de treino
python -m src.training.train
```

**O que acontece:**
1.  O script verifica se há GPU (`cuda`) disponível.
2.  Carrega os dados e inicia o loop de épocas.
3.  Imprime `Loss` e `Acc` a cada época.
4.  Salva o melhor modelo em `data/SAIMP_v1.pth`.

---

## 6. Próximos Passos (Fase 5 - Produção) 🚀

O cérebro está criado. Agora precisamos colocá-lo no corpo do robô.

- [ ] **Inference Engine**: Script que carrega o `.pth` e roda previsões em tempo real conectado ao `stream.py`.
- [ ] **Risk Manager**: Módulo que decide o tamanho da posição baseado na confiança da IA.
- [ ] **Execution Algo**: O "dedo no gatilho" que envia ordens via API da Binance.

---
> *SAIMP - Inteligência Artificial aplicada à Microestrutura de Mercado.*
