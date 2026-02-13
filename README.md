# 🧠 BTCR: Deep Market Intelligence & Sniper Decision Engine

> **Versão**: 5.4 (The Monolith Edition - Execution Diary)  
> **Status**: 🟢 Operacional (Coleta em Tempo Real & Auditoria)  
> **Filosofia**: "O Mercado não é uma linha 2D, é uma topografia 4D moldada pelo fluxo de ordens."

O **BTCR (Bitcoin Transformer Decision Engine)** é um sistema de inteligência artificial de alta performance projetado para decodificar a microestrutura do mercado e identificar **rompimentos estruturais** com alta precisão. Utilizando uma arquitetura de visão computacional adaptada (**ViViT Transformers**), o sistema trata o histórico de preços e o fluxo de ordens como quadros de um vídeo, identificando padrões complexos de agressão e exaustão.

---

## 📋 Índice Mestre

1.  [Introdução e Capacidades do Modelo](#-introdução-e-capacidades-do-modelo)
2.  [Arquitetura do Sistema](#2-arquitetura-do-sistema)
3.  [Capítulo I: Coleta de Dados (Data Ingestion)](#3-capítulo-i-coleta-de-dados)
4.  [Capítulo II: Estruturação e Simulação (Refinaria)](#4-capítulo-ii-estruturação-e-simulação)
5.  [Capítulo III: Labeling e Metodologia (Triple Barrier)](#5-capítulo-iii-labeling-e-metodologia)
6.  [Capítulo IV: O Cérebro (SAIMPViViT)](#6-capítulo-iv-o-cérebro-saimpvivit)
7.  [Capítulo V: Testes e Qualidade (QA)](#7-capítulo-v-testes-e-qualidade)
8.  [Capítulo VI: Cockpit Operacional (Live Trading)](#8-capítulo-vi-cockpit-operacional-live-trading)
    - [8.1 Diário de Execução (Sniper Shots)](#81-diário-de-execução-sniper-shots)
9.  [Guia de Instalação e Configuração](#9-guia-de-instalação-e-configuração)
10. [Manual de Operação](#10-manual-de-opereração)
11. [Estrutura do Projeto](#11-estrutura-do-projeto)
12. [Roadmap e Próximos Passos](#12-roadmap-e-próximos-passos)

---

## 📖 Introdução e Capacidades do Modelo

### 🎯 O que este Modelo Faz?
Ele opera como um **Analista de Microestrutura Autônomo**. O sistema captura cada trade individual através do WebSocket da Binance e reconstrói o **Order Flow Index (OFI)** em tempo real. Esses dados são convertidos em tensores espaciais-temporais que o "Brain" (SniperBrain) processa para identificar o momento exato em que a agressão (Takers) supera a liquidez passiva (Makers).

### ✅ O que ele Preve?
*   **Direcionalidade de Alta Convicção**: Identifica se a probabilidade estatística favorece uma **Compra (Long)** ou **Venda (Short)**.
*   **Rompimentos Reais vs. Falsos**: Cruza a predição da Rede Neural com o OFI bruto para validar se um movimento de preço tem "combustível" real ou se é apenas uma armadilha de liquidez (*Spoofing*).
*   **Janela de Alvo**: O modelo é treinado sob a metodologia *Triple Barrier*, buscando prever se o mercado atingirá um alvo de **1.5% (Gain)** antes de recuar **0.75% (Stop)**.

### 💹 Performance e Acurácia (Metrics)
O modelo foi treinado em um dataset histórico de alta densidade (2023-2026). Os resultados dos últimos benchmarks são:
*   **Acurácia de Treino (Train Acc)**: ~62.75%
*   **Melhor Acurácia de Validação (Best Val Acc)**: **59.85%**
*   **Precisão Sniper**: Devido ao filtro de probabilidade (só registrar sinais > 50-60%), a precisão em sinais de execução real tende a ser superior à acurácia base.

### 📈 Sinais Consecutivos e Direcionalidade
O modelo possui capacidade de identificar **Momentum e Tendências Estendidas**:
*   **Altas/Quedas Consecutivas**: Caso a força do fluxo (OFI) e a estrutura de liquidez permaneçam favoráveis, o modelo emitirá **Previsões Consecutivas** da mesma classe. Isso indica uma tendência de forte convicção onde múltiplos "frames" de mercado apontam para o mesmo alvo de 1.5%.
*   **Persistent Outlook**: Diferente de modelos que mudam de opinião a cada candle, o BTCR tende a manter a "cor" do sinal enquanto o embasamento de fluxo de ordens (Delta de Agressão) não for exaurido.

### 🔭 Horizonte de Visão: Até onde ele enxerga?
*   **Memória Contextual (Lookback)**: O modelo analisa as últimas **8 horas** de dados (`SEQ_LEN=32`) para entender a construção da tendência atual e a memória dos níveis de suporte e resistência.
*   **Janela de Previsão (Lookahead)**: Sua inferência é focada em um horizonte de **4 horas** (`LABEL_WINDOW_HOURS`). É o tempo estimado para que a tese de rompimento ou defesa se concreteize.

### ❌ O que ele NÃO Preve?
*   **Fundamentos e Notícias**: O modelo é puramente técnico/quantitativo. Ele não enxerga notícias externas, tweets ou decisões de bancos centrais.
*   **Scalping de Segundos**: Não é um robô de arbitragem ou de frequência ultrarrápida. Ele busca movimentos sólidos com alvo de ~1.5%.
*   **Cisnes Negros**: Eventos globais extremos que geram volatilidade irracional e instantânea podem invalidar a análise estrutural clássica.

---

## 2. Arquitetura do Sistema

O pipeline foi desenhado para processar Terabytes de dados históricos com latência mínima, utilizando **Rust (Polars)** para ETL e **CUDA (PyTorch)** para Deep Learning.

```mermaid
graph TD
    subgraph "Nível 1: Ingestão (Data Lake)"
        A["Binance Vision API"] -->|Historical Downloader| B("Raw Parquet: aggTrades")
        A -->|Historical Downloader| C("Raw Parquet: Klines")
        W["WebSocket Stream"] -->|Live Recorder| D("Raw Stream: Trades/Depth")
    end

    subgraph "Nível 2: Refinaria (ETL On-the-Fly)"
        B -->|Simulation Engine| E["Order Book Reconstructor"]
        C -->|Labeling Engine| F["Triple Barrier Method"]
        E -->|Feature Engineering| G["OFI calculation"]
        G -->|Tensor Builder| H["4D Tensor (B, T, C, H)"]
    end

    subgraph "Nível 3: Inteligência Artificial (Brain)"
        H -->|Spatial Features| I["CNN 1D Encoder"]
        I -->|Temporal Features| J["Transformer Encoder"]
        J -->|Decision Head| K["Probabilidades (Softmax)"]
    end
```

---

## 3. Capítulo I: Coleta de Dados

> **Referência**: `docs/COLETA_DADOS_README_CAP_I.md`

Este módulo é responsável pela ingestão bruta de dados com foco em **baixa latência** e **eficiência de armazenamento**.

### Stack Tecnológica
- **Data Engine**: `Polars` (Rust-backed, High Performance).
- **Storage**: `Apache Parquet` (Compressão `zstd`, Colunar).
- **Async I/O**: `asyncio`, `aiohttp`, `websockets`.

### Dicionário de Dados

#### A. Histórico (`data/raw/historical/`)
Dados oficiais da Binance Vision, consolidados mensalmente.

| Tipo | Nome do Arquivo | Conteúdo Principal | Uso |
|:---|:---|:---|:---|
| **Klines** | `klines_YYYY-MM.parquet` | OHLCV (1m), Volume, Taker Buy Vol | Contexto Macro, Tendência |
| **Trades** | `aggTrades_YYYY-MM.parquet` | Preço, Qtd, Tempo, IsBuyerMaker | Análise de Fluxo, OFI, Delta |

#### B. Streaming (`data/raw/stream/`)
Dados proprietários gravados em tempo real (Live Trading).

| Tipo | Conteúdo | Estrutura | Uso |
|:---|:---|:---|:---|
| **Depth** | Order Book (Top 20 levels) | **bids**: `[[price, qty], ...]`<br>**asks**: `[[price, qty], ...]` | Identificar Liquidez, Spoofing |
| **Trade** | Execuções em Tempo Real | Formato Binance Futures: `p` (price), `q` (qty), `T` (time), `m` (isMaker) | Sincronizar erosão da liquidez |

#### C. Otimizações Binance Futures
O coletor `src/collectors/stream.py` foi atualizado para suportar:
- **Multiplexed Streams**: Captura simultânea de trades e profundidade.
- **Normalização Automática**: Conversão de chaves curtas do WebSocket para o padrão do Data Lake.
- **Fragmentação Inteligente**: Gravação de arquivos compactos (chunks) para evitar perda de dados por queda de rede.

### Protocolo de Recuperação (Disaster Recovery)
1.  **PC Desligou?**: Reinicie o script `stream.py` imediatamente.
2.  **Gap**: O período offline será um "buraco" nos dados. O pipeline de treino ignorará janelas com gaps > 15min.
3.  **Backfill**: Use o `historical.py` para baixar dias perdidos quando disponíveis na Binance.

---

## 4. Capítulo II: Estruturação e Simulação

> **Referência**: `docs/ESTRUTURACAO_DADOS_README_CAP_II.md`

Este módulo transforma logs financeiros em **Tensores Quadridimensionais (4D)**.

### A. Simulação de Order Book (Volume Profile Reconstructor)
Como não temos o Order Book completo de anos passados, utilizamos o conceito de **Restauração de Perfil de Volume**:
*   **Trade = Erosão**: Cada trade agressivo "cavou" um buraco na liquidez.
*   **Inversão Lógica**: Se houve um *Market Buy* de 10 BTC, sabemos que existia um *Limit Sell* (Ask) naquele preço.
*   **Resultado**: Recriamos a silhueta das montanhas de liquidez apenas olhando para onde a água (trades) bateu.

### B. Engenharia de Features
Calculamos em `src/processing/features.py`:
1.  **OFI (Order Flow Imbalance)**: $OFI_t = Vol_{Buy} - Vol_{Sell}$. O "vento" que empurra o preço.
2.  **Volatilidade Local**: Log Returns Std. Usada para normalizar os inputs (regime switching).

### C. Tensores 4D: A Visão da IA
A IA recebe um Tensor `(Batch, Time, Channels, Height)`.

| Canal (Index) | Nome | O que representa? | Significado Visual |
|:---|:---|:---|:---|
| **0** | **Bids (Liquidez Compra)** | Onde os Vendedores bateram. | **Suporte**. Montanhas verdes. |
| **1** | **Asks (Liquidez Venda)** | Onde os Compradores bateram. | **Resistência**. Montanhas vermelhas. |
| **2** | **OFI (Fluxo Líquido)** | Saldo $Buy - Sell$ no nível. | **Direção**. Intensidade do rompimento. |
| **3** | **Activity (Calor)** | Contagem de Trades / Volatilidade. | **Mapas de Calor**. Onde a batalha ocorre. |

### D. Normalização de Tensores (Crítico)
Para garantir a convergência da rede neural (que odeia números grandes), aplicamos em `src/processing/tensor_builder.py`:
1.  **Canais de Volume (0, 1, 3)**: Aplicação de `Log1p` seguida de divisão por escalar global (`/ 10.0`).
2.  **Canal de OFI (2)**: Aplicação de Tangente Hiperbólica (`tanh`) para comprimir o fluxo entre `[-1, 1]`.
3.  **Clipping Global**: Garantia de que nenhum valor exceda o intervalo `[-1.0, 1.0]`.

---

## 5. Capítulo III: Labeling e Metodologia

> **Referência**: `docs/LABELLING_DADOS_README_CAP_III.md`

Ensinamos a IA a responder não apenas "O preço vai subir?", mas "Vale a pena o risco?".

### O Método das Três Barreiras (Triple Barrier)
Simula a realidade de um trader com **Stop Loss** e **Take Profit**.

| Barreira | Tipo | Evento | Label (Classe) | Significado |
|:---|:---|:---|:---|:---|
| **Superior** | Horizontal | Preço toca `High * (1 + Alvo)` | **2 (Long/Buy)** | "Lucro provável antes do stop." |
| **Inferior** | Horizontal | Preço toca `Low * (1 - Stop)` | **1 (Short/Stop)** | "Risco de stop é alto. Venda." |
| **Vertical** | Tempo | Nenhuma barreira tocada em N horas | **0 (Neutral/Hold)** | "Mercado lateral. Evite taxas." |

### Parâmetros Atuais (v5.3 em `src/config.py`)
*   **Janela Temporal (`LABEL_WINDOW_HOURS`)**: **4 Horas**. (Busca movimentos de curto prazo).
*   **Alvo de Lucro (`LABEL_TARGET_PCT`)**: **1.5%** (0.015).
*   **Stop Loss (`LABEL_STOP_PCT`)**: **0.75%** (0.0075).
*   **Relação Risco/Retorno**: 2:1.

---

## 6. Capítulo IV: O Cérebro (SAIMPViViT)

> **Referência**: `docs/MODEL_DADOS_README_CAP_IV.md`

O modelo `SAIMPViViT` combina Visão Computacional e Processamento Sequencial.

### Arquitetura Híbrida
1.  **Spatial Feature Extractor (CNN 1D)**:
    *   **Função**: "Olhos". Analisa cada snapshot individualmente.
    *   **Mecanismo**: Convoluções 1D varrem os **128 níveis** de preço.
    *   **Output**: Vetor latente (`d_model=128`) para cada instante.
2.  **Temporal Transformer Encoder**:
    *   **Função**: "Memória". Conecta os pontos no tempo.
    *   **Contexto**: **32 Snapshots** (8 Horas) de histórico.
    *   **Mecanismo**: Self-Attention (`MultiHeadAttention`).
    *   **Output**: Probabilidades (Softmax) para as 3 classes.

### Pipeline de Treinamento (On-the-Fly)
Para economizar RAM (Treinar TBs em 32GB RAM), usamos **Lazy Loading**:
1.  **Carrega um dia** do disco.
2.  **Processa em Memória** (Simulação + Labeling).
3.  **Treina** a GPU.
4.  **Descarta** e carrega o próximo dia.

---

## 7. Capítulo V: Testes e Qualidade

> **Referência**: `docs/TESTS_README_CAP_V.md`

Adotamos a pirâmide de testes expandida:

### A. Testes Unitários (`pytest tests/`)
*   **Tensor Builder**: Valida se a normalização `tanh` está mantendo o OFI entre -1 e 1.
*   **Features**: Garante que a matemática do OFI está correta.
*   **Labeling**: Testa se o sistema prioriza o **Stop Loss** sobre o Take Profit (Conservadorismo).

### B. Teste de Integração (Smoke Test)
`python tests/test_integration.py` (ou `test_simulation.py` para simulação)
*   **Objetivo**: Rodar o pipeline do início ao fim com dados reais para garantir que nada crashe ("CHECK-MATE").

### C. Auditoria de Dados
*   **Histórico**: `python src/audit/check_completeness.py` (Busca buracos/arquivos vazios).
*   **Stream**: `python src/audit/check_stream.py` (Verifica se o sistema está vivo).
*   **Visual**: `python src/visualization/verify_labels.py` (Autópsia visual dos trades).

### D. Backtesting de Alta Fidelidade (Sniper Mode)
O script `src/evaluation/backtest_stream.py` realiza a validação definitiva do modelo usando dados **offline**.
- **Fonte de Dados**: O script busca dados já persistidos em disco (`.parquet`). Ele **não** lê o buffer de memória de um processo de stream ativo.
- **Hierarquia de Busca**:
    1. **Arquivos Históricos**: Procura o `.parquet` consolidado de 1 mês (ex: Binance Vision).
    2. **Arquivos de Stream (Disk)**: Se não houver histórico consolidado (mês atual), ele varre a pasta `data/raw/stream/trades/` e une todos os arquivos gravados por sessões anteriores do coletor.
- **Unificação de Chunks**: Combina automaticamente os fragmentos de trades salvos entre "ontem e hoje" em um único bloco contínuo, permitindo que a IA tenha contexto suficiente (SEQ_LEN) para prever.

---


---

## 8. Capítulo VI: Cockpit Operacional (Live Trading)

> **Referência**: `src/dashboard/app.py`

O **SAIMP Sniper Cockpit** é a interface visual de comando para a fase de produção. Ele transforma os dados brutos de microestrutura e as previsões da IA em uma ferramenta de decisão para o trader humano (**HFT-Human Hybrid**).

### 🎯 O Que é Este Painel? (Manual do Piloto)
Imagine que este painel é o **painel de instrumentos de um caça**. Você não precisa saber como o motor a jato (a Rede Neural) funciona por dentro; você só precisa saber ler os mostradores para não cair e para acertar o alvo. 

O robô (Navegador) analisa milhares de transações por segundo e calcula as probabilidades. **Você (Piloto)** aperta o botão de execução na corretora.

### 🩺 Anatomia do Sniper Cockpit

#### A. Status do Sistema (Health Check)
Exibe a saúde do robô e o estado do buffer de memória.
- **Warm-up**: O robô baixa automaticamente o histórico recente via API REST para preencher a memória (8 horas de contexto).
- **Regra de Ouro**: Se aparecer "Aguardando dados...", **NÃO OPERE**. O cérebro ainda está "acordando".

#### B. Triângulo de Decisão (O Veredito)
Três indicadores que devem ser lidos em convergência:

1.  **SINAL (A Direção)**:
    - ⚪ **NEUTRO (Hold)**: A IA não vê oportunidade clara ou tem certeza que o mercado está perigoso. Ficar fora também é uma posição.
    - 🟢 **COMPRA (Long)**: Padrão matemático de alta probabilidade de subida nas próximas 4 horas.
    - 🔴 **VENDA (Short)**: Padrão matemático de alta probabilidade de queda iminente.

2.  **CONFIANÇA (O Velocímetro)**: Nível de convicção da IA.
    - **33% a 45% (Dúvida)**: O robô está "chutando". Ignore o sinal.
    - **45% a 60% (Moderado)**: Padrão interessante. Operar com gerenciamento conservador (mão leve).
    - **Acima de 60% (Sniper Mode)**: Convicção extrema. Oportunidade de alta probabilidade.

3.  **RAIO-X / OFI (O Detector de Mentiras)**:
    - **OFI Positivo (+)**: Dinheiro real entrando (Compradores agredindo).
    - **OFI Negativo (-)**: Dinheiro real saindo (Vendedores agredindo).
    - **Divergência**: Se o preço sobe, mas o OFI cai, é uma **armadilha**. O preço está subindo "vazio" (sem volume real). O OFI te salva dessas furadas.

#### C. Telemetria & Validação (Auditoria de Performance)
O sistema possui um "Gravador de Caixa Preta" (`src/live/predictor.py`) que registra cada decisão tomada pela IA para auditoria posterior.

1.  **Onde Fica?**: `data/prediction_log.csv`
2.  **O que Grava?**:
    *   `timestamp`: Hora exata da decisão.
    *   `price`: Preço de execução.
    *   `signal`: Direção (COMPRA/VENDA/NEUTRO).
    *   `confidence`: Probabilidade bruta (0.0 a 1.0).
    *   `ofi`: Valor do fluxo no momento do sinal.
    *   `verdict`: Texto completo da análise de barreiras.
    *   `result`: Status do trade (preenchido post-factum).

> **Visualização**: No rodapé do Cockpit, a seção **"🚦 Auditoria de Performance"** exibe esses logs em tempo real, colorindo o OFI (Verde/Vermelho) e marcando o resultado (✅ Win / ❌ Loss).

#### D. Manual de Leitura Visual (Guia de Legendas)
O gráfico não é apenas velas; é um mapa tático.

| Componente | Estilo Visual | Significado | Ação Sugerida |
|:---|:---|:---|:---|
| **ZONA DE TESTE** | Linha Sólida + Grossa (Opacidade 0.6) | **Guerra Imediata**. O preço está "brigando" para passar. | Atenção redobrada. Aguarde rompimento ou rejeição. |
| **ESTRUTURA** | Linha Tracejada (`dash`) | **Concreto**. Suporte/Resistência histórico com volume real. | Alta chance de segurar o preço. Bom alvo de Take Profit. |
| **PSICOLÓGICO** | Linha Traço-Ponto (`dashdot`) | **Vidro**. Nível matemático (ex: 100k) sem histórico recente. | Pode quebrar fácil. Não confie cegamente. |
| **COR** | 🟢 Verde Neon / 🔴 Vermelho Alerta | Polaridade (Suporte vs Resistência). | Verde = Compradores defendendo. Vermelho = Vendedores defendendo. |

### 🧭 Como Operar: Checklist Mental de 5 Segundos
Antes de abrir a corretora para clicar, faça esta checagem:

| Passo | Pergunta | Requisito para VÁLIDO |
|:---:|:---|:---|
| **1** | **Sinal Direcional?** | Deve ser 🟢 ou 🔴 (Evite ⚪) |
| **2** | **Convicção Alta?** | Probabilidade idealmente **> 50%** |
| **3** | **Convergência?** | Compra pede OFI (+) / Venda pede OFI (-) |

> **Exemplo Real (Caso Neutro)**: Se o sinal for **NEUTRO** com **70% de Confiança**, a IA está te dando um aviso forte: "Tenho certeza absoluta de que não é hora de operar, mesmo que o fluxo (OFI) pareça bom."

### 🛡️ Análise de Barreiras (Inteligência de Fluxo)
O painel classifica a força das regiões de preço e identifica manipulações:

*   **Análise Multi-Timeframe (MTF)**: O Cockpit exibe quatro visões síncronas para garantir que você nunca opere contra a macro-tendência:
    *   **Microestrutura (15m)**: O "campo de batalha" imediato.
    *   **Tendência Intraday (1h)**: Filtra o ruído e mostra o fluxo da hora.
    *   **Contexto de Inferência (4h)**: O horizonte de visão da IA (Configurável).
    *   **Visão Diária (1d)**: As grandes paredes institucionais das últimas 24h.
*   **Score de "Realidade" (Anti-Spoofing)**: As linhas de Suporte e Resistência exibem uma porcentagem de "Realidade".
    *   **Como funciona a lógica**: O sistema compara o volume acumulado naquele preço com a **execução real** (trades realizados).
    *   **Execução > 0**: Se houver negócios sendo fechados naquele nível, o sistema atribui alta probabilidade de ser uma barreira real (~95%), pois o mercado está "testando" e consumindo a ordem.
    *   **Execução = 0**: Se houver uma parede enorme de ordens mas zero negócios realizados, a probabilidade cai (~45%), sinalizando que pode ser **Spoofing** (ordens fantasmas colocadas para manipular o preço).
*   **Visual Dinâmico & Inteligência de Estrutura (Motor v2)**: 
    *   **Polaridade Absoluta & Mapa de Cores**: O sistema aplica cores técnicas rigorosas: **Verde Neon** para suportes (abaixo do preço) e **Vermelho Alerta** para resistências (acima do preço).
    *   **Estratégia de Balde Duplo (Double Bucket)**: Garante visibilidade equilibrada, exibindo obrigatoriamente os **3 níveis técnicos mais próximos acima e os 3 abaixo** do preço real.
    *   **Prioridade Histórica (Deep Scan)**: O sistema prioriza níveis reais encontrados em até 72h de histórico. Níveis psicológicos só são ativados se o Deep Scan não encontrar estrutura anterior (ex: All Time High).
    *   **Lógica de "Zona de Briga" (💥 ZONA DE TESTE)**: Quando o preço desafia um nível (distância < 0.05%), a linha torna-se **sólida, semi-transparente (0.6)** e levemente mais grossa, indicando teste ativo sem esconder o candle.
    *   **Cláusula de Segurança (🧠 PSICOLÓGICO)**: Projeções automáticas em zonas sem histórico.
        *   **Passo Dinâmico**: Saltos de **$500** (para preços > $50k) ou **$100** (preços < $50k) para evitar poluição visual.
    *   **Layering Profissional**: Todas as linhas de barreira são desenhadas **atrás dos candles** (layer below), garantindo que a ação de preço (pavios e corpos) seja sempre protagonista.
    *   **Zonas de Confluência**: Agrupamento automático de níveis próximos (< 0.1%) com reforço visual (width 3+).
    *   **Tipos de Traçado & Validação**:
        *   Linhas **Tracejadas (`dash`)**: Barreiras **REAIS** (Volume confirmado por execução de trades).
        *   Linhas **Traço-Ponto (`dashdot`)**: Suspeita de **SPOOFING** (Volume estacionário/sem execução).

---

## 9. Guia de Instalação e Configuração

> **Referência**: `docs/INSTALL_PYTORCH_CUDA.md`

### Pré-requisitos
*   **Python 3.10+**
*   **GPU NVIDIA** (Essencial para treino - CUDA 12.x).

### Instalação Passo-a-Passo
```powershell
# 1. Clone e Ambiente
git clone https://github.com/seu-usuario/saimp.git
cd BTCR
python -m venv .venv
.\.venv\Scripts\activate  # Windows

# 2. Instale Dependências
pip install -r requirements.txt

# 3. Force Instalação do PyTorch com CUDA (Crítico!)
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Configuração (.env)
echo "ENV=development" > .env
```

### Verificação (Pre-Flight)
```powershell
python src/scripts/pre_flight_check.py
```
*Saída esperada*: `Hardware: cuda`, `VRAM: OK`.

---

## 10. Manual de Operação

### 🔴 Terminal 1: Coleta de Dados Históricos
Para baixar dados passados da Binance:
```powershell
python -m src.collectors.historical
```

### 🚜 Terminal 2: Treinamento da IA
Para iniciar o treinamento (usa `src/config.py`):
```powershell
python -m src.training.train
```
*   **Logs**: `logs/train_run_TIMESTAMP.txt`.
*   **Backup**: `data/saimp_best.pth`.

### 🔍 Terminal 3: Auditoria Visual
Para inspecionar o que a IA está aprendendo:
```powershell
python src/visualization/verify_labels.py
```

### 🎯 Terminal 4: Backtest Sniper (Validação)
Para rodar o backtest sobre os dados coletados (Histórico ou Stream):
```powershell
python src/evaluation/backtest_stream.py
```
> **Dica**: Altere `STREAM_MONTH` no script para alternar entre Janeiro (Histórico) e Fevereiro (Stream do dia).

### 📊 Terminal 5: Painel Operacional (Cockpit)
Para visualizar as decisões da IA em tempo real com interface visual:
```powershell
# Certifique-se de estar no ambiente virtual
streamlit run src/dashboard/app.py
```

### 8.1 Diário de Execução (Sniper Shots)
O sistema possui um mecanismo de **Autenticação de Resultado** integrado:
*   **Log de Telemetria**: Salvo em `data/prediction_log.csv`.
*   **Filtro Sniper**: O sistema ignora ruídos ("Neutro") e registra apenas entradas reais de Compra/Venda.
*   **Auditoria Automática**: Passados 15 minutos de uma entrada, o validador compara o preço de saída com o de entrada e calcula o **P&L (Profit & Loss)** real, classificando o trade como `WIN` ou `LOSS`.
*   **Visualização**: Exibida no dashboard na tabela "🎯 Diário de Execução (Tiros do Sniper)".

### Ajuste Fino (Tunning)
Edite `src/config.py` para alterar:
*   Horizonte de Previsão (`LABEL_WINDOW_HOURS`)
*   Alvos de Lucro/Stop (`LABEL_TARGET_PCT`)
*   Hiperparâmetros de Treino (`BATCH_SIZE`, `LR`)

---

## 11. Estrutura do Projeto

```bash
BTCR/
├── data/                       # Data Warehouse (Ignorado no Git)
│   ├── raw/historical/         # Dados para treino (Parquet)
│   ├── raw/stream/             # Dados ao vivo (Inferência)
│   └── processed/              # Tensores temporários
├── docs/                       # Documentação Original Detalhada (.md)
├── logs/                       # Logs de execução
├── src/                        # Código Fonte Principal
│   ├── audit/                  # Scripts de verificação de integridade
│   ├── collectors/             # Crawlers da Binance
│   ├── debug/                  # Ferramentas de inspeção visual
│   ├── models/                 # Arquiteturas Neurais (ViViT)
│   ├── processing/             # ETL Core (Simulation, Labeling, Features)
│   ├── scripts/                # Utilitários (Pre-flight, etc)
│   ├── training/               # Loop de Treinamento
│   ├── utils/                  # Loggers e helpers
│   ├── visualization/          # Plotting scripts
│   └── config.py               # ⚙️ Configuração Centralizada
├── tests/                      # Súite de Testes
├── .env                        # Variáveis de ambiente
├── requirements.txt            # Dependências
└── README.md                   # Este arquivo (A Fonte da Verdade)
```

---

## 12. MAPA DO IMPÉRIO ATUALIZADO (Roadmap v2.0)

Este é o guia definitivo para levar o projeto do estágio "Protótipo Funcional" para "Hedge Fund Pessoal".

### 🏁 FASE 1: O TESTE DE FOGO (O "Agora")
**Objetivo**: Validar que o software não quebra e que a lógica básica funciona no mundo real.

1. **Validação Visual (Estabilidade)**
   - **Ação**: Deixe o Dashboard rodando localmente por 1 a 2 horas.
   - **O que checar**:
     - O "Warm-up" carrega sem erros?
     - O preço na tela bate com o da Binance?
     - O velocímetro de probabilidade oscila (está vivo) ou travou?
   - **Meta**: Zero erros de conexão ou estouro de memória (RAM/VRAM).

   - **Meta**: Zero erros de conexão ou estouro de memória (RAM/VRAM).

2. **Paper Trading Automatizado (Acurácia)**
   - **Status**: ✅ **AUTOMATIZADO**.
   - **Como funciona**: O robô agora registra suas próprias previsões no `data/prediction_log.csv` e valida o resultado (WIN/LOSS) sozinho.
   - **Ação**: Basta monitorar a aba "Diário de Execução" no Dashboard.
   - **Meta**: Validar se o Win Rate estatístico está alinhado com o esperado antes de liberar capital real.

### ☁️ FASE 2: INFRAESTRUTURA & DADOS PREMIUM
**Objetivo**: Profissionalizar a execução (sair do PC Gamer) e refinar a "gasolina" do modelo.

1. **Migração para Cloud (RunPod / Vast.ai)**
   - **Por que?** Servidores dedicados têm uptime de 99.9%. Evita quedas de luz ou internet.
   - **Ação**: Alugar instância (CPU robusta ou GPU básica) para rodar o Dashboard 24/7.
   - **Extra**: Configurar acesso via celular para monitorar o robô de qualquer lugar.

2. **Upgrade de Dados (Tardis.dev / Kaiko)**
   - **O Problema**: Dados públicos da Binance têm pequenos "gaps" e são agregados.
   - **A Solução**: Integrar `Tardis.dev` para acesso ao histórico tick-by-tick e replay de Order Book (L2).
   - **Impacto**: Ver o mercado em 4K em vez de HD, capturando micro-padrões invisíveis.

### 🌊 FASE 3: EVOLUÇÃO CIENTÍFICA (O "Quant God")
**Objetivo**: Aumentar o Win Rate usando Matemática Avançada e Segunda IA.

1. **Wavelets (Denoising Matemático)**
   - **Conceito**: Limpar o sinal do mercado para focar apenas na tendência estrutural.
   - **Ação**: Implementar Transformada Wavelet no `tensor_builder.py`.

2. **Arquitetura de Dupla IA (Stacking / Ensemble)**
   - **IA 1 (O Visionário)**: Modelo ViViT atual. Olha o gráfico e prevê a direção.
   - **IA 2 (O Gerente de Risco)**: Novo modelo (XGBoost) treinado em dados tabulares para "vetar" sinais em horários de baixo volume ou ruído.
   - **Resultado**: Redução drástica de falsos positivos.

---

### ✅ CHECKLIST UNIFICADO DE PRIORIDADES

**HOJE (Fase 1)**
- [ ] Rodar `streamlit run src/dashboard/app.py`.
- [ ] Validar Warm-up e estabilidade por 2 horas.
- [ ] Fazer 3 a 5 "Paper Trades" (anotar e conferir resultado).

**SEMANA QUE VEM (Fase 2)**
- [ ] Criar conta na RunPod/Vast.ai e subir o projeto.
- [ ] (Opcional) Avaliar custo do Tardis.dev para dataset de treino mais preciso.

**FUTURO PRÓXIMO (Fase 3)**
- [ ] Implementar Wavelets (Limpeza de Sinal).
- [ ] Treinar a 2ª IA (XGBoost) para filtrar os sinais do ViViT.

---
> **SAIMP Project** - *Decoding the Matrix.* 🐺🚀
