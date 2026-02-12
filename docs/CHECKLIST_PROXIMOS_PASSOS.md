# 🗺️ O Mapa da Mina: Checklist de Evolução (Do Local à Nuvem)

Este é o plano detalhado para transformar o código atual em um Hedge Fund autônomo.

## 🟢 FASE 1: Auditoria & Validação Local ✅ CONCLUÍDA
O objetivo é garantir que a lógica básica funciona antes de escalar.

- [x] **Executar `verify_labels.py`**:
    - [x] **Critério de Sucesso**: Se 90% dos gráficos fizerem sentido visualmente, a lógica de rotulagem está aprovada.
    - [x] Verificar **Fundo VERDE**: O preço tocou na linha pontilhada verde (Alvo) antes da vermelha?
    - [x] Verificar **Fundo VERMELHO**: O preço tocou na linha vermelha (Stop) antes da verde?
    - [x] Verificar **Fundo CINZA**: O preço ficou "sambando" no meio até o fim do gráfico?

- [ ] **Verificar Resultado do `train.py` (Profissional)**:
    - [ ] **Métrica**: Acurácia de Validação (Val Acc) > 33% (Melhor que aleatório).
    - [ ] **Loss**: Deve cair consistentemente ao longo das épocas.

## 🟡 FASE 2: Rigor Científico (Chronological Split) ✅ CONCLUÍDA
Aqui corrigimos o viés de "olhar para o futuro" que o random_split introduz.

- [x] **Implementar Divisão Cronológica no `train.py`**:
    - **Conceito**: Em séries temporais, nunca embaralhamos os dados. Treinamos no passado para prever o futuro.
    - **Ação**: ✅ Script reescrito com listas explícitas:
        - **Treino**: Novembro/2025 + Dezembro/2025.
        - **Validação**: Janeiro/2026.
    - **Por que**: Se o modelo acertar Janeiro sem nunca tê-lo visto, ele é robusto.

- [x] **Técnicas Profissionais de ML Engineering**:
    - [x] **Gradient Accumulation**: Batch efetivo de 32 (4 físico × 8 accumulation steps).
    - [x] **Mixed Precision (AMP)**: Treino em FP16 para economizar VRAM.
    - [x] **Memory Optimization**: `pin_memory=True`, `gc.collect()` estratégico.

- [x] **Estratégia de Dados (Histórico vs. Stream)**:
    - **O Histórico (A Escola)**: Usado apenas para criar o arquivo .pth (o cérebro).
    - **O Stream (O Trabalho)**: O script de inferência (`dashboard_live.py`) carrega o .pth e processa o Stream em tempo real. Ele não treina, apenas executa.
    - **O Ciclo (Feedback Loop)**:
        - Dia 1: Stream salva dados em disco (`raw/stream/`).
        - Dia 30: Movemos esses dados para `raw/historical/`.
        - Dia 31: Re-treinamos o modelo com o novo mês incluído.


## 🟠 FASE 3: Visão Computacional Financeira (Os "Olhos")
Aqui transformamos números em intuição visual.

- [ ] **Heatmap com Inferência Real**:
    - Criar `dashboard_live.py`. Ele lê o último snapshot do Stream, passa no modelo e plota o Heatmap.

- [ ] **Decodificação de Fluxo**:
    - Sobrepor setas no Heatmap indicando a pressão do OFI (Order Flow Imbalance).

- [ ] **Detecção de Paredes Reais vs. Falsas (Attention Map)**:
    - **Técnica**: Extrair os pesos de atenção do Transformer (`model.transformer_encoder.layers[-1].self_attn`).
    - **Visual**:
        - Se o peso de atenção é alto numa coordenada $(Preço, Tempo)$, desenhar BBox Sólido (A IA "confiou" nessa liquidez).
        - Se a liquidez é alta mas a atenção é zero, desenhar BBox Tracejado (A IA ignorou = Spoofing provável).

## 🔵 FASE 4: Escala na Nuvem (Heavy Lifting)
Sua GPU de 2GB não aguenta o treino massivo (2023-2026).

- [ ] **Infraestrutura (GCP/AWS/Lambda)**:
    - Subir uma VM com GPU (T4 ou A100).

- [ ] **Migração de Dados**:
    - Enviar a pasta `data/raw/historical` para um Bucket S3 ou GCS.

- [ ] **Treino Full-Scale**:
    - Rodar o `train.py` com todos os 37 meses.
    - Aumentar Batch Size para 64/128.
    - Usar ChronologicalSplit (Treino: 2023-2025 | Validação: 2026).

---
