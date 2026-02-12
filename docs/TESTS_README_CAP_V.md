# 🧪 SAIMP: Quality Assurance & Testing Module (CAPÍTULO V)

> **Status**: Concluído (Fase 5) ✅  
> **Versão**: 5.1 (Full Coverage + Audit)
> **Dependência**: Requer `MODEL_DADOS_README_CAP_IV.md` (Architecture)

## 1. Visão Geral
Este documento detalha o "Sistema Imunológico" do **SAIMP**. Em Finanças Quantitativas e Deep Learning, um bug silencioso pode custar milhões ou invalidar meses de pesquisa. Por isso, implementamos uma bateria rigorosa de testes unitários, de integração e auditoria de dados.

O objetivo do Capítulo V é garantir que a matemática descrita nos capítulos anteriores (I a IV) esteja sendo executada fielmente pelo código e que os dados (combustível) estejam íntegros.

### Filosofia de Testes
Adotamos a pirâmide de testes expandida:
1.  **Unitários (Pytest)**: Validação matemática e lógica de funções isoladas.
2.  **Integração (Smoke Test)**: Validação do pipeline completo (do disco à rede neural) com dados reais.
3.  **Auditoria (Health Checks)**: Monitoramento da integridade dos dados históricos e do fluxo em tempo real.

---

## 2. A Suíte de Testes Unitários (`tests/`)

Utilizamos o framework `pytest` para execução automatizada de testes granulares.

### Módulos Cobertos

| Teste | O que valida? | Descrição |
|:---|:---|:---|
| **Features** | Fluxo e Volatilidade | Garante que $OFI = Vol_{Buy} - Vol_{Sell}$ e que o desvio padrão de preços (volatilidade) nunca seja zero em mercado ativo. |
| **Simulação** | Book Reconstruction | Verifica se trades de venda a mercado consomen corretamente a liquidez do lado do Bid. |
| **Tensores** | Dimensões 4D | Assegura que o output seja sempre `(Time, 4, 128)`, respeitando a estrutura de canais da CNN. |
| **Labeling** | Triple Barrier | **Crítico**: Testa se o sistema prioriza o **Stop Loss** sobre o Take Profit em caso de ambiguidade (Princípio do Conservadorismo). |
| **Modelo** | ViViT Forward Pass | Testa se a rede neural aceita os tensores e retorna probabilidades válidas (sem travar ou dar NaN). |

---

## 3. Testes de Integração & Smoke (`test_integration.py`)

Testes unitários usam dados falsos (Mock). O **Smoke Test** usa dados reais para provar que o sistema funciona no mundo real.

### O Script `test_integration.py`
Este script simula um ciclo de produção completo:
1.  **Leitura Real**: Carrega 500k linhas de um arquivo `aggTrades` real do disco.
2.  **Simulação em Massa**: Reconstrói ~100 snapshots de Order Book.
3.  **Inferência**: Alimenta o modelo ViViT com esses dados.

> **Objetivo**: Se este script rodar sem erros ("CHECK-MATE"), significa que não há incompatibilidade de shapes, tipos de dados ou memória entre os módulos.

---

## 4. Auditoria de Dados & Health Checks

Não adianta ter um motor de Ferrari (IA) e colocar gasolina adulterada (Dados ruins).

### A. Auditoria Histórica (`check_completeness.py`)
Verifica se o nosso "Lago de Dados" tem buracos.
*   **Varredura**: Checa mês a mês (desde Jan/2023) se existem os arquivos `klines` e `aggTrades`.
*   **Detecção de Corrupção**: Alerta se encontrar arquivos vazios ou muito pequenos (<1KB).
*   **Amostragem**: Tenta ler um arquivo aleatório para garantir que o Parquet é válido.

### B. Auditoria de Stream (`check_stream.py`)
Verifica se o sistema está "VIVO" agora.
*   **Batimentos Cardíacos**: Alerta se o último arquivo inserido pelo stream tem mais de 20 minutos (indica crash ou desconexão).
*   **Biópsia**: Abre o último arquivo e verifica se contém colunas de `trades` ou `depth` com dados reais, não apenas cabeçalhos vazios.

---

## 5. Como Executar o Protocolo de QA

Para rodar a bateria completa e verificar a saúde do sistema:

### Passo 1: Testes Matemáticos (Rápido)
```powershell
pytest tests/
```
*Saída Esperada: 12 passed.*

### Passo 2: Teste de Fumaça (Integração)
```powershell
python tests/integration/test_integration.py
```
*Saída Esperada: "CHECK-MATE! O Pipeline está 100% blindado".*

### Passo 3: Auditoria de Dados
```powershell
python src/audit/check_completeness.py  # Para histórico
python src/audit/check_stream.py        # Para tempo real
```
*Saída Esperada: "Auditoria Concluída: DADOS ÍNTEGROS" e "O Stream está VIVO".*

---

## 6. Próximos Passos (Fase 6 - Deployment) 🚀

Com o sistema documentado, implementado, testado e auditado, a base tecnológica está concluída.

- [ ] **Integração Contínua (CI)**: Configurar GitHub Actions para rodar `pytest` a cada commit.
- [ ] **Training Run**: Iniciar o treinamento do modelo com o dataset completo auditado.
- [ ] **Live Deployment**: Conectar o modelo treinado ao script de inferência.

---
> *SAIMP - Confiança através da Verificação.*
