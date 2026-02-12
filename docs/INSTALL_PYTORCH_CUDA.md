# 🔧 Guia de Instalação: PyTorch com CUDA

## ❌ Problema Identificado

Sua instalação atual do PyTorch é **CPU-only**:
```
torch==2.10.0+cpu
```

O sufixo `+cpu` indica que o PyTorch foi compilado sem suporte CUDA, por isso a GPU não está sendo utilizada.

## ✅ Solução: Reinstalar PyTorch com CUDA

### Passo 1: Verificar Versão do CUDA

Primeiro, verifique qual versão do CUDA Toolkit está instalada:

```powershell
nvidia-smi
```

Procure pela linha "CUDA Version" no canto superior direito. Exemplo:
```
CUDA Version: 12.1
```

### Passo 2: Desinstalar PyTorch CPU-only

```powershell
pip uninstall torch torchvision torchaudio
```

Confirme com `y` quando solicitado.

### Passo 3: Instalar PyTorch com CUDA

Escolha o comando apropriado baseado na sua versão do CUDA:

#### Para CUDA 12.1 (Recomendado - Mais Recente)
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Para CUDA 11.8 (Alternativa Estável)
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Passo 4: Verificar Instalação

Após a instalação, verifique se CUDA foi detectado:

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Saída esperada:**
```
PyTorch: 2.10.0+cu121  # Note o +cu121 em vez de +cpu
CUDA Available: True
GPU: NVIDIA GeForce RTX 3050
```

### Passo 5: Re-executar Pre-Flight Check

```powershell
python pre_flight_check.py
```

Deve mostrar:
```
🖥️ Hardware: cuda
   GPU: NVIDIA GeForce RTX 3050
   VRAM: 2.00 GB
   Mixed Precision (AMP): ✅ Enabled
```

### Passo 6: Reiniciar Treinamento

```powershell
python -m src.training.train
```

Agora o treino deve usar a GPU!

---

## 🔍 Troubleshooting

### Problema: "CUDA Available: False" mesmo após reinstalação

**Causa:** Driver NVIDIA desatualizado ou CUDA Toolkit não instalado.

**Solução:**
1. Atualize o driver NVIDIA: https://www.nvidia.com/Download/index.aspx
2. Instale o CUDA Toolkit: https://developer.nvidia.com/cuda-downloads

### Problema: "RuntimeError: CUDA out of memory"

**Causa:** Batch size muito grande para 2GB VRAM.

**Solução:** Edite `src/training/train.py`:
```python
BATCH_SIZE = 2  # Reduza de 4 para 2
ACCUMULATION_STEPS = 16  # Aumente de 8 para 16
```

Isso mantém o batch efetivo em 32 (2 × 16).

---

## 📚 Referências

- PyTorch Installation Guide: https://pytorch.org/get-started/locally/
- CUDA Toolkit Archive: https://developer.nvidia.com/cuda-toolkit-archive
