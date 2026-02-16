
import argparse
import json
import os
import re
import subprocess
import sys
import shutil
from pathlib import Path

# Configurações de Caminho
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "src" / "config.py"
PARAMS_PATH = BASE_DIR / "best_l2_params.json" # Atualizado para bater com optimize_l2_vivit_raw.py
OPT_SCRIPT = BASE_DIR / "scripts" / "optimize_l2_vivit_raw.py"
TRAIN_MODULE = "src.models.train"

def backup_config():
    print(f"BACKUP: Criando backup de {CONFIG_PATH.name}...")
    shutil.copy(CONFIG_PATH, f"{CONFIG_PATH}.bak")

def update_config(params):
    print("💉 Injetando hiperparâmetros no config.py...")
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    mapping = {
        'batch_size': 'BATCH_SIZE',
        'd_model': 'D_MODEL',
        'nhead': 'NHEAD',
        'lr': 'LEARNING_RATE',
        'dropout': 'DROPOUT'
    }

    processed_keys = set()
    
    for line in lines:
        updated = False
        # Hiperparâmetros do Optuna
        for json_key, config_key in mapping.items():
            if line.strip().startswith(f"{config_key}:"):
                val = params[json_key]
                # Preservar o prefixo do tipo se houver
                prefix = line.split("=")[0]
                new_lines.append(f"{prefix}= {val}\n")
                updated = True
                processed_keys.add(config_key)
                break
        
        if updated: continue
        
        # Parâmetros Fixos Swing
        if line.strip().startswith("SEQ_LEN:"):
            prefix = line.split("=")[0]
            new_lines.append(f"{prefix}= 720\n")
        elif line.strip().startswith("NUM_FEATURES:"):
            prefix = line.split("=")[0]
            new_lines.append(f"{prefix}= 9\n")
        elif line.strip().startswith("DATA_PATH:"):
            prefix = line.split("=")[0]
            new_lines.append(f'{prefix}= "data/processed/l2_features_1min_final.parquet"\n')
        elif line.strip().startswith("PREDICTION_HORIZON:"):
            prefix = line.split("=")[0]
            new_lines.append(f"{prefix}= 60\n")
        else:
            new_lines.append(line)

    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("DONE: Configuracao atualizada com sucesso.")

def run_command(command):
    """Executa comando subprocess e faz stream do output em tempo real"""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        shell=True if os.name == 'nt' else False
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode

def main():
    parser = argparse.ArgumentParser(description="QuantGod Swing Trade Master Pipeline")
    parser.add_argument("--skip-opt", action="store_true", help="Pular otimização e usar JSON existente")
    parser.add_argument("--run-opt", action="store_true", default=True, help="Executar Optuna + Visual Audit")
    args = parser.parse_args()

    # Se explicitamente pediu skip, sobrescrever run_opt
    if args.skip_opt:
        args.run_opt = False

    print("\n" + "="*60)
    print("QUANTGOD MLOPS PIPELINE - MASTER AUTOMATION")
    print("="*60)

    # 1. Backup
    backup_config()

    # 2. Otimização ou Carregamento
    if not args.run_opt:
        if not PARAMS_PATH.exists():
            print(f"ERROR: {PARAMS_PATH} nao encontrado. Execute sem --skip-opt primeiro.")
            sys.exit(1)
        print(f"INFO: Carregando parametros otimizados de {PARAMS_PATH}...")
    else:
        print("SEARCH: Iniciando Otimizacao + Verificacao Visual...")
        exit_code = run_command([sys.executable, str(OPT_SCRIPT)])
        if exit_code != 0:
            print("ERROR: Falha no script de otimizacao/auditoria.")
            sys.exit(1)
            
    # Carregar Parâmetros
    with open(PARAMS_PATH, 'r') as f:
        best_params = json.load(f)

    # 3. Injeção
    update_config(best_params)

    # 4. Treino Final
    print("\nTRAIN: Iniciando Treino Final com Configuracao Otimizada (Swing)...")
    train_code = run_command([sys.executable, "-m", TRAIN_MODULE])
    
    if train_code == 0:
        print("\nDONE: Pipeline executado com SUCESSO!")
    else:
        print("\nWARNING: O treino final terminou com avisos ou erros.")

    # 5. Rollback Interativo
    print("\n" + "-"*60)
    revert = input("🔄 Deseja reverter o src/config.py para o original? [y/N]: ").strip().lower()
    if revert == 'y':
        shutil.move(f"{CONFIG_PATH}.bak", CONFIG_PATH)
        print("⏪ Configuração original restaurada.")
    else:
        print(f"📌 Configuração otimizada mantida em {CONFIG_PATH.name}.")

if __name__ == "__main__":
    main()
