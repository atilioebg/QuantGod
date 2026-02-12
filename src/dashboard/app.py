import streamlit as st
import time
from pathlib import Path
import sys

# Setup Project Path
# Assumes app.py is in src/dashboard/ and running from project root
sys.path.append(str(Path.cwd()))

from src.live.connector import BinanceConnector
from src.live.predictor import SniperBrain

st.set_page_config(page_title="Sniper Cockpit", layout="wide", page_icon="🎯")

# Use st.cache_resource for objects that persist (Connector and Model)
@st.cache_resource
def get_system():
    # Conector
    conn = BinanceConnector(symbol="BTCUSDT")
    conn.start()
    conn.warm_up()
    
    # Cérebro
    # Tenta carregar do caminho padrão
    brain = SniperBrain()
    
    return conn, brain

st.title("🎯 SAIMP: Cockpit Operacional")
st.markdown("---")

col_sinal, col_conf, col_preco = st.columns(3)
st.markdown("### 🛡️ Análise de Barreiras & Estrutura")
col_barreira, col_ofi = st.columns(2)

# Load System
try:
    with st.spinner("Inicializando Conexão e IA..."):
        conn, brain = get_system()
except Exception as e:
    st.error(f"Erro ao inicializar sistema: {e}")
    st.stop()

if st.checkbox("🔴 SISTEMA LIGADO", value=True):
    # Loop de atualização (Streamlit Rerun Loop)
    
    # Create empty container to hold updates
    placeholder = st.empty()
    
    # We don't actually loop *inside* Streamlit script typically, 
    # but for "Live Dashboard" we can use st.rerun() with sleep.
    # However, st.empty() is better for updating elements without full reload if loop is inside.
    # But user code used `while True` with `st.rerun()`. This refreshes the whole page.
    # Let's stick to the user's pattern but add safety break.
    
    # Fetch Data
    # 500 minutes to ensure enough history for simulation
    df = conn.get_data(minutes=500)
    
    if df is not None and df.height > 100:
        result = brain.analyze(df)
        
        if result:
            # Lógica de Exibição
            sinal = result['signal']
            conf = result['confidence']
            ofi = result['ofi']
            price = result['price']
            
            # 1. Painel Principal
            labels = ["⏸️ NEUTRO / AGUARDAR", "🔴 VENDA (SHORT)", "🟢 COMPRA (LONG)"]
            colors = ["gray", "red", "green"]
            
            col_sinal.markdown(f"## :{colors[sinal]}[{labels[sinal]}]")
            col_conf.metric("Probabilidade IA", f"{conf*100:.1f}%")
            col_preco.metric("Preço Atual", f"${price:,.2f}")
            
            # 2. Análise de Barreiras (Verdadeiro vs Falso)
            barrier_msg = "Sem definição clara"
            if sinal == 2: # Compra
                if ofi > 0: barrier_msg = "✅ Suporte VERDADEIRO (Fluxo Comprador Confirmado)"
                else: barrier_msg = "⚠️ Suporte FALSO (Divergência de Fluxo)"
            elif sinal == 1: # Venda
                if ofi < 0: barrier_msg = "✅ Resistência VERDADEIRA (Fluxo Vendedor Confirmado)"
                else: barrier_msg = "⚠️ Resistência FALSA (Possível Armadilha)"
            else:
                barrier_msg = "Aguardando sinal direcional..."
            
            col_barreira.info(f"**Veredito:** {barrier_msg}")
            col_ofi.metric("Order Flow (OFI)", f"{ofi:.2f}", delta=result['trend_intent'])
            
        else:
            st.warning("⏳ Dados insuficientes para inferência (Buffer < 32 snapshots ou erro).")
    else:
        st.warning("⏳ Carregando dados (Warm-up)... Aguarde buffer encher.")
    
    time.sleep(5)
    st.rerun()
