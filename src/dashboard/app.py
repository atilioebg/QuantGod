import streamlit as st
import time
import plotly.graph_objects as go
import polars as pl
from datetime import timedelta
from src.config import settings
from pathlib import Path
import sys

# Setup Project Path
# Assumes app.py is in src/dashboard/ and running from project root
sys.path.append(str(Path.cwd()))

from src.live.connector import BinanceConnector
from src.live.predictor import SniperBrain

st.set_page_config(page_title="Sniper Cockpit", layout="wide", page_icon="🎯")

# Use st.cache_resource for objects that persist (Connector and Model)
@st.cache_resource(on_release=lambda x: x[0].stop())
def get_system():
    # Conector
    conn = BinanceConnector(symbol="BTCUSDT")
    conn.start()
    conn.warm_up()
    
    # Cérebro
    brain = SniperBrain()
    
    return conn, brain

# Gerenciamento de Ciclo de Vida do Recurso (Estabilidade)
conn_obj, brain_obj = get_system()

def get_candles(df_trades, window="15m"):
    """Agrega trades em velas OHLC"""
    if df_trades is None: return None
    
    # Garantir que timestamp é datetime e converter para Brasília (UTC-3)
    # TRUQUE: Converter para TZ correto e depois REMOVER a info de TZ (Naive)
    # Isso impede o Plotly de tentar "re-converter" para local time do browser e cagar o offset.
    df = df_trades.with_columns(
        pl.from_epoch("timestamp", time_unit="ms").alias("dt")
    ).with_columns(
        pl.col("dt").dt.replace_time_zone("UTC").dt.convert_time_zone("America/Sao_Paulo").dt.replace_time_zone(None)
    ).sort("dt")
    
    # Agrupar por janela
    candles = (
        df.group_by_dynamic("dt", every=window)
        .agg([
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("quantity").sum().alias("volume")
        ])
    )
    return candles

st.title("🎯 SAIMP: Cockpit Operacional")
st.markdown("---")

# 5. ZONA DE COMANDO (FRAGMENTO AO VIVO)
# O Fragmento permite atualizar apenas esta parte da tela a cada 5 segundos
# Isso elimina o "pisca-pisca" da página inteira e resolve erros de ID duplicado.
@st.fragment(run_every=10)
def render_live_dashboard(conn, brain):
    # Lock de Execução para evitar death spiral (overlapping runs)
    if st.session_state.get('analyzing', False):
        return
    
    st.session_state.analyzing = True
    try:
        # Fetch Data
        df = conn.get_data(minutes=10080)
        
        if df is not None and df.height > 100:
            result = brain.analyze(df)
            
            if result:
                # 1. Métricas de Cabeçalho (Sinal e Confiança)
                col_sinal, col_conf, col_preco = st.columns(3)
                
                sinal = result['signal']
                conf = result['confidence']
                ofi = result['ofi']
                price = result['price']
                
                labels = ["⏸️ NEUTRO / AGUARDAR", "🔴 VENDA (SHORT)", "🟢 COMPRA (LONG)"]
                colors = ["gray", "red", "green"]
                
                col_sinal.markdown(f"## :{colors[sinal]}[{labels[sinal]}]")
                col_conf.metric("Probabilidade IA", f"{conf*100:.1f}%")
                col_preco.metric("Preço Atual", f"${price:,.2f}")
    
                # 2. Análise de Fluxo e Barreiras (Abaixo das métricas)
                st.markdown("### 🛡️ Análise de Barreiras & Estrutura")
                col_barreira, col_ofi_metric = st.columns(2)
                
                barrier_msg = "Lateralidade: Aguardando rompimento de fluxo ou estrutura."
                if sinal == 0:
                    barrier_msg = "Lateralidade: Aguardando rompimento de fluxo ou estrutura."
                elif sinal == 2: 
                    barrier_msg = "✅ SUPORTE VERDADEIRO, o fluxo confirma a zona de liquidez, logo temos uma barreira real." if ofi > 0 else "⚠️ SUPORTE FALSO, o volume não é validado pelo fluxo, logo temos suspeita de spoofing."
                elif sinal == 1: 
                    barrier_msg = "✅ RESISTÊNCIA VERDADEIRA, o fluxo confirma a zona de liquidez, logo temos uma barreira real." if ofi < 0 else "⚠️ RESISTÊNCIA FALSA, o volume não é validado pelo fluxo, logo temos suspeita de spoofing."
                
                col_barreira.info(f"**Veredito:** {barrier_msg}")
                # Correção de Coerência OFI (Precisão de 4 casas decimais)
                desc_ofi = f"{ofi:.4f} ({result['trend_intent']})"
                col_ofi_metric.metric("Order Flow (OFI)", desc_ofi, delta=round(ofi, 4))
    
                # Frase Descritiva Acionável (README.md logic)
                with st.expander("📌 O QUE FAZER AGORA? (Manual do Piloto)", expanded=True):
                    if sinal == 0: # Neutro
                        st.warning("⚪ **ESTRATÉGIA: AGUARDAR**, a IA não vê entrada clara ou o mercado está sem direção definida, logo ficar fora é uma posição protegida.")
                    else:
                        direcao = "ALTA" if sinal == 2 else "BAIXA"
                        emoji = "🟢" if sinal == 2 else "🔴"
                        conf_msg = "🔥 ALTA CONVICÇÃO" if conf > 0.6 else "⚖️ CONVICÇÃO MODERADA"
                        
                        # Convergência OFI
                        conv = (sinal == 2 and ofi > 0) or (sinal == 1 and ofi < 0)
                        conv_msg = "✅ FLUXO CONFIRMA" if conv else "❌ FLUXO DIVERGENTE (Cuidado!)"
                        
                        st.success(f"{emoji} **ORDEM: {labels[sinal]}**, o mercado apresenta viés de {direcao} com {conf_msg}, logo a análise de {conv_msg}.")
                        st.info(f"**Checklist Mental:** 1. Sinal Direcional {emoji} | 2. Confiança >50% | 3. OFI Convergente? {'Sim' if conv else 'Não'}")
    
                st.markdown("---")
                st.caption("🕒 **Fuso Horário:** America/Sao_Paulo (Brasília/UTC-3)")
                
                # Motor de Validação de Estrutura v2: Deep Scan, Volatility Step e Layering
                def render_trading_chart(df_data, title, window, current_live_price, height=500, levels=None, s_color="rgba(0, 230, 118, ", r_color="rgba(255, 82, 82, "):
                    st.markdown(f"### {title}")
                    candles = get_candles(df_data, window=window)
                    
                    # Cálculo de Delta (Time Difference) para geometria
                    unit_map = {"m": "minutes", "h": "hours", "d": "days"}
                    val, unit = int(window[:-1]), window[-1]
                    delta = timedelta(**{unit_map[unit]: val})
                    
                    if candles is not None:
                        fig = go.Figure()
                        
                        # 1. Preparação de Níveis (Calculados ANTES para Layering)
                        plot_traces = []
                        if levels:
                            # Filtro de Proximidade (±5% do preço atual para escala limpa)
                            prox_levels = [lv for lv in levels if abs(current_live_price - lv['price']) / current_live_price <= 0.05]
                            
                            # Clustering @ 0.1% de confluência
                            def cluster_levels(l_list):
                                if not l_list: return []
                                l_list.sort(key=lambda x: x['price'])
                                clustered = []
                                cluster = [l_list[0]]
                                for i in range(1, len(l_list)):
                                    if (l_list[i]['price'] - l_list[i-1]['price']) / l_list[i-1]['price'] <= 0.001:
                                        cluster.append(l_list[i])
                                    else:
                                        avg_p = sum(c['price'] for c in cluster) / len(cluster)
                                        avg_r = sum(c['realness'] for c in cluster) / len(cluster)
                                        clustered.append({'price': avg_p, 'realness': avg_r, 'confluence': len(cluster)})
                                        cluster = [l_list[i]]
                                avg_p = sum(c['price'] for c in cluster) / len(cluster)
                                avg_r = sum(c['realness'] for c in cluster) / len(cluster)
                                clustered.append({'price': avg_p, 'realness': avg_r, 'confluence': len(cluster)})
                                return clustered
    
                            clustered_prox = cluster_levels(prox_levels)
                            
                            # Double Bucket Strategy
                            res_list = sorted([lv for lv in clustered_prox if lv['price'] > current_live_price], key=lambda x: x['price'])
                            sup_list = sorted([lv for lv in clustered_prox if lv['price'] <= current_live_price], key=lambda x: x['price'], reverse=True)
                            
                            # --- MOTOR DE VALIDAÇÃO: Deep Scan vs Psicologia ---
                            if not res_list:
                                deep_res = sorted([lv for lv in levels if lv['price'] > current_live_price], key=lambda x: x['price'])
                                if deep_res: res_list = cluster_levels(deep_res)
                                else:
                                    step = 500 if current_live_price > 50000 else 100
                                    base = (current_live_price // step + 1) * step
                                    res_list = [{'price': base + i*step, 'realness': 0.5, 'confluence': 1, 'psicologico': True} for i in range(3)]
                            
                            if not sup_list:
                                deep_sup = sorted([lv for lv in levels if lv['price'] <= current_live_price], key=lambda x: x['price'], reverse=True)
                                if deep_sup: sup_list = cluster_levels(deep_sup)
                                else:
                                    step = 500 if current_live_price > 50000 else 100
                                    base = (current_live_price // step) * step
                                    sup_list = [{'price': base - i*step, 'realness': 0.5, 'confluence': 1, 'psicologico': True} for i in range(3)]
                            
                            # 2. Desenhar Níveis Estruturais (Finite Scatter para evitar poluição)
                            for lv in (res_list[:3] + sup_list[:3]):
                                dist_pct = abs(current_live_price - lv['price']) / current_live_price
                                is_battle = dist_pct < 0.0005
                                is_real = lv['realness'] > 0.5
                                
                                if is_battle: opacity, width, dash = 0.6, 3, "solid"
                                else:
                                    opacity = 0.8 if is_real else 0.4
                                    width, dash = (3 if lv.get('confluence', 1) > 1 else 2), ("dash" if is_real else "dashdot")
                                
                                # Polaridade Dinâmica
                                if current_live_price < lv['price']:
                                    color, type_label = f"{r_color}{opacity})", "RESISTÊNCIA"
                                else:
                                    color, type_label = f"{s_color}{opacity})", "SUPORTE"
                                    
                                label_final = f"{type_label} ({window})"
                                if is_battle: label_final = f"💥 ZONA DE TESTE ({type_label})"
                                elif lv.get('psicologico'): label_final = f"🧠 PSICOLÓGICO ({type_label})"
    
                                # Ajuste de Geometria: Intraday para EXATAMENTE na vela atual, Macro para no meio
                                line_ext = timedelta(0) if window != "1d" else (delta * 0.5)
                                fig.add_trace(go.Scatter(
                                    x=[candles["dt"].min(), candles["dt"].max() + line_ext], y=[lv['price'], lv['price']],
                                    mode="lines", name=label_final,
                                    line=dict(color=color, width=width, dash=dash),
                                    showlegend=True
                                ))
    
                        # 3. Add Candlestick POR CIMA (Layering)
                        fig.add_trace(go.Candlestick(
                            x=candles["dt"], open=candles["open"], high=candles["high"],
                            low=candles["low"], close=candles["close"], name=f"Price ({window})"
                        ))
    
                        # 4. Viewport Tático (Presente + Futuro em Branco)
                        zoom_n_map = {"15m": 12, "1h": 10, "4h": 12, "1d": 10}
                        n_zoom = zoom_n_map.get(window, 12)
                        
                        if len(candles) > 1:
                            last_ts = candles["dt"].max()
                            # Espaço à direita: 15 candles para evitar corte da última vela
                            range_end = last_ts + (delta * 15)
                            range_start = last_ts - (delta * n_zoom)
                            xaxis_range = [range_start, range_end]
                        else:
                            xaxis_range = None
    
                        fig.update_layout(
                            template="plotly_dark", xaxis_rangeslider_visible=False, 
                            height=500, margin=dict(l=10, r=10, t=10, b=10),
                            legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)"),
                            xaxis=dict(
                                type="date", # Força escala temporal linear
                                range=xaxis_range, tickformat="%d/%m %H:%M", tickangle=-45,
                                showgrid=True, gridcolor="rgba(128, 128, 128, 0.1)"
                            ),
                            yaxis=dict(fixedrange=False, autorange=True, side="right")
                        )
                        
                        # Configuração de Zoom Independente e Modebar
                        st.plotly_chart(fig, width='stretch', key=f"chart_{window}", config={
                            'scrollZoom': True,
                            'displayModeBar': True,
                            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraselayer']
                        })
    
                # 3. Sequência de Gráficos (15m, 1h, 4h, 1d)
                # Gráfico 1: 15m (Microestrutura)
                levels_15m = result.get('supports_15m', []) + result.get('resistances_15m', [])
                render_trading_chart(df, "📈 Microestrutura (15m)", "15m", price, 
                                     levels=levels_15m)
    
                # Gráfico 2: 1h (Curto Prazo)
                levels_1h = result.get('supports_1h', []) + result.get('resistances_1h', [])
                render_trading_chart(df, "🕒 Tendência Intraday (1h)", "1h", price, 
                                     levels=levels_1h)
    
                # Gráfico 3: Timeframe de Predição (Dinamico)
                pred_window = f"{settings.LABEL_WINDOW_HOURS}h"
                levels_pred = result.get('supports_pred', []) + result.get('resistances_pred', [])
                render_trading_chart(df, f"🌐 Contexto de Inferência ({pred_window})", pred_window, price,
                                     levels=levels_pred)
    
                # Gráfico 4: 1 Dia (Macro Contexto)
                levels_1d = result.get('supports_1d', []) + result.get('resistances_1d', [])
                render_trading_chart(df, "📅 Visão Diária (1d)", "1d", price, 
                                     levels=levels_1d)
            
            else:
                st.warning("⏳ **DADOS INSUFICIENTES**, o sistema ainda não processou o histórico necessário, logo aguarde a conclusão da análise.")
        else:
            st.warning("⏳ **SISTEMA EM WARM-UP**, o buffer de mercado ainda está sendo preenchido, logo o cockpit estará operacional em breve.")
    finally:
        st.session_state.analyzing = False

# Execução do Dashboard
if st.checkbox("🔴 SISTEMA LIGADO", value=True):
    render_live_dashboard(conn_obj, brain_obj)
else:
    st.info("💡 **SISTEMA EM STANDBY**, o monitoramento em tempo real está pausado, logo ligue o interruptor acima para iniciar.")
