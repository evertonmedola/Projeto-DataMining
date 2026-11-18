"""
Dashboard do CloudCast (Streamlit). Consome a API -- não acessa modelo nem
banco diretamente, então mostra exatamente o que o sistema serve.

    streamlit run dashboard/app.py         (API em CLOUDCAST_API_URL, padrão http://localhost:8000)

Indicadores pedidos no enunciado do tema 4:
  consumo histórico · previsão · tendência · custo projetado · alertas de crescimento
"""

import os
from datetime import timedelta

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

API = os.environ.get("CLOUDCAST_API_URL", "http://localhost:8000")
AZUL, CINZA, VERMELHO, VERDE = "#2563eb", "#475569", "#dc2626", "#059669"
COR_SEVERIDADE = {"CRITICO": "🔴", "ATENCAO": "🟠", "INFO": "🔵"}

st.set_page_config(page_title="CloudCast · Capacity Planning", page_icon="☁️", layout="wide")


@st.cache_data(ttl=30)
def chamar(caminho: str, **params):
    r = httpx.get(f"{API}{caminho}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def usd(valor) -> str:
    return "—" if valor is None else f"US$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ----------------------------------------------------------------------------
try:
    saude = chamar("/health")
    info = chamar("/model/info")
except Exception as exc:  # API fora do ar
    st.error(
        f"Não foi possível acessar a API em {API}. Suba com `uvicorn src.api.main:app` ou "
        f"`docker compose up`.\n\nDetalhe: {exc}"
    )
    st.stop()

with st.sidebar:
    st.header("☁️ CloudCast")
    st.caption("Capacity Planning de infraestrutura cloud")
    horizonte = st.slider("Horizonte da previsão (dias)", 7, 90, 30, step=1)
    dias_hist = st.slider("Histórico exibido (dias)", 30, 720, 180, step=30)
    simular = st.checkbox(
        "Simular a partir de uma data passada",
        help="Prevê como se hoje fosse a data escolhida, usando só o histórico anterior a ela. "
        "Útil para ver como o sistema teria antecipado a alta de novembro.",
    )
    referencia = st.date_input("Data de referência", value=pd.Timestamp("2024-10-29")) if simular else None
    st.divider()
    st.markdown(f"**Modelo em produção:** {info['algoritmo']} v{info['versao']}")
    st.markdown(f"Treinado até **{info['treinado_ate']}** ({info['n_observacoes']} dias)")
    st.markdown(f"MAPE no teste: **{info['metricas_teste']['MAPE_%']:.2f}%**")
    st.caption(f"commit {info['git_commit']} · dados {info['dados_sha256']}")
    if st.button("Atualizar dados"):
        st.cache_data.clear()
        st.rerun()

params_prev = {"horizonte": horizonte}
if referencia:
    params_prev["referencia"] = str(referencia)
prev = chamar("/forecast", **params_prev)
hist = pd.DataFrame(chamar("/historico", dias=dias_hist + (0 if not referencia else 400)))
hist["data"] = pd.to_datetime(hist["data"])
if referencia:  # na simulação, o "futuro" real fica visível para comparar com a previsão
    futuro_real = hist[hist["data"] >= pd.Timestamp(referencia)].head(horizonte)
    hist = hist[hist["data"] < pd.Timestamp(referencia)].tail(dias_hist)
else:
    futuro_real = hist.iloc[0:0]
fc = pd.DataFrame(prev["previsao"])
fc["data"] = pd.to_datetime(fc["data"])
resumo = prev["resumo"]

st.title("Previsão de consumo e custo de infraestrutura cloud")
st.caption(f"Previsão gerada a partir de {prev['gerado_a_partir_de']} · último dia observado: {saude['ultima_data']}")
if prev.get("simulacao"):
    st.info(
        f"Modo simulação: previsão como se hoje fosse {prev['gerado_a_partir_de']}. A linha pontilhada "
        "mostra o custo que de fato ocorreu depois dessa data. (O modelo em produção já viu esse período "
        "no treino; a avaliação sem vazamento está no relatório.)"
    )

# --- KPIs ---------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Custo previsto amanhã",
    usd(resumo["custo_amanha"]),
    f"{(resumo['custo_amanha'] / resumo['media_diaria_ultimos_7_dias'] - 1) * 100:+.1f}% vs média 7d",
    delta_color="inverse",
)
k2.metric(
    "Próxima semana (7 dias)",
    usd(resumo["custo_proxima_semana"]),
    f"{resumo['variacao_semana_vs_recente_pct']:+.1f}% vs semana recente",
    delta_color="inverse",
)
k3.metric(f"Custo projetado ({horizonte} dias)", usd(resumo["custo_projetado_periodo"]))
k4.metric(
    "Tendência no período",
    f"{resumo['variacao_tendencia_periodo_pct']:+.2f}%",
    help="Variação do componente de tendência do Prophet (sem sazonalidade) ao longo do horizonte.",
)

# --- Alertas ------------------------------------------------------------------
st.subheader("Alertas")
if prev["alertas"]:
    for a in prev["alertas"]:
        texto = (
            f"{COR_SEVERIDADE.get(a['severidade'], '⚪')} **{a['tipo'].replace('_', ' ').title()}** — {a['mensagem']}"
        )
        (st.error if a["severidade"] == "CRITICO" else st.warning if a["severidade"] == "ATENCAO" else st.info)(texto)
else:
    st.success("Nenhum alerta para o horizonte selecionado.")
recentes = [a for a in chamar("/alertas", limite=20) if a["tipo"] in ("ANOMALIA_DE_CUSTO", "CUSTO_ABAIXO_DO_ESPERADO")]
for a in recentes[:3]:
    st.error(f"{COR_SEVERIDADE.get(a['severidade'], '⚪')} **Dia registrado fora do esperado** — {a['mensagem']}")

# --- Histórico + previsão + tendência ------------------------------------------
st.subheader("Custo diário: histórico, previsão e tendência")
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist["data"], y=hist["custo_usd"], name="Custo real", line=dict(color=CINZA, width=1.4)))
fig.add_trace(
    go.Scatter(
        x=pd.concat([fc["data"], fc["data"][::-1]]),
        y=pd.concat([fc["custo_limite_superior"], fc["custo_limite_inferior"][::-1]]),
        fill="toself",
        fillcolor="rgba(37,99,235,0.15)",
        line=dict(width=0),
        name="Intervalo de 90%",
        hoverinfo="skip",
    )
)
fig.add_trace(go.Scatter(x=fc["data"], y=fc["custo_previsto"], name="Previsão", line=dict(color=AZUL, width=2.2)))
fig.add_trace(go.Scatter(x=fc["data"], y=fc["tendencia"], name="Tendência", line=dict(color=VERMELHO, dash="dash")))
if len(futuro_real):
    fig.add_trace(
        go.Scatter(
            x=futuro_real["data"],
            y=futuro_real["custo_usd"],
            name="Custo que ocorreu",
            line=dict(color=CINZA, dash="dot", width=1.4),
        )
    )
fig.update_layout(
    height=420,
    margin=dict(l=10, r=10, t=10, b=10),
    yaxis_title="USD/dia",
    legend=dict(orientation="h", y=1.08),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# --- Consumo histórico e previsto ----------------------------------------------
st.subheader("Consumo de recursos")
c1, c2 = st.columns(2)
with c1:
    f2 = make_subplots(specs=[[{"secondary_y": True}]])
    f2.add_trace(
        go.Scatter(
            x=hist["data"], y=hist["requisicoes"], name="Requisições (real)", line=dict(color="#6366f1", width=1.2)
        )
    )
    if "requisicoes_previstas" in fc:
        f2.add_trace(
            go.Scatter(
                x=fc["data"],
                y=fc["requisicoes_previstas"],
                name="Requisições (prev.)",
                line=dict(color="#6366f1", dash="dot"),
            )
        )
    f2.add_trace(
        go.Scatter(
            x=hist["data"], y=hist["instancias_ativas"], name="Instâncias (real)", line=dict(color="#0ea5e9", width=1.2)
        ),
        secondary_y=True,
    )
    if "instancias_previstas" in fc:
        f2.add_trace(
            go.Scatter(
                x=fc["data"],
                y=fc["instancias_previstas"],
                name="Instâncias (prev.)",
                line=dict(color="#0ea5e9", dash="dot"),
            ),
            secondary_y=True,
        )
    f2.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=30, b=10),
        title="Tráfego e capacidade",
        legend=dict(orientation="h", y=-0.2),
    )
    f2.update_yaxes(title_text="requisições/dia", secondary_y=False)
    f2.update_yaxes(title_text="instâncias", secondary_y=True)
    st.plotly_chart(f2, use_container_width=True)
with c2:
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=hist["data"], y=hist["cpu_pct"], name="CPU %", line=dict(color="#f97316", width=1.2)))
    f3.add_trace(
        go.Scatter(x=hist["data"], y=hist["memoria_pct"], name="Memória %", line=dict(color="#a855f7", width=1.2))
    )
    f3.add_hline(y=80, line_dash="dash", line_color=VERMELHO, annotation_text="80% (saturação)")
    f3.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=30, b=10),
        title="Utilização média das instâncias",
        yaxis_range=[0, 100],
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(f3, use_container_width=True)

# --- Tabela de previsão + custo projetado por semana ----------------------------
st.subheader("Custo projetado")
semanal = fc.set_index("data")["custo_previsto"].resample("W-SUN").sum().reset_index()
semanal["semana"] = semanal["data"].apply(lambda d: f"{(d - timedelta(days=6)):%d/%m} a {d:%d/%m}")
c3, c4 = st.columns([1, 1])
with c3:
    f4 = go.Figure(
        go.Bar(
            x=semanal["semana"],
            y=semanal["custo_previsto"],
            marker_color=AZUL,
            text=semanal["custo_previsto"].map(lambda v: f"{v:,.0f}"),
            textposition="outside",
        )
    )
    f4.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        title="Custo previsto por semana (USD) — 1ª e última podem ser parciais",
    )
    st.plotly_chart(f4, use_container_width=True)
with c4:
    tabela = fc[["data", "custo_previsto", "custo_limite_inferior", "custo_limite_superior"]].copy()
    tabela["data"] = tabela["data"].dt.strftime("%d/%m/%Y")
    tabela.columns = ["Data", "Previsto (USD)", "Mín. 90% (USD)", "Máx. 90% (USD)"]
    st.dataframe(tabela, hide_index=True, height=320, use_container_width=True)

# --- Operação: registrar um dia, monitoramento e auditoria ---------------------
st.subheader("Operação")
t1, t2, t3 = st.tabs(["Registrar consumo do dia", "Monitoramento do modelo", "Histórico de alertas e previsões"])
with t1:
    ultimo = hist.iloc[-1]
    with st.form("obs"):
        a, b, c = st.columns(3)
        data = a.date_input("Data", value=pd.Timestamp(saude["ultima_data"]) + timedelta(days=1))
        custo = a.number_input("Custo do dia (USD)", min_value=0.01, value=float(round(resumo["custo_amanha"], 2)))
        req = b.number_input("Requisições", min_value=0.0, value=float(round(ultimo["requisicoes"])))
        inst = b.number_input("Instâncias ativas", min_value=1.0, value=float(ultimo["instancias_ativas"]))
        cpu = c.number_input("CPU %", 0.0, 100.0, float(round(ultimo["cpu_pct"], 1)))
        mem = c.number_input("Memória %", 0.0, 100.0, float(round(ultimo["memoria_pct"], 1)))
        arm = a.number_input("Armazenamento (GB)", min_value=0.0, value=float(round(ultimo["armazenamento_gb"], 1)))
        if st.form_submit_button("Registrar e verificar anomalia"):
            r = httpx.post(
                f"{API}/observacoes",
                timeout=60,
                json={
                    "data": str(data),
                    "custo_usd": custo,
                    "requisicoes": req,
                    "instancias_ativas": inst,
                    "cpu_pct": cpu,
                    "memoria_pct": mem,
                    "armazenamento_gb": arm,
                },
            )
            if r.status_code == 201:
                resp = r.json()
                st.cache_data.clear()
                if resp["alerta"]:
                    st.error(f"{resp['alerta']['tipo']}: {resp['alerta']['mensagem']}")
                else:
                    st.success(
                        f"Dia registrado. Custo dentro do esperado (previsto {usd(resp['custo_previsto'])}, "
                        f"limite {usd(resp['limite_superior'])})."
                    )
            else:
                st.error(f"Erro {r.status_code}: {r.json().get('detail')}")
with t2:
    mon = chamar("/monitoramento")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAPE em produção", "—" if mon["mape_producao_pct"] is None else f"{mon['mape_producao_pct']:.2f}%")
    m2.metric("MAPE no teste (referência)", f"{info['metricas_teste']['MAPE_%']:.2f}%")
    m3.metric("Dias avaliados", mon["dias_avaliados"])
    if mon["recomenda_retreino"]:
        st.warning(f"Erro recente acima de {mon['limite_pct']}%: recomenda-se retreinar (`python -m src.pipeline`).")
    else:
        st.caption(mon["mensagem"])
with t3:
    st.dataframe(pd.DataFrame(chamar("/alertas", limite=50)), hide_index=True, use_container_width=True)
    st.dataframe(pd.DataFrame(chamar("/previsoes", limite=50)), hide_index=True, use_container_width=True)
