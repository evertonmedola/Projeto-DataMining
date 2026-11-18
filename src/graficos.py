"""Geração das figuras usadas no relatório (fases de Preparação, Modelagem e Avaliação).

Nada aqui é chumbado no código: o período do gráfico vem dos próprios dados,
a lista de modelos vem de config.MODELOS e o "melhor modelo" é escolhido pelo
MAE registrado em metricas.json (antes o Prophet estava fixo no título da
figura de resíduos, o que ficaria errado se outro modelo vencesse).
"""

import json

import matplotlib

matplotlib.use("Agg")  # backend sem janela: o script só grava arquivos

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    ARQ_CV,
    ARQ_HORIZONTE,
    ARQ_IMPORTANCIAS,
    ARQ_METRICAS,
    ARQ_PREVISOES,
    COR_REAL,
    DATASET_BRUTO,
    DATASET_PREPARADO,
    DIAS_TESTE,
    MODELOS,
    PASTA_FIGURAS,
    PASTA_MODELOS,
    cor,
    garantir_pastas,
    rotulo,
)

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 11,
    }
)

COLUNAS_AUSENTES = ["requisicoes", "cpu_pct", "memoria_pct", "armazenamento_gb"]


def salvar(fig, nome: str) -> None:
    caminho = PASTA_FIGURAS / nome
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"  {caminho.name}")


def carregar_metricas() -> dict:
    with open(ARQ_METRICAS, encoding="utf-8") as f:
        return json.load(f)


def melhor_modelo(metricas: dict) -> str:
    """Chave do modelo com menor MAE, ignorando o baseline ingênuo."""
    candidatos = [c for c in MODELOS if c != "baseline" and rotulo(c) in metricas]
    return min(candidatos, key=lambda c: metricas[rotulo(c)]["MAE"])


def fig_serie_historica():
    df = pd.read_csv(DATASET_PREPARADO, parse_dates=["data"])
    corte_teste = df["data"].iloc[-DIAS_TESTE]
    ano_inicio, ano_fim = df["data"].dt.year.min(), df["data"].dt.year.max()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["data"], df["custo_usd"], color=COR_REAL, linewidth=1.1, label="Custo diário (USD)")
    ax.axvline(
        corte_teste,
        color="#9333ea",
        linestyle="--",
        linewidth=1.3,
        label=f"Início do período de teste ({DIAS_TESTE} dias)",
    )
    ax.set_title(f"Custo diário de infraestrutura cloud — histórico ({ano_inicio}–{ano_fim})")
    ax.set_ylabel("Custo (USD)")
    ax.set_xlabel("Data")
    ax.legend(loc="upper left")
    fig.tight_layout()
    salvar(fig, "01_serie_historica_custo.png")


def fig_qualidade_dados():
    df_bruto = pd.read_csv(DATASET_BRUTO, parse_dates=["data"])
    df_prep = pd.read_csv(DATASET_PREPARADO, parse_dates=["data"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ausentes = df_bruto[COLUNAS_AUSENTES].isna().sum()
    axes[0].bar(ausentes.index, ausentes.values, color="#f59e0b")
    axes[0].set_title("Valores ausentes no dataset bruto")
    axes[0].set_ylabel("Nº de dias")
    axes[0].tick_params(axis="x", rotation=25)

    incidentes = int(df_prep["houve_incidente"].sum())
    normais = len(df_prep) - incidentes
    axes[1].pie(
        [normais, incidentes],
        labels=["Dias normais", "Dias com incidente\n(outlier tratado)"],
        autopct="%1.1f%%",
        colors=["#93c5fd", "#ef4444"],
        startangle=90,
    )
    axes[1].set_title("Dias marcados como incidente após tratamento")

    fig.tight_layout()
    salvar(fig, "02_qualidade_dados.png")


def fig_previsao_vs_real():
    df = pd.read_csv(ARQ_PREVISOES, parse_dates=["data"])

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["data"], df["custo_usd"], color=COR_REAL, linewidth=2, label="Real", marker="o", markersize=3)
    for chave in MODELOS:
        coluna = f"previsao_{chave}"
        if coluna not in df.columns:
            continue
        # O baseline entra pontilhado e discreto: é régua, não concorrente.
        e_baseline = chave == "baseline"
        ax.plot(
            df["data"],
            df[coluna],
            color=cor(chave),
            linewidth=1.0 if e_baseline else 1.4,
            linestyle=":" if e_baseline else "-",
            label=rotulo(chave),
            alpha=0.85,
        )
    ax.set_title(f"Previsão de custo diário — período de teste (últimos {DIAS_TESTE} dias)")
    ax.set_ylabel("Custo (USD)")
    ax.set_xlabel("Data")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.tight_layout()
    salvar(fig, "03_previsao_vs_real.png")


def fig_comparacao_metricas():
    metricas = carregar_metricas()
    chaves = [c for c in MODELOS if rotulo(c) in metricas]
    nomes = [rotulo(c).replace(" (", "\n(") for c in chaves]
    cores = [cor(c) for c in chaves]

    painéis = [("MAE", "MAE (USD)"), ("RMSE", "RMSE (USD)"), ("MAPE_%", "MAPE (%)")]
    fig, axes = plt.subplots(1, len(painéis), figsize=(13, 4.6))

    for ax, (chave_metrica, titulo) in zip(axes, painéis):
        valores = [metricas[rotulo(c)][chave_metrica] for c in chaves]
        barras = ax.bar(nomes, valores, color=cores)
        ax.set_title(titulo)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_ylim(0, max(valores) * 1.18)
        for b, v in zip(barras, valores):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Comparação de métricas de erro entre os modelos (menor = melhor)")
    fig.tight_layout()
    salvar(fig, "04_comparacao_metricas.png")


def fig_importancia_features():
    imp = pd.read_csv(ARQ_IMPORTANCIAS, index_col=0).sort_values("permutacao_media")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    axes[0].barh(imp.index, imp["importancia"], color=cor("gb"))
    axes[0].set_title("Importância por impureza (treino)")
    axes[0].set_xlabel("Importância relativa")
    axes[1].barh(imp.index, imp["permutacao_media"], xerr=imp["permutacao_desvio"], color="#047857", alpha=0.85)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Importância por permutação (teste, 30 repetições)")
    axes[1].set_xlabel("Aumento do MAE da variação diária ao embaralhar")
    fig.suptitle("Gradient Boosting — quais atributos o modelo usa")
    fig.tight_layout()
    salvar(fig, "05_importancia_features.png")


def fig_residuos():
    df = pd.read_csv(ARQ_PREVISOES, parse_dates=["data"])
    chave = melhor_modelo(carregar_metricas())
    nome = rotulo(chave)
    residuo = df["custo_usd"] - df[f"previsao_{chave}"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df["data"], residuo, color=cor(chave), marker="o", markersize=3)
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].axhline(
        residuo.mean(),
        color="#ef4444",
        linewidth=1.0,
        linestyle=":",
        label=f"Viés médio: {residuo.mean():+.2f} USD",
    )
    axes[0].set_title(f"Resíduos do melhor modelo ({nome}) ao longo do tempo")
    axes[0].set_ylabel("Real − Previsto (USD)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(loc="upper left", fontsize=9)

    axes[1].hist(residuo, bins=15, color=cor(chave), edgecolor="white")
    axes[1].set_title(f"Distribuição dos resíduos — {nome}")
    axes[1].set_xlabel("Resíduo (USD)")

    fig.tight_layout()
    salvar(fig, f"06_residuos_{chave}.png")


def fig_consumo_eda():
    """EDA: as séries de consumo de recursos que formam o custo."""
    df = pd.read_csv(DATASET_PREPARADO, parse_dates=["data"])
    paineis = [
        ("requisicoes", "Requisições/dia", "#6366f1"),
        ("instancias_ativas", "Instâncias ativas", "#0ea5e9"),
        ("cpu_pct", "CPU média (%)", "#f97316"),
        ("memoria_pct", "Memória média (%)", "#a855f7"),
        ("armazenamento_gb", "Armazenamento (GB)", "#64748b"),
        ("custo_usd", "Custo diário (USD)", COR_REAL),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 8.5), sharex=True)
    for ax, (col, titulo, c) in zip(axes.ravel(), paineis):
        ax.plot(df["data"], df[col], color=c, linewidth=0.7, alpha=0.6)
        ax.plot(df["data"], df[col].rolling(28, center=True).mean(), color=c, linewidth=2)
        ax.set_title(titulo, fontsize=11)
    for ax in axes[-1]:
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Consumo de recursos e custo — série diária (linha grossa = média móvel de 28 dias)")
    fig.tight_layout()
    salvar(fig, "07_eda_consumo.png")


def fig_correlacao():
    df = pd.read_csv(DATASET_PREPARADO, parse_dates=["data"])
    cols = [
        "requisicoes",
        "instancias_ativas",
        "cpu_pct",
        "memoria_pct",
        "armazenamento_gb",
        "custo_usd",
        "fim_de_semana",
        "temporada_alta",
        "custo_lag_1",
        "custo_lag_7",
    ]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(cols)), cols, fontsize=9)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            ax.text(
                j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white" if abs(v) > 0.6 else "black"
            )
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Matriz de correlação (Pearson)")
    fig.tight_layout()
    salvar(fig, "08_correlacao.png")


def fig_sazonalidade():
    df = pd.read_csv(DATASET_PREPARADO, parse_dates=["data"])
    nomes_dia = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]
    nomes_mes = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    axes[0].boxplot([df.loc[df["dia_semana"] == d, "custo_usd"] for d in range(7)], tick_labels=nomes_dia)
    axes[0].set_title("Custo por dia da semana")
    axes[0].set_ylabel("Custo (USD)")
    axes[1].boxplot([df.loc[df["mes"] == m, "custo_usd"] for m in range(1, 13)], tick_labels=nomes_mes)
    axes[1].set_title("Custo por mês")
    fig.tight_layout()
    salvar(fig, "09_sazonalidade.png")


def fig_validacao_cruzada():
    if not ARQ_CV.exists():
        return
    cv = json.loads(ARQ_CV.read_text(encoding="utf-8"))
    chaves = [c for c in MODELOS if c in cv]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, c in enumerate(chaves):
        folds = [f["MAE"] for f in cv[c]["melhor"]["folds"]]
        ax.scatter(np.full(len(folds), i) + np.linspace(-0.12, 0.12, len(folds)), folds, color=cor(c), alpha=0.7, s=28)
        m, d = cv[c]["melhor"]["MAE_media"], cv[c]["melhor"]["MAE_desvio"]
        ax.errorbar(i + 0.28, m, yerr=d, fmt="D", color=cor(c), capsize=6, markersize=7)
        ax.text(i + 0.36, m, f"{m:.2f} ± {d:.2f}", va="center", fontsize=9)
    ax.set_xticks(range(len(chaves)), [rotulo(c) for c in chaves], fontsize=9)
    ax.set_xlim(-0.5, len(chaves) - 0.1)
    ax.set_ylabel("MAE na dobra (USD)")
    ax.set_title("Backtest temporal (5 dobras de 30 dias) — melhor configuração de cada modelo")
    fig.tight_layout()
    salvar(fig, "10_validacao_cruzada.png")


def fig_horizonte():
    if not ARQ_HORIZONTE.exists():
        return
    h = pd.read_csv(ARQ_HORIZONTE)
    fig, ax = plt.subplots(figsize=(9, 4.3))
    for c in MODELOS:
        sub = h[h["modelo"] == rotulo(c)]
        if sub.empty:
            continue
        ax.plot(
            sub["horizonte"],
            sub["MAPE_pct"],
            marker="o",
            color=cor(c),
            label=rotulo(c),
            linestyle=":" if c == "baseline" else "-",
        )
    ax.set_xlabel("Horizonte (dias à frente)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Erro por horizonte de previsão — holdout Nov-Dez/2024")
    ax.legend(fontsize=9)
    fig.tight_layout()
    salvar(fig, "11_erro_por_horizonte.png")


def fig_componentes_prophet():
    """Interpretação do modelo final: tendência + sazonalidades semanal e anual."""
    try:
        from src.registro import carregar_producao

        modelos, meta = carregar_producao(PASTA_MODELOS)
    except FileNotFoundError:
        return
    m = modelos["custo"]
    futuro = m.make_future_dataframe(periods=0)
    comp = m.predict(futuro)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(comp["ds"], comp["trend"], color=cor("prophet"))
    axes[0].set_title("Tendência (USD/dia)")
    axes[0].tick_params(axis="x", rotation=25)
    semana = comp.assign(d=comp["ds"].dt.dayofweek).groupby("d")["weekly"].mean() * 100
    axes[1].bar(["seg", "ter", "qua", "qui", "sex", "sáb", "dom"], semana, color=cor("prophet"))
    axes[1].set_title("Efeito semanal (% sobre a tendência)")
    ano = comp.assign(doy=comp["ds"].dt.dayofyear).groupby("doy")["yearly"].mean() * 100
    axes[2].plot(ano.index, ano.values, color=cor("prophet"))
    axes[2].set_title("Efeito anual (% sobre a tendência)")
    axes[2].set_xlabel("Dia do ano")
    fig.suptitle(f"Componentes do Prophet em produção (v{meta['versao']}, sazonalidade multiplicativa)")
    fig.tight_layout()
    salvar(fig, "12_componentes_prophet.png")


def main():
    garantir_pastas()
    print("Gerando figuras...")
    fig_serie_historica()
    fig_qualidade_dados()
    fig_previsao_vs_real()
    fig_comparacao_metricas()
    fig_importancia_features()
    fig_residuos()
    fig_consumo_eda()
    fig_correlacao()
    fig_sazonalidade()
    fig_validacao_cruzada()
    fig_horizonte()
    fig_componentes_prophet()
    print(f"Figuras geradas em {PASTA_FIGURAS}")


if __name__ == "__main__":
    main()
