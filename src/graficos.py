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
import pandas as pd

from config import (
    ARQ_IMPORTANCIAS,
    ARQ_METRICAS,
    ARQ_PREVISOES,
    COR_REAL,
    DATASET_BRUTO,
    DATASET_PREPARADO,
    DIAS_TESTE,
    MODELOS,
    PASTA_FIGURAS,
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
    ax.plot(
        df["data"], df["custo_usd"], color=COR_REAL, linewidth=2, label="Real", marker="o", markersize=3
    )
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
            ax.text(
                b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9
            )

    fig.suptitle("Comparação de métricas de erro entre os modelos (menor = melhor)")
    fig.tight_layout()
    salvar(fig, "04_comparacao_metricas.png")


def fig_importancia_features():
    imp = pd.read_csv(ARQ_IMPORTANCIAS, index_col=0)
    coluna = imp.columns[0]
    imp = imp.sort_values(coluna)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(imp.index, imp[coluna], color=cor("gb"))
    ax.set_title("Importância dos atributos — Gradient Boosting")
    ax.set_xlabel("Importância relativa")
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


def main():
    garantir_pastas()
    print("Gerando figuras...")
    fig_serie_historica()
    fig_qualidade_dados()
    fig_previsao_vs_real()
    fig_comparacao_metricas()
    fig_importancia_features()
    fig_residuos()
    print(f"Figuras geradas em {PASTA_FIGURAS}")


if __name__ == "__main__":
    main()
