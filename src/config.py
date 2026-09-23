"""
Configuração central do projeto: caminhos e constantes compartilhadas.

Os caminhos são resolvidos a partir da localização deste arquivo, de forma
que o projeto funcione em qualquer máquina/sistema operacional.
"""

from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

PASTA_DADOS = RAIZ / "data"
PASTA_SAIDA = RAIZ / "outputs"
PASTA_FIGURAS = PASTA_SAIDA / "figuras"

DATASET_BRUTO = PASTA_DADOS / "cloud_infra_bruto.csv"
DATASET_PREPARADO = PASTA_DADOS / "cloud_infra_preparado.csv"

ARQ_PREVISOES = PASTA_SAIDA / "previsoes_teste.csv"
ARQ_METRICAS = PASTA_SAIDA / "metricas.json"
ARQ_IMPORTANCIAS = PASTA_SAIDA / "importancia_features_gb.csv"

RANDOM_SEED = 42

# Tamanho do conjunto de teste (dias no fim da série, sem embaralhamento).
# Compartilhado entre modelagem e gráficos para não haver divergência.
DIAS_TESTE = 60

# Modelos avaliados: chave interna -> (rótulo no relatório, cor no gráfico).
# A chave interna também nomeia a coluna de previsão ("previsao_<chave>").
MODELOS = {
    "baseline": ("Baseline sazonal (7 dias)", "#9ca3af"),
    "sarimax": ("SARIMAX (ARIMA sazonal)", "#ef4444"),
    "prophet": ("Prophet", "#3b82f6"),
    "gb": ("Gradient Boosting", "#10b981"),
}

COR_REAL = "#1f2937"


def rotulo(chave: str) -> str:
    return MODELOS[chave][0]


def cor(chave: str) -> str:
    return MODELOS[chave][1]


def garantir_pastas() -> None:
    """Cria as pastas de saída caso ainda não existam."""
    for pasta in (PASTA_DADOS, PASTA_SAIDA, PASTA_FIGURAS):
        pasta.mkdir(parents=True, exist_ok=True)
