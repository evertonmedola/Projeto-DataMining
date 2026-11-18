"""
Configuração central do projeto: caminhos e parâmetros compartilhados.

Os caminhos são resolvidos a partir da localização deste arquivo, de forma
que o projeto funcione em qualquer máquina/sistema operacional, e os
parâmetros (sementes, janelas, grades de busca, limites de alerta) vêm de
params.yaml, na raiz do repositório.

Algumas pastas podem ser sobrescritas por variável de ambiente (útil nos
testes e no Docker): CLOUDCAST_MODELS_DIR e CLOUDCAST_DB_PATH.
"""

import os
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent.parent

with open(RAIZ / "params.yaml", encoding="utf-8") as _f:
    PARAMS = yaml.safe_load(_f)

PASTA_DADOS = RAIZ / "data"
PASTA_SAIDA = RAIZ / "outputs"
PASTA_FIGURAS = PASTA_SAIDA / "figuras"
PASTA_MODELOS = Path(os.environ.get("CLOUDCAST_MODELS_DIR", RAIZ / "models"))
CAMINHO_DB = Path(os.environ.get("CLOUDCAST_DB_PATH", PASTA_DADOS / "cloudcast.db"))
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(RAIZ / 'mlruns' / 'mlflow.db').as_posix()}")
MLFLOW_ARTEFATOS = (RAIZ / "mlruns" / "artifacts").as_uri()

DATASET_BRUTO = PASTA_DADOS / "cloud_infra_bruto.csv"
DATASET_PREPARADO = PASTA_DADOS / "cloud_infra_preparado.csv"
GABARITO_INCIDENTES = PASTA_DADOS / "incidentes_injetados.csv"

ARQ_QUALIDADE = PASTA_SAIDA / "relatorio_qualidade.json"
ARQ_PREPARACAO = PASTA_SAIDA / "relatorio_preparacao.json"
ARQ_PREVISOES = PASTA_SAIDA / "previsoes_teste.csv"
ARQ_METRICAS = PASTA_SAIDA / "metricas.json"
ARQ_CV = PASTA_SAIDA / "resultados_cv.json"
ARQ_TEMPOS = PASTA_SAIDA / "tempos_cv.json"  # não versionado: muda a cada execução
ARQ_HORIZONTE = PASTA_SAIDA / "metricas_horizonte.csv"
ARQ_IMPORTANCIAS = PASTA_SAIDA / "importancia_features_gb.csv"
ARQ_MELHORES_PARAMS = PASTA_SAIDA / "melhores_params.json"

RANDOM_SEED = PARAMS["seed"]

# Tamanho do conjunto de teste (dias no fim da série, sem embaralhamento).
# Compartilhado entre modelagem e gráficos para não haver divergência.
DIAS_TESTE = PARAMS["avaliacao"]["dias_teste"]

# Modelos avaliados: chave interna -> (rótulo no relatório, cor no gráfico).
# A chave interna também nomeia a coluna de previsão ("previsao_<chave>").
MODELOS = {
    "baseline": ("Baseline sazonal (7 dias)", "#9ca3af"),
    "sarimax": ("SARIMAX (ARIMA sazonal)", "#ef4444"),
    "prophet": ("Prophet", "#3b82f6"),
    "gb": ("Gradient Boosting", "#10b981"),
}

COR_REAL = "#1f2937"

# Colunas de consumo de recursos (entrada do sistema) e variável-alvo.
COLUNAS_BASE = [
    "data",
    "requisicoes",
    "instancias_ativas",
    "cpu_pct",
    "memoria_pct",
    "armazenamento_gb",
    "custo_usd",
]
ALVO = "custo_usd"


def rotulo(chave: str) -> str:
    return MODELOS[chave][0]


def cor(chave: str) -> str:
    return MODELOS[chave][1]


def garantir_pastas() -> None:
    """Cria as pastas de saída caso ainda não existam."""
    for pasta in (PASTA_DADOS, PASTA_SAIDA, PASTA_FIGURAS, PASTA_MODELOS):
        pasta.mkdir(parents=True, exist_ok=True)
