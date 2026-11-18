"""
Rastreamento de experimentos com MLflow.

Toda execução de modelagem e de treino registra parâmetros, métricas e
artefatos no MLflow (backend SQLite em mlruns/mlflow.db, com a interface em
`mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`).

O rastreamento é opcional em tempo de execução: se o MLflow não estiver
instalado, ou se CLOUDCAST_DISABLE_MLFLOW=1 (usado nos testes), as funções
viram no-ops e o pipeline continua funcionando normalmente.
"""

import contextlib
import os
import subprocess

from src.config import MLFLOW_ARTEFATOS, MLFLOW_URI, RAIZ

_HABILITADO = os.environ.get("CLOUDCAST_DISABLE_MLFLOW", "0") != "1"

try:  # pragma: no cover - depende do ambiente
    if _HABILITADO:
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        import logging

        import mlflow

        logging.getLogger("mlflow").setLevel(logging.ERROR)
    else:
        mlflow = None
except ImportError:  # pragma: no cover
    mlflow = None


def ativo() -> bool:
    return mlflow is not None


def commit_git() -> str:
    """Hash do commit atual, para amarrar cada execução a uma versão do código."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "sem-git"


def preparar_experimento(nome: str) -> None:
    if not ativo():
        return
    (RAIZ / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)
    if mlflow.get_experiment_by_name(nome) is None:
        mlflow.create_experiment(nome, artifact_location=MLFLOW_ARTEFATOS)
    mlflow.set_experiment(nome)


@contextlib.contextmanager
def execucao(nome: str, aninhada: bool = False, tags: dict | None = None):
    if not ativo():
        yield None
        return
    with mlflow.start_run(run_name=nome, nested=aninhada) as run:
        mlflow.set_tags({"git_commit": commit_git(), **(tags or {})})
        yield run


def registrar_params(params: dict) -> None:
    if ativo():
        mlflow.log_params({k: str(v) for k, v in params.items()})


def registrar_metricas(metricas: dict, prefixo: str = "") -> None:
    if ativo():
        mlflow.log_metrics({f"{prefixo}{k}".replace("%", "pct"): float(v) for k, v in metricas.items()})


def registrar_artefato(caminho) -> None:
    if ativo() and os.path.exists(caminho):
        mlflow.log_artifact(str(caminho))
