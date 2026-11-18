"""
Configuração dos testes.

As variáveis de ambiente são definidas ANTES de qualquer import de `src`,
para que os testes usem uma pasta de modelos e um banco temporários e nunca
toquem nos artefatos reais do projeto (models/, data/cloudcast.db, mlruns/).
"""

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="cloudcast_testes_"))
os.environ["CLOUDCAST_MODELS_DIR"] = str(_TMP / "models")
os.environ["CLOUDCAST_DB_PATH"] = str(_TMP / "teste.db")
os.environ["CLOUDCAST_DISABLE_MLFLOW"] = "1"

import logging  # noqa: E402

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src import gerar_dados, preparar_dados  # noqa: E402
from src.config import DATASET_PREPARADO, PASTA_MODELOS  # noqa: E402

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


@pytest.fixture(scope="session")
def bruto() -> pd.DataFrame:
    return gerar_dados.gerar()


@pytest.fixture(scope="session")
def preparado(bruto) -> pd.DataFrame:
    df, _ = preparar_dados.preparar(bruto.copy())
    return df


@pytest.fixture(scope="session")
def modelo_treinado(preparado):
    """Treina e registra um modelo de produção na pasta temporária (≈3 s)."""
    from src.modelagem import ajustar_prophet
    from src.registro import salvar_versao

    if not DATASET_PREPARADO.exists():  # checkout limpo: a API popula o banco a partir deste CSV
        DATASET_PREPARADO.parent.mkdir(parents=True, exist_ok=True)
        preparado.to_csv(DATASET_PREPARADO, index=False)

    series = {"custo": "custo_usd", "requisicoes": "requisicoes", "instancias": "instancias_ativas"}
    modelos = {n: ajustar_prophet(preparado[["data", c]], coluna=c) for n, c in series.items()}
    meta = {
        "modelo": "cloudcast-custo",
        "algoritmo": "Prophet",
        "hiperparametros": {},
        "treinado_ate": f"{preparado['data'].max():%Y-%m-%d}",
        "n_observacoes": len(preparado),
        "dados_sha256": "teste",
        "git_commit": "teste",
        "metricas_teste": {"MAE": 8.0, "RMSE": 10.0, "R2": 0.1, "MAPE_%": 8.0},
        "metricas_teste_consumo": {},
    }
    versao = salvar_versao(modelos, meta, promover=True, pasta=PASTA_MODELOS)
    return versao
