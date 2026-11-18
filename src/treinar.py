"""
Treino do modelo de produção (fase "Deployment" do CRISP-DM).

Depois que a modelagem escolheu o algoritmo e os hiperparâmetros (com o
holdout usado uma única vez), o modelo final é reajustado em TODO o
histórico -- inclusive o período de teste, que agora é passado conhecido --
e registrado como uma nova versão.

São treinadas três séries com a mesma configuração do Prophet:
  - custo_usd          (alvo principal, avaliado no relatório)
  - requisicoes        (consumo: tráfego)
  - instancias_ativas  (consumo: capacidade)

Regra de promoção: a versão só vai para produção se o MAPE do Prophet no
holdout (outputs/metricas.json) estiver dentro da meta de negócio (<= 10%).
"""

import hashlib
import json
import logging
import sys
import warnings

import numpy as np
import pandas as pd

from src import rastreamento as rt
from src.config import (
    ARQ_MELHORES_PARAMS,
    ARQ_METRICAS,
    DATASET_PREPARADO,
    DIAS_TESTE,
    PARAMS,
    PASTA_MODELOS,
    garantir_pastas,
)
from src.modelagem import ajustar_prophet, calcular_metricas
from src.registro import ler_registro, salvar_versao

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

META_MAPE = 10.0
SERIES = {"custo": "custo_usd", "requisicoes": "requisicoes", "instancias": "instancias_ativas"}


def hash_dados(caminho) -> str:
    return hashlib.sha256(open(caminho, "rb").read()).hexdigest()[:16]


def avaliar_serie_auxiliar(df: pd.DataFrame, coluna: str, hp: dict, passo: int = 7) -> dict:
    """Walk-forward simples no holdout para as séries de consumo (mesmo protocolo do custo)."""
    dev, teste = df.iloc[:-DIAS_TESTE], df.iloc[-DIAS_TESTE:]
    historico = dev[["data", coluna]].copy()
    prev = np.empty(len(teste))
    for i in range(0, len(teste), passo):
        bloco = teste.iloc[i : i + passo]
        modelo = ajustar_prophet(historico, coluna=coluna, **hp)
        prev[i : i + len(bloco)] = modelo.predict(pd.DataFrame({"ds": bloco["data"]}))["yhat"].to_numpy()
        historico = pd.concat([historico, bloco[["data", coluna]]], ignore_index=True)
    return calcular_metricas(teste[coluna], prev)


def main():
    garantir_pastas()
    df = pd.read_csv(DATASET_PREPARADO, parse_dates=["data"]).sort_values("data").reset_index(drop=True)
    hp = json.loads(ARQ_MELHORES_PARAMS.read_text(encoding="utf-8"))["prophet"]
    metricas_modelagem = json.loads(ARQ_METRICAS.read_text(encoding="utf-8"))
    metricas_custo = metricas_modelagem["Prophet"]

    # Idempotência: se a versão em produção já foi treinada com os mesmos dados,
    # hiperparâmetros e métricas, não registra uma cópia. Sem isso, cada
    # `docker compose up` (que roda o pipeline) criaria models/v2, v3, ...
    # idênticas. Use `python -m src.treinar --forcar` para registrar mesmo assim.
    em_producao = ler_registro().get("producao")
    if em_producao and "--forcar" not in sys.argv:
        atual = json.loads((PASTA_MODELOS / f"v{em_producao}" / "metadata.json").read_text(encoding="utf-8"))
        if (
            atual.get("dados_sha256") == hash_dados(DATASET_PREPARADO)
            and atual.get("hiperparametros") == hp
            and atual.get("metricas_teste") == metricas_custo
        ):
            print(f"Modelo v{em_producao} já está em produção com os mesmos dados e parâmetros; nada a registrar.")
            return

    print(f"Treinando Prophet final em {len(df)} dias ({df['data'].min():%Y-%m-%d} a {df['data'].max():%Y-%m-%d})")
    print(f"Hiperparâmetros (escolhidos no backtest): {hp}")

    metricas_aux = {}
    for nome, coluna in SERIES.items():
        if nome != "custo":
            metricas_aux[nome] = avaliar_serie_auxiliar(df, coluna, hp)
            print(f"  holdout {coluna:18s} -> {metricas_aux[nome]}")

    modelos = {nome: ajustar_prophet(df[["data", coluna]], coluna=coluna, **hp) for nome, coluna in SERIES.items()}

    promover = metricas_custo["MAPE_%"] <= META_MAPE
    metadata = {
        "modelo": PARAMS["produto"]["modelo_nome"],
        "algoritmo": "Prophet",
        "hiperparametros": hp,
        "treinado_ate": f"{df['data'].max():%Y-%m-%d}",
        "n_observacoes": len(df),
        "dados_sha256": hash_dados(DATASET_PREPARADO),
        "git_commit": rt.commit_git(),
        "metricas_teste": metricas_custo,
        "metricas_teste_consumo": metricas_aux,
        "meta_mape": META_MAPE,
        "custo_medio_ultimos_30d": round(float(df["custo_usd"].tail(30).mean()), 2),
    }
    versao = salvar_versao(modelos, metadata, promover=promover)
    status = "PROMOVIDA para produção" if promover else f"NÃO promovida (MAPE > {META_MAPE}%)"
    print(f"Versão v{versao} registrada -> {status}")

    rt.preparar_experimento("cloudcast-treino")
    with rt.execucao(f"treino-v{versao}", tags={"etapa": "treino", "versao": str(versao)}):
        rt.registrar_params({**hp, "treinado_ate": metadata["treinado_ate"], "dados_sha256": metadata["dados_sha256"]})
        rt.registrar_metricas(metricas_custo, "teste_custo_")
        for nome, m in metricas_aux.items():
            rt.registrar_metricas(m, f"teste_{nome}_")
        if rt.ativo():
            try:  # registra também no MLflow Model Registry, com o mesmo número de versão
                import mlflow
                import mlflow.prophet

                mlflow.prophet.log_model(
                    modelos["custo"], name="modelo_custo", registered_model_name=PARAMS["produto"]["modelo_nome"]
                )
            except Exception as exc:  # pragma: no cover
                print(f"(aviso) Registro no MLflow Model Registry ignorado: {exc}")


if __name__ == "__main__":
    main()
