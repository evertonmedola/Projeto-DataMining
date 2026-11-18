"""
Mineração e modelagem (fases "Modeling" e "Evaluation" do CRISP-DM).

Tarefa: prever o CUSTO DIÁRIO de infraestrutura cloud (custo_usd).
Compara os três algoritmos sugeridos no enunciado do tema 4 -- SARIMAX
(variante sazonal do ARIMA), Prophet e Gradient Boosting -- contra um
baseline ingênuo sazonal ("o custo de hoje é o do mesmo dia da semana
passada"), que serve de régua: todo modelo precisa vencê-lo.

PROTOCOLO (tudo com divisão TEMPORAL: treina no passado, testa no futuro)
-----------------------------------------------------------------------
  1. Holdout final: os últimos DIAS_TESTE dias (Nov-Dez/2024). Ficam
     trancados até o passo 4 e são usados uma única vez.
  2. Backtest (validação cruzada temporal com janela expansível): no
     restante, N_FOLDS janelas consecutivas de DIAS_POR_FOLD dias; em cada
     dobra o modelo treina com tudo o que vem antes da janela e prevê a
     janela em walk-forward (1 dia à frente).
  3. Busca de hiperparâmetros: grade de params.yaml, avaliada pelo MAE médio
     do backtest. Resultados por dobra em outputs/resultados_cv.json.
  4. Teste: cada modelo, com os melhores hiperparâmetros, treina em todo o
     período de desenvolvimento e prevê o holdout em walk-forward (h = 1).
  5. Multi-horizonte: a partir de origens semanais dentro do holdout, cada
     modelo prevê os 7 dias seguintes de uma vez (h = 1..7), que é o uso
     real do produto ("consumo da próxima semana").

Walk-forward = ao prever o dia D, nenhum modelo enxerga qualquer informação
de D ou posterior -- apenas o histórico até D-1 e o calendário:
  - SARIMAX: estimado uma vez; as observações reais são incorporadas ao
    estado do filtro de Kalman (append com refit=False);
  - Prophet: reajustado a cada `passo_reajuste_prophet` dias;
  - Gradient Boosting: atributos de lag/média móvel já usam só D-1.

A primeira versão deste projeto comparava um GB de 1 dia à frente contra
SARIMAX/Prophet prevendo 60 dias à frente de uma vez, o que penalizava os
modelos de série temporal e produzia R² negativo para todos -- foi a
principal iteração Avaliação -> Modelagem do ciclo CRISP-DM.
"""

import itertools
import json
import logging
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src import rastreamento as rt
from src.config import (
    ARQ_CV,
    ARQ_HORIZONTE,
    ARQ_IMPORTANCIAS,
    ARQ_MELHORES_PARAMS,
    ARQ_METRICAS,
    ARQ_PREVISOES,
    ARQ_TEMPOS,
    DATASET_PREPARADO,
    DIAS_TESTE,
    MODELOS,
    PARAMS,
    RANDOM_SEED,
    garantir_pastas,
    rotulo,
)
from src.preparar_dados import engenharia_atributos

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

_AV = PARAMS["avaliacao"]
N_FOLDS = _AV["n_folds"]
DIAS_POR_FOLD = _AV["dias_por_fold"]
HORIZONTE_MAX = _AV["horizonte_max"]
PASSO_REAJUSTE_PROPHET = _AV["passo_reajuste_prophet"]

# Sazonalidade semanal: o custo de 7 dias atrás é o palpite ingênuo natural,
# porque já preserva o efeito "dia útil x fim de semana".
LAG_BASELINE = "custo_lag_7"

# Atributos do Gradient Boosting: apenas variáveis de calendário e atributos
# ESTACIONÁRIOS (razões/variações). Nenhum atributo em nível absoluto -- ver
# a explicação em ajustar_gb().
COLUNAS_FEATURES_GB = [
    "dia_semana",
    "mes",
    "fim_de_semana",
    "temporada_alta",
    "var_diaria",
    "var_semanal",
    "razao_lag1_mm7",
    "razao_lag7_mm7",
    "razao_mm7_mm14",
    "coef_variacao_7",
]


# ---------------------------------------------------------------------------
# Dados e métricas
# ---------------------------------------------------------------------------
def carregar_dados():
    df = pd.read_csv(DATASET_PREPARADO, parse_dates=["data"])
    df = df.sort_values("data").reset_index(drop=True)
    dev = df.iloc[:-DIAS_TESTE].reset_index(drop=True)
    teste = df.iloc[-DIAS_TESTE:].reset_index(drop=True)
    return df, dev, teste


def calcular_metricas(y_real, y_previsto) -> dict:
    y_real, y_previsto = np.asarray(y_real, float), np.asarray(y_previsto, float)
    mae = mean_absolute_error(y_real, y_previsto)
    rmse = float(np.sqrt(mean_squared_error(y_real, y_previsto)))
    r2 = r2_score(y_real, y_previsto) if len(y_real) > 1 else float("nan")
    mape = float(np.mean(np.abs((y_real - y_previsto) / y_real)) * 100)
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 4), "MAPE_%": round(mape, 2)}


def folds_temporais(n: int, n_folds: int = N_FOLDS, tamanho: int = DIAS_POR_FOLD) -> list[tuple[int, int]]:
    """Janelas de validação consecutivas no FIM do período de desenvolvimento.

    Devolve pares (inicio_validacao, fim_validacao) em índices de linha. O treino
    de cada dobra é tudo o que vem antes de inicio_validacao (janela expansível),
    exatamente como o TimeSeriesSplit do scikit-learn.
    """
    folds = []
    for k in range(n_folds):
        fim = n - (n_folds - 1 - k) * tamanho
        folds.append((fim - tamanho, fim))
    if folds[0][0] < 365:
        raise ValueError("Histórico insuficiente: cada dobra precisa de ao menos 1 ano de treino.")
    return folds


# ---------------------------------------------------------------------------
# Modelos (todos em walk-forward, 1 dia à frente)
# ---------------------------------------------------------------------------
def prever_baseline(treino: pd.DataFrame, alvo: pd.DataFrame, **_) -> np.ndarray:
    """Palpite ingênuo: o custo de hoje é o do mesmo dia da semana anterior."""
    return alvo[LAG_BASELINE].to_numpy()


def ajustar_sarimax(serie: pd.Series, order, seasonal_order):
    return SARIMAX(
        serie.asfreq("D"),
        order=tuple(order),
        seasonal_order=tuple(seasonal_order),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)


def prever_sarimax(treino: pd.DataFrame, alvo: pd.DataFrame, order=(2, 1, 2), seasonal_order=(1, 0, 1, 7)):
    serie_treino = treino.set_index("data")["custo_usd"].asfreq("D")
    if serie_treino.isna().any():
        raise ValueError("A série de treino tem buracos na grade diária.")
    resultado = ajustar_sarimax(serie_treino, order, seasonal_order)
    serie_alvo = alvo.set_index("data")["custo_usd"].asfreq("D")
    # Estende o estado do filtro com as observações SEM reestimar parâmetros;
    # com dynamic=False cada previsão é de 1 passo: ao prever D, o filtro só
    # processou observações até D-1. É o walk-forward em um único passe.
    resultado = resultado.append(serie_alvo, refit=False)
    previsao = resultado.get_prediction(start=serie_alvo.index[0], end=serie_alvo.index[-1], dynamic=False)
    return previsao.predicted_mean.to_numpy()


def ajustar_prophet(
    historico: pd.DataFrame, changepoint_prior_scale=0.05, seasonality_mode="multiplicative", coluna: str = "custo_usd"
):
    from prophet import Prophet

    modelo = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        interval_width=0.9,
    )
    modelo.fit(historico.rename(columns={"data": "ds", coluna: "y"})[["ds", "y"]])
    return modelo


def prever_prophet(treino: pd.DataFrame, alvo: pd.DataFrame, passo: int = PASSO_REAJUSTE_PROPHET, **hp):
    historico = treino[["data", "custo_usd"]].copy()
    previsoes = np.empty(len(alvo))
    for inicio in range(0, len(alvo), passo):
        bloco = alvo.iloc[inicio : inicio + passo]
        modelo = ajustar_prophet(historico, **hp)
        previsoes[inicio : inicio + len(bloco)] = modelo.predict(pd.DataFrame({"ds": bloco["data"]}))["yhat"].to_numpy()
        # A origem avança: o bloco recém-previsto vira histórico observado.
        historico = pd.concat([historico, bloco[["data", "custo_usd"]]], ignore_index=True)
    return previsoes


def ajustar_gb(treino: pd.DataFrame, n_estimators=300, learning_rate=0.03, max_depth=3):
    """Gradient Boosting sobre a VARIAÇÃO RELATIVA do custo.

    Árvores não extrapolam: só respondem com valores vistos no treino. Numa
    série com tendência de alta -- cujo período de teste (Nov/Dez) é o mais
    caro do histórico -- treinar no nível absoluto, com lags em USD, gera
    erro sistemático. Por isso o alvo é custo/custo_lag_1 - 1 (estacionário)
    e os atributos são só calendário e razões entre lags; o nível volta ao
    multiplicar a variação prevista pelo custo do dia anterior.
    Isso derrubou o viés do modelo de +14,0 para cerca de +0,3 USD no teste.
    """
    alvo = treino["custo_usd"] / treino["custo_lag_1"] - 1
    modelo = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.9,
        random_state=RANDOM_SEED,
    )
    modelo.fit(treino[COLUNAS_FEATURES_GB], alvo)
    return modelo


def prever_gb(treino: pd.DataFrame, alvo: pd.DataFrame, **hp):
    modelo = ajustar_gb(treino, **hp)
    variacao = modelo.predict(alvo[COLUNAS_FEATURES_GB])
    return alvo["custo_lag_1"].to_numpy() * (1 + variacao)


PREVISORES = {
    "baseline": prever_baseline,
    "sarimax": prever_sarimax,
    "prophet": prever_prophet,
    "gb": prever_gb,
}


def grade(chave: str) -> list[dict]:
    if chave == "baseline":
        return [{}]
    espaco = PARAMS["busca"][chave]
    nomes = list(espaco)
    return [dict(zip(nomes, valores)) for valores in itertools.product(*espaco.values())]


# ---------------------------------------------------------------------------
# Backtest + busca de hiperparâmetros
# ---------------------------------------------------------------------------
def backtest(chave: str, dev: pd.DataFrame, hp: dict) -> dict:
    por_fold = []
    for inicio, fim in folds_temporais(len(dev)):
        treino, validacao = dev.iloc[:inicio], dev.iloc[inicio:fim]
        previsto = PREVISORES[chave](treino, validacao, **hp)
        por_fold.append(calcular_metricas(validacao["custo_usd"], previsto))
    resumo = {}
    for m in ("MAE", "RMSE", "MAPE_%"):
        valores = [f[m] for f in por_fold]
        resumo[f"{m}_media"] = round(float(np.mean(valores)), 3)
        resumo[f"{m}_desvio"] = round(float(np.std(valores, ddof=1)), 3)
    return {"params": hp, **resumo, "folds": por_fold}


def buscar_hiperparametros(dev: pd.DataFrame) -> tuple[dict, dict]:
    resultados, melhores = {}, {}
    for chave in MODELOS:
        candidatos = []
        for hp in grade(chave):
            t0 = time.time()
            r = backtest(chave, dev, hp)
            r["tempo_s"] = round(time.time() - t0, 1)
            candidatos.append(r)
            print(f"  {rotulo(chave):26s} {str(hp):70s} MAE CV = {r['MAE_media']:.3f} ± {r['MAE_desvio']:.3f}")
            with rt.execucao(f"cv-{chave}", aninhada=True, tags={"modelo": chave, "etapa": "backtest"}):
                rt.registrar_params(hp)
                rt.registrar_metricas({k: v for k, v in r.items() if k.endswith(("_media", "_desvio"))}, "cv_")
        melhor = min(candidatos, key=lambda r: r["MAE_media"])
        melhores[chave] = melhor["params"]
        resultados[chave] = {"melhor": melhor, "candidatos": candidatos}
    return resultados, melhores


# ---------------------------------------------------------------------------
# Avaliação multi-horizonte (previsão de 1 a 7 dias à frente)
# ---------------------------------------------------------------------------
def _previsao_multi_gb(historico: pd.DataFrame, datas, modelo) -> np.ndarray:
    """Previsão recursiva: cada dia previsto vira 'histórico' para o seguinte."""
    hist = historico[["data", "custo_usd"]].copy()
    saida = []
    for d in datas:
        bruto = pd.concat([hist, pd.DataFrame({"data": [d], "custo_usd": [np.nan]})], ignore_index=True)
        feats, _ = engenharia_atributos(bruto)
        linha = feats.iloc[[-1]]
        valor = float(linha["custo_lag_1"].iloc[0] * (1 + modelo.predict(linha[COLUNAS_FEATURES_GB])[0]))
        saida.append(valor)
        hist.loc[len(hist)] = [d, valor]
    return np.array(saida)


def avaliar_multi_horizonte(df: pd.DataFrame, dev: pd.DataFrame, melhores: dict) -> pd.DataFrame:
    n_dev = len(dev)
    origens = list(range(n_dev, len(df) - HORIZONTE_MAX + 1, HORIZONTE_MAX))
    registros = []

    sarimax_base = ajustar_sarimax(dev.set_index("data")["custo_usd"], **melhores["sarimax"])
    modelo_gb = ajustar_gb(dev, **melhores["gb"])

    for origem in origens:
        historico = df.iloc[:origem]
        futuro = df.iloc[origem : origem + HORIZONTE_MAX]
        real = futuro["custo_usd"].to_numpy()
        datas = futuro["data"].to_list()

        prev = {}
        # Baseline sazonal: para h <= 7, o mesmo dia da semana passada já é conhecido.
        prev["baseline"] = df["custo_usd"].iloc[origem - 7 : origem - 7 + HORIZONTE_MAX].to_numpy()
        res = sarimax_base
        if origem > n_dev:
            res = sarimax_base.append(df.iloc[n_dev:origem].set_index("data")["custo_usd"].asfreq("D"), refit=False)
        prev["sarimax"] = res.forecast(steps=HORIZONTE_MAX).to_numpy()
        prophet = ajustar_prophet(historico[["data", "custo_usd"]], **melhores["prophet"])
        prev["prophet"] = prophet.predict(pd.DataFrame({"ds": datas}))["yhat"].to_numpy()
        prev["gb"] = _previsao_multi_gb(historico, datas, modelo_gb)

        for chave, valores in prev.items():
            for h in range(HORIZONTE_MAX):
                registros.append(
                    {
                        "modelo": rotulo(chave),
                        "origem": f"{df['data'].iloc[origem]:%Y-%m-%d}",
                        "horizonte": h + 1,
                        "real": real[h],
                        "previsto": valores[h],
                    }
                )
    return pd.DataFrame(registros)


def resumir_horizonte(det: pd.DataFrame) -> pd.DataFrame:
    det = det.assign(
        erro_abs=(det["real"] - det["previsto"]).abs(),
        erro_pct=((det["real"] - det["previsto"]).abs() / det["real"]) * 100,
    )
    por_h = (
        det.groupby(["modelo", "horizonte"]).agg(MAE=("erro_abs", "mean"), MAPE_pct=("erro_pct", "mean")).reset_index()
    )
    # Erro no TOTAL da semana: é o que importa para orçamento.
    semana = det.groupby(["modelo", "origem"]).agg(real=("real", "sum"), previsto=("previsto", "sum")).reset_index()
    semana["erro_pct"] = (semana["real"] - semana["previsto"]).abs() / semana["real"] * 100
    total = semana.groupby("modelo")["erro_pct"].mean().round(2).rename("erro_total_semana_pct").reset_index()
    return por_h.round(3).merge(total, on="modelo")


# ---------------------------------------------------------------------------
def main(rapido: bool = False):
    garantir_pastas()
    df, dev, teste = carregar_dados()
    y_teste = teste["custo_usd"].to_numpy()
    print(
        f"Desenvolvimento: {len(dev)} dias | Teste (holdout): {len(teste)} dias "
        f"({teste['data'].min():%Y-%m-%d} a {teste['data'].max():%Y-%m-%d})"
    )

    rt.preparar_experimento("cloudcast-comparacao")
    with rt.execucao("comparacao-modelos", tags={"etapa": "modelagem"}):
        rt.registrar_params(
            {
                "dias_teste": DIAS_TESTE,
                "n_folds": N_FOLDS,
                "dias_por_fold": DIAS_POR_FOLD,
                "seed": RANDOM_SEED,
                "rapido": rapido,
            }
        )

        # 1-3) Backtest + busca de hiperparâmetros
        if rapido and ARQ_MELHORES_PARAMS.exists():
            melhores = json.loads(ARQ_MELHORES_PARAMS.read_text(encoding="utf-8"))
            print("Modo rápido: reutilizando melhores_params.json (sem busca).")
        else:
            print(f"\n[1] Backtest temporal ({N_FOLDS} dobras x {DIAS_POR_FOLD} dias) + busca de hiperparâmetros")
            resultados_cv, melhores = buscar_hiperparametros(dev)
            # Os tempos de execução variam a cada rodada; ficam fora do arquivo
            # versionado (senão o hash em dvc.lock nunca bateria) e vão para um
            # arquivo próprio, ignorado pelo Git.
            tempos = {c: [r.pop("tempo_s") for r in v["candidatos"]] for c, v in resultados_cv.items()}
            for v in resultados_cv.values():
                v["melhor"].pop("tempo_s", None)
            ARQ_TEMPOS.write_text(json.dumps(tempos, indent=2), encoding="utf-8", newline="\n")
            ARQ_CV.write_text(json.dumps(resultados_cv, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
            ARQ_MELHORES_PARAMS.write_text(json.dumps(melhores, indent=2), encoding="utf-8", newline="\n")
            rt.registrar_artefato(ARQ_CV)

        # 4) Holdout, usado uma única vez
        print(f"\n[2] Teste final (holdout de {DIAS_TESTE} dias, walk-forward h=1)")
        previsoes = {}
        for chave in MODELOS:
            t0 = time.time()
            previsoes[chave] = PREVISORES[chave](dev, teste, **melhores[chave])
            print(f"  {rotulo(chave):26s} ok ({time.time() - t0:.1f}s)")
        metricas = {rotulo(c): calcular_metricas(y_teste, p) for c, p in previsoes.items()}
        for nome, m in metricas.items():
            print(f"  {nome:28s} -> {m}")
            rt.registrar_metricas(m, f"teste_{nome.split(' ')[0].lower()}_")

        # 5) Multi-horizonte
        print(f"\n[3] Avaliação multi-horizonte (h = 1..{HORIZONTE_MAX})")
        det = avaliar_multi_horizonte(df, dev, melhores)
        horizonte = resumir_horizonte(det)
        horizonte.to_csv(ARQ_HORIZONTE, index=False, lineterminator="\n")
        print(horizonte.groupby("modelo")[["MAE", "MAPE_pct", "erro_total_semana_pct"]].mean().round(2))

        melhor = min((c for c in MODELOS if c != "baseline"), key=lambda c: metricas[rotulo(c)]["MAE"])
        print(f"\nMelhor modelo no teste (MAE): {rotulo(melhor)}")

        # Persistência dos resultados (gráficos, relatório e testes leem daqui)
        resultados = teste[["data", "custo_usd"]].copy()
        for chave in MODELOS:
            resultados[f"previsao_{chave}"] = previsoes[chave]
        resultados.to_csv(ARQ_PREVISOES, index=False, lineterminator="\n")
        ARQ_METRICAS.write_text(
            json.dumps({**metricas, "melhor_modelo": rotulo(melhor)}, indent=2, ensure_ascii=False),
            encoding="utf-8",
            newline="\n",
        )

        # Interpretação do GB: importância por impureza e por permutação (no teste)
        modelo_gb = ajustar_gb(dev, **melhores["gb"])
        alvo_teste = teste["custo_usd"] / teste["custo_lag_1"] - 1
        perm = permutation_importance(
            modelo_gb,
            teste[COLUNAS_FEATURES_GB],
            alvo_teste,
            n_repeats=30,
            random_state=RANDOM_SEED,
            scoring="neg_mean_absolute_error",
        )
        importancias = pd.DataFrame(
            {
                "importancia": modelo_gb.feature_importances_,
                "permutacao_media": perm.importances_mean,
                "permutacao_desvio": perm.importances_std,
            },
            index=pd.Index(COLUNAS_FEATURES_GB, name="atributo"),
        ).sort_values("importancia", ascending=False)
        importancias.to_csv(ARQ_IMPORTANCIAS, lineterminator="\n")

        for arq in (ARQ_METRICAS, ARQ_PREVISOES, ARQ_HORIZONTE, ARQ_IMPORTANCIAS):
            rt.registrar_artefato(arq)

    print(f"\nResultados salvos em {ARQ_PREVISOES.parent}")


if __name__ == "__main__":
    import sys

    main(rapido="--rapido" in sys.argv)
