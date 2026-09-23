"""
Mineração e modelagem (fase "Modeling" do CRISP-DM).

Tarefa: prever o CUSTO DIÁRIO de infraestrutura cloud (custo_usd) com
horizonte de 1 dia à frente, comparando os três algoritmos sugeridos no
enunciado do projeto:
  - SARIMAX (variante do ARIMA com componente sazonal semanal)
  - Prophet
  - Gradient Boosting Regressor
Mais um baseline ingênuo, usado como régua para interpretar as métricas.

PROTOCOLO DE AVALIAÇÃO (walk-forward / origem móvel)
----------------------------------------------------
Para que a comparação seja justa, os modelos são avaliados sob o mesmo
protocolo: ao prever o dia D, nenhum deles enxerga qualquer informação de D
ou posterior -- apenas o histórico até D-1 e variáveis de calendário, que
são conhecidas antecipadamente.

Isso é implementado avançando a origem da previsão dia a dia:
  - SARIMAX: o modelo é estimado uma vez no treino e, a cada dia, a
    observação real de D-1 é incorporada ao estado (append com refit=False)
    antes de prever D;
  - Prophet: reajustado a cada PASSO_REAJUSTE_PROPHET dias sobre todo o
    histórico disponível até aquele ponto (reajustar todo dia custaria 60
    ajustes; como o Prophet não tem componente autorregressivo, o ganho
    seria marginal);
  - Gradient Boosting: os atributos de lag/média móvel já usam apenas
    informação até D-1, então basta prever linha a linha.

A versão anterior deste script comparava um GB de 1 dia à frente contra
SARIMAX/Prophet prevendo 60 dias à frente de uma só vez, o que penalizava
injustamente os modelos de série temporal (e produzia R² negativo para
todos os modelos).

Divisão temporal (sem embaralhar, como manda um problema de série temporal):
  - Treino: todos os dias, exceto os últimos DIAS_TESTE
  - Teste:  últimos DIAS_TESTE dias
"""

import json
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import (
    ARQ_IMPORTANCIAS,
    ARQ_METRICAS,
    ARQ_PREVISOES,
    DATASET_PREPARADO,
    DIAS_TESTE,
    MODELOS,
    RANDOM_SEED,
    garantir_pastas,
    rotulo,
)

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

# Reajuste do Prophet a cada N dias do período de teste (1 = todo dia).
PASSO_REAJUSTE_PROPHET = 7

# Sazonalidade semanal: o custo de 7 dias atrás é o palpite ingênuo natural,
# porque já preserva o efeito "dia útil x fim de semana".
LAG_BASELINE = "custo_lag_7"

# Atributos do Gradient Boosting: apenas variáveis de calendário e atributos
# ESTACIONÁRIOS (razões/variações). Nenhum atributo em nível absoluto -- ver a
# explicação em rodar_gradient_boosting().
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


def carregar_dados():
    df = pd.read_csv(DATASET_PREPARADO, parse_dates=["data"])
    df = df.sort_values("data").reset_index(drop=True)
    treino = df.iloc[:-DIAS_TESTE].reset_index(drop=True)
    teste = df.iloc[-DIAS_TESTE:].reset_index(drop=True)
    return df, treino, teste


def rodar_baseline(treino: pd.DataFrame, teste: pd.DataFrame) -> np.ndarray:
    """Palpite ingênuo: o custo de hoje é o do mesmo dia da semana anterior."""
    return teste[LAG_BASELINE].to_numpy()


def rodar_sarimax(treino: pd.DataFrame, teste: pd.DataFrame) -> np.ndarray:
    serie_treino = treino.set_index("data")["custo_usd"].asfreq("D")
    if serie_treino.isna().any():
        raise ValueError("A serie de treino tem buracos na grade diaria.")

    resultado = SARIMAX(
        serie_treino,
        order=(2, 1, 2),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    serie_teste = teste.set_index("data")["custo_usd"].asfreq("D")

    # Estende o estado do filtro de Kalman com as observacoes de teste SEM
    # reestimar os parametros (refit=False). Com dynamic=False, cada previsao
    # do periodo estendido e a previsao de 1 passo a frente: ao prever D o
    # filtro so processou observacoes ate D-1. E exatamente o walk-forward,
    # obtido em um unico passe.
    resultado = resultado.append(serie_teste, refit=False)
    previsao = resultado.get_prediction(
        start=serie_teste.index[0], end=serie_teste.index[-1], dynamic=False
    )
    return previsao.predicted_mean.to_numpy()


def rodar_prophet(treino: pd.DataFrame, teste: pd.DataFrame) -> np.ndarray:
    from prophet import Prophet

    def ajustar(historico: pd.DataFrame):
        modelo = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
            # Multiplicativa: a amplitude da sazonalidade cresce junto com o
            # nivel da serie (a queda de fim de semana vale ~28% do trafego,
            # nao um valor fixo em USD). Com "additive" o modelo subestimava
            # sistematicamente o pico de Nov/Dez (vies de +5,15 USD contra
            # +1,77 no modo multiplicativo).
            seasonality_mode="multiplicative",
            interval_width=0.9,
        )
        modelo.fit(historico.rename(columns={"data": "ds", "custo_usd": "y"}))
        return modelo

    historico = treino[["data", "custo_usd"]].copy()
    previsoes = np.empty(len(teste))

    for inicio in range(0, len(teste), PASSO_REAJUSTE_PROPHET):
        bloco = teste.iloc[inicio : inicio + PASSO_REAJUSTE_PROPHET]
        modelo = ajustar(historico)
        futuro = pd.DataFrame({"ds": bloco["data"].to_numpy()})
        previsoes[inicio : inicio + len(bloco)] = modelo.predict(futuro)["yhat"].to_numpy()
        # A origem avanca: o bloco recem-previsto vira historico observado.
        historico = pd.concat([historico, bloco[["data", "custo_usd"]]], ignore_index=True)

    return previsoes


def rodar_gradient_boosting(treino: pd.DataFrame, teste: pd.DataFrame):
    """Gradient Boosting sobre a VARIACAO RELATIVA do custo.

    Arvores de decisao nao extrapolam: elas so respondem com valores vistos
    no treino. Numa serie com tendencia de alta -- e cujo periodo de teste
    (Nov/Dez) e o trecho mais caro de todo o historico -- treinar no nivel
    absoluto do custo, usando lags em nivel como atributos, produz um erro
    sistematico grande: no teste os lags caem fora da faixa aprendida e o
    modelo devolve o valor da folha extrema.

    A correcao tem duas partes, ambas para tirar o NIVEL da equacao:
      - alvo: a razao custo_usd / custo_lag_1 - 1 (variacao relativa), que e
        estacionaria; o nivel volta ao multiplicar pelo lag na predicao;
      - atributos: apenas calendario e razoes entre lags/medias moveis,
        nenhum valor em USD.

    Isso derrubou o vies do modelo de +14,0 para +0,3 USD no teste.
    """
    alvo_treino = treino["custo_usd"] / treino["custo_lag_1"] - 1

    modelo = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.9,
        random_state=RANDOM_SEED,
    )
    modelo.fit(treino[COLUNAS_FEATURES_GB], alvo_treino)

    variacao_prevista = modelo.predict(teste[COLUNAS_FEATURES_GB])
    return teste["custo_lag_1"].to_numpy() * (1 + variacao_prevista), modelo


def calcular_metricas(y_real, y_previsto) -> dict:
    mae = mean_absolute_error(y_real, y_previsto)
    rmse = float(np.sqrt(mean_squared_error(y_real, y_previsto)))
    r2 = r2_score(y_real, y_previsto)
    mape = float(np.mean(np.abs((y_real - y_previsto) / y_real)) * 100)
    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 4),
        "MAPE_%": round(mape, 2),
    }


def main():
    garantir_pastas()

    df, treino, teste = carregar_dados()
    y_real = teste["custo_usd"].to_numpy()

    print(
        f"Treino: {len(treino)} dias | Teste: {len(teste)} dias "
        f"({teste['data'].min():%Y-%m-%d} a {teste['data'].max():%Y-%m-%d})"
    )

    previsoes = {}

    print("Baseline sazonal...")
    previsoes["baseline"] = rodar_baseline(treino, teste)

    print("Treinando SARIMAX (walk-forward)...")
    previsoes["sarimax"] = rodar_sarimax(treino, teste)

    print(f"Treinando Prophet (reajuste a cada {PASSO_REAJUSTE_PROPHET} dias)...")
    previsoes["prophet"] = rodar_prophet(treino, teste)

    print("Treinando Gradient Boosting...")
    previsoes["gb"], modelo_gb = rodar_gradient_boosting(treino, teste)

    metricas = {rotulo(c): calcular_metricas(y_real, p) for c, p in previsoes.items()}

    print(f"\n=== Metricas no conjunto de teste (ultimos {DIAS_TESTE} dias) ===")
    for nome, m in metricas.items():
        print(f"{nome:28s} -> {m}")

    melhor = min(metricas, key=lambda nome: metricas[nome]["MAE"])
    print(f"\nMelhor modelo por MAE: {melhor}")

    # Persistencia dos resultados para uso posterior (graficos e relatorio)
    resultados = teste[["data", "custo_usd"]].copy()
    for chave in MODELOS:
        resultados[f"previsao_{chave}"] = previsoes[chave]
    resultados.to_csv(ARQ_PREVISOES, index=False)

    with open(ARQ_METRICAS, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    importancias = pd.Series(
        modelo_gb.feature_importances_, index=COLUNAS_FEATURES_GB, name="importancia"
    ).sort_values(ascending=False)
    importancias.to_csv(ARQ_IMPORTANCIAS)

    print(f"\nResultados salvos em {ARQ_PREVISOES.parent}")


if __name__ == "__main__":
    main()
