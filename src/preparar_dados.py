"""
Preparação e limpeza dos dados (fase "Data Preparation" do CRISP-DM).

Trata os problemas de qualidade injetados propositalmente no dataset bruto:
  1. Remoção de linhas duplicadas;
  2. Imputação de valores ausentes por interpolação temporal;
  3. Tratamento de outliers/incidentes (winsorização do resíduo por IQR),
     preservando um marcador "houve_incidente" para não perder essa
     informação;
  4. Engenharia de atributos: variáveis de calendário e variáveis
     autorregressivas (lags e médias móveis) do custo, usadas depois pelos
     modelos de mineração.
"""

import numpy as np
import pandas as pd

from config import DATASET_BRUTO, DATASET_PREPARADO, garantir_pastas

# Colunas de métricas sujeitas a falha de coleta (valores ausentes).
COLUNAS_METRICAS = ["requisicoes", "cpu_pct", "memoria_pct", "armazenamento_gb", "custo_usd"]

# Colunas em que faz sentido procurar incidentes (picos anômalos).
COLUNAS_OUTLIERS = ["requisicoes", "cpu_pct", "custo_usd"]

JANELA_MEDIANA = 15  # dias da mediana móvel usada como referência do "nível normal"
FATOR_IQR = 2.5  # quanto mais alto, mais conservador na marcação de outliers
LAGS = [1, 2, 7, 14]


def carregar(caminho) -> pd.DataFrame:
    return pd.read_csv(caminho, parse_dates=["data"])


def remover_duplicatas(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Em uma série diária, cada data deve aparecer uma única vez."""
    n_antes = len(df)
    n_linhas_identicas = int(df.duplicated().sum())
    df = df.sort_values("data").drop_duplicates(subset=["data"], keep="first").reset_index(drop=True)
    return df, {
        "linhas_identicas": n_linhas_identicas,
        "datas_repetidas_removidas": n_antes - len(df),
    }


def validar_continuidade(df: pd.DataFrame) -> int:
    """Garante que a série é diária e contínua.

    A interpolação temporal e o SARIMAX (que usa asfreq("D")) assumem uma
    grade diária sem buracos; se houver datas faltando, elas são inseridas
    aqui e ficam como ausentes para serem imputadas na etapa seguinte.
    """
    grade_completa = pd.date_range(df["data"].min(), df["data"].max(), freq="D")
    n_faltantes = len(grade_completa) - len(df)
    if n_faltantes:
        df = df.set_index("data").reindex(grade_completa).rename_axis("data").reset_index()
    return df, n_faltantes


def imputar_ausentes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    contagem_antes = {c: int(df[c].isna().sum()) for c in COLUNAS_METRICAS}

    df = df.sort_values("data").set_index("data")
    # interpolate(method="time") respeita o espaçamento real entre as datas;
    # bfill/ffill cobrem eventuais ausências nas pontas da série.
    df[COLUNAS_METRICAS] = df[COLUNAS_METRICAS].interpolate(method="time").bfill().ffill()
    df = df.reset_index()

    restantes = int(df[COLUNAS_METRICAS].isna().sum().sum())
    if restantes:
        raise ValueError(f"Ainda restaram {restantes} valores ausentes após a imputação.")

    return df, contagem_antes


def tratar_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Detecta outliers em cima do RESÍDUO em relação a uma mediana móvel,
    e não do valor bruto -- essencial aqui porque a série tem tendência de
    crescimento e sazonalidade anual (Nov/Dez), o que faria um IQR aplicado
    ao valor bruto confundir "pico sazonal legítimo" com "incidente".

    Os valores marcados são winsorizados (trazidos de volta ao limite), e não
    descartados: em série temporal, remover a linha abriria um buraco na
    grade diária.
    """
    df = df.copy()
    df["houve_incidente"] = 0

    for col in COLUNAS_OUTLIERS:
        mediana_movel = df[col].rolling(window=JANELA_MEDIANA, center=True, min_periods=5).median()
        residuo = df[col] - mediana_movel

        q1, q3 = residuo.quantile([0.25, 0.75])
        iqr = q3 - q1
        limite_inf = q1 - FATOR_IQR * iqr
        limite_sup = q3 + FATOR_IQR * iqr

        marcados = residuo.gt(limite_sup) | residuo.lt(limite_inf)
        df.loc[marcados, "houve_incidente"] = 1

        valor_corrigido = mediana_movel + residuo.clip(lower=limite_inf, upper=limite_sup)
        df[col] = df[col].mask(marcados, valor_corrigido)

    return df, int(df["houve_incidente"].sum())


def engenharia_atributos(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = df.sort_values("data").reset_index(drop=True)

    # Variáveis de calendário (conhecidas antecipadamente -> seguras para forecasting)
    df["dia_semana"] = df["data"].dt.dayofweek
    df["mes"] = df["data"].dt.month
    df["fim_de_semana"] = (df["dia_semana"] >= 5).astype(int)
    df["temporada_alta"] = df["mes"].isin([11, 12]).astype(int)
    # Índice contínuo de tempo, atribuído ANTES do dropna para que a origem
    # temporal continue sendo o primeiro dia observado da série.
    df["indice_tempo"] = np.arange(len(df))

    # Variáveis autorregressivas do custo. Todas usam shift(1) antes da janela
    # móvel: no dia D só entra informação até D-1, sem vazamento do alvo.
    colunas_derivadas = []
    for lag in LAGS:
        nome = f"custo_lag_{lag}"
        df[nome] = df["custo_usd"].shift(lag)
        colunas_derivadas.append(nome)

    custo_passado = df["custo_usd"].shift(1)
    df["custo_media_movel_7"] = custo_passado.rolling(window=7).mean()
    df["custo_media_movel_14"] = custo_passado.rolling(window=14).mean()
    df["custo_std_movel_7"] = custo_passado.rolling(window=7).std()
    colunas_derivadas += ["custo_media_movel_7", "custo_media_movel_14", "custo_std_movel_7"]

    # Versões ESTACIONÁRIAS dos atributos acima (razões e variações relativas).
    # Motivo: modelos baseados em árvore (Gradient Boosting) não extrapolam --
    # eles só sabem responder com valores vistos no treino. Como a série tem
    # tendência de alta, no período de teste os lags em nível caem fora da
    # faixa aprendida e a árvore erra sistematicamente. Razões entre lags são
    # livres de nível e, por isso, continuam válidas fora da faixa de treino.
    df["var_diaria"] = df["custo_lag_1"] / df["custo_lag_2"] - 1
    df["var_semanal"] = df["custo_lag_1"] / df["custo_lag_7"] - 1
    df["razao_lag1_mm7"] = df["custo_lag_1"] / df["custo_media_movel_7"]
    df["razao_lag7_mm7"] = df["custo_lag_7"] / df["custo_media_movel_7"]
    df["razao_mm7_mm14"] = df["custo_media_movel_7"] / df["custo_media_movel_14"]
    df["coef_variacao_7"] = df["custo_std_movel_7"] / df["custo_media_movel_7"]
    colunas_derivadas += [
        "var_diaria",
        "var_semanal",
        "razao_lag1_mm7",
        "razao_lag7_mm7",
        "razao_mm7_mm14",
        "coef_variacao_7",
    ]

    # As primeiras linhas ficam com NaN por causa dos lags/janelas -> removidas
    n_antes = len(df)
    df = df.dropna(subset=colunas_derivadas).reset_index(drop=True)
    return df, n_antes - len(df)


def main():
    garantir_pastas()

    df = carregar(DATASET_BRUTO)
    linhas_brutas = len(df)

    df, duplicatas = remover_duplicatas(df)
    df, n_datas_faltantes = validar_continuidade(df)
    df, ausentes_por_coluna = imputar_ausentes(df)
    df, n_incidentes = tratar_outliers(df)
    df_final, n_descartadas = engenharia_atributos(df)

    df_final.to_csv(DATASET_PREPARADO, index=False)

    print("=== Relatório de qualidade e preparação de dados ===")
    print(f"Linhas no dataset bruto:               {linhas_brutas}")
    print(f"Linhas totalmente idênticas:           {duplicatas['linhas_identicas']}")
    print(f"Datas repetidas removidas:             {duplicatas['datas_repetidas_removidas']}")
    print(f"Datas faltantes reinseridas na grade:  {n_datas_faltantes}")
    print(f"Valores ausentes por coluna (antes):   {ausentes_por_coluna}")
    print(f"Dias marcados como incidente/outlier:  {n_incidentes}")
    print(f"Linhas descartadas por lag/janela:     {n_descartadas}")
    print(f"Linhas no dataset final preparado:     {len(df_final)}")
    print(f"Período coberto: {df_final['data'].min():%Y-%m-%d} a {df_final['data'].max():%Y-%m-%d}")
    print(f"Colunas finais: {list(df_final.columns)}")
    print(f"\nSalvo em: {DATASET_PREPARADO}")


if __name__ == "__main__":
    main()
