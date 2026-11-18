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

import json

import numpy as np
import pandas as pd

from src.config import ARQ_PREPARACAO, DATASET_BRUTO, DATASET_PREPARADO, GABARITO_INCIDENTES, PARAMS, garantir_pastas

# Colunas de métricas sujeitas a falha de coleta (valores ausentes).
COLUNAS_METRICAS = ["requisicoes", "cpu_pct", "memoria_pct", "armazenamento_gb", "custo_usd"]

# Colunas em que faz sentido procurar incidentes (picos anômalos).
COLUNAS_OUTLIERS = ["requisicoes", "cpu_pct", "custo_usd"]

_P = PARAMS["preparacao"]
JANELA_MEDIANA = _P["janela_mediana"]  # dias da mediana móvel usada como referência do "nível normal"
FATOR_IQR = _P["fator_iqr"]  # quanto mais alto, mais conservador na marcação de outliers
LAGS = _P["lags"]


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
        # Janela CENTRADA (olha 7 dias para trás e 7 para frente). Testamos uma
        # janela só com o passado: ela acertou os mesmos 10 incidentes, mas
        # gerou 7 falsos positivos -- 6 deles nos primeiros dias de novembro,
        # porque um degrau sazonal legítimo (Black Friday) parece "pico" para
        # quem só olha o passado, e a winsorização teria apagado o início da
        # alta temporada. A limpeza do HISTÓRICO é retrospectiva, então usar
        # vizinhos dos dois lados é legítimo aqui; os atributos de previsão
        # continuam usando apenas D-1. Em produção, a detecção de anomalias
        # de um dia novo é feita pela API, com o intervalo de previsão.
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


def preparar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Executa toda a preparação, na ordem, e devolve o dataset final + estatísticas."""
    linhas_brutas = len(df)
    df, duplicatas = remover_duplicatas(df)
    df, n_datas_faltantes = validar_continuidade(df)
    df, ausentes_por_coluna = imputar_ausentes(df)
    df, n_incidentes = tratar_outliers(df)
    df_final, n_descartadas = engenharia_atributos(df)
    estatisticas = {
        "linhas_brutas": linhas_brutas,
        "linhas_identicas": duplicatas["linhas_identicas"],
        "datas_repetidas_removidas": duplicatas["datas_repetidas_removidas"],
        "datas_faltantes_reinseridas": n_datas_faltantes,
        "valores_ausentes_imputados": ausentes_por_coluna,
        "dias_com_incidente": n_incidentes,
        "linhas_descartadas_por_lag": n_descartadas,
        "linhas_finais": len(df_final),
        "colunas_finais": len(df_final.columns),
        "periodo_final": [f"{df_final['data'].min():%Y-%m-%d}", f"{df_final['data'].max():%Y-%m-%d}"],
    }
    return df_final, estatisticas


def avaliar_deteccao(df_final: pd.DataFrame) -> dict:
    """Compara os dias marcados como incidente com o gabarito da simulação."""
    verdade = set(pd.read_csv(GABARITO_INCIDENTES)["data"])
    inicio = f"{df_final['data'].min():%Y-%m-%d}"
    verdade = {d for d in verdade if d >= inicio}
    marcados = set(df_final.loc[df_final["houve_incidente"] == 1, "data"].dt.strftime("%Y-%m-%d"))
    vp = len(verdade & marcados)
    return {
        "verdadeiros_positivos": vp,
        "falsos_positivos": len(marcados - verdade),
        "falsos_negativos": len(verdade - marcados),
        "precisao": round(vp / len(marcados), 3) if marcados else None,
        "recall": round(vp / len(verdade), 3) if verdade else None,
    }


def main():
    garantir_pastas()

    df_final, est = preparar(carregar(DATASET_BRUTO))
    df_final.to_csv(DATASET_PREPARADO, index=False, float_format="%.6f", lineterminator="\n")

    relatorio = {"preparacao": est}
    if GABARITO_INCIDENTES.exists():
        relatorio["deteccao_incidentes"] = avaliar_deteccao(df_final)
        print(f"Detecção de incidentes vs. gabarito: {relatorio['deteccao_incidentes']}")
    ARQ_PREPARACAO.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")

    print("=== Relatório de qualidade e preparação de dados ===")
    for chave, valor in est.items():
        print(f"{chave:32s} {valor}")
    print(f"Colunas finais: {list(df_final.columns)}")
    print(f"\nSalvo em: {DATASET_PREPARADO}")


if __name__ == "__main__":
    main()
