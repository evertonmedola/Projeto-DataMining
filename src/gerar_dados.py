"""
Geração do dataset de consumo e custo de infraestrutura cloud.

Como não há acesso direto a uma conta cloud real (AWS/Azure/GCP) para este
trabalho acadêmico, o dataset é SIMULADO com base em padrões reais de
consumo e cobrança de provedores cloud (crescimento de uso ao longo do
tempo, sazonalidade semanal - menor tráfego em finais de semana -,
sazonalidade anual - pico em novembro/dezembro por Black Friday/Natal -,
autoscaling de instâncias correlacionado à demanda, e formação de custo
como função de instâncias ativas + armazenamento + transferência de dados).

Isso está alinhado ao enunciado do projeto, que permite explicitamente o uso
de "datasets públicos de cloud ou dados simulados com base em métricas
reais".

Também são injetados propositalmente problemas de qualidade de dados
(valores ausentes, outliers/incidentes, linhas duplicadas) para que a etapa
de Preparação de Dados do CRISP-DM tenha problemas reais para tratar.
"""

import numpy as np
import pandas as pd

from config import DATASET_BRUTO, RANDOM_SEED, garantir_pastas

np.random.seed(RANDOM_SEED)

N_DIAS = 730  # 2 anos de histórico diário
DATA_INICIAL = "2023-01-01"


def gerar_serie_base(n_dias: int) -> pd.DataFrame:
    datas = pd.date_range(start=DATA_INICIAL, periods=n_dias, freq="D")
    t = np.arange(n_dias)

    # --- Requisições diárias (tráfego) ---
    tendencia_crescimento = 1 + 0.0009 * t  # crescimento orgânico do negócio
    dia_semana = datas.dayofweek  # 0=segunda ... 6=domingo
    fator_semanal = np.where(dia_semana >= 5, 0.72, 1.0)  # queda no fim de semana

    mes = datas.month
    fator_sazonal_anual = np.where(np.isin(mes, [11, 12]), 1.55, 1.0)  # Black Friday/Natal
    fator_sazonal_anual = np.where(np.isin(mes, [1]), 0.9, fator_sazonal_anual)  # ressaca pós-festas

    base_requisicoes = 120_000
    ruido_requisicoes = np.random.normal(0, 4_000, n_dias)
    requisicoes = (
        base_requisicoes
        * tendencia_crescimento
        * fator_semanal
        * fator_sazonal_anual
        + ruido_requisicoes
    )
    requisicoes = np.clip(requisicoes, 20_000, None)

    # --- Instâncias ativas (autoscaling reage à demanda, com suavização) ---
    demanda_normalizada = requisicoes / requisicoes.mean()
    instancias_alvo = 18 * demanda_normalizada
    instancias_ativas = pd.Series(instancias_alvo).rolling(3, min_periods=1).mean().to_numpy()
    instancias_ativas = instancias_ativas + np.random.normal(0, 0.6, n_dias)
    instancias_ativas = np.clip(np.round(instancias_ativas), 4, None)

    # --- Utilização de CPU e memória (%) ---
    # A CPU depende da carga POR instância: se o autoscaling acompanha bem a
    # demanda, a utilização fica estável; picos aparecem quando o tráfego sobe
    # mais rápido do que o número de instâncias.
    carga_por_instancia = requisicoes / instancias_ativas
    cpu_pct = 40 + 25 * (carga_por_instancia / carga_por_instancia.mean())
    cpu_pct = cpu_pct + np.random.normal(0, 4, n_dias)
    cpu_pct = np.clip(cpu_pct, 5, 99)

    memoria_pct = 0.8 * cpu_pct + 15 + np.random.normal(0, 5, n_dias)
    memoria_pct = np.clip(memoria_pct, 5, 99)

    # --- Armazenamento (GB, crescimento quase monotônico) ---
    armazenamento_gb = 500 + 1.8 * t + np.cumsum(np.random.normal(0.3, 0.8, n_dias))
    armazenamento_gb = np.clip(armazenamento_gb, 500, None)

    return pd.DataFrame(
        {
            "data": datas,
            "requisicoes": requisicoes,
            "instancias_ativas": instancias_ativas,
            "cpu_pct": cpu_pct,
            "memoria_pct": memoria_pct,
            "armazenamento_gb": armazenamento_gb,
        }
    )


def calcular_custo(df: pd.DataFrame) -> pd.DataFrame:
    """Compõe o custo diário a partir dos recursos consumidos."""
    df = df.copy()

    # Preços aproximados (USD) inspirados em tabelas públicas de provedores cloud
    preco_instancia_hora = 0.096  # ~ t3.medium sob demanda
    preco_storage_gb_mes = 0.023  # ~ S3 standard
    preco_por_milhao_requisicoes = 0.40
    custo_fixo_diario = 12.0  # suporte, load balancer, IPs reservados etc.
    # Uma requisição carrega, em média, algumas centenas de KB de resposta;
    # o fator abaixo converte "milhões de requisições" no volume de saída
    # efetivamente cobrado pelo provedor.
    fator_volume_transferencia = 180

    custo_computacao = df["instancias_ativas"] * 24 * preco_instancia_hora
    custo_storage = df["armazenamento_gb"] * (preco_storage_gb_mes / 30)
    custo_transferencia = (
        (df["requisicoes"] / 1_000_000) * preco_por_milhao_requisicoes * fator_volume_transferencia
    )

    ruido_custo = np.random.normal(0, 8, len(df))
    df["custo_usd"] = (
        custo_fixo_diario + custo_computacao + custo_storage + custo_transferencia + ruido_custo
    )
    df["custo_usd"] = df["custo_usd"].clip(lower=10)
    return df


def injetar_problemas_qualidade(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    rng = np.random.default_rng(RANDOM_SEED)

    # 1) Valores ausentes (~2% em colunas de métricas), simulando falha de coleta
    for col in ["cpu_pct", "memoria_pct", "requisicoes", "armazenamento_gb"]:
        idx_nulos = rng.choice(n, size=int(n * 0.02), replace=False)
        df.loc[idx_nulos, col] = np.nan

    # 2) Outliers/incidentes pontuais (picos anômalos de tráfego/CPU em ~10 dias)
    idx_incidentes = rng.choice(n, size=10, replace=False)
    df.loc[idx_incidentes, "requisicoes"] *= rng.uniform(2.2, 3.5, size=10)
    df.loc[idx_incidentes, "cpu_pct"] = np.clip(
        df.loc[idx_incidentes, "cpu_pct"] * rng.uniform(1.4, 1.8, size=10), 0, 100
    )
    df.loc[idx_incidentes, "custo_usd"] *= rng.uniform(1.3, 1.9, size=10)

    # 3) Linhas duplicadas (erro comum de ETL / reprocessamento)
    linhas_duplicadas = df.sample(n=5, random_state=RANDOM_SEED)
    df = pd.concat([df, linhas_duplicadas], ignore_index=True)

    return df


def main():
    garantir_pastas()

    df = gerar_serie_base(N_DIAS)
    df = calcular_custo(df)
    df = injetar_problemas_qualidade(df)
    df = df.sort_values("data").reset_index(drop=True)

    df.to_csv(DATASET_BRUTO, index=False)
    print(f"Dataset bruto gerado com {len(df)} linhas -> {DATASET_BRUTO}")
    print(df.describe(include="all").T[["count", "mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
