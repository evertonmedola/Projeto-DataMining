"""
Validação de dados (fase "Data Understanding" do CRISP-DM).

Roda ANTES da preparação e responde, com números, à pergunta "o dado bruto
está em condições de ser usado?". Há dois tipos de verificação:

  - CRÍTICAS (bloqueiam o pipeline): esquema/colunas, tipos, faixas físicas
    impossíveis (CPU acima de 100%, custo negativo...). Se falharem, o
    pipeline para com código de saída 1 -- é melhor parar do que treinar um
    modelo sobre dado corrompido.
  - INFORMATIVAS (são tratadas na preparação): valores ausentes, linhas e
    datas duplicadas, buracos na grade diária e picos suspeitos.

O resultado é gravado em outputs/relatorio_qualidade.json e alimenta a
seção "Relatório de qualidade" do relatório.
"""

import json
import sys

import numpy as np
import pandas as pd

from src.config import ARQ_QUALIDADE, COLUNAS_BASE, DATASET_BRUTO, garantir_pastas

# Faixas fisicamente possíveis para cada métrica (mínimo, máximo).
FAIXAS = {
    "requisicoes": (0, None),
    "instancias_ativas": (1, None),
    "cpu_pct": (0, 100),
    "memoria_pct": (0, 100),
    "armazenamento_gb": (0, None),
    "custo_usd": (0, None),
}


def verificar_esquema(df: pd.DataFrame) -> list[str]:
    erros = []
    faltando = [c for c in COLUNAS_BASE if c not in df.columns]
    if faltando:
        erros.append(f"Colunas obrigatórias ausentes: {faltando}")
    if "data" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["data"]):
        erros.append("Coluna 'data' não está em formato de data.")
    for col in FAIXAS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            erros.append(f"Coluna '{col}' não é numérica.")
    return erros


def verificar_faixas(df: pd.DataFrame) -> tuple[list[str], dict]:
    erros, violacoes = [], {}
    for col, (minimo, maximo) in FAIXAS.items():
        if col not in df.columns:
            continue
        serie = df[col].dropna()
        fora = 0
        if minimo is not None:
            fora += int((serie < minimo).sum())
        if maximo is not None:
            fora += int((serie > maximo).sum())
        violacoes[col] = fora
        if fora:
            erros.append(f"{fora} valor(es) fora da faixa física em '{col}'.")
    return erros, violacoes


def contar_picos_suspeitos(serie: pd.Series, fator: float = 3.0) -> int:
    """Picos acima de mediana móvel (só passado) + fator x desvio absoluto mediano."""
    referencia = serie.rolling(15, min_periods=5).median().shift(1)
    desvio = (serie - referencia).abs().rolling(30, min_periods=10).median().shift(1)
    return int(((serie - referencia) > fator * 1.4826 * desvio).sum())


def validar(df: pd.DataFrame) -> dict:
    erros_criticos = verificar_esquema(df)
    if erros_criticos:
        return {"aprovado": False, "erros_criticos": erros_criticos}

    erros_faixa, violacoes = verificar_faixas(df)
    erros_criticos += erros_faixa

    datas_unicas = df["data"].drop_duplicates().sort_values()
    grade = pd.date_range(datas_unicas.min(), datas_unicas.max(), freq="D")
    ordenado = df.sort_values("data").drop_duplicates(subset=["data"])

    n = len(df)
    ausentes = {c: int(df[c].isna().sum()) for c in COLUNAS_BASE if c != "data"}
    relatorio = {
        "aprovado": not erros_criticos,
        "erros_criticos": erros_criticos,
        "linhas": n,
        "colunas": len(df.columns),
        "periodo": [f"{grade.min():%Y-%m-%d}", f"{grade.max():%Y-%m-%d}"],
        "valores_ausentes": ausentes,
        "valores_ausentes_pct": {c: round(100 * v / n, 2) for c, v in ausentes.items()},
        "linhas_duplicadas": int(df.duplicated().sum()),
        "datas_repetidas": int(df["data"].duplicated().sum()),
        "datas_faltantes_na_grade": int(len(grade) - len(datas_unicas)),
        "violacoes_de_faixa": violacoes,
        "picos_suspeitos": {
            c: contar_picos_suspeitos(ordenado.set_index("data")[c].interpolate())
            for c in ["requisicoes", "cpu_pct", "custo_usd"]
        },
        "estatisticas": {
            c: {
                "media": round(float(df[c].mean()), 2),
                "desvio": round(float(df[c].std()), 2),
                "min": round(float(df[c].min()), 2),
                "max": round(float(df[c].max()), 2),
            }
            for c in FAIXAS
        },
    }
    return relatorio


def main():
    garantir_pastas()
    df = pd.read_csv(DATASET_BRUTO, parse_dates=["data"])
    relatorio = validar(df)

    with open(ARQ_QUALIDADE, "w", encoding="utf-8", newline="\n") as f:
        json.dump(
            relatorio,
            f,
            indent=2,
            ensure_ascii=False,
            default=lambda o: o.item() if isinstance(o, np.generic) else str(o),
        )

    print("=== Validação do dataset bruto ===")
    for chave in (
        "linhas",
        "periodo",
        "valores_ausentes",
        "linhas_duplicadas",
        "datas_repetidas",
        "datas_faltantes_na_grade",
        "picos_suspeitos",
    ):
        print(f"{chave:28s} {relatorio.get(chave)}")
    if not relatorio["aprovado"]:
        print("REPROVADO:", relatorio["erros_criticos"])
        sys.exit(1)
    print(f"Aprovado. Relatório em {ARQ_QUALIDADE}")


if __name__ == "__main__":
    main()
