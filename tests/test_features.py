"""Testes de TRANSFORMAÇÃO: engenharia de atributos sem vazamento de dados."""

import numpy as np
import pandas as pd
import pytest

from src.preparar_dados import engenharia_atributos, imputar_ausentes


@pytest.fixture
def serie_simples():
    datas = pd.date_range("2024-01-01", periods=40, freq="D")
    return pd.DataFrame({"data": datas, "custo_usd": np.arange(1, 41, dtype=float) * 10})


def test_lags_usam_somente_o_passado(serie_simples):
    df, _ = engenharia_atributos(serie_simples)
    original = serie_simples.set_index("data")["custo_usd"]
    for _, linha in df.iterrows():
        d = linha["data"]
        assert linha["custo_lag_1"] == original[d - pd.Timedelta(days=1)]
        assert linha["custo_lag_7"] == original[d - pd.Timedelta(days=7)]


def test_media_movel_nao_inclui_o_proprio_dia(serie_simples):
    df, _ = engenharia_atributos(serie_simples)
    linha = df.iloc[0]
    janela = serie_simples[serie_simples["data"] < linha["data"]].tail(7)["custo_usd"]
    assert linha["custo_media_movel_7"] == pytest.approx(janela.mean())


def test_alterar_o_alvo_do_dia_nao_altera_seus_atributos(serie_simples):
    """Prova direta de ausência de vazamento: os atributos do dia D independem de custo(D)."""
    base, _ = engenharia_atributos(serie_simples)
    alterada = serie_simples.copy()
    alterada.loc[alterada.index[-1], "custo_usd"] = 99999
    nova, _ = engenharia_atributos(alterada)
    colunas = [c for c in base.columns if c not in ("custo_usd",)]
    pd.testing.assert_series_equal(base.iloc[-1][colunas], nova.iloc[-1][colunas], check_names=False)


def test_atributos_estacionarios(serie_simples):
    df, _ = engenharia_atributos(serie_simples)
    esperado = df["custo_lag_1"] / df["custo_lag_2"] - 1
    pd.testing.assert_series_equal(df["var_diaria"], esperado, check_names=False)
    assert (df["razao_lag1_mm7"] > 0).all()


def test_calendario(serie_simples):
    df, descartadas = engenharia_atributos(serie_simples)
    assert descartadas == 14  # maior janela = 14 dias
    assert set(df["fim_de_semana"].unique()) <= {0, 1}
    assert (df.loc[df["dia_semana"] >= 5, "fim_de_semana"] == 1).all()


def test_imputacao_preenche_buracos_por_interpolacao():
    df = pd.DataFrame(
        {
            "data": pd.date_range("2024-01-01", periods=3, freq="D"),
            "requisicoes": [100.0, np.nan, 300.0],
            "cpu_pct": [10.0, 20.0, 30.0],
            "memoria_pct": [1.0, 1.0, 1.0],
            "armazenamento_gb": [5.0, 5.0, 5.0],
            "custo_usd": [1.0, 2.0, 3.0],
        }
    )
    imputado, contagem = imputar_ausentes(df)
    assert contagem["requisicoes"] == 1
    assert imputado.loc[1, "requisicoes"] == pytest.approx(200.0)
