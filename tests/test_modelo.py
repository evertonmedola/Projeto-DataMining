"""Testes de MODELO: métricas, protocolo de validação, determinismo, desempenho mínimo e alertas."""

import json

import numpy as np
import pandas as pd
import pytest

from src import modelagem
from src.config import ARQ_METRICAS, DIAS_TESTE
from src.previsao import ServicoPrevisao


def test_metricas_com_valores_conhecidos():
    m = modelagem.calcular_metricas([100, 200], [110, 190])
    assert m["MAE"] == 10
    assert m["RMSE"] == 10
    assert m["MAPE_%"] == 7.5


def test_folds_temporais_sao_ordenados_e_sem_sobreposicao():
    folds = modelagem.folds_temporais(656, n_folds=5, tamanho=30)
    assert folds[-1][1] == 656
    for (_i0, f0), (i1, f1) in zip(folds, folds[1:]):
        assert f0 == i1 and f1 - i1 == 30
    assert folds[0][0] >= 365  # pelo menos 1 ano de treino na primeira dobra


def test_divisao_temporal_teste_e_o_fim_da_serie(preparado):
    dev, teste = preparado.iloc[:-DIAS_TESTE], preparado.iloc[-DIAS_TESTE:]
    assert dev["data"].max() < teste["data"].min()


def test_baseline_e_o_custo_da_semana_anterior(preparado):
    alvo = preparado.tail(10)
    np.testing.assert_array_equal(modelagem.prever_baseline(None, alvo), alvo["custo_lag_7"].to_numpy())


def test_gradient_boosting_e_deterministico(preparado):
    dev, teste = preparado.iloc[:-DIAS_TESTE], preparado.iloc[-DIAS_TESTE:]
    p1 = modelagem.prever_gb(dev, teste, n_estimators=50)
    p2 = modelagem.prever_gb(dev, teste, n_estimators=50)
    np.testing.assert_array_equal(p1, p2)


def test_gb_vence_o_baseline_no_teste(preparado):
    dev, teste = preparado.iloc[:-DIAS_TESTE], preparado.iloc[-DIAS_TESTE:]
    real = teste["custo_usd"]
    mae_gb = modelagem.calcular_metricas(real, modelagem.prever_gb(dev, teste))["MAE"]
    mae_base = modelagem.calcular_metricas(real, modelagem.prever_baseline(dev, teste))["MAE"]
    assert mae_gb < mae_base


@pytest.mark.skipif(not ARQ_METRICAS.exists(), reason="rode o pipeline antes (gera outputs/metricas.json)")
def test_metricas_registradas_atingem_a_meta():
    m = json.loads(ARQ_METRICAS.read_text(encoding="utf-8"))
    assert m["Prophet"]["MAPE_%"] <= 10.0  # meta de negócio da seção 1
    base = m["Baseline sazonal (7 dias)"]["MAE"]
    for nome in ("SARIMAX (ARIMA sazonal)", "Prophet", "Gradient Boosting"):
        assert m[nome]["MAE"] < base, f"{nome} não venceu o baseline"


# --- regras de alerta (independem do modelo) --------------------------------
def _previsao(valores, instancias=None):
    n = len(valores)
    df = pd.DataFrame({"custo_previsto": valores, "tendencia": np.linspace(100, 101, n)})
    if instancias is not None:
        df["instancias_previstas"] = instancias
    return df


def test_alerta_de_crescimento():
    alertas = ServicoPrevisao.gerar_alertas(_previsao([130.0] * 7), pd.Series([100.0] * 7), orcamento_mensal=1e9)
    tipos = {a.tipo: a for a in alertas}
    assert "CRESCIMENTO_DE_CUSTO" in tipos and tipos["CRESCIMENTO_DE_CUSTO"].severidade == "CRITICO"


def test_alerta_de_orcamento():
    alertas = ServicoPrevisao.gerar_alertas(_previsao([110.0] * 30), pd.Series([110.0] * 7), orcamento_mensal=3000)
    assert [a.tipo for a in alertas] == ["ORCAMENTO_EXCEDIDO"]


def test_sem_alerta_quando_estavel():
    alertas = ServicoPrevisao.gerar_alertas(
        _previsao([90.0] * 30, [20.0] * 30), pd.Series([90.0] * 7), orcamento_mensal=3000
    )
    assert alertas == []


def test_servico_preve_horizonte_pedido(modelo_treinado):
    servico = ServicoPrevisao()
    prev = servico.prever(7, pd.Timestamp("2024-12-31").date())
    assert len(prev) == 7
    assert (prev["custo_limite_inferior"] <= prev["custo_previsto"]).all()
    assert (prev["custo_previsto"] <= prev["custo_limite_superior"]).all()
    # determinismo: a mesma consulta devolve o mesmo intervalo
    pd.testing.assert_frame_equal(prev, servico.prever(7, pd.Timestamp("2024-12-31").date()))


# --- protocolo de avaliação (versões curtas, para rodar rápido) ----------------
def test_backtest_resume_media_e_desvio_por_dobra(preparado):
    dev = preparado.iloc[:-DIAS_TESTE]
    r = modelagem.backtest("gb", dev, {"n_estimators": 30})
    assert len(r["folds"]) == modelagem.N_FOLDS
    assert r["MAE_media"] == pytest.approx(np.mean([f["MAE"] for f in r["folds"]]), abs=1e-3)
    assert r["MAE_desvio"] >= 0


def test_sarimax_e_prophet_walk_forward_vencem_o_baseline(preparado):
    dev, teste = preparado.iloc[:-DIAS_TESTE], preparado.iloc[-DIAS_TESTE:].head(21)
    real = teste["custo_usd"]
    base = modelagem.calcular_metricas(real, modelagem.prever_baseline(dev, teste))["MAE"]
    sar = modelagem.prever_sarimax(dev, teste, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7))
    pro = modelagem.prever_prophet(dev, teste)
    assert len(sar) == len(pro) == 21
    assert modelagem.calcular_metricas(real, sar)["MAE"] < base
    assert modelagem.calcular_metricas(real, pro)["MAE"] < base


def test_grade_de_hiperparametros_cobre_todas_as_combinacoes():
    assert modelagem.grade("baseline") == [{}]
    assert len(modelagem.grade("gb")) == 8
    assert len(modelagem.grade("prophet")) == 6


def test_previsao_recursiva_do_gb_tem_o_horizonte_pedido(preparado):
    dev = preparado.iloc[:-DIAS_TESTE]
    modelo = modelagem.ajustar_gb(dev, n_estimators=30)
    datas = preparado["data"].iloc[len(dev) : len(dev) + 7].to_list()
    prev = modelagem._previsao_multi_gb(dev, datas, modelo)
    assert prev.shape == (7,) and np.isfinite(prev).all() and (prev > 0).all()
