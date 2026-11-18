"""Testes de DADOS: geração, validação de esquema/faixas e qualidade após a preparação."""

import numpy as np
import pandas as pd

from src import gerar_dados, preparar_dados, validar_dados
from src.config import COLUNAS_BASE


def test_geracao_e_deterministica(bruto):
    outra = gerar_dados.gerar()
    pd.testing.assert_frame_equal(bruto, outra)


def test_dataset_bruto_tem_o_esquema_esperado(bruto):
    assert list(bruto.columns) == COLUNAS_BASE
    assert len(bruto) == 730 + 5  # 2 anos + 5 duplicatas injetadas


def test_validacao_aprova_e_quantifica_problemas(bruto):
    rel = validar_dados.validar(bruto)
    assert rel["aprovado"]
    assert rel["linhas_duplicadas"] == 5
    assert rel["valores_ausentes"]["cpu_pct"] == 14
    assert rel["valores_ausentes"]["custo_usd"] == 0
    assert rel["datas_faltantes_na_grade"] == 0


def test_validacao_reprova_valor_fisicamente_impossivel(bruto):
    ruim = bruto.copy()
    ruim.loc[3, "cpu_pct"] = 150
    rel = validar_dados.validar(ruim)
    assert not rel["aprovado"]
    assert any("cpu_pct" in e for e in rel["erros_criticos"])


def test_validacao_reprova_esquema_incompleto(bruto):
    rel = validar_dados.validar(bruto.drop(columns=["custo_usd"]))
    assert not rel["aprovado"]


def test_preparado_sem_nulos_sem_duplicatas_e_grade_diaria(preparado):
    assert not preparado.isna().any().any()
    assert preparado["data"].is_unique
    assert (preparado["data"].diff().dropna() == pd.Timedelta(days=1)).all()


def test_preparado_respeita_faixas(preparado):
    assert preparado["cpu_pct"].between(0, 100).all()
    assert (preparado["custo_usd"] > 0).all()


def test_deteccao_encontra_todos_os_incidentes_injetados(bruto, preparado):
    verdade = set(bruto.attrs["datas_incidentes"])
    marcados = set(preparado.loc[preparado["houve_incidente"] == 1, "data"].dt.strftime("%Y-%m-%d"))
    assert verdade <= marcados  # recall = 100%
    assert len(marcados - verdade) <= 2  # poucos falsos positivos


def test_incidente_e_winsorizado_e_nao_removido(bruto, preparado):
    """Cada dia de incidente continua na série, com pelo menos uma métrica trazida de volta ao limite."""
    bruto_unico = bruto.drop_duplicates(subset=["data"]).set_index("data")
    tratado = preparado.set_index("data")
    for dia in bruto.attrs["datas_incidentes"]:
        assert pd.Timestamp(dia) in tratado.index
        reduziu = [
            tratado.at[pd.Timestamp(dia), c] < bruto_unico.at[pd.Timestamp(dia), c]
            for c in ("requisicoes", "cpu_pct", "custo_usd")
            if np.isfinite(bruto_unico.at[pd.Timestamp(dia), c])
        ]
        assert any(reduziu), dia


def test_remover_duplicatas_mantem_uma_linha_por_data():
    df = pd.DataFrame({"data": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]), "x": [1, 1, 2]})
    limpo, info = preparar_dados.remover_duplicatas(df)
    assert len(limpo) == 2 and info["datas_repetidas_removidas"] == 1


def test_validar_continuidade_reinsere_datas_faltantes():
    df = pd.DataFrame({"data": pd.to_datetime(["2024-01-01", "2024-01-04"]), "x": [1.0, 4.0]})
    completo, faltantes = preparar_dados.validar_continuidade(df)
    assert faltantes == 2 and len(completo) == 4
