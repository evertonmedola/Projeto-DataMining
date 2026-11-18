"""Testes de API: contrato, validação de entrada (422), persistência e alertas."""

import pytest
from fastapi.testclient import TestClient

OBS = {
    "data": "2024-12-31",
    "requisicoes": 185000,
    "instancias_ativas": 27,
    "cpu_pct": 66.5,
    "memoria_pct": 68.1,
    "armazenamento_gb": 1950.0,
    "custo_usd": 95.0,
}


@pytest.fixture(scope="module")
def cliente(modelo_treinado):
    from src.api.main import app

    with TestClient(app) as c:
        yield c


def test_health(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_model_info(cliente, modelo_treinado):
    info = cliente.get("/model/info").json()
    assert info["versao"] == modelo_treinado and info["algoritmo"] == "Prophet"


def test_forecast_contrato(cliente):
    r = cliente.get("/forecast", params={"horizonte": 7})
    assert r.status_code == 200
    corpo = r.json()
    assert len(corpo["previsao"]) == 7
    soma = sum(p["custo_previsto"] for p in corpo["previsao"])
    assert corpo["resumo"]["custo_proxima_semana"] == pytest.approx(soma, abs=0.05)
    assert {"tendencia", "requisicoes_previstas", "instancias_previstas"} <= set(corpo["previsao"][0])


@pytest.mark.parametrize("horizonte", [0, -1, 91])
def test_forecast_horizonte_invalido_retorna_422(cliente, horizonte):
    assert cliente.get("/forecast", params={"horizonte": horizonte}).status_code == 422


def test_forecast_e_persistido_para_auditoria(cliente):
    antes = len(cliente.get("/previsoes", params={"limite": 5000}).json())
    cliente.get("/forecast", params={"horizonte": 3})
    depois = cliente.get("/previsoes", params={"limite": 5000}).json()
    assert len(depois) == antes + 3 and all("versao_modelo" in p for p in depois)


def test_simulacao_retroativa_nao_e_persistida(cliente):
    antes = len(cliente.get("/previsoes", params={"limite": 5000}).json())
    r = cliente.get("/forecast", params={"horizonte": 30, "referencia": "2024-10-29"})
    assert r.status_code == 200 and r.json()["simulacao"] is True
    assert "CRESCIMENTO_DE_CUSTO" in {a["tipo"] for a in r.json()["alertas"]}
    assert len(cliente.get("/previsoes", params={"limite": 5000}).json()) == antes


def test_observacao_invalida_retorna_422(cliente):
    assert cliente.post("/observacoes", json={**OBS, "cpu_pct": 150}).status_code == 422
    assert cliente.post("/observacoes", json={**OBS, "custo_usd": -5}).status_code == 422
    assert cliente.post("/observacoes", json={k: v for k, v in OBS.items() if k != "data"}).status_code == 422


def test_observacao_normal_e_anomala(cliente):
    r = cliente.post("/observacoes", json=OBS)
    assert r.status_code == 201
    r2 = cliente.post("/observacoes", json={**OBS, "data": "2025-01-01", "custo_usd": 400.0})
    assert r2.status_code == 201
    assert r2.json()["alerta"]["tipo"] == "ANOMALIA_DE_CUSTO"
    assert any(a["tipo"] == "ANOMALIA_DE_CUSTO" for a in cliente.get("/alertas").json())


def test_observacao_fora_de_sequencia_retorna_422(cliente):
    assert cliente.post("/observacoes", json={**OBS, "data": "2025-03-01"}).status_code == 422


def test_monitoramento_cruza_previsao_com_realizado(cliente):
    cliente.get("/forecast", params={"horizonte": 1})  # prevê o dia seguinte ao último observado
    ultima = cliente.get("/health").json()["ultima_data"]
    from datetime import date, timedelta

    proximo = (date.fromisoformat(ultima) + timedelta(days=1)).isoformat()
    cliente.post("/observacoes", json={**OBS, "data": proximo})
    mon = cliente.get("/monitoramento").json()
    assert mon["dias_avaliados"] >= 1 and mon["mape_producao_pct"] is not None


def test_reload_do_modelo(cliente, modelo_treinado):
    assert cliente.post("/model/reload").json()["versao_atual"] == modelo_treinado
