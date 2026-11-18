"""
API do CloudCast -- Capacity Planning de infraestrutura cloud.

Executar:  uvicorn src.api.main:app --reload     (documentação em /docs)

Endpoints
  GET  /health                  -> status do serviço e versão do modelo
  GET  /model/info              -> metadados da versão em produção
  POST /model/reload            -> recarrega o modelo após um novo treino (sem derrubar a API)
  GET  /forecast?horizonte=7    -> previsão de custo/consumo, tendência, custo projetado e alertas
                                   (&referencia=AAAA-MM-DD simula a previsão a partir de uma data passada)
  GET  /historico?dias=90       -> consumo e custo históricos
  POST /observacoes             -> registra o consumo real de um dia e verifica anomalia
  GET  /alertas                 -> alertas gerados
  GET  /previsoes               -> previsões servidas (auditoria, com versão do modelo)
  GET  /monitoramento           -> erro do modelo em produção e recomendação de retreino
"""

from contextlib import asynccontextmanager
from datetime import date

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status

from src.api.db import Banco
from src.api.schemas import InfoModelo, Observacao, RespostaObservacao, RespostaPrevisao
from src.config import PARAMS
from src.previsao import ServicoPrevisao, mape, proximo_dia

_PROD = PARAMS["produto"]
MIN_DIAS_MONITORAMENTO = 7
estado: dict = {}


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    banco = Banco()
    inseridas = banco.popular_se_vazio()
    if inseridas:
        print(f"Banco inicializado com {inseridas} dias de histórico.")
    estado["banco"] = banco
    estado["servico"] = ServicoPrevisao()
    yield
    estado.clear()


app = FastAPI(
    title="CloudCast API",
    description="Previsão de consumo e custo de infraestrutura cloud (tema 4 -- Mineração de Dados, FAG).",
    version="1.0.0",
    lifespan=ciclo_de_vida,
)


def _servico() -> ServicoPrevisao:
    return estado["servico"]


def _banco() -> Banco:
    return estado["banco"]


@app.get("/health", tags=["operação"])
def health():
    return {"status": "ok", "versao_modelo": _servico().versao, "ultima_data": str(_banco().ultima_data())}


@app.get("/model/info", response_model=InfoModelo, tags=["modelo"])
def info_modelo():
    m = _servico().metadata
    return InfoModelo(nome=m["modelo"], **{k: m[k] for k in InfoModelo.model_fields if k != "nome"})


@app.post("/model/reload", tags=["modelo"])
def recarregar_modelo():
    anterior = _servico().versao
    estado["servico"] = ServicoPrevisao()
    return {"versao_anterior": anterior, "versao_atual": estado["servico"].versao}


@app.get("/forecast", response_model=RespostaPrevisao, tags=["previsão"])
def forecast(
    horizonte: int = Query(
        _PROD["horizonte_padrao"],
        ge=1,
        le=_PROD["horizonte_maximo"],
        description="Dias à frente (1 = amanhã, 7 = próxima semana)",
    ),
    referencia: date | None = Query(
        None,
        description="Simulação retroativa: prever como se hoje fosse esta "
        "data (usa só o histórico anterior a ela). Não é persistida.",
    ),
):
    banco, servico = _banco(), _servico()
    ultima = banco.ultima_data()
    simulacao = referencia is not None
    if simulacao:
        if referencia > ultima:
            raise HTTPException(status_code=422, detail=f"A data de referência deve ser até {ultima}.")
        historico = banco.historico()
        historico = historico[historico["data"] < pd.Timestamp(referencia)].tail(30)
        inicio = referencia
    else:
        historico = banco.historico(dias=30)
        inicio = proximo_dia(ultima)

    previsao = servico.prever(horizonte, inicio)
    resumo = servico.resumir(previsao, historico["custo_usd"])
    alertas = servico.gerar_alertas(previsao, historico["custo_usd"])

    if not simulacao:  # simulações não entram na auditoria nem no monitoramento
        banco.registrar_previsoes(previsao, servico.versao)
        banco.registrar_alertas(alertas, servico.versao)
    return {
        "versao_modelo": servico.versao,
        "gerado_a_partir_de": inicio,
        "horizonte": horizonte,
        "simulacao": simulacao,
        "resumo": resumo,
        "previsao": previsao.replace({np.nan: None}).to_dict(orient="records"),
        "alertas": [a.__dict__ for a in alertas],
    }


@app.get("/historico", tags=["consumo"])
def historico(dias: int = Query(90, ge=1, le=3650)):
    df = _banco().historico(dias=dias)
    df["data"] = df["data"].dt.strftime("%Y-%m-%d")
    return df.to_dict(orient="records")


@app.post("/observacoes", response_model=RespostaObservacao, status_code=status.HTTP_201_CREATED, tags=["consumo"])
def registrar_observacao(obs: Observacao):
    banco, servico = _banco(), _servico()
    ultima = banco.ultima_data()
    if ultima and (obs.data - ultima).days > 1:
        raise HTTPException(
            status_code=422,
            detail=f"Dia fora de sequência: o último registrado é {ultima}. Envie os dias em ordem, sem buracos.",
        )
    atualizado = banco.inserir_observacao({**obs.model_dump(), "data": obs.data.isoformat()})
    previsto, _, superior = servico.intervalo_do_dia(obs.data)
    alerta = servico.verificar_anomalia(obs.data, obs.custo_usd)
    banco.registrar_alertas([alerta] if alerta else [], servico.versao)
    return {
        "data": obs.data,
        "atualizado": atualizado,
        "custo_previsto": round(previsto, 2),
        "limite_superior": round(superior, 2),
        "alerta": alerta.__dict__ if alerta else None,
    }


@app.get("/alertas", tags=["alertas"])
def listar_alertas(limite: int = Query(50, ge=1, le=1000)):
    return _banco().listar("alertas", limite)


@app.get("/previsoes", tags=["auditoria"])
def listar_previsoes(limite: int = Query(100, ge=1, le=5000)):
    return _banco().listar("previsoes", limite)


@app.get("/monitoramento", tags=["operação"])
def monitoramento(dias: int = Query(30, ge=1, le=365)):
    erros = _banco().erro_em_producao(dias)
    limite = _PROD["limite_mape_monitoramento"]
    if erros.empty:
        return {
            "dias_avaliados": 0,
            "mape_producao_pct": None,
            "limite_pct": limite,
            "recomenda_retreino": False,
            "mensagem": "Ainda não há dias observados via API com previsão de 1 dia registrada.",
        }
    valor = round(mape(erros["custo_real"], erros["custo_previsto"]), 2)
    # Um único dia ruim (ex.: um incidente) não justifica retreinar: a
    # recomendação só vale com pelo menos uma semana de dias avaliados.
    suficiente = len(erros) >= MIN_DIAS_MONITORAMENTO
    if not suficiente:
        mensagem = f"Erro calculado sobre {len(erros)} dia(s); a recomendação exige {MIN_DIAS_MONITORAMENTO}."
    elif valor > limite:
        mensagem = "Erro recente acima do limite: recomenda-se retreinar o modelo."
    else:
        mensagem = "Erro em produção dentro do limite."
    return {
        "dias_avaliados": len(erros),
        "mape_producao_pct": valor,
        "limite_pct": limite,
        "recomenda_retreino": suficiente and valor > limite,
        "mape_teste_pct": _servico().metadata["metricas_teste"]["MAPE_%"],
        "mensagem": mensagem,
    }
