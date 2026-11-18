"""Contratos de entrada e saída da API (validados pelo Pydantic -> erro 422 automático)."""

from datetime import date

from pydantic import BaseModel, Field


class Observacao(BaseModel):
    """Métricas de um dia de operação, como exportadas do provedor cloud."""

    data: date
    requisicoes: float = Field(ge=0, description="Requisições recebidas no dia")
    instancias_ativas: float = Field(ge=1, description="Média de instâncias ativas")
    cpu_pct: float = Field(ge=0, le=100)
    memoria_pct: float = Field(ge=0, le=100)
    armazenamento_gb: float = Field(ge=0)
    custo_usd: float = Field(gt=0, description="Custo total do dia em USD")

    model_config = {
        "json_schema_extra": {
            "example": {
                "data": "2024-12-31",
                "requisicoes": 185000,
                "instancias_ativas": 27,
                "cpu_pct": 66.5,
                "memoria_pct": 68.1,
                "armazenamento_gb": 1950.0,
                "custo_usd": 104.30,
            }
        }
    }


class AlertaOut(BaseModel):
    tipo: str
    severidade: str
    mensagem: str
    valor: float | None = None


class PontoPrevisao(BaseModel):
    data: date
    custo_previsto: float
    custo_limite_inferior: float
    custo_limite_superior: float
    tendencia: float
    requisicoes_previstas: float | None = None
    instancias_previstas: float | None = None


class ResumoPrevisao(BaseModel):
    custo_amanha: float
    custo_proxima_semana: float
    custo_projetado_periodo: float
    custo_projetado_30_dias: float | None
    media_diaria_ultimos_7_dias: float
    variacao_semana_vs_recente_pct: float | None
    variacao_tendencia_periodo_pct: float


class RespostaPrevisao(BaseModel):
    versao_modelo: int
    gerado_a_partir_de: date
    horizonte: int
    simulacao: bool = False
    resumo: ResumoPrevisao
    previsao: list[PontoPrevisao]
    alertas: list[AlertaOut]


class RespostaObservacao(BaseModel):
    data: date
    atualizado: bool
    custo_previsto: float
    limite_superior: float
    alerta: AlertaOut | None


class InfoModelo(BaseModel):
    nome: str
    versao: int
    algoritmo: str
    hiperparametros: dict
    treinado_ate: str
    n_observacoes: int
    metricas_teste: dict
    metricas_teste_consumo: dict
    dados_sha256: str
    git_commit: str
    criado_em: str
