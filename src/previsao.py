"""
Camada de serviço de previsão (usada pela API e pelos testes).

Separa a regra de negócio do transporte HTTP: aqui ficam a previsão de
custo e consumo, a tendência, o custo projetado e as regras de alerta do
Capacity Planning. A API só converte entrada/saída e persiste resultados.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.config import PARAMS, PASTA_MODELOS, RANDOM_SEED
from src.registro import carregar_producao

_PROD = PARAMS["produto"]


def _usd(valor: float) -> str:
    """Formato brasileiro: US$ 3.081,02."""
    return f"US$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


@dataclass
class Alerta:
    tipo: str
    severidade: str  # INFO | ATENCAO | CRITICO
    mensagem: str
    valor: float | None = None


class ServicoPrevisao:
    def __init__(self, pasta_modelos=PASTA_MODELOS):
        self.modelos, self.metadata = carregar_producao(pasta_modelos)
        self.versao = self.metadata["versao"]

    # ------------------------------------------------------------------
    def _predict(self, serie: str, futuro: pd.DataFrame) -> pd.DataFrame:
        # O intervalo de incerteza do Prophet é estimado por simulação; fixar a
        # semente torna a resposta da API determinística (mesma entrada, mesma saída).
        np.random.seed(RANDOM_SEED)
        return self.modelos[serie].predict(futuro)

    def prever(self, horizonte: int, inicio: date) -> pd.DataFrame:
        """Previsão diária de custo (com intervalo de 90%), tendência e consumo."""
        datas = pd.date_range(inicio, periods=horizonte, freq="D")
        futuro = pd.DataFrame({"ds": datas})
        custo = self._predict("custo", futuro)
        saida = pd.DataFrame(
            {
                "data": datas.date,
                "custo_previsto": custo["yhat"].clip(lower=0).round(2),
                "custo_limite_inferior": custo["yhat_lower"].clip(lower=0).round(2),
                "custo_limite_superior": custo["yhat_upper"].round(2),
                "tendencia": custo["trend"].round(2),
            }
        )
        if "requisicoes" in self.modelos:
            saida["requisicoes_previstas"] = self._predict("requisicoes", futuro)["yhat"].clip(lower=0).round(0)
        if "instancias" in self.modelos:
            saida["instancias_previstas"] = self._predict("instancias", futuro)["yhat"].clip(lower=1).round(1)
        return saida

    def intervalo_do_dia(self, dia: date) -> tuple[float, float, float]:
        p = self._predict("custo", pd.DataFrame({"ds": [pd.Timestamp(dia)]})).iloc[0]
        return float(p["yhat"]), float(p["yhat_lower"]), float(p["yhat_upper"])

    # ------------------------------------------------------------------
    @staticmethod
    def resumir(previsao: pd.DataFrame, custo_recente: pd.Series) -> dict:
        """Indicadores do dashboard: amanhã, próxima semana, 30 dias, tendência."""
        media_recente = float(custo_recente.tail(7).mean()) if len(custo_recente) else float("nan")
        media_prevista_7 = float(previsao["custo_previsto"].head(7).mean())
        tendencia = previsao["tendencia"].to_numpy()
        variacao_tendencia = float((tendencia[-1] / tendencia[0] - 1) * 100) if len(tendencia) > 1 else 0.0
        return {
            "custo_amanha": round(float(previsao["custo_previsto"].iloc[0]), 2),
            "custo_proxima_semana": round(float(previsao["custo_previsto"].head(7).sum()), 2),
            "custo_projetado_periodo": round(float(previsao["custo_previsto"].sum()), 2),
            "custo_projetado_30_dias": round(float(previsao["custo_previsto"].head(30).sum()), 2)
            if len(previsao) >= 30
            else None,
            "media_diaria_ultimos_7_dias": round(media_recente, 2),
            "variacao_semana_vs_recente_pct": round((media_prevista_7 / media_recente - 1) * 100, 2)
            if media_recente
            else None,
            "variacao_tendencia_periodo_pct": round(variacao_tendencia, 2),
        }

    @staticmethod
    def gerar_alertas(
        previsao: pd.DataFrame,
        custo_recente: pd.Series,
        orcamento_mensal: float = _PROD["orcamento_mensal_usd"],
        limite_crescimento: float = _PROD["limite_alerta_crescimento"],
    ) -> list[Alerta]:
        alertas: list[Alerta] = []
        media_recente = float(custo_recente.tail(7).mean()) if len(custo_recente) else 0.0
        media_prevista = float(previsao["custo_previsto"].head(7).mean())

        if media_recente:
            variacao = media_prevista / media_recente - 1
            if variacao >= limite_crescimento:
                sev = "CRITICO" if variacao >= 2 * limite_crescimento else "ATENCAO"
                alertas.append(
                    Alerta(
                        "CRESCIMENTO_DE_CUSTO",
                        sev,
                        f"Custo médio previsto para os próximos 7 dias é {variacao:.0%} maior "
                        f"que a média dos últimos 7 dias.",
                        round(variacao * 100, 2),
                    )
                )
            elif variacao <= -limite_crescimento:
                alertas.append(
                    Alerta(
                        "QUEDA_DE_DEMANDA",
                        "INFO",
                        f"Custo previsto {abs(variacao):.0%} menor que a semana recente: "
                        "oportunidade de reduzir instâncias reservadas.",
                        round(variacao * 100, 2),
                    )
                )

        janela = previsao.head(30)
        projetado_30 = float(janela["custo_previsto"].sum()) * (30 / len(janela))
        if projetado_30 > orcamento_mensal:
            alertas.append(
                Alerta(
                    "ORCAMENTO_EXCEDIDO",
                    "CRITICO" if projetado_30 > 1.1 * orcamento_mensal else "ATENCAO",
                    f"Custo projetado para 30 dias ({_usd(projetado_30)}) excede o orçamento "
                    f"mensal de {_usd(orcamento_mensal)}.",
                    round(projetado_30, 2),
                )
            )

        if "instancias_previstas" in previsao:
            pico = float(previsao["instancias_previstas"].max())
            atual = float(previsao["instancias_previstas"].iloc[0])
            if atual and pico / atual - 1 >= limite_crescimento:
                alertas.append(
                    Alerta(
                        "CRESCIMENTO_DE_CAPACIDADE",
                        "ATENCAO",
                        f"Previsão de pico de {pico:.0f} instâncias no período "
                        f"({pico / atual - 1:.0%} acima do início).",
                        round(pico, 1),
                    )
                )
        return alertas

    def verificar_anomalia(self, dia: date, custo_real: float) -> Alerta | None:
        """Um dia observado fora do intervalo de 90% da previsão é anômalo."""
        previsto, inferior, superior = self.intervalo_do_dia(dia)
        if custo_real > superior:
            excesso = custo_real / previsto - 1
            return Alerta(
                "ANOMALIA_DE_CUSTO",
                "CRITICO" if excesso > 0.5 else "ATENCAO",
                f"Custo de {dia:%d/%m/%Y} ({_usd(custo_real)}) acima do limite esperado "
                f"({_usd(superior)}); {excesso:.0%} acima do previsto.",
                round(custo_real, 2),
            )
        if custo_real < inferior:
            return Alerta(
                "CUSTO_ABAIXO_DO_ESPERADO",
                "INFO",
                f"Custo de {dia:%d/%m/%Y} ({_usd(custo_real)}) abaixo do limite inferior "
                f"({_usd(inferior)}). Verifique se houve queda de serviço.",
                round(custo_real, 2),
            )
        return None


def proximo_dia(ultima_data: date) -> date:
    return ultima_data + timedelta(days=1)


def mape(reais, previstos) -> float:
    reais, previstos = np.asarray(reais, float), np.asarray(previstos, float)
    return float(np.mean(np.abs((reais - previstos) / reais)) * 100)
