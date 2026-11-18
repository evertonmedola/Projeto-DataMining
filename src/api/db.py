"""
Persistência do CloudCast (SQLite via sqlite3 da biblioteca padrão).

Tabelas:
  metricas_diarias -- histórico de consumo e custo (1 linha por dia). É
                      populada a partir do dataset preparado na primeira
                      execução e recebe novos dias via POST /observacoes.
  previsoes        -- cada previsão servida pela API, com a versão do modelo
                      (auditoria e monitoramento de erro em produção).
  alertas          -- alertas gerados (crescimento, orçamento, anomalia...).

SQLite atende ao escopo acadêmico (um arquivo, zero configuração). Como o
acesso é feito só por este módulo, trocar por PostgreSQL exige mudar apenas
a conexão e alguns detalhes de SQL.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from src.config import CAMINHO_DB, COLUNAS_BASE, DATASET_PREPARADO

ESQUEMA = """
CREATE TABLE IF NOT EXISTS metricas_diarias (
    data              TEXT PRIMARY KEY,
    requisicoes       REAL NOT NULL,
    instancias_ativas REAL NOT NULL,
    cpu_pct           REAL NOT NULL,
    memoria_pct       REAL NOT NULL,
    armazenamento_gb  REAL NOT NULL,
    custo_usd         REAL NOT NULL,
    origem            TEXT NOT NULL DEFAULT 'historico'
);
CREATE TABLE IF NOT EXISTS previsoes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    criado_em       TEXT NOT NULL DEFAULT (datetime('now')),
    data_alvo       TEXT NOT NULL,
    horizonte       INTEGER NOT NULL,
    custo_previsto  REAL NOT NULL,
    limite_inferior REAL NOT NULL,
    limite_superior REAL NOT NULL,
    versao_modelo   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_previsoes_data ON previsoes (data_alvo);
CREATE TABLE IF NOT EXISTS alertas (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    criado_em     TEXT NOT NULL DEFAULT (datetime('now')),
    tipo          TEXT NOT NULL,
    severidade    TEXT NOT NULL,
    mensagem      TEXT NOT NULL,
    valor         REAL,
    versao_modelo INTEGER
);
"""


class Banco:
    def __init__(self, caminho: Path = CAMINHO_DB):
        self.caminho = Path(caminho)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        with self.conexao() as con:
            con.executescript(ESQUEMA)

    @contextmanager
    def conexao(self):
        con = sqlite3.connect(self.caminho)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    # --- histórico -----------------------------------------------------
    def popular_se_vazio(self, csv: Path = DATASET_PREPARADO) -> int:
        with self.conexao() as con:
            if con.execute("SELECT COUNT(*) FROM metricas_diarias").fetchone()[0]:
                return 0
            df = pd.read_csv(csv, parse_dates=["data"])[COLUNAS_BASE]
            df["data"] = df["data"].dt.strftime("%Y-%m-%d")
            con.executemany(
                "INSERT INTO metricas_diarias (data, requisicoes, instancias_ativas, cpu_pct, memoria_pct, "
                "armazenamento_gb, custo_usd) VALUES (?, ?, ?, ?, ?, ?, ?)",
                df.itertuples(index=False, name=None),
            )
            return len(df)

    def historico(self, dias: int | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM metricas_diarias ORDER BY data"
        with self.conexao() as con:
            df = pd.read_sql_query(sql, con, parse_dates=["data"])
        return df.tail(dias).reset_index(drop=True) if dias else df

    def ultima_data(self):
        with self.conexao() as con:
            valor = con.execute("SELECT MAX(data) FROM metricas_diarias").fetchone()[0]
        return pd.Timestamp(valor).date() if valor else None

    def inserir_observacao(self, obs: dict) -> bool:
        """Insere (ou substitui) um dia. Devolve True se o dia já existia."""
        with self.conexao() as con:
            existia = con.execute("SELECT 1 FROM metricas_diarias WHERE data = ?", (obs["data"],)).fetchone()
            con.execute(
                "INSERT OR REPLACE INTO metricas_diarias (data, requisicoes, instancias_ativas, cpu_pct, "
                "memoria_pct, armazenamento_gb, custo_usd, origem) VALUES (?, ?, ?, ?, ?, ?, ?, 'api')",
                (
                    obs["data"],
                    obs["requisicoes"],
                    obs["instancias_ativas"],
                    obs["cpu_pct"],
                    obs["memoria_pct"],
                    obs["armazenamento_gb"],
                    obs["custo_usd"],
                ),
            )
        return existia is not None

    # --- previsões e alertas -------------------------------------------
    def registrar_previsoes(self, previsao: pd.DataFrame, versao: int) -> None:
        linhas = [
            (
                str(r.data),
                h + 1,
                float(r.custo_previsto),
                float(r.custo_limite_inferior),
                float(r.custo_limite_superior),
                versao,
            )
            for h, r in enumerate(previsao.itertuples(index=False))
        ]
        with self.conexao() as con:
            con.executemany(
                "INSERT INTO previsoes (data_alvo, horizonte, custo_previsto, limite_inferior, limite_superior, "
                "versao_modelo) VALUES (?, ?, ?, ?, ?, ?)",
                linhas,
            )

    def registrar_alertas(self, alertas, versao: int) -> None:
        if not alertas:
            return
        with self.conexao() as con:
            con.executemany(
                "INSERT INTO alertas (tipo, severidade, mensagem, valor, versao_modelo) VALUES (?, ?, ?, ?, ?)",
                [(a.tipo, a.severidade, a.mensagem, a.valor, versao) for a in alertas],
            )

    def listar(self, tabela: str, limite: int = 100) -> list[dict]:
        if tabela not in {"previsoes", "alertas"}:
            raise ValueError("Tabela inválida.")
        with self.conexao() as con:
            linhas = con.execute(f"SELECT * FROM {tabela} ORDER BY id DESC LIMIT ?", (limite,)).fetchall()
        return [dict(linha) for linha in linhas]

    def erro_em_producao(self, dias: int = 30) -> pd.DataFrame:
        """Cruza a previsão mais recente de horizonte 1 de cada dia com o custo observado."""
        sql = """
            SELECT p.data_alvo AS data, p.custo_previsto, m.custo_usd AS custo_real, p.versao_modelo
            FROM previsoes p
            JOIN metricas_diarias m ON m.data = p.data_alvo AND m.origem = 'api'
            WHERE p.horizonte = 1
              AND p.id = (SELECT MAX(id) FROM previsoes q WHERE q.data_alvo = p.data_alvo AND q.horizonte = 1)
            ORDER BY p.data_alvo DESC LIMIT ?
        """
        with self.conexao() as con:
            return pd.read_sql_query(sql, con, params=(dias,))
