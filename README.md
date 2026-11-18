# CloudCast — Previsão de Consumo e Custo de Infraestrutura Cloud

Projeto de Mineração de Dados (Engenharia de Software — FAG), **tema 4**.
Equipe: **Everton Medola** e **Luis Cerutti**.

Sistema de *Capacity Planning* que prevê o custo diário e o consumo (requisições e instâncias) de uma infraestrutura
cloud, projeta o gasto da próxima semana/mês e gera alertas de crescimento, estouro de orçamento e anomalias.
Metodologia **CRISP-DM** + práticas de **MLOps** (pipeline versionado, rastreamento de experimentos, registro de
modelos, testes, CI e Docker).

## Resultados (holdout Nov–Dez/2024, 60 dias, `seed = 42`)

| Modelo | MAE (USD) | RMSE (USD) | MAPE | Erro no total da semana (h = 1..7) |
|---|---|---|---|---|
| Baseline sazonal (7 dias) | 13,24 | 16,39 | 12,93% | 8,25% |
| SARIMAX (1,1,1)(0,1,1,7) | 9,74 | 11,92 | 9,53% | 7,79% |
| **Prophet (multiplicativo, cps = 0,05)** | **8,67** | **11,00** | **8,46%** | **4,64%** |
| Gradient Boosting | 9,92 | 12,15 | 9,76% | 8,54% |

Backtest temporal (5 dobras × 30 dias, MAE médio ± desvio): Prophet 6,86 ± 0,85 · SARIMAX 7,06 ± 1,24 ·
GB 7,60 ± 0,86 · Baseline 9,92 ± 1,60. Detalhes em `outputs/` e no relatório.

## Como reproduzir

Pré-requisito: Python 3.13 (ou Docker).

```bash
git clone https://github.com/evertonmedola/Projeto-DataMining.git
cd Projeto-DataMining
python -m venv .venv && .venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python -m src.pipeline            # dados -> validação -> preparação -> modelagem -> treino -> figuras (~1 min)
python -m pytest                  # 45 testes
uvicorn src.api.main:app          # API em http://localhost:8000/docs
streamlit run dashboard/app.py    # dashboard em http://localhost:8501 (com a API no ar)
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db   # experimentos em http://localhost:5000
```

Com Docker (sobe pipeline, API, dashboard e MLflow, nessa ordem):

```bash
docker compose up --build
```

Com DVC (refaz só as etapas que mudaram): `dvc repro`. Atalhos: `make pipeline`, `make testes`, `make api`, ...

## Estrutura

```
├── params.yaml            # sementes, janelas, grades de hiperparâmetros, limites de alerta
├── dvc.yaml / dvc.lock    # pipeline declarativo e hashes de cada etapa
├── src/
│   ├── gerar_dados.py     # ingestão: dataset simulado com base em métricas reais (+ gabarito de incidentes)
│   ├── validar_dados.py   # esquema, faixas físicas, ausentes, duplicatas -> relatorio_qualidade.json
│   ├── preparar_dados.py  # limpeza, imputação, outliers, engenharia de atributos (sem vazamento)
│   ├── modelagem.py       # baseline, SARIMAX, Prophet, GB; backtest, busca, teste, multi-horizonte
│   ├── treinar.py         # modelo final + registro de versão (models/ e MLflow Model Registry)
│   ├── registro.py        # model registry (versões e ponteiro de produção)
│   ├── previsao.py        # serviço de previsão, resumo e regras de alerta
│   ├── rastreamento.py    # MLflow
│   ├── graficos.py        # figuras do relatório
│   ├── pipeline.py        # orquestra todas as etapas
│   └── api/               # FastAPI (main.py), contratos (schemas.py), persistência SQLite (db.py)
├── dashboard/app.py       # Streamlit
├── tests/                 # dados, transformações, modelo, API
├── data/ models/ outputs/ # artefatos versionados (pequenos)
├── docs/                  # diagrama de arquitetura e capturas de tela
├── Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```

## API

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | status e versão do modelo |
| GET | `/model/info` | metadados da versão em produção (métricas, hash dos dados, commit) |
| POST | `/model/reload` | recarrega o modelo após novo treino |
| GET | `/forecast?horizonte=7` | previsão diária com intervalo de 90%, tendência, consumo, custo projetado e alertas |
| GET | `/forecast?horizonte=30&referencia=2024-10-29` | simulação retroativa (não é persistida) |
| GET | `/historico?dias=90` | consumo e custo históricos |
| POST | `/observacoes` | registra o consumo real de um dia e verifica anomalia |
| GET | `/alertas`, `/previsoes` | histórico de alertas e de previsões servidas (auditoria) |
| GET | `/monitoramento` | MAPE em produção e recomendação de retreino |

## Dados

Dataset **simulado com base em métricas e preços reais de provedores cloud** (permitido pelo enunciado do tema 4):
730 dias (01/01/2023–30/12/2024), crescimento orgânico, sazonalidade semanal e anual (Black Friday/Natal),
autoscaling e custo = computação + armazenamento + transferência + fixo. Problemas de qualidade injetados de
propósito (2% de ausentes, 5 duplicatas, 10 incidentes) para exercitar a preparação. Uso acadêmico.
