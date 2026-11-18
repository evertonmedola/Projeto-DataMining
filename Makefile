# Atalhos do projeto. No Windows sem `make`, rode os comandos da direita direto.
PY ?= python

.PHONY: instalar pipeline rapido api dashboard mlflow testes lint docker

instalar:      ## instala as dependências fixadas
	$(PY) -m pip install -r requirements.txt

pipeline:      ## dados -> validação -> preparação -> modelagem -> treino -> figuras
	$(PY) -m src.pipeline

rapido:        ## mesmo pipeline, sem refazer a busca de hiperparâmetros
	$(PY) -m src.pipeline --rapido

api:           ## API em http://localhost:8000/docs
	$(PY) -m uvicorn src.api.main:app --reload --port 8000

dashboard:     ## dashboard em http://localhost:8501 (precisa da API no ar)
	$(PY) -m streamlit run dashboard/app.py

mlflow:        ## interface do MLflow em http://localhost:5000
	mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

testes:        ## testes automatizados com cobertura
	$(PY) -m pytest --cov=src --cov-report=term-missing

lint:
	$(PY) -m ruff check src tests dashboard

docker:        ## sobe API + dashboard + MLflow
	docker compose up --build
