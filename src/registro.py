"""
Registro de modelos (model registry) do CloudCast.

Cada treino gera uma VERSÃO numerada em models/v<N>/, com os modelos
serializados e um metadata.json (parâmetros, métricas, hash dos dados,
commit do Git). O arquivo models/registry.json aponta qual versão está em
PRODUÇÃO -- é ela que a API carrega. Promover ou reverter uma versão é só
trocar esse ponteiro, sem retreinar nada.

A mesma versão também é registrada no MLflow Model Registry (quando o MLflow
está disponível); o número da versão é o mesmo nos dois lugares.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from prophet.serialize import model_from_json, model_to_json

from src.config import PASTA_MODELOS

ARQ_REGISTRO = "registry.json"


def _caminho_registro(pasta: Path) -> Path:
    return Path(pasta) / ARQ_REGISTRO


def ler_registro(pasta: Path = PASTA_MODELOS) -> dict:
    caminho = _caminho_registro(pasta)
    if not caminho.exists():
        return {"producao": None, "versoes": []}
    return json.loads(caminho.read_text(encoding="utf-8"))


def _gravar_registro(registro: dict, pasta: Path) -> None:
    Path(pasta).mkdir(parents=True, exist_ok=True)
    _caminho_registro(pasta).write_text(
        json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )


def salvar_versao(modelos: dict, metadata: dict, promover: bool, pasta: Path = PASTA_MODELOS) -> int:
    """Grava os modelos Prophet como uma nova versão e (opcionalmente) a promove."""
    registro = ler_registro(pasta)
    versao = max((v["versao"] for v in registro["versoes"]), default=0) + 1
    destino = Path(pasta) / f"v{versao}"
    destino.mkdir(parents=True, exist_ok=True)

    for nome, modelo in modelos.items():
        (destino / f"{nome}.json").write_text(model_to_json(modelo), encoding="utf-8", newline="\n")

    metadata = {
        **metadata,
        "versao": versao,
        "criado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "series": sorted(modelos),
    }
    (destino / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )

    registro["versoes"].append(
        {
            "versao": versao,
            "criado_em": metadata["criado_em"],
            "promovido": promover,
            "metricas_teste": metadata.get("metricas_teste", {}),
        }
    )
    if promover:
        registro["producao"] = versao
    _gravar_registro(registro, pasta)
    return versao


def promover(versao: int, pasta: Path = PASTA_MODELOS) -> None:
    registro = ler_registro(pasta)
    if versao not in {v["versao"] for v in registro["versoes"]}:
        raise ValueError(f"Versão {versao} não existe no registro.")
    registro["producao"] = versao
    _gravar_registro(registro, pasta)


def carregar_producao(pasta: Path = PASTA_MODELOS) -> tuple[dict, dict]:
    """Devolve ({serie: modelo Prophet}, metadata) da versão em produção."""
    registro = ler_registro(pasta)
    versao = registro.get("producao")
    if versao is None:
        raise FileNotFoundError(
            f"Nenhum modelo em produção em {pasta}. Rode `python -m src.pipeline` (ou `make treinar`)."
        )
    origem = Path(pasta) / f"v{versao}"
    metadata = json.loads((origem / "metadata.json").read_text(encoding="utf-8"))
    modelos = {s: model_from_json((origem / f"{s}.json").read_text(encoding="utf-8")) for s in metadata["series"]}
    return modelos, metadata
