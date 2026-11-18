"""
Orquestrador do pipeline completo: dados -> validação -> preparação ->
modelagem -> treino/registro -> figuras.

    python -m src.pipeline            # execução completa (com busca de hiperparâmetros)
    python -m src.pipeline --rapido   # reaproveita melhores_params.json (usado na CI)

É o equivalente, sem dependências extras, ao `dvc repro` (ver dvc.yaml): as
mesmas etapas, na mesma ordem, com os mesmos artefatos.
"""

import sys
import time

from src import gerar_dados, graficos, modelagem, preparar_dados, treinar, validar_dados

ETAPAS = [
    ("1/6 Ingestão (geração do dataset)", gerar_dados.main),
    ("2/6 Validação de qualidade", validar_dados.main),
    ("3/6 Preparação e engenharia de atributos", preparar_dados.main),
    ("4/6 Modelagem, backtest e teste", None),  # recebe a flag --rapido
    ("5/6 Treino final e registro do modelo", treinar.main),
    ("6/6 Figuras do relatório", graficos.main),
]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    rapido = "--rapido" in argv
    inicio_total = time.time()
    for nome, etapa in ETAPAS:
        print(f"\n{'=' * 70}\n{nome}\n{'=' * 70}")
        t0 = time.time()
        if etapa is None:
            modelagem.main(rapido=rapido)
        else:
            etapa()
        print(f"-> concluída em {time.time() - t0:.1f}s")
    print(f"\nPipeline completo em {time.time() - inicio_total:.1f}s")


if __name__ == "__main__":
    main()
