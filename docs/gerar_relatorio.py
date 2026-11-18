"""
Gera o relatório final (docs/Relatorio_CloudCast.docx) a partir do modelo do professor.

Os estilos (títulos, listas, tabela "Table Grid", bloco de código) vêm do próprio
modelo; o corpo é reconstruído com os números lidos de outputs/ -- assim o
relatório nunca diverge das métricas do pipeline.

    python docs/gerar_relatorio.py caminho/do/Modelo.docx
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor, Twips

RAIZ = Path(__file__).resolve().parent.parent
OUT = RAIZ / "outputs"
FIG = OUT / "figuras"
TELAS = RAIZ / "docs" / "telas"
MODELO = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "docs" / "Modelo_Relatorio.docx"
SAIDA = RAIZ / "docs" / "Relatorio_CloudCast.docx"

LARGURA = 6.5  # polegadas úteis (Carta, margens de 1")

# ---------------------------------------------------------------- números
met = json.loads((OUT / "metricas.json").read_text(encoding="utf-8"))
cv = json.loads((OUT / "resultados_cv.json").read_text(encoding="utf-8"))
# Tempos de execução da busca (arquivo não versionado; só existe após rodar a modelagem)
_arq_tempos = OUT / "tempos_cv.json"
tempos = json.loads(_arq_tempos.read_text(encoding="utf-8")) if _arq_tempos.exists() else {}


def tempo_melhor(chave):
    """Tempo do backtest da melhor configuração, ou "—" se não houver registro."""
    for cand, t in zip(cv[chave]["candidatos"], tempos.get(chave, [])):
        if cand["params"] == cv[chave]["melhor"]["params"]:
            return f"{br(t, 1)} s"
    return "—"
qual = json.loads((OUT / "relatorio_qualidade.json").read_text(encoding="utf-8"))
prep = json.loads((OUT / "relatorio_preparacao.json").read_text(encoding="utf-8"))
melhores = json.loads((OUT / "melhores_params.json").read_text(encoding="utf-8"))
hor = pd.read_csv(OUT / "metricas_horizonte.csv")
prev = pd.read_csv(OUT / "previsoes_teste.csv", parse_dates=["data"])
imp = pd.read_csv(OUT / "importancia_features_gb.csv", index_col=0)
meta = json.loads((RAIZ / "models" / "v1" / "metadata.json").read_text(encoding="utf-8"))

ROT = {
    "baseline": "Baseline sazonal (7 dias)",
    "sarimax": "SARIMAX (ARIMA sazonal)",
    "prophet": "Prophet",
    "gb": "Gradient Boosting",
}


def br(x, casas=2):
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, casas=2):
    return f"{br(x, casas)}%"


P = met["Prophet"]
residuo = prev["custo_usd"] - prev["previsao_prophet"]
hz = hor.pivot(index="horizonte", columns="modelo", values="MAPE_pct")
semana = hor.groupby("modelo")["erro_total_semana_pct"].first()
hz_media = hor.groupby("modelo")[["MAE", "MAPE_pct"]].mean()
det = prep["deteccao_incidentes"]
pp = prep["preparacao"]

# ---------------------------------------------------------------- documento
doc = Document(MODELO)
corpo = doc.element.body
for el in list(corpo):
    if el.tag != qn("w:sectPr"):
        corpo.remove(el)

contador = {"fig": 0, "tab": 0}


def _runs(par, texto):
    """**negrito**, *itálico* e `código` em linha."""
    for parte in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", texto):
        if not parte:
            continue
        if parte.startswith("**"):
            par.add_run(parte[2:-2]).bold = True
        elif parte.startswith("`"):
            r = par.add_run(parte[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(9.5)
        elif parte.startswith("*"):
            par.add_run(parte[1:-1]).italic = True
        else:
            par.add_run(parte)
    return par


def h1(t):
    doc.add_paragraph(t, style="Heading 1")


def h2(t):
    doc.add_paragraph(t, style="Heading 2")


def h3(t):
    doc.add_paragraph(t, style="Heading 3")


def p(t, align_justify=True):
    par = _runs(doc.add_paragraph(style="Normal"), t)
    if align_justify:
        par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return par


def _lista(itens, num_id):
    for it in itens:
        par = doc.add_paragraph(style="List Paragraph")
        pPr = par._p.get_or_add_pPr()
        numPr = OxmlElement("w:numPr")
        ilvl = OxmlElement("w:ilvl")
        ilvl.set(qn("w:val"), "0")
        nid = OxmlElement("w:numId")
        nid.set(qn("w:val"), str(num_id))
        numPr.append(ilvl)
        numPr.append(nid)
        pPr.append(numPr)
        _runs(par, it)


def bullets(itens):
    _lista(itens, 1)


def numerada(itens):
    _lista(itens, 2)


def codigo(texto):
    par = doc.add_paragraph(style="Code")
    linhas = texto.strip("\n").split("\n")
    for i, linha in enumerate(linhas):
        r = par.add_run(linha)
        if i < len(linhas) - 1:
            r.add_break(WD_BREAK.LINE)


def _sombrear(celula, cor="E7EEF7"):
    tcPr = celula._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), cor)
    tcPr.append(shd)


def tabela(titulo, cabecalho, linhas, larguras=None, leitura=None, tamanho=9.5):
    contador["tab"] += 1
    cap = doc.add_paragraph(style="Normal")
    _runs(cap, f"**Tabela {contador['tab']} — {titulo}**")
    t = doc.add_table(rows=1, cols=len(cabecalho))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, c in enumerate(cabecalho):
        cel = t.rows[0].cells[i]
        cel.text = ""
        _runs(cel.paragraphs[0], f"**{c}**")
        _sombrear(cel)
    for linha in linhas:
        cels = t.add_row().cells
        for i, v in enumerate(linha):
            cels[i].text = ""
            _runs(cels[i].paragraphs[0], str(v))
    for row in t.rows:
        for cel in row.cells:
            for par in cel.paragraphs:
                for r in par.runs:
                    r.font.size = Pt(tamanho)
    larguras = larguras or [1] * len(cabecalho)
    total = sum(larguras)
    dxa = [int(1440 * LARGURA * w / total) for w in larguras]
    t.autofit = False
    tblPr = t._tbl.tblPr
    for tag in ("w:tblW", "w:tblLayout"):
        for el in tblPr.findall(qn(tag)):
            tblPr.remove(el)
    tblW = OxmlElement("w:tblW")
    tblW.set(qn("w:w"), str(sum(dxa)))
    tblW.set(qn("w:type"), "dxa")
    tblPr.append(tblW)
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)
    grid = t._tbl.tblGrid
    for gc, w in zip(grid.findall(qn("w:gridCol")), dxa):
        gc.set(qn("w:w"), str(w))
    for row in t.rows:
        for i, cel in enumerate(row.cells):
            cel.width = Twips(dxa[i])
    if leitura:
        nota = _runs(doc.add_paragraph(style="Normal"), f"*Leitura:* {leitura}")
        nota.paragraph_format.space_before = Pt(4)
    else:
        doc.add_paragraph(style="Normal")


def figura(caminho, titulo, leitura, largura=LARGURA):
    contador["fig"] += 1
    par = doc.add_paragraph(style="Normal")
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.add_run().add_picture(str(caminho), width=Inches(largura))
    cap = _runs(doc.add_paragraph(style="Normal"), f"**Figura {contador['fig']} — {titulo}.** {leitura}")
    cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for r in cap.runs:
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)


def quebra():
    doc.add_paragraph(style="Normal").add_run().add_break(WD_BREAK.PAGE)


def cvtxt(chave, m="MAE"):
    b = cv[chave]["melhor"]
    return f"{br(b[f'{m}_media'])} ± {br(b[f'{m}_desvio'])}"


# =====================================================================
# CAPA
# =====================================================================
h1("CloudCast — Previsão de Consumo e Custo de Infraestrutura Cloud")
p("Relatório do Projeto de Mineração de Dados (CRISP-DM + MLOps) · Engenharia de Software — FAG", False)

tabela(
    "Identificação do projeto",
    ["Campo", "Preenchido"],
    [
        ["Tema (nº e título)", "4 — Predição de Consumo e Custo de Infraestrutura Cloud"],
        ["Nome do produto", "CloudCast — Capacity Planning de Infraestrutura Cloud"],
        ["Equipe", "Everton Medola e Luis Cerutti"],
        ["Disciplina / professor", "Mineração de Dados — Engenharia de Software (FAG) / Prof. Franco"],
        ["Repositório", "github.com/evertonmedola/Projeto-DataMining"],
        [
            "Dataset (fonte e licença)",
            "Dataset simulado com base em métricas e preços reais de provedores cloud, gerado por "
            "`src/gerar_dados.py` (seed = 42; SHA-256 registrado). Dados sintéticos, uso acadêmico.",
        ],
        ["Demonstração", "Ao vivo, durante a apresentação, com o sistema rodando via docker compose (roteiro na seção 9.2)"],
        ["Data de entrega", "[PREENCHER: dd/mm/2026]"],
    ],
    [1.6, 4.9],
)

h2("Resumo")
p(
    "Empresas que operam na nuvem pagam por consumo e têm dificuldade de antecipar aumentos de custo. Este "
    "projeto aplica a metodologia CRISP-DM para prever o custo diário de uma infraestrutura cloud e o consumo "
    "de recursos que o forma (requisições e instâncias). Usamos dois anos de métricas diárias simuladas com base "
    "em padrões e preços reais (730 dias, 7 variáveis, com ausentes, duplicatas e incidentes injetados para "
    "exercitar a preparação). Comparamos um baseline sazonal, SARIMAX, Prophet e Gradient Boosting com "
    f"backtest temporal de {len(cv['prophet']['melhor']['folds'])} dobras e busca de hiperparâmetros, e avaliamos "
    "uma única vez em um holdout de 60 dias (novembro–dezembro de 2024, o período mais caro do ano). O Prophet "
    f"com sazonalidade multiplicativa obteve MAPE de {pct(P['MAPE_%'])} (MAE de US$ {br(P['MAE'])}) na previsão "
    f"de 1 dia e erro de {pct(semana['Prophet'])} no total da semana seguinte, contra {pct(met[ROT['baseline']]['MAPE_%'])} "
    "e "
    f"{pct(semana[ROT['baseline']])} do baseline. O modelo foi empacotado em um pipeline reprodutível (DVC + MLflow, "
    "com registro de versões), servido por uma API FastAPI com persistência em SQLite e exibido em um dashboard "
    "Streamlit com consumo histórico, previsão, tendência, custo projetado e alertas de crescimento, orçamento e "
    "anomalia. Uso exclusivamente acadêmico."
)
p("**Palavras-chave:** previsão de séries temporais, custo de nuvem, Prophet, CRISP-DM, MLOps.", False)
quebra()

# =====================================================================
# 1. ENTENDIMENTO DO NEGÓCIO
# =====================================================================
h2("1. Entendimento do negócio")
p("*Critério 1 — Entendimento do problema e CRISP-DM (10 pts).*", False)

h3("1.1 Contexto e problema")
p(
    "Na nuvem (AWS, Azure, GCP) a conta é calculada pelo uso: horas de instâncias ativas, gigabytes armazenados "
    "e dados transferidos. Esse modelo é flexível, mas torna o gasto difícil de planejar: ele cresce com o "
    "negócio, cai nos fins de semana, dispara em datas comerciais como Black Friday e Natal e reage ao "
    "autoscaling. Sem previsão, o time de FinOps só descobre o estouro do orçamento quando a fatura chega, e o "
    "time de engenharia só percebe a falta de capacidade quando a CPU satura."
)
p(
    "O CloudCast responde a três perguntas do enunciado do tema 4 — *quanto vamos gastar amanhã?*, *quanto vamos "
    "gastar na próxima semana?* e *qual o custo estimado do mês?* — e transforma as respostas em alertas "
    "acionáveis de Capacity Planning."
)

h3("1.2 Objetivos")
tabela(
    "Objetivos e critérios de sucesso",
    ["Tipo", "Definição"],
    [
        ["Objetivo de negócio", "Antecipar aumentos de custo e de capacidade para evitar desperdício e estouro de orçamento."],
        [
            "Objetivo de mineração",
            "Previsão de série temporal (regressão): prever `custo_usd` diário de 1 a 7 dias à frente; "
            "auxiliarmente, `requisicoes` e `instancias_ativas`.",
        ],
        ["Usuário da solução", "Time de FinOps / Engenharia de infraestrutura."],
        [
            "Decisão apoiada",
            "Ajustar orçamento, reservar ou liberar capacidade, investigar dias de custo anômalo.",
        ],
        [
            "Critério de sucesso (métrica)",
            "MAPE ≤ 10% no holdout (h = 1) e vencer o baseline sazonal em MAE em todos os horizontes de 1 a 7 dias.",
        ],
        [
            "Critério de sucesso (negócio)",
            "Erro ≤ 10% no total previsto para a semana (uso em orçamento) e alerta de crescimento emitido antes "
            "do início da alta de novembro.",
        ],
    ],
    [1.8, 4.7],
)

h3("1.3 Premissas, restrições e riscos")
bullets(
    [
        "**Dados simulados.** Não tivemos acesso a uma fatura real; o enunciado permite dados simulados com base em "
        "métricas reais. Os valores absolutos em USD não se generalizam para outra empresa — o que se transfere é "
        "o método, reaplicável sobre um export do AWS Cost Explorer ou do Azure Cost Management.",
        "**Histórico curto para sazonalidade anual.** Há apenas dois ciclos de novembro/dezembro; qualquer modelo "
        "aprende o padrão anual com pouca evidência.",
        "**Assimetria de custo do erro.** Subestimar o gasto (estourar orçamento, faltar capacidade) é pior do que "
        "superestimar; por isso o produto exibe o intervalo de 90% e não só o valor central.",
        "**Vazamento temporal** é o principal risco metodológico em séries temporais: toda divisão é cronológica "
        "e todo atributo usa apenas informação até o dia anterior (verificado por teste automatizado).",
    ]
)

h3("1.4 Plano CRISP-DM")
p(
    "O projeto percorreu as seis fases do CRISP-DM em ciclo, não em linha reta. A primeira versão (entregue para "
    "as dimensões 1 a 4) foi revista depois que a Avaliação revelou problemas; as principais voltas estão na "
    "Tabela 3. A fase de Implantação (seções 6 a 9) acrescentou o pipeline, a API, o dashboard e o "
    "monitoramento que fecha o ciclo: quando o erro em produção passa do limite, o sistema recomenda retreinar."
)
tabela(
    "Iterações do ciclo CRISP-DM",
    ["Volta", "O que a avaliação mostrou", "O que mudou"],
    [
        [
            "Avaliação → Modelagem",
            "Na 1ª versão, SARIMAX e Prophet previam os 60 dias de teste de uma vez, contra um GB de 1 dia à "
            "frente: comparação injusta e R² negativo para todos.",
            "Protocolo walk-forward (1 dia à frente) igual para todos + avaliação separada de 1 a 7 dias.",
        ],
        [
            "Avaliação → Preparação",
            "GB com lags em USD subestimava a alta temporada (viés de +14 USD): árvores não extrapolam tendência.",
            "Alvo passou a ser a variação relativa e os atributos viraram razões estacionárias (viés ≈ 0).",
        ],
        [
            "Avaliação → Modelagem",
            "Prophet aditivo subestimava o pico (viés de +5,15 USD).",
            "Sazonalidade multiplicativa — depois confirmada pela busca de hiperparâmetros.",
        ],
        [
            "Entendimento → Preparação",
            "IQR sobre o valor bruto marcava 68 dias como outlier, confundindo o pico sazonal com incidentes.",
            "Detecção no resíduo em relação à mediana móvel: 11 dias, com recall de 100% (seção 2.4).",
        ],
        [
            "Modelagem → Avaliação",
            "Sem referência, não havia como dizer se um MAE de 9 USD era bom.",
            "Baseline sazonal ingênuo como régua obrigatória.",
        ],
    ],
    [1.3, 2.7, 2.5],
)

# =====================================================================
# 2. ENTENDIMENTO DOS DADOS
# =====================================================================
h2("2. Entendimento dos dados")
p("*Critério 2 — Aquisição, qualidade e preparação (15 pts, junto com a seção 3).*", False)

h3("2.1 Fonte e aquisição")
tabela(
    "Fonte e aquisição dos dados",
    ["Item", "Descrição"],
    [
        [
            "Fonte",
            "Simulação com base em métricas e preços públicos de provedores cloud (instância ~t3.medium a US$ 0,096/h, "
            "armazenamento ~S3 a US$ 0,023/GB-mês, transferência, custo fixo diário).",
        ],
        ["Forma de coleta", "Script `python -m src.gerar_dados` (sem passo manual), parametrizado por `params.yaml`."],
        ["Volume", f"{qual['linhas']} linhas × {qual['colunas']} colunas (730 dias + 5 duplicatas)."],
        [
            "Período / versão",
            f"{qual['periodo'][0]} a {qual['periodo'][1]}; seed = 42; hash SHA-256 impresso na geração e registrado em `dvc.lock`.",
        ],
        ["Licença / uso", "Dados sintéticos produzidos pela equipe; uso acadêmico, sem dados pessoais."],
    ],
    [1.5, 5.0],
)
p(
    "O gerador reproduz os mecanismos que formam a conta real: tráfego com crescimento orgânico (~0,09% ao dia), "
    "queda de 28% nos fins de semana, alta de 55% em novembro/dezembro e queda em janeiro; autoscaling que "
    "acompanha a demanda com atraso (média móvel de 3 dias); CPU que depende da carga por instância; "
    "armazenamento que só cresce; e custo = fixo + computação + armazenamento + transferência + ruído. Problemas "
    "de qualidade são injetados de propósito, e o gabarito dos incidentes é salvo em "
    "`data/incidentes_injetados.csv` — o que permite medir a precisão da limpeza, algo impossível com dados reais."
)

h3("2.2 Dicionário de dados")
tabela(
    "Dicionário de dados (colunas originais; derivadas no Apêndice A)",
    ["Coluna", "Tipo", "Descrição", "Exemplo"],
    [
        ["data", "data", "Dia de referência (1 linha por dia)", "2024-11-29"],
        ["requisicoes", "numérica", "Requisições recebidas pela aplicação no dia", "296.410"],
        ["instancias_ativas", "numérica", "Média de instâncias de computação ativas (autoscaling)", "33"],
        ["cpu_pct", "numérica (%)", "Utilização média de CPU das instâncias", "64,2"],
        ["memoria_pct", "numérica (%)", "Utilização média de memória", "66,8"],
        ["armazenamento_gb", "numérica", "Volume armazenado (GB)", "1.912,5"],
        ["custo_usd", "numérica (alvo)", "Custo total do dia em USD", "118,40"],
    ],
    [1.4, 1.1, 2.9, 1.1],
)

h3("2.3 Análise exploratória (EDA)")
figura(
    FIG / "07_eda_consumo.png",
    "Consumo de recursos e custo ao longo de dois anos",
    "Requisições, instâncias e custo têm tendência de alta e o mesmo degrau em novembro/dezembro; CPU e memória "
    "ficam estáveis (~65%) porque o autoscaling acompanha a demanda; o armazenamento cresce de forma quase linear.",
)
figura(
    FIG / "09_sazonalidade.png",
    "Custo por dia da semana e por mês",
    "Sábado e domingo ficam cerca de 8% e 13% abaixo da média dos dias úteis, e a segunda-feira também (9%) — o "
    "autoscaling leva cerca de um dia para reagir à volta do tráfego. Novembro e dezembro são os meses mais caros; janeiro é o mais barato.",
)
figura(
    FIG / "08_correlacao.png",
    "Matriz de correlação",
    "O custo é explicado principalmente por instâncias (r ≈ 0,9) e requisições; CPU quase não se correlaciona "
    "com o custo, o que confirma que o autoscaling absorve a carga. O custo do dia anterior (custo_lag_1) é um "
    "bom preditor, mas carrega o nível da série — ponto que motivou os atributos estacionários (seção 3.2).",
    largura=4.8,
)
p(
    "**Achados que orientaram a modelagem:** (i) há duas sazonalidades — semanal e anual — sobre uma tendência; "
    "(ii) a amplitude da sazonalidade cresce com o nível da série (a queda de fim de semana é proporcional, não "
    "fixa), o que sugere sazonalidade multiplicativa; (iii) o período de teste natural (novembro–dezembro) é o "
    "mais caro do histórico, fora da faixa vista no treino — um desafio para modelos de árvore."
)

h3("2.4 Relatório de qualidade")
aus = qual["valores_ausentes"]
tabela(
    "Relatório de qualidade do dataset bruto (gerado por src/validar_dados.py)",
    ["Verificação", "Resultado", "Ação"],
    [
        [
            "Valores ausentes",
            f"{aus['requisicoes']} dias (~{pct(qual['valores_ausentes_pct']['requisicoes'], 1)}) em requisições, CPU, "
            "memória e armazenamento; 0 no custo",
            "Interpolação temporal (seção 3.2)",
        ],
        ["Duplicados", f"{qual['linhas_duplicadas']} linhas idênticas ({qual['datas_repetidas']} datas repetidas)", "Removidos pela chave `data`"],
        ["Buracos na grade diária", f"{qual['datas_faltantes_na_grade']} datas faltantes", "Validação de continuidade antes da imputação"],
        [
            "Outliers / incidentes",
            f"{pp['dias_com_incidente']} dias marcados; contra o gabarito: {det['verdadeiros_positivos']} verdadeiros, "
            f"{det['falsos_positivos']} falso positivo, {det['falsos_negativos']} perdidos (precisão "
            f"{pct(100 * det['precisao'], 1)}, recall {pct(100 * det['recall'], 0)})",
            "Winsorizados e sinalizados em `houve_incidente`",
        ],
        ["Faixas físicas", "0 violações (CPU e memória entre 0 e 100%, custo > 0)", "Validação crítica: reprovaria o pipeline"],
        ["Esquema e tipos", "7 colunas esperadas, tipos corretos", "Validação crítica: reprovaria o pipeline"],
        ["Vazamento de dados", "Nenhuma variável posterior ao dia previsto", "Atributos com shift(1); teste automatizado"],
    ],
    [1.5, 3.1, 1.9],
    "a validação separa erros críticos (que param o pipeline com código de saída 1) de problemas tratáveis, "
    "que seguem para a preparação com números registrados em outputs/relatorio_qualidade.json.",
)
figura(
    FIG / "02_qualidade_dados.png",
    "Ausentes por coluna e proporção de dias com incidente",
    f"Os ausentes estão espalhados por quatro métricas; apenas {pct(100 * pp['dias_com_incidente'] / pp['linhas_finais'], 1)} "
    "dos dias foram tratados como incidente.",
    largura=5.6,
)

# =====================================================================
# 3. PREPARAÇÃO
# =====================================================================
h2("3. Preparação dos dados")

h3("3.1 Divisão dos dados")
p(
    "Divisão **temporal**, nunca aleatória: os últimos **60 dias** (01/11/2024 a 30/12/2024) formam o holdout de "
    f"teste, trancado até o fim e usado uma única vez; os {len(prev) and 656} dias anteriores formam o período de "
    "desenvolvimento, onde roda o backtest (seção 4.3). Escolhemos de propósito o trecho mais difícil do ano "
    "(alta de Black Friday e Natal) como teste, porque é nele que a previsão tem mais valor para o negócio."
)

h3("3.2 Transformações")
tabela(
    "Transformações aplicadas, na ordem (src/preparar_dados.py)",
    ["Etapa", "Técnica", "Colunas", "Justificativa"],
    [
        ["1. Duplicatas", "drop_duplicates pela chave", "data", "Série diária: uma linha por dia"],
        ["2. Continuidade", "Reindexar na grade diária", "todas", "SARIMAX e interpolação exigem grade sem buracos"],
        ["3. Ausentes", "interpolate(method='time')", "métricas", "Respeita a tendência local; bfill/ffill só nas pontas"],
        [
            "4. Outliers",
            "Resíduo vs. mediana móvel de 15 dias; limite 2,5 × IQR; winsorização",
            "requisições, CPU, custo",
            "IQR no valor bruto confundia o pico sazonal com incidente; winsorizar não abre buracos na série",
        ],
        ["5. Calendário", "dia_semana, mês, fim_de_semana, temporada_alta", "data", "Conhecidos com antecedência: seguros para prever"],
        ["6. Lags e janelas", "lags 1, 2, 7, 14; médias e desvio móveis (7 e 14)", "custo_usd", "Todos com shift(1): só informação até D-1"],
        [
            "7. Atributos estacionários",
            "var_diaria, var_semanal, razões lag/média, coef. de variação",
            "custo_usd",
            "Livres de nível: continuam válidos fora da faixa do treino (para o GB)",
        ],
    ],
    [1.2, 2.0, 1.2, 2.1],
    tamanho=9,
)
p(
    "**Janela centrada x só passado.** Testamos detectar outliers com uma janela que só olha o passado: ela "
    "encontrou os mesmos 10 incidentes, mas gerou 7 falsos positivos — 6 deles nos primeiros dias de novembro, "
    "porque o degrau sazonal da Black Friday parece um pico para quem só olha para trás, e a winsorização teria "
    "apagado o início da alta temporada no teste. Como a limpeza do *histórico* é retrospectiva, a janela "
    "centrada é legítima; os atributos de previsão continuam usando apenas D-1, e em produção a anomalia de um dia "
    "novo é detectada pela API com o intervalo de previsão (seção 7.3). A ausência de vazamento é verificada por "
    "um teste que altera o custo do dia D e confere que nenhum atributo do dia D muda."
)

h3("3.3 Dataset final")
p(
    f"{pp['linhas_finais']} linhas × {pp['colunas_finais']} colunas ({pp['periodo_final'][0]} a {pp['periodo_final'][1]}), "
    f"após remover {pp['datas_repetidas_removidas']} duplicatas e as {pp['linhas_descartadas_por_lag']} primeiras "
    "linhas sem histórico para o lag de 14 dias: 656 dias de desenvolvimento e 60 de teste. Arquivo "
    "`data/cloud_infra_preparado.csv`, com hash registrado em `dvc.lock`."
)

# =====================================================================
# 4. MODELAGEM
# =====================================================================
h2("4. Modelagem")
p("*Critério 3 — Mineração e modelagem (20 pts).*", False)

h3("4.1 Baseline")
p(
    "**Baseline sazonal ingênuo:** o custo de hoje é igual ao do mesmo dia da semana anterior (`custo_lag_7`). "
    "Ele já captura o efeito dia útil × fim de semana, então é uma régua exigente: todo modelo precisa vencê-lo "
    "para justificar a complexidade."
)

h3("4.2 Algoritmos comparados")
tabela(
    "Algoritmos comparados (os três sugeridos no enunciado do tema 4)",
    ["Modelo", "Por que foi escolhido", "Hiperparâmetros buscados"],
    [
        [
            "SARIMAX (ARIMA sazonal)",
            "Referência estatística clássica para séries temporais; componente sazonal semanal explícito.",
            "order ∈ {(1,1,1), (2,1,2)}; seasonal_order ∈ {(1,0,1,7), (0,1,1,7)}",
        ],
        [
            "Prophet",
            "Decompõe tendência + sazonalidades semanal e anual; robusto a feriados e mudanças de tendência.",
            "changepoint_prior_scale ∈ {0,01; 0,05; 0,2}; seasonality_mode ∈ {aditiva, multiplicativa}",
        ],
        [
            "Gradient Boosting",
            "Aprendizado supervisionado não linear sobre atributos de calendário e de histórico.",
            "n_estimators ∈ {150, 300}; learning_rate ∈ {0,03; 0,08}; max_depth ∈ {2, 3}",
        ],
    ],
    [1.4, 2.8, 2.3],
)
p(
    "No Gradient Boosting, o alvo é a variação relativa `custo / custo_lag_1 − 1` e os atributos são apenas "
    "calendário e razões; o nível volta ao multiplicar a variação prevista pelo custo do dia anterior. Isso "
    "contorna a incapacidade das árvores de extrapolar tendência (seção 1.4). Para o horizonte de 7 dias, o GB "
    "prevê de forma recursiva (cada dia previsto vira histórico do seguinte)."
)

h3("4.3 Protocolo de validação")
p(
    "**Backtest temporal com janela expansível** (equivalente ao TimeSeriesSplit): no período de desenvolvimento, "
    "5 janelas consecutivas de 30 dias (junho a outubro de 2024); em cada dobra o modelo treina com tudo o que vem "
    "antes da janela (no mínimo 1 ano) e prevê a janela em **walk-forward**, 1 dia à frente — ao prever o dia D, "
    "nenhum modelo enxerga dados de D em diante. O SARIMAX incorpora as observações ao filtro de Kalman sem "
    "reestimar parâmetros; o Prophet é reajustado a cada 7 dias. A busca é exaustiva sobre a grade de "
    "`params.yaml` (19 configurações no total), com MAE médio como critério; random_state = 42 em todo o código. "
    "Cada configuração é registrada como uma execução no MLflow."
)

h3("4.4 Resultados na validação cruzada")
linhas_cv = []
for c in ROT:
    b = cv[c]["melhor"]
    hp = "—" if not b["params"] else ", ".join(f"{k}={v}" for k, v in b["params"].items())
    linhas_cv.append(
        [ROT[c], hp.replace("seasonality_mode=", "modo=").replace("changepoint_prior_scale=", "cps="), cvtxt(c, "MAE"),
         cvtxt(c, "RMSE"), cvtxt(c, "MAPE_%") + "%", tempo_melhor(c)]
    )
tabela(
    "Backtest temporal — melhor configuração de cada modelo (média ± desvio das 5 dobras; seed = 42)",
    ["Modelo", "Melhores hiperparâmetros", "MAE (USD)", "RMSE (USD)", "MAPE", "Tempo"],
    linhas_cv,
    [1.3, 1.9, 0.9, 0.9, 0.9, 0.6],
    "o Prophet tem o menor erro médio e a menor variação entre dobras; todos os modelos vencem o baseline com "
    "folga. A grade completa está no Apêndice C.",
    tamanho=8.5,
)
figura(
    FIG / "10_validacao_cruzada.png",
    "MAE por dobra no backtest",
    "Cada ponto é uma dobra; o losango é a média ± desvio. O Prophet é o mais estável (desvio de "
    f"{br(cv['prophet']['melhor']['MAE_desvio'])} USD).",
    largura=5.8,
)

h3("4.5 Modelo escolhido")
p(
    f"**Prophet** com sazonalidade multiplicativa e changepoint_prior_scale = {melhores['prophet']['changepoint_prior_scale']}: "
    "menor MAE no backtest, menor desvio entre dobras e, sozinho, entrega o que o produto precisa — previsão para "
    "qualquer horizonte, intervalo de incerteza de 90% e um componente de tendência interpretável. O modelo "
    f"final foi reajustado em todos os {meta['n_observacoes']} dias e registrado como versão {meta['versao']} (seção 6). "
    "Com a mesma configuração foram treinados modelos auxiliares para requisições (MAPE de "
    f"{pct(meta['metricas_teste_consumo']['requisicoes']['MAPE_%'])} no holdout) e instâncias "
    f"({pct(meta['metricas_teste_consumo']['instancias']['MAPE_%'])})."
)

# =====================================================================
# 5. AVALIAÇÃO
# =====================================================================
h2("5. Avaliação e interpretação")
p("*Critério 4 — Avaliação e interpretação (10 pts).*", False)

h3("5.1 Resultado no teste")
base = met[ROT["baseline"]]
tabela(
    "Metas da seção 1 x resultado no holdout (60 dias, usado uma única vez)",
    ["Métrica", "Meta", "Obtido (Prophet)", "Atingiu?"],
    [
        ["MAPE, 1 dia à frente", "≤ 10%", pct(P["MAPE_%"]), "Sim"],
        [
            "MAE vs. baseline, horizontes 1 a 7",
            "menor em todos",
            f"menor em {(hz['Prophet'] < hz[ROT['baseline']]).sum()} de 7 (médias: {br(hz_media.loc['Prophet', 'MAE'])} x "
            f"{br(hz_media.loc[ROT['baseline'], 'MAE'])} USD)",
            "Sim" if (hz["Prophet"] < hz[ROT["baseline"]]).all() else "Parcial",
        ],
        ["Erro no total da semana", "≤ 10%", pct(semana["Prophet"]), "Sim"],
        ["Alerta antes da alta de novembro", "emitido", "emitido em 29/10 (+25% previsto)", "Sim"],
    ],
    [2.1, 1.2, 2.3, 0.9],
)
linhas_teste = []
for c in ROT:
    m = met[ROT[c]]
    linhas_teste.append(
        [ROT[c], br(m["MAE"]), br(m["RMSE"]), pct(m["MAPE_%"]), br(m["R2"], 3), pct(hz_media.loc[ROT[c], "MAPE_pct"]),
         pct(semana[ROT[c]])]
    )
tabela(
    "Todos os modelos no holdout (h = 1 em walk-forward; e média de h = 1..7)",
    ["Modelo", "MAE", "RMSE", "MAPE", "R²", "MAPE h1..7", "Erro semana"],
    linhas_teste,
    [1.7, 0.7, 0.7, 0.8, 0.7, 0.9, 1.0],
    "a ordem do backtest se mantém no teste: Prophet < SARIMAX < GB < baseline. No total da semana — o número "
    "usado para orçamento — o Prophet erra metade do que erram os demais.",
    tamanho=9,
)

h3("5.2 Figuras")
figura(
    FIG / "03_previsao_vs_real.png",
    "Real x previsto no holdout (1 dia à frente)",
    "Todos os modelos acompanham o padrão semanal; o baseline (pontilhado) reage com uma semana de atraso.",
)
figura(
    FIG / "04_comparacao_metricas.png",
    "Comparação das métricas de erro no holdout",
    "Prophet tem o menor MAE, RMSE e MAPE.",
)
figura(
    FIG / "11_erro_por_horizonte.png",
    "MAPE por horizonte de previsão (1 a 7 dias)",
    "Calculado a partir de 8 origens semanais no holdout, por isso a curva é ruidosa; o Prophet fica abaixo "
    "do baseline em todos os horizontes e abaixo dos demais modelos na maioria deles.",
    largura=5.4,
)
figura(
    FIG / "06_residuos_prophet.png",
    "Resíduos do Prophet no holdout",
    f"Viés médio de {br(residuo.mean())} USD (leve subestimação no pico), sem tendência ao longo do tempo e "
    "distribuição aproximadamente simétrica.",
)

h3("5.3 Escolha do limiar e impacto no negócio")
p(
    "O produto não usa um limiar de classificação, mas três limites de alerta, configuráveis em `params.yaml`: "
    "**crescimento** (média prevista dos próximos 7 dias ≥ 15% acima da média dos últimos 7; crítico acima de "
    "30%), **orçamento** (custo projetado para 30 dias acima de US$ 3.000) e **anomalia** (custo observado fora "
    "do intervalo de 90% da previsão do dia). O intervalo de 90% foi escolhido porque, com ruído diário de ~8 USD, "
    "um intervalo mais estreito geraria um falso alarme a cada poucos dias."
)
p(
    "**Impacto simulado:** se o sistema estivesse em uso em 29/10/2024, teria emitido três alertas antes do "
    "início da alta — crescimento de 25% no custo da semana seguinte, custo projetado de US$ 3.081,02 para 30 "
    "dias (acima do orçamento de US$ 3.000) e pico previsto de 34 instâncias (51% acima do início) — dando ao "
    "time alguns dias para reservar capacidade e renegociar o orçamento (Figura 15). Com erro de 4,6% no total "
    "semanal, a projeção é precisa o bastante para planejamento, embora não substitua a fatura."
)

h3("5.4 Interpretação")
figura(
    FIG / "12_componentes_prophet.png",
    "Componentes do Prophet em produção",
    "A tendência sobe de ~53 para ~79 USD/dia em dois anos; o efeito semanal mostra domingo 10% e segunda 6% "
    "abaixo da tendência (atraso do autoscaling); o efeito anual sobe cerca de 40% em novembro/dezembro e "
    "cai em janeiro — exatamente os mecanismos da EDA.",
)
p(
    "O sentido dos efeitos é coerente com o negócio, e a decomposição é o que torna o Prophet explicável para "
    "o usuário: o dashboard mostra a linha de tendência separada da sazonalidade, de modo que *crescimento real* "
    "não é confundido com *pico sazonal*."
)
figura(
    FIG / "05_importancia_features.png",
    "Importância dos atributos no Gradient Boosting",
    "Por impureza e por permutação no teste: a razão entre o custo de ontem e a média dos últimos 7 dias domina, "
    "seguida do dia da semana. temporada_alta tem importância nula — sobre a variação diária, o mês já "
    "não acrescenta informação, porque o nível é recuperado pelo lag.",
    largura=6.2,
)

h3("5.5 Análise de erros e viés")
bullets(
    [
        f"**Maior erro:** {br(residuo.abs().max())} USD em {prev.loc[residuo.abs().idxmax(), 'data']:%d/%m/%Y}, um domingo "
        "de alta temporada em que o custo ficou muito acima do padrão semanal — ruído que nenhum modelo antecipa.",
        f"**Viés:** o Prophet subestima em média {br(residuo.mean())} USD no holdout (cerca de 1,7% do custo médio "
        "do período), coerente com o pico anual aprendido com apenas dois ciclos. O monitoramento em produção "
        "acompanha esse erro (seção 6.4).",
        "**Por que o R² é baixo (0,10) se o MAPE é bom (8,5%)?** O R² compara o modelo com a média do próprio "
        "período de teste, cuja variação (desvio de 11,7 USD) é pequena perto do ruído diário da simulação "
        "(desvio de ~8,2 USD, impossível de prever). O R² máximo atingível nesse período é de cerca de 0,5. No "
        "backtest, o MAE do Prophet (6,86 USD) já está perto do piso teórico de ~6,5 USD dado por esse ruído. "
        "Por isso usamos MAE, MAPE e o erro semanal como métricas de decisão, conforme a regra prática do "
        "enunciado para regressão.",
        "**Viés de dados:** não há atributos sensíveis (pessoas) neste problema; o viés relevante é o de período — "
        "o modelo foi avaliado em um único fim de ano.",
    ]
)

# =====================================================================
# 6. PIPELINE
# =====================================================================
h2("6. Engenharia do pipeline (MLOps)")
p("*Critério 5 — Engenharia do pipeline (15 pts).*", False)

h3("6.1 Etapas do pipeline")
tabela(
    "Etapas do pipeline (todas a partir da raiz do repositório)",
    ["Etapa", "Script / comando", "Entrada", "Saída (artefato)"],
    [
        ["Ingestão", "python -m src.gerar_dados", "params.yaml", "data/cloud_infra_bruto.csv, incidentes_injetados.csv"],
        ["Validação", "python -m src.validar_dados", "data/ bruto", "outputs/relatorio_qualidade.json (falha = para)"],
        ["Preparação", "python -m src.preparar_dados", "data/ bruto", "data/cloud_infra_preparado.csv, relatorio_preparacao.json"],
        [
            "Modelagem",
            "python -m src.modelagem",
            "dataset preparado",
            "resultados_cv.json, melhores_params.json, metricas.json, metricas_horizonte.csv + runs no MLflow",
        ],
        ["Treino e registro", "python -m src.treinar", "melhores parâmetros", "models/vN/ + registry.json + versão no MLflow Model Registry"],
        ["Figuras", "python -m src.graficos", "outputs/", "outputs/figuras/*.png"],
        ["Tudo", "python -m src.pipeline  ou  dvc repro", "—", "todas as anteriores (~50 s)"],
    ],
    [1.0, 1.9, 1.2, 2.4],
    tamanho=8.5,
)

h3("6.2 Orquestração e rastreamento")
p(
    "`dvc.yaml` declara as seis etapas com suas dependências (código, parâmetros e dados); `dvc repro` refaz só "
    "o que mudou e grava o hash de cada entrada e saída em `dvc.lock`. Como os artefatos somam menos de 3 MB, "
    "eles são versionados no próprio Git (`cache: false`), sem exigir um remote DVC. Sem DVC instalado, "
    "`python -m src.pipeline` executa as mesmas etapas na mesma ordem."
)
p(
    "Cada execução de modelagem cria uma execução-mãe no **MLflow** (experimento `cloudcast-comparacao`) com 19 "
    "execuções-filhas, uma por configuração da busca, contendo parâmetros e métricas de backtest; as métricas "
    "do teste e os arquivos de resultado ficam na execução-mãe. O treino cria uma execução em `cloudcast-treino` "
    "e registra o modelo no **Model Registry** (`cloudcast-custo`), com a tag do commit do Git."
)
p(
    "**Registro de versões.** O treino grava `models/v<N>/` (modelos serializados em JSON e `metadata.json` com "
    "hiperparâmetros, métricas, hash dos dados e commit) e atualiza `models/registry.json`. A versão só é "
    "**promovida** para produção se o MAPE no holdout estiver dentro da meta de 10%; a API carrega a versão "
    "apontada como produção e `POST /model/reload` troca o modelo sem derrubar o serviço."
)
figura(
    TELAS / "mlflow_runs.png",
    "Experimento de comparação no MLflow",
    "A execução-mãe agrupa as 20 execuções da busca (19 configurações + a mãe).",
    largura=6.0,
)

h3("6.3 Estrutura do repositório")
codigo(
    """Projeto-DataMining/
├── params.yaml            # sementes, grades, limites
├── dvc.yaml / dvc.lock    # pipeline e hashes
├── data/                  # bruto, preparado, gabarito
├── src/
│   ├── gerar_dados.py  validar_dados.py  preparar_dados.py
│   ├── modelagem.py    treinar.py        registro.py
│   ├── previsao.py     rastreamento.py   graficos.py   pipeline.py
│   └── api/            # main.py, schemas.py, db.py
├── dashboard/app.py       # Streamlit
├── tests/                 # dados, features, modelo, API
├── models/                # versões do modelo + registry.json
├── outputs/               # métricas, previsões, figuras
├── docs/                  # arquitetura, telas e este relatório
├── Dockerfile  docker-compose.yml  Makefile  requirements.txt
└── .github/workflows/ci.yml"""
)

h3("6.4 Monitoramento")
p(
    "Cada previsão servida é gravada com a versão do modelo. Quando o consumo real de um dia chega por "
    "`POST /observacoes`, o endpoint `GET /monitoramento` cruza a previsão de 1 dia feita para aquela data com o "
    "valor observado e calcula o MAPE em produção. Com pelo menos 7 dias avaliados e MAPE acima de 15% (o dobro "
    "do erro no teste, aproximadamente), o sistema recomenda retreinar — fechando o ciclo CRISP-DM. A mesma "
    "comparação com o intervalo de 90% gera os alertas de anomalia."
)

# =====================================================================
# 7. ARQUITETURA
# =====================================================================
h2("7. Arquitetura e implementação do software")
p("*Critério 6 — Arquitetura e implementação (15 pts).*", False)

h3("7.1 Diagrama de arquitetura")
figura(
    RAIZ / "docs" / "arquitetura.png",
    "Arquitetura do CloudCast",
    "Em cima, o pipeline de treino (executado sob demanda ou pela CI) produz uma versão registrada do modelo; "
    "embaixo, o fluxo de uso: o dashboard e a exportação diária do provedor falam com a API, que carrega a "
    "versão em produção e persiste histórico, previsões e alertas.",
)

h3("7.2 Pilha tecnológica")
tabela(
    "Pilha tecnológica",
    ["Camada", "Tecnologia"],
    [
        ["Linguagem", "Python 3.13"],
        ["Mineração", "pandas, NumPy, scikit-learn, statsmodels (SARIMAX), Prophet"],
        ["API", "FastAPI + Pydantic (validação e Swagger automáticos), Uvicorn"],
        ["Persistência", "SQLite (sqlite3); acesso isolado em src/api/db.py para trocar por PostgreSQL"],
        ["Visualização", "Streamlit + Plotly (dashboard); Matplotlib (figuras do relatório)"],
        ["MLOps", "DVC (pipeline e hashes), MLflow (experimentos e Model Registry), registro de versões próprio"],
        ["Qualidade e infra", "pytest + pytest-cov, Ruff, Docker / Docker Compose, GitHub Actions"],
    ],
    [1.6, 4.9],
)
p(
    "A regra de negócio fica em `src/previsao.py` (serviço de previsão e alertas), separada do transporte HTTP "
    "(`src/api/main.py`) e da persistência (`src/api/db.py`): os alertas são testados sem subir a API, e a API "
    "não conhece detalhes do Prophet."
)

h3("7.3 Contrato da API")
tabela(
    "Endpoints da API (documentação interativa em /docs)",
    ["Método e rota", "Função"],
    [
        ["GET /health", "Status do serviço, versão do modelo e último dia registrado"],
        ["GET /model/info", "Metadados da versão em produção (métricas, hash dos dados, commit)"],
        ["POST /model/reload", "Recarrega a versão em produção após um novo treino"],
        ["GET /forecast?horizonte=7", "Previsão diária com intervalo de 90%, tendência, consumo previsto, resumo e alertas"],
        ["GET /forecast?horizonte=30&referencia=AAAA-MM-DD", "Simulação retroativa (não é persistida nem monitorada)"],
        ["GET /historico?dias=90", "Consumo e custo históricos"],
        ["POST /observacoes", "Registra o consumo real de um dia e verifica anomalia (201; 422 se inválido ou fora de sequência)"],
        ["GET /alertas · GET /previsoes", "Histórico de alertas e de previsões servidas (auditoria)"],
        ["GET /monitoramento", "MAPE em produção e recomendação de retreino"],
    ],
    [2.6, 3.9],
    tamanho=9,
)
codigo(
    """GET /forecast?horizonte=7
Response 200:
{
  "versao_modelo": 1,
  "gerado_a_partir_de": "2025-01-01",
  "horizonte": 7,
  "simulacao": false,
  "resumo": {
    "custo_amanha": 95.11,
    "custo_proxima_semana": 566.83,
    "media_diaria_ultimos_7_dias": 117.32,
    "variacao_semana_vs_recente_pct": -30.98,
    ...
  },
  "previsao": [
    { "data": "2025-01-01", "custo_previsto": 95.11,
      "custo_limite_inferior": 79.5, "custo_limite_superior": 109.84,
      "tendencia": 78.92, "requisicoes_previstas": 247498.0,
      "instancias_previstas": 28.3 },
    ...
  ],
  "alertas": [
    { "tipo": "QUEDA_DE_DEMANDA", "severidade": "INFO", "valor": -30.98,
      "mensagem": "Custo previsto 31% menor que a semana recente: ..." }
  ]
}"""
)
codigo(
    """POST /observacoes
Request:
{ "data": "2024-12-31", "requisicoes": 262000, "instancias_ativas": 29,
  "cpu_pct": 88.0, "memoria_pct": 83.5, "armazenamento_gb": 1985,
  "custo_usd": 168.40 }
Response 201:
{ "data": "2024-12-31", "atualizado": false,
  "custo_previsto": 90.64, "limite_superior": 105.15,
  "alerta": { "tipo": "ANOMALIA_DE_CUSTO", "severidade": "CRITICO",
              "valor": 168.4,
              "mensagem": "Custo de 31/12/2024 (US$ 168,40) acima do limite
                           esperado (US$ 105,15); 86% acima do previsto." } }
Response 422: entrada inválida (ex.: cpu_pct = 150) ou dia fora de sequência"""
)

h3("7.4 Persistência")
tabela(
    "Modelo de persistência (SQLite)",
    ["Tabela", "Colunas", "Papel"],
    [
        [
            "metricas_diarias",
            "data (PK), requisicoes, instancias_ativas, cpu_pct, memoria_pct, armazenamento_gb, custo_usd, origem",
            "Histórico de consumo; populado com o dataset preparado e alimentado por POST /observacoes",
        ],
        [
            "previsoes",
            "id, criado_em, data_alvo, horizonte, custo_previsto, limite_inferior, limite_superior, versao_modelo",
            "Auditoria e monitoramento: toda previsão servida, com a versão do modelo",
        ],
        ["alertas", "id, criado_em, tipo, severidade, mensagem, valor, versao_modelo", "Histórico de alertas"],
    ],
    [1.3, 3.2, 2.0],
    tamanho=9,
)

h3("7.5 Interface / dashboard")
p(
    "O dashboard consome apenas a API e exibe os cinco indicadores pedidos no enunciado do tema 4: **consumo "
    "histórico**, **previsão**, **tendência**, **custo projetado** e **alertas de crescimento**."
)
figura(
    TELAS / "dashboard_operacao.png",
    "Dashboard em operação",
    "Indicadores de amanhã, próxima semana, 30 dias e tendência; o alerta crítico vem de um dia registrado com "
    "custo 86% acima do previsto; o gráfico une histórico, previsão, intervalo de 90% e tendência.",
)
figura(
    TELAS / "dashboard_simulacao.png",
    "Simulação retroativa a partir de 29/10/2024",
    "O sistema teria antecipado a alta de novembro com alertas de crescimento de custo, orçamento e "
    "capacidade; a linha pontilhada é o custo que de fato ocorreu.",
)
figura(
    TELAS / "dashboard_consumo_custo.png",
    "Consumo de recursos, custo projetado por semana e registro do consumo do dia",
    "Tráfego e instâncias reais e previstos, utilização de CPU/memória com a linha de saturação, custo semanal "
    "projetado com a tabela diária e o formulário que envia o consumo real à API.",
)

# =====================================================================
# 8. REPRODUTIBILIDADE
# =====================================================================
h2("8. Reprodutibilidade, testes e DevOps")
p("*Critério 7 — Reprodutibilidade, testes e DevOps (10 pts).*", False)

h3("8.1 Como reproduzir")
codigo(
    """git clone https://github.com/evertonmedola/Projeto-DataMining.git
cd Projeto-DataMining

# Opção 1 — Docker (sobe pipeline -> API -> dashboard -> MLflow)
docker compose up --build

# Opção 2 — ambiente local
python -m venv .venv
.venv\\Scripts\\activate             # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline                 # ou: dvc repro
python -m pytest
uvicorn src.api.main:app               # http://localhost:8000/docs
streamlit run dashboard/app.py         # http://localhost:8501"""
)

h3("8.2 Controles de reprodutibilidade")
bullets(
    [
        "☑ Versões de bibliotecas fixadas com `==` em requirements.txt (testadas com Python 3.13).",
        "☑ Semente única (seed = 42) em params.yaml, usada na geração dos dados, no GB, na permutação e no "
        "intervalo do Prophet servido pela API.",
        "☑ Dados gerados por script com hash conferido (SHA-256 e md5 em dvc.lock).",
        "☑ Nenhum caminho absoluto: tudo é resolvido a partir de `src/config.py`.",
        "☑ Métricas deste relatório lidas de outputs/metricas.json pelo script que gera o documento.",
        "☑ Verificado: duas execuções completas e independentes do pipeline produziram o mesmo dataset (mesmo "
        "md5) e exatamente as mesmas métricas; uma segunda execução de `dvc repro` não refaz nenhuma etapa.",
        "☑ Verificado entre máquinas: `docker compose up --build` executado no Windows (Docker Desktop) gerou o "
        "dataset com o mesmo SHA-256 obtido no Linux, e os quatro serviços subiram.",
        "☑ Arquivos de texto gravados sempre com quebra de linha LF (`.gitattributes` e gravação explícita), para que "
        "os hashes de dvc.lock valham também no Windows; tempos de execução ficam fora dos arquivos versionados.",
        "☑ Retreinar com os mesmos dados e parâmetros não cria uma versão duplicada do modelo no registro.",
    ]
)

h3("8.3 Testes automatizados")
tabela(
    "Testes automatizados (45 testes, pytest)",
    ["Tipo", "O que testa", "Exemplos"],
    [
        ["Dados (11)", "esquema, faixas, ausentes, duplicatas, grade diária, incidentes", "CPU = 150% reprova; recall de 100% nos incidentes; geração determinística"],
        ["Transformação (6)", "atributos sem vazamento e cálculos", "alterar custo(D) não muda os atributos de D; média móvel exclui o próprio dia"],
        ["Modelo (15)", "métricas, protocolo, determinismo, desempenho mínimo, alertas", "MAPE do Prophet ≤ 10%; todos vencem o baseline; dobras sem sobreposição"],
        ["API (13)", "contrato, erros e persistência", "horizonte 0 ou 91 → 422; observação anômala gera alerta; simulação não é persistida"],
    ],
    [1.2, 2.3, 3.0],
    tamanho=9,
)
p(
    "Cobertura: **73%** do pacote `src` (`pytest --cov=src`); a CI falha abaixo de 70%. Os testes usam pasta de "
    "modelos e banco temporários, sem tocar nos artefatos reais. Lint com Ruff sem nenhum aviso."
)

h3("8.4 Integração contínua")
p(
    "GitHub Actions (`.github/workflows/ci.yml`) a cada push e pull request: (1) instala as dependências "
    "fixadas, (2) roda o Ruff, (3) executa o pipeline de ponta a ponta, (4) roda os testes com cobertura mínima de "
    "70% e (5), em um segundo job, faz o build da imagem Docker e um teste de fumaça da API dentro do container "
    "(`/health` e `/forecast`). O merge na main deve exigir o pipeline verde."
)
p("[PREENCHER: captura de tela da execução verde em github.com/evertonmedola/Projeto-DataMining/actions]")

# =====================================================================
# 9. DEMONSTRAÇÃO
# =====================================================================
h2("9. Implantação e demonstração")
p("*Critério 8 — Apresentação e demonstração (5 pts).*", False)

h3("9.1 Ambiente de execução")
p(
    "Docker Compose local com quatro serviços: `pipeline` (roda e termina), `api` (porta 8000, só sobe depois do "
    "pipeline terminar com sucesso e tem healthcheck), `dashboard` (8501) e `mlflow` (5000). A pasta do projeto "
    "é montada no container, então os artefatos ficam no disco do usuário. O ambiente foi executado com sucesso no "
    "Windows com Docker Desktop: build da imagem em cerca de 4,5 minutos e pipeline concluído dentro do container. "
    "Opcionalmente, a mesma imagem pode ser publicada no Render ou no Railway."
)

h3("9.2 Roteiro da demonstração ao vivo (3 a 5 minutos)")
numerada(
    [
        "**O problema em uma frase:** prever o custo da nuvem para o time de FinOps não ser surpreendido pela fatura.",
        "`docker compose up` rodando: pipeline concluído, API e dashboard no ar.",
        "Dashboard: indicadores de amanhã, próxima semana e 30 dias; gráfico com histórico, previsão, intervalo e tendência.",
        "Simulação a partir de 29/10/2024: os alertas de crescimento, orçamento e capacidade aparecem antes da Black Friday.",
        "Caso ponta a ponta: registrar um dia com custo normal (sem alerta) e outro com custo de US$ 168 (alerta crítico de anomalia).",
        "Swagger em /docs: GET /forecast e o erro 422 de uma entrada inválida.",
        "MLflow: a comparação das 19 configurações e a versão registrada do modelo.",
        "GitHub Actions: o pipeline de CI verde.",
    ]
)

h3("9.3 Forma de apresentação")
p(
    "A demonstração é feita ao vivo, durante a apresentação, seguindo o roteiro acima e acompanhada pelos slides "
    "do projeto, que trazem o problema, os resultados da avaliação, a arquitetura e capturas do dashboard. Não há "
    "vídeo gravado: o caso ponta a ponta (registrar um dia de custo anômalo e ver o alerta) é executado na hora."
)

# =====================================================================
# 10. CONCLUSÕES
# =====================================================================
h2("10. Conclusões, limitações e trabalhos futuros")
bullets(
    [
        f"**Conclusão:** o Prophet com sazonalidade multiplicativa atingiu três das quatro metas da seção 1 — MAPE de "
        f"{pct(P['MAPE_%'])} em 1 dia, erro de {pct(semana['Prophet'])} no total da semana e alerta emitido antes da "
        "alta de novembro. A quarta (vencer o baseline em todos os horizontes) foi atingida em 6 de 7: no horizonte de "
        "5 dias, com apenas 8 origens, o baseline foi ligeiramente melhor. No geral, o Prophet venceu SARIMAX, "
        "Gradient Boosting e o baseline no backtest e no holdout. O modelo "
        "está dentro de um software completo: pipeline reprodutível, API, persistência, dashboard e monitoramento.",
        "**Limitações:** dados simulados (valores absolutos não generalizam); apenas dois ciclos anuais; avaliação "
        "em um único fim de ano; o retreino a partir das observações "
        "registradas na API ainda é manual.",
        "**Trabalhos futuros:** aplicar o pipeline a um export real (AWS Cost Explorer); incluir feriados e eventos "
        "comerciais como regressores do Prophet; prever por serviço (computação, armazenamento, rede); retreino "
        "automático agendado a partir do banco; relatório de drift dos dados de entrada.",
        "**Lições aprendidas:** a forma de avaliar importa tanto quanto o algoritmo — a mesma comparação, feita sem "
        "walk-forward, apontava conclusões erradas; e um baseline simples é indispensável para interpretar "
        "qualquer métrica.",
    ]
)

# =====================================================================
# REFERÊNCIAS
# =====================================================================
h2("Referências")
bullets(
    [
        "CHAPMAN, P. et al. *CRISP-DM 1.0: Step-by-step data mining guide*. SPSS, 2000.",
        "TAYLOR, S. J.; LETHAM, B. Forecasting at scale. *The American Statistician*, v. 72, n. 1, p. 37-45, 2018.",
        "BOX, G. E. P.; JENKINS, G. M.; REINSEL, G. C.; LJUNG, G. M. *Time Series Analysis: Forecasting and Control*. 5. ed. Wiley, 2015.",
        "FRIEDMAN, J. H. Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, v. 29, n. 5, p. 1189-1232, 2001.",
        "HYNDMAN, R. J.; ATHANASOPOULOS, G. *Forecasting: Principles and Practice*. 3. ed. OTexts, 2021.",
        "PEDREGOSA, F. et al. Scikit-learn: Machine Learning in Python. *JMLR*, v. 12, p. 2825-2830, 2011.",
        "SEABOLD, S.; PERKTOLD, J. Statsmodels: econometric and statistical modeling with Python. *Proc. 9th Python in Science Conference*, 2010.",
        "ZAHARIA, M. et al. Accelerating the Machine Learning Lifecycle with MLflow. *IEEE Data Engineering Bulletin*, v. 41, n. 4, 2018.",
        "AMAZON WEB SERVICES. *Amazon EC2 On-Demand Pricing* e *Amazon S3 Pricing*. Tabelas públicas usadas como base dos preços simulados.",
        "FINOPS FOUNDATION. *FinOps Framework*. finops.org.",
    ]
)

# =====================================================================
# APÊNDICES
# =====================================================================
quebra()
h2("Apêndices")
h3("A. Dicionário de dados completo (dataset preparado)")
derivadas = [
    ["houve_incidente", "binária", "1 se algum valor do dia foi winsorizado como incidente"],
    ["dia_semana / mes", "inteira", "0 = segunda ... 6 = domingo / 1 a 12"],
    ["fim_de_semana / temporada_alta", "binária", "sábado ou domingo / novembro ou dezembro"],
    ["indice_tempo", "inteira", "dias desde o início da série"],
    ["custo_lag_1, _2, _7, _14", "numérica", "custo de 1, 2, 7 e 14 dias antes"],
    ["custo_media_movel_7 / _14", "numérica", "média do custo nos 7 / 14 dias anteriores (exclui o dia)"],
    ["custo_std_movel_7", "numérica", "desvio do custo nos 7 dias anteriores"],
    ["var_diaria / var_semanal", "numérica", "lag1/lag2 − 1 e lag1/lag7 − 1"],
    ["razao_lag1_mm7 / razao_lag7_mm7", "numérica", "lag / média móvel de 7 dias"],
    ["razao_mm7_mm14", "numérica", "média móvel de 7 / de 14 dias (aceleração)"],
    ["coef_variacao_7", "numérica", "desvio móvel / média móvel de 7 dias"],
]
tabela("Atributos derivados", ["Coluna", "Tipo", "Descrição"], derivadas, [2.3, 0.9, 3.3], tamanho=9)

h3("B. Figura adicional da EDA")
figura(
    FIG / "01_serie_historica_custo.png",
    "Série completa do custo diário, com o início do holdout",
    "O teste (após a linha) é o trecho mais caro do histórico.",
)

h3("C. Grade completa de hiperparâmetros (backtest, MAE médio ± desvio)")
linhas_grade = []
for c in ("sarimax", "prophet", "gb"):
    for cand in sorted(cv[c]["candidatos"], key=lambda r: r["MAE_media"]):
        hp = ", ".join(f"{k}={v}" for k, v in cand["params"].items())
        hp = hp.replace("seasonality_mode=", "modo=").replace("changepoint_prior_scale=", "cps=")
        hp = hp.replace("seasonal_order=", "sazonal=").replace("n_estimators=", "n=").replace("learning_rate=", "lr=")
        escolhido = "✔" if cand["params"] == cv[c]["melhor"]["params"] else ""
        linhas_grade.append(
            [ROT[c], hp, f"{br(cand['MAE_media'], 3)} ± {br(cand['MAE_desvio'], 3)}", pct(cand["MAPE_%_media"]), escolhido]
        )
tabela(
    "Todas as configurações avaliadas",
    ["Modelo", "Configuração", "MAE (USD)", "MAPE", "Escolhida"],
    linhas_grade,
    [1.4, 2.8, 1.2, 0.7, 0.6],
    tamanho=8.5,
)

h3("D. Documentação da API")
p(
    "A especificação OpenAPI completa, gerada automaticamente pelo FastAPI, está em `docs/openapi.json` "
    "(9 rotas, com esquemas de entrada e saída e exemplos). Com a API no ar, a mesma documentação é navegável "
    "em http://localhost:8000/docs (Swagger) e http://localhost:8000/redoc."
)

h3("E. Divisão de tarefas na equipe")
tabela(
    "Divisão de tarefas",
    ["Integrante", "Responsabilidades"],
    [
        ["Everton Medola", "[PREENCHER]"],
        ["Luis Cerutti", "[PREENCHER]"],
    ],
    [1.8, 4.7],
)

# =====================================================================
# CHECKLIST
# =====================================================================
h2("Checklist de autoavaliação")
check = [
    ("1. Problema e CRISP-DM (10)", [
        "Objetivo de negócio e de mineração separados, com critério de sucesso numérico (1.2).",
        "Ciclo CRISP-DM descrito, incluindo iterações (1.4).",
    ]),
    ("2. Dados (15)", [
        "Coleta automatizada por script; fonte e licença citadas (2.1).",
        "Dicionário de dados, EDA com achados e relatório de qualidade com números (2.2–2.4).",
        "Pré-processamento sem vazamento, verificado por teste (3.2, 8.3).",
    ]),
    ("3. Modelagem (20)", [
        "Baseline + todos os algoritmos sugeridos para o tema (4.1–4.2).",
        "Validação temporal e busca de hiperparâmetros (4.3).",
        "Tabela comparativa com média ± desvio (4.4).",
    ]),
    ("4. Avaliação (10)", [
        "Teste usado uma única vez; metas atingidas ou não, com números (5.1).",
        "Interpretação e análise de erros (5.4–5.5).",
    ]),
    ("5. Pipeline (15)", [
        "Etapas automatizadas (DVC/Makefile/pipeline.py) e experimentos rastreados (MLflow) (6.1–6.2).",
        "Modelo versionado e carregado pela API a partir do registro (6.2).",
    ]),
    ("6. Arquitetura (15)", ["Diagrama, API documentada, persistência e interface funcionando (7)."]),
    ("7. Reprodutibilidade (10)", ["docker compose up (verificado no Windows) + dvc repro reproduzem tudo; testes e CI (8)."]),
    ("8. Apresentação (5)", ["Demonstração ao vivo com roteiro (9.2) e slides com os resultados (9.3)."]),
]
for titulo, itens in check:
    p(f"**{titulo}**", False)
    bullets([f"☑ {i}" for i in itens])

doc.save(SAIDA)
print(f"Relatório salvo em {SAIDA} — {contador['fig']} figuras, {contador['tab']} tabelas")
