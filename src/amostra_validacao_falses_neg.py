"""Amostra estratificada dos itens classificados como False, para achar falsos negativos.

A amostra de ``amostra_validacao.py`` sai só dos arquivos True e mede a precisão.
Para estimar a cobertura (quantos itens de alimentação escolar ficaram de fora), é
preciso validar também os False. Um sorteio simples quase não acharia falsos
negativos: nos serviços, 96% dos False nem têm contexto alimentar. Por isso cada
item cai em um estrato, do mais ao menos suspeito, e cada estrato tem o seu
tamanho de amostra.

A planilha ``output/periodo/amostra_validacao_falses_neg_<período>.xlsx`` tem uma
aba por classificador e a aba "Estratos", com o total de itens e o peso de cada
estrato. O peso (itens do estrato / itens sorteados) expande os falsos negativos
achados na amostra para o total do período.

Exemplo:
    python src/amostra_validacao_falses_neg.py --inicio 2025-01 --fim 2025-03
"""

import argparse

import numpy as np
import pandas as pd
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from main import (
    ARQUIVOS_PERIODO,
    COLUNAS_CHAVE_ITEM,
    PASTA_PERIODO,
    SAIDA_PADRAO,
    caminhos_itens,
    caminhos_servicos,
    chave_item,
    meses_do_periodo,
    parse_ano_mes,
    remover_versoes_antigas,
    sufixo_periodo,
)

COLUNAS_ITEM = ["coleta_ano", "coleta_mes", "numeroControlePNCP", "numeroItem", "descricao"]
COLUNAS_CLASSIFICACAO = {
    "materiais": [
        "nome_pdm_detectado",
        "descricao_item_catmat",
        "grupo_taxonomico",
        "objetoCompra",
        "amparoLegal.codigo",
        "termos_escolar_fraco_objeto",
        "termos_escolar_indicio_objeto",
        "termos_outro_contexto_objeto",
        "unidadeOrgao.nomeUnidade",
        "orgaoEntidade.razaoSocial",
        "orgaoEntidade.esferaId",
    ],
    "servicos": [
        "objetoCompra",
        "amparoLegal.codigo",
        "nome_servico_catser",
        "similaridade",
        "grupo_proposto",
        "metodo_classificacao",
        "termos_alimentar_objeto",
        "termos_outro_contexto_objeto",
    ],
}

# Estratos em ordem de prioridade: o item fica no primeiro em que se encaixa.
# O número é o tamanho da amostra do estrato.
ESTRATOS = {
    "materiais": {
        "termo fraco anulado por outro contexto": 50,
        "só indício escolar": 30,
        "unidade ou órgão de educação": 50,
        "demais municipais": 40,
        "demais": 30,
    },
    "servicos": {
        "sem contexto escolar": 90,
        "abaixo de 0,4 sem função": 40,
        "faixa 0,4 a 0,5 sem termos": 30,
        "sem contexto alimentar": 40,
    },
}
# Nome da unidade compradora ou do órgão que indica a área de educação.
PADRAO_EDUCACAO = r"educa|escol|ensino"

COLUNAS_VALIDACAO = ["deveria_ser_true", "observacao"]
OPCOES_VALIDACAO = "Sim,Não,Dúvida"
NOME_ABA = {"materiais": "Alimentos False", "servicos": "Serviços False"}


def ler_materiais(sufixos: list[str], periodo: str) -> pd.DataFrame:
    """Alimentos candidatos sem contexto escolar no período, com o estrato."""
    falsos = pd.read_csv(
        PASTA_PERIODO / ARQUIVOS_PERIODO["alimentos_false"].format(periodo),
        dtype=str,
        keep_default_na=False,
    )
    educacao = falsos["unidadeOrgao.nomeUnidade"].str.contains(
        PADRAO_EDUCACAO, case=False
    ) | falsos["orgaoEntidade.razaoSocial"].str.contains(PADRAO_EDUCACAO, case=False)
    condicoes = [
        falsos["termos_escolar_fraco_objeto"].ne(""),
        falsos["termos_escolar_indicio_objeto"].ne(""),
        educacao,
        falsos["orgaoEntidade.esferaId"].eq("M"),
    ]
    nomes = list(ESTRATOS["materiais"])
    falsos["estrato"] = np.select(condicoes, nomes[:-1], default=nomes[-1])
    return falsos


def ler_servicos(sufixos: list[str], periodo: str) -> pd.DataFrame:
    """Serviços classificados como False no período, com o estrato.

    O período não guarda os False dos serviços. Eles são lidos dos arquivos
    mensais e, como em ``main.juntar_periodo``, um item coletado de novo num
    mês posterior vale pela versão mais recente.
    """
    colunas = set(COLUNAS_ITEM + COLUNAS_CLASSIFICACAO["servicos"] + COLUNAS_CHAVE_ITEM)
    tabelas, chaves_base = [], []
    for sufixo in sufixos:
        tabelas.append(
            pd.read_csv(
                caminhos_servicos(sufixo)["false"],
                dtype=str,
                keep_default_na=False,
                usecols=lambda coluna: coluna in colunas,
            )
        )
        base = pd.read_csv(
            caminhos_itens(SAIDA_PADRAO, sufixo)["todos"],
            dtype=str,
            keep_default_na=False,
            usecols=COLUNAS_CHAVE_ITEM,
        )
        chaves_base.append(set(chave_item(base)))
    falsos = pd.concat(remover_versoes_antigas(tabelas, chaves_base), ignore_index=True)
    falsos["estrato"] = falsos["metodo_classificacao"]
    return falsos


def sortear(falsos: pd.DataFrame, tipo: str, semente: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sorteia cada estrato e devolve a amostra e o resumo dos estratos."""
    colunas = ["estrato", *COLUNAS_ITEM, *COLUNAS_CLASSIFICACAO[tipo]]
    fora = set(falsos["estrato"]).difference(ESTRATOS[tipo])
    if fora:
        print(f"Aviso: estratos sem amostra definida ficam fora do sorteio: {sorted(fora)}")

    partes, resumo = [], []
    for estrato, tamanho in ESTRATOS[tipo].items():
        grupo = falsos.loc[falsos["estrato"].eq(estrato), colunas]
        parte = grupo.sample(n=min(tamanho, len(grupo)), random_state=semente)
        partes.append(parte.sort_values(["coleta_ano", "coleta_mes", "numeroControlePNCP"]))
        resumo.append(
            {
                "classificador": tipo,
                "estrato": estrato,
                "itens_no_estrato": len(grupo),
                "itens_sorteados": len(parte),
                "peso": round(len(grupo) / len(parte), 2) if len(parte) else None,
            }
        )
    amostra = pd.concat(partes, ignore_index=True)
    for coluna in COLUNAS_VALIDACAO:
        amostra[coluna] = ""
    print(f"{NOME_ABA[tipo]}: {len(amostra)} de {len(falsos):,} itens")
    return amostra, pd.DataFrame(resumo)


def gravar_planilha(amostras: dict[str, pd.DataFrame], estratos: pd.DataFrame, destino) -> None:
    """Grava uma aba por amostra e a aba dos estratos, com filtro e lista de opções."""
    with pd.ExcelWriter(destino, engine="openpyxl") as escritor:
        tabelas = {NOME_ABA[tipo]: amostra for tipo, amostra in amostras.items()}
        tabelas["Estratos"] = estratos
        for nome, tabela in tabelas.items():
            tabela.to_excel(escritor, sheet_name=nome, index=False)
            aba = escritor.sheets[nome]
            aba.auto_filter.ref = aba.dimensions
            aba.freeze_panes = "A2"
            for posicao, coluna in enumerate(tabela.columns, start=1):
                maior = max([len(coluna), *tabela[coluna].astype(str).str.len().head(50)])
                aba.column_dimensions[get_column_letter(posicao)].width = min(maior + 2, 60)
            if "deveria_ser_true" in tabela:
                letra = get_column_letter(tabela.columns.get_loc("deveria_ser_true") + 1)
                opcoes = DataValidation(type="list", formula1=f'"{OPCOES_VALIDACAO}"')
                opcoes.add(f"{letra}2:{letra}{len(tabela) + 1}")
                aba.add_data_validation(opcoes)


def main() -> None:
    """Lê os argumentos, sorteia os estratos e grava a planilha."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inicio", type=parse_ano_mes, required=True, help="Mês inicial (AAAA-MM)")
    parser.add_argument("--fim", type=parse_ano_mes, required=True, help="Mês final (AAAA-MM)")
    parser.add_argument("--semente", type=int, default=42, help="Semente do sorteio")
    args = parser.parse_args()

    periodo = sufixo_periodo(args.inicio, args.fim)
    sufixos = [sufixo_periodo(mes, mes) for mes in meses_do_periodo(args.inicio, args.fim)]
    leitores = {"materiais": ler_materiais, "servicos": ler_servicos}
    amostras, estratos = {}, []
    for tipo, ler in leitores.items():
        amostras[tipo], resumo = sortear(ler(sufixos, periodo), tipo, args.semente)
        estratos.append(resumo)
    estratos = pd.concat(estratos, ignore_index=True)
    print(estratos.to_string(index=False))

    destino = PASTA_PERIODO / f"amostra_validacao_falses_neg_{periodo}.xlsx"
    gravar_planilha(amostras, estratos, destino)
    print(f"Planilha: {destino}")


if __name__ == "__main__":
    main()
