"""Amostra dos resultados do pipeline para validação manual.

Sorteia, com semente fixa, linhas dos arquivos do período gerados por
``src/main.py``: os alimentos candidatos com contexto escolar (o arquivo True;
os sem contexto escolar ficam fora da amostra) e os serviços classificados como
alimentação escolar. Cada amostra vai para uma aba
de ``output/periodo/amostra_validacao_<período>.xlsx``, com as colunas usadas
na classificação e colunas em branco para a validação.

Exemplo:
    python src/amostra_validacao.py --inicio 2025-01 --fim 2025-03
"""

import argparse

import pandas as pd
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from main import ARQUIVOS_PERIODO, PASTA_PERIODO, parse_ano_mes, sufixo_periodo

# Arquivo do período (ARQUIVOS_PERIODO, em main.py) de onde sai cada amostra.
ARQUIVO_AMOSTRA = {"materiais": "alimentos_true", "servicos": "servicos"}
COLUNAS_ITEM = ["coleta_ano", "coleta_mes", "numeroControlePNCP", "numeroItem", "descricao"]
# Colunas que explicam a decisão de cada classificador.
COLUNAS_CLASSIFICACAO = {
    "materiais": [
        "nome_pdm_detectado",
        "codigo_item_catmat",
        "descricao_item_catmat",
        "similaridade",
        "grupo_taxonomico",
        "subgrupo_taxonomico",
        "metodo_classificacao",
        "objetoCompra",
        "metodo_contexto_escolar",
        "unidadeOrgao.nomeUnidade",
        "orgaoEntidade.razaoSocial",
    ],
    "servicos": [
        "objetoCompra",
        "amparoLegal.codigo",
        "codigo_servico_catser",
        "nome_servico_catser",
        "similaridade",
        "grupo_proposto",
        "metodo_classificacao",
    ],
}
COLUNAS_VALIDACAO = ["classificacao_correta", "grupo_correto", "observacao"]
OPCOES_CLASSIFICACAO = "Sim,Não,Dúvida"
NOME_ABA = {"materiais": "Alimentos contexto escolar", "servicos": "Serviços True"}


def sortear(sufixo_total: str, tipo: str, tamanho: int, semente: int) -> pd.DataFrame:
    """Sorteia as linhas de um resultado do período e acrescenta as colunas da validação."""
    colunas = COLUNAS_ITEM + COLUNAS_CLASSIFICACAO[tipo]
    resultado = pd.read_csv(
        PASTA_PERIODO / ARQUIVOS_PERIODO[ARQUIVO_AMOSTRA[tipo]].format(sufixo_total),
        dtype=str,
        keep_default_na=False,
        usecols=colunas,
    )[colunas]
    amostra = resultado.sample(n=min(tamanho, len(resultado)), random_state=semente)
    amostra = amostra.sort_values(["coleta_ano", "coleta_mes", "numeroControlePNCP"])
    for coluna in COLUNAS_VALIDACAO:
        amostra[coluna] = ""
    print(f"{NOME_ABA[tipo]}: {len(amostra)} de {len(resultado):,} linhas")
    return amostra


def gravar_planilha(amostras: dict[str, pd.DataFrame], destino) -> None:
    """Grava uma aba por amostra, com filtro, cabeçalho fixo e lista de opções."""
    with pd.ExcelWriter(destino, engine="openpyxl") as escritor:
        for tipo, amostra in amostras.items():
            amostra.to_excel(escritor, sheet_name=NOME_ABA[tipo], index=False)
            aba = escritor.sheets[NOME_ABA[tipo]]
            aba.auto_filter.ref = aba.dimensions
            aba.freeze_panes = "A2"
            for posicao, coluna in enumerate(amostra.columns, start=1):
                maior = max([len(coluna), *amostra[coluna].str.len().head(50)])
                aba.column_dimensions[get_column_letter(posicao)].width = min(maior + 2, 60)

            letra = get_column_letter(amostra.columns.get_loc("classificacao_correta") + 1)
            opcoes = DataValidation(type="list", formula1=f'"{OPCOES_CLASSIFICACAO}"')
            opcoes.add(f"{letra}2:{letra}{len(amostra) + 1}")
            aba.add_data_validation(opcoes)


def main() -> None:
    """Lê os argumentos, sorteia as amostras e grava a planilha."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inicio", type=parse_ano_mes, required=True, help="Mês inicial (AAAA-MM)")
    parser.add_argument("--fim", type=parse_ano_mes, required=True, help="Mês final (AAAA-MM)")
    parser.add_argument("--tamanho", type=int, default=200, help="Linhas por amostra")
    parser.add_argument("--semente", type=int, default=42, help="Semente do sorteio")
    args = parser.parse_args()

    periodo = sufixo_periodo(args.inicio, args.fim)
    amostras = {
        tipo: sortear(periodo, tipo, args.tamanho, args.semente) for tipo in ARQUIVO_AMOSTRA
    }
    destino = PASTA_PERIODO / f"amostra_validacao_{periodo}.xlsx"
    gravar_planilha(amostras, destino)
    print(f"Planilha: {destino}")


if __name__ == "__main__":
    main()
