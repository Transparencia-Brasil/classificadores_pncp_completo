"""Acrescenta aos alimentos candidatos os dados da contratação de cada item.

O classificador de alimentos olha só a descrição do item e responde se ele é um
alimento candidato ao PNAE. Se a compra é para alimentação escolar depende da
contratação: objeto, amparo legal e quem compra. Esta etapa roda depois dos
embeddings e lê o CSV de alimentos candidatos do lote, liga cada item à sua
contratação pela chave do endpoint e grava o mesmo CSV com as colunas da
contratação. Assim, mudanças nesta etapa não exigem refazer os embeddings.

Exemplo:
    python contexto_escolar_alimentos.py --sufixo 2025-01_a_2025-01 \\
        --contratacoes ../../modulo-construcao-da-base/output/contratacoes_2025-01_a_2025-01.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from classifica_alimentos import PASTA_OUTPUT, caminhos_lote
from classifica_alimentos_servicos import PADRAO_CHAVE_ENDPOINT

# Campos da contratação levados aos alimentos. Os nomes seguem o PNCP sem o
# prefixo "data.", como no CSV de serviços.
COLUNAS_CHAVE_CONTRATACAO = {
    "data.orgaoEntidade.cnpj": "cnpj_orgao",
    "data.anoCompra": "ano_compra",
    "data.sequencialCompra": "sequencial_compra",
}
COLUNAS_CONTRATACAO_ALIMENTOS = {
    "data.objetoCompra": "objetoCompra",
    "data.amparoLegal.codigo": "amparoLegal.codigo",
    "data.amparoLegal.nome": "amparoLegal.nome",
    "data.orgaoEntidade.razaoSocial": "orgaoEntidade.razaoSocial",
    "data.orgaoEntidade.esferaId": "orgaoEntidade.esferaId",
    "data.unidadeOrgao.nomeUnidade": "unidadeOrgao.nomeUnidade",
    "data.unidadeOrgao.municipioNome": "unidadeOrgao.municipioNome",
    "data.unidadeOrgao.ufSigla": "unidadeOrgao.ufSigla",
}
# As colunas da contratação entram logo depois desta coluna, como nos serviços.
COLUNA_ANTERIOR = "problema_codificacao"


def carregar_contratacoes(caminho: Path) -> pd.DataFrame:
    """Lê as contratações do lote com a chave usada no endpoint dos itens."""
    colunas = {**COLUNAS_CHAVE_CONTRATACAO, **COLUNAS_CONTRATACAO_ALIMENTOS}
    contratacoes = pd.read_csv(
        caminho, usecols=lambda coluna: coluna in colunas, dtype=str, keep_default_na=False
    )
    ausentes = set(colunas).difference(contratacoes.columns)
    if ausentes:
        raise ValueError(f"Colunas ausentes nas contratações: {sorted(ausentes)}")
    contratacoes = contratacoes[list(colunas)].rename(columns=colunas)
    contratacoes["chave_contratacao"] = (
        contratacoes["cnpj_orgao"]
        + "/" + contratacoes["ano_compra"]
        + "/" + contratacoes["sequencial_compra"]
    )
    if contratacoes["chave_contratacao"].duplicated().any():
        raise ValueError("O arquivo de contratações tem chaves repetidas.")
    return contratacoes[["chave_contratacao", *COLUNAS_CONTRATACAO_ALIMENTOS.values()]]


def ligar_contratacoes(alimentos: pd.DataFrame, contratacoes: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta a cada alimento as colunas da contratação a que pertence.

    Os alimentos sem contratação correspondente (endpoint com ano inválido, por
    exemplo) continuam no resultado, com as colunas da contratação vazias.
    """
    novas = list(COLUNAS_CONTRATACAO_ALIMENTOS.values())
    # Uma nova execução substitui as colunas gravadas pela anterior.
    base = alimentos.drop(columns=[coluna for coluna in novas if coluna in alimentos])
    chave = base["endpoint"].str.extract(PADRAO_CHAVE_ENDPOINT)
    resultado = base.assign(
        chave_contratacao=chave[0] + "/" + chave[1] + "/" + chave[2]
    ).merge(contratacoes, on="chave_contratacao", how="left", validate="many_to_one")

    sem_contratacao = resultado["objetoCompra"].isna()
    if sem_contratacao.any():
        print(
            f"Aviso: {int(sem_contratacao.sum())} alimentos sem contratação "
            "correspondente ficam com as colunas da contratação vazias."
        )
    resultado[novas] = resultado[novas].fillna("")

    colunas = list(base.columns)
    posicao = colunas.index(COLUNA_ANTERIOR) + 1 if COLUNA_ANTERIOR in colunas else len(colunas)
    return resultado[colunas[:posicao] + novas + colunas[posicao:]]


def executar(caminho_alimentos: Path, caminho_contratacoes: Path) -> pd.DataFrame:
    """Liga os alimentos do lote às contratações e regrava o CSV de alimentos."""
    alimentos = pd.read_csv(caminho_alimentos, dtype=str, keep_default_na=False)
    resultado = ligar_contratacoes(alimentos, carregar_contratacoes(caminho_contratacoes))
    # A gravação temporária evita deixar o CSV de alimentos pela metade.
    temporario = caminho_alimentos.with_suffix(".tmp.csv")
    resultado.to_csv(temporario, index=False, encoding="utf-8")
    temporario.replace(caminho_alimentos)
    return resultado


def main() -> None:
    """Lê os argumentos e acrescenta as colunas da contratação ao lote."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sufixo", required=True, help="Lote (ex.: 2025-01_a_2025-01)")
    parser.add_argument(
        "--contratacoes", type=Path, required=True, help="CSV de contratações do lote"
    )
    parser.add_argument("--saida", type=Path, default=PASTA_OUTPUT, help="Pasta dos resultados")
    args = parser.parse_args()

    caminho = caminhos_lote(args.sufixo, args.saida)["alimentos"]
    resultado = executar(caminho, args.contratacoes)
    print(f"{len(resultado):,} alimentos com as colunas da contratação -> {caminho}")


if __name__ == "__main__":
    main()
