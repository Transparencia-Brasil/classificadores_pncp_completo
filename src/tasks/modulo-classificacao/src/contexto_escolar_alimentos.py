"""Marca o contexto escolar dos alimentos candidatos pela contratação de cada item.

O classificador de alimentos olha só a descrição do item e responde se ele é um
alimento candidato ao PNAE. Se a compra é para alimentação escolar depende da
contratação: objeto, amparo legal e quem compra. Esta etapa roda depois dos
embeddings e lê o CSV de alimentos candidatos do lote. Ela:

1. liga cada item à sua contratação pela chave do endpoint e acrescenta objeto,
   amparo, órgão e unidade;
2. marca o contexto escolar: amparo 138/139, ou termo escolar forte ou de função
   no objeto da compra, ou termo escolar fraco no objeto sem termo de outro
   contexto (o mesmo dicionário e a mesma regra do passo 3b dos serviços).

Nenhum item é removido: o CSV é regravado com as novas colunas e o recorte de
alimentação escolar é ``contexto_escolar == True``. Mudanças nesta etapa, como
novos termos no dicionário, não exigem refazer os embeddings.

Exemplo:
    python contexto_escolar_alimentos.py --sufixo 2025-01_a_2025-01 \\
        --contratacoes ../../modulo-construcao-da-base/output/contratacoes_2025-01_a_2025-01.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from classifica_alimentos import PASTA_OUTPUT, caminhos_lote
from classifica_alimentos_servicos import (
    CODIGOS_AMPARO_PNAE,
    GRUPOS_CONTEXTO_ESCOLAR,
    GRUPOS_TERMOS,
    METODO_AMPARO,
    METODO_SEM_CONTEXTO_ESCOLAR,
    PADRAO_CHAVE_ENDPOINT,
    buscar_termos_objeto,
    carregar_padroes_termos,
)

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

# Como cada item chegou (ou não) ao contexto escolar, em ordem de prioridade.
METODO_TERMO_FORTE = "termo escolar forte ou função no objeto"
METODO_TERMO_FRACO = "termo escolar fraco no objeto"
# Colunas da marcação, gravadas depois das colunas da contratação.
COLUNAS_CONTEXTO = [
    *(f"termos_{grupo}_objeto" for grupo in GRUPOS_TERMOS),
    "contexto_escolar",
    "metodo_contexto_escolar",
]


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
    base = alimentos.drop(
        columns=[coluna for coluna in novas + COLUNAS_CONTEXTO if coluna in alimentos]
    )
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


def marcar_contexto_escolar(alimentos: pd.DataFrame, padroes: dict) -> pd.DataFrame:
    """Aplica o dicionário ao objeto da compra e marca o contexto escolar.

    A regra é a do passo 3b dos serviços, restrita ao objeto: a descrição de um
    material ("Arroz beneficiado") quase nunca diz para quem é a compra.
    """
    resultado = buscar_termos_objeto(alimentos, padroes)

    def tem(grupos: list[str]) -> pd.Series:
        return resultado[[f"termos_{grupo}_objeto" for grupo in grupos]].ne("").any(axis=1)

    amparo = resultado["amparoLegal.codigo"].isin(CODIGOS_AMPARO_PNAE)
    forte = tem(GRUPOS_CONTEXTO_ESCOLAR)
    fraco = tem(["escolar_fraco"]) & ~tem(["outro_contexto"])
    resultado["contexto_escolar"] = amparo | forte | fraco

    resultado["metodo_contexto_escolar"] = METODO_SEM_CONTEXTO_ESCOLAR
    resultado.loc[fraco, "metodo_contexto_escolar"] = METODO_TERMO_FRACO
    resultado.loc[forte, "metodo_contexto_escolar"] = METODO_TERMO_FORTE
    resultado.loc[amparo, "metodo_contexto_escolar"] = METODO_AMPARO

    # As colunas da marcação ficam logo depois das colunas da contratação.
    colunas = [coluna for coluna in resultado.columns if coluna not in COLUNAS_CONTEXTO]
    ultima = list(COLUNAS_CONTRATACAO_ALIMENTOS.values())[-1]
    posicao = colunas.index(ultima) + 1
    return resultado[colunas[:posicao] + COLUNAS_CONTEXTO + colunas[posicao:]]


def executar(caminho_alimentos: Path, caminho_contratacoes: Path) -> pd.DataFrame:
    """Liga os alimentos às contratações, marca o contexto e regrava o CSV."""
    alimentos = pd.read_csv(caminho_alimentos, dtype=str, keep_default_na=False)
    resultado = ligar_contratacoes(alimentos, carregar_contratacoes(caminho_contratacoes))
    resultado = marcar_contexto_escolar(resultado, carregar_padroes_termos())
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
    print(
        f"{len(resultado):,} alimentos, {int(resultado['contexto_escolar'].sum()):,} "
        f"com contexto escolar -> {caminho}"
    )


if __name__ == "__main__":
    main()
