"""Novo pipeline do classificador de serviços ligados à alimentação escolar.

O fluxo segue o desenho `pipeline_servicos_completo.excalidraw` do repositório
alimentacao-escolar-analises. O lote PNCP (itens e contratações) e o sufixo dos
arquivos de saída são informados na execução.

Passo 1: a tabela de contratações recebe a coluna `servico_alimentacao_escolar`.
Os amparos legais 138 e 139 são os dois artigos da Lei 11.947/2009 (PNAE)
cadastrados no PNCP: a chamada pública da agricultura familiar (Art. 14, § 1º)
e a compra emergencial de gêneros alimentícios (Art. 21). As contratações com
esses amparos são marcadas como True. As demais ficam como False até passarem
pelas buscas textual e semântica.

Passo 2: os itens do PNCP são reduzidos aos serviços com descrição e ligados à
contratação a que pertencem pela chave CNPJ, ano e sequencial do `endpoint`.
Cada serviço recebe o objeto da compra, o amparo legal, a modalidade e a
marcação `servico_alimentacao_escolar` do passo 1. Em seguida vem o vínculo
direto: o serviço cujo `catalogoCodigoItem` é exatamente um dos 29 códigos da
aba "Alimentação escolar" é True, sem passar pelo filtro do passo 3 nem pela
similaridade. Só vale o código informado no catálogo do Compras.gov.br; no
catálogo "Outros" o número é do próprio órgão e não identifica o CATSER.

Passo 3: o dicionário `referencias/termos_contexto_new_pipeline.csv` é aplicado
à descrição do item. Ele parte do dicionário da versão melhoria e acrescenta o
grupo escolar função (merendeira e preparo da alimentação escolar ou da
merenda), que descreve o trabalho contratado. Os trechos de exclusão são
apagados antes da busca dos demais grupos. A busca funciona como filtro: um
serviço ainda False sem termo alimentar, escolar forte ou de função na
descrição termina como False, com o método "sem contexto alimentar", e não
segue para a busca semântica. Os termos de todos os grupos ficam no resultado
para auditoria.

Passo 4: a descrição é comparada por embeddings com os 29 serviços CATSER da
aba "Alimentação escolar", e cada serviço conserva o mais semelhante e o grupo
proposto (S01 a S07) dele. Todas as descrições são vetorizadas para reaproveitar
o cache das versões anteriores, mas as regras só valem para os serviços que
chegam a este passo. Os serviços do amparo 138/139 e os do vínculo direto pelo
código apenas registram a similaridade e continuam True. Para os que vieram do
filtro do passo 3:
- similaridade de 0,5 ou mais: True, pela semântica;
- de 0,4 a 0,5 (resgate escolar): True com termo alimentar na descrição e
  termo escolar forte ou de função na descrição ou no objeto da compra;
- abaixo de 0,4 (resgate por função): True com termo alimentar e termo de
  função escolar, ambos na descrição. O objeto da compra não conta aqui, para
  não puxar todos os itens de uma contratação mista.
Só o grupo alimentar conta como termo alimentar nos resgates: "pnae" ou
"11.947" sozinhos citam o programa e aparecem em assessorias e locações.

Passo 5: o resultado é gravado em dois CSVs, um com os serviços True e outro,
compactado, com os False. Os dois têm as mesmas colunas, incluindo o método de
classificação, os termos encontrados na descrição e no objeto da compra e o
grupo proposto do serviço CATSER mais semelhante. O grupo é só rótulo: não
decide True ou False.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from classifica_alimentos import (
    MODELO_PADRAO,
    PASTA_CACHE,
    PASTA_OUTPUT,
    PASTA_REFERENCIAS,
    carregar_itens,
    carregar_modelo,
    corrigir_codificacao,
)
from utils_servicos import (
    CAMINHO_CATSER,
    DATA_REFERENCIA_CATSER,
    LIMITE_TOKENS_CONSULTAS,
    aplicar_exclusoes,
    carregar_dicionario_termos,
    compilar_padroes_por_grupo,
    encontrar_termos,
    normalizar_busca,
    obter_embeddings_consultas,
    padronizar_textos_exportados,
    preparar_referencia,
)


# Campos da contratação levados aos itens. Os nomes seguem o PNCP sem o prefixo
# "data.", como nas colunas dos itens (catalogo.id, catalogo.nome).
COLUNAS_CONTRATACOES = {
    "data.orgaoEntidade.cnpj": "cnpj_orgao",
    "data.anoCompra": "ano_compra",
    "data.sequencialCompra": "sequencial_compra",
    "data.objetoCompra": "objetoCompra",
    "data.amparoLegal": "amparoLegal",
    "data.amparoLegal.descricao": "amparoLegal.descricao",
    "data.amparoLegal.nome": "amparoLegal.nome",
    "data.amparoLegal.codigo": "amparoLegal.codigo",
    "data.modalidadeNome": "modalidadeNome",
}
# Lei 11.947/2009: Art. 14, § 1º (138) e Art. 21 (139).
CODIGOS_AMPARO_PNAE = {"138", "139"}
# Colunas da contratação acrescentadas aos itens, nesta ordem.
COLUNAS_CONTRATACAO_ITENS = [
    "objetoCompra",
    "servico_alimentacao_escolar",
    "amparoLegal",
    "amparoLegal.descricao",
    "amparoLegal.nome",
    "amparoLegal.codigo",
    "modalidadeNome",
]
# O endpoint dos itens contém a mesma chave da contratação.
PADRAO_CHAVE_ENDPOINT = r"/orgaos/(\d+)/compras/(\d+)/(\d+)/itens"
# catalogo.id do "Catálogo do Compras.gov.br", o único em que catalogoCodigoItem
# é um código CATSER. O id 2 ("Outros") traz numerações dos próprios órgãos.
ID_CATALOGO_COMPRAS = 1

# O dicionário da versão melhoria continua intacto. Este acrescenta o grupo
# escolar função e retira "merendeir" do grupo escolar forte.
CAMINHO_TERMOS = PASTA_REFERENCIAS / "termos_contexto_new_pipeline.csv"
# Grupos na ordem das colunas de termos do resultado. A exclusão fica por último
# porque é aplicada antes dos demais.
GRUPOS_TERMOS = [
    "alimentar",
    "escolar_forte",
    "escolar_funcao",
    "escolar_fraco",
    "outro_contexto",
    "exclusao",
]
# Um destes grupos na descrição leva o serviço à busca semântica.
GRUPOS_CONTEXTO_ALIMENTAR = ["alimentar", "escolar_forte", "escolar_funcao"]
METODO_SEM_CONTEXTO = "sem contexto alimentar"

CAMINHO_REFERENCIA = PASTA_OUTPUT / "referencia_servicos_alimentacao_catser.csv"
LIMITE_PADRAO = 0.5
LIMITE_RESGATE_PADRAO = 0.4
# Métodos atribuídos no passo 4, na ordem do desenho.
METODO_AMPARO = "amparo legal 138/139"
METODO_CODIGO = "código CATSER informado"
METODO_SEMANTICA = "semântica"
METODO_RESGATE_ESCOLAR = "resgate escolar"
METODO_FAIXA_SEM_TERMOS = "faixa 0,4 a 0,5 sem termos"
METODO_RESGATE_FUNCAO = "resgate por função"
METODO_ABAIXO_SEM_FUNCAO = "abaixo de 0,4 sem função"



def caminhos_lote(sufixo: str, pasta_saida: Path = PASTA_OUTPUT) -> dict[str, Path]:
    """Caminhos dos resultados e do cache de consultas de um lote do PNCP."""
    # Quase todos os serviços terminam como False. O arquivo é compactado, como o
    # diagnóstico das versões anteriores.
    return {
        "true": pasta_saida / f"servicos_alimentacao_escolar_true_pncp_{sufixo}.csv",
        "false": pasta_saida / f"servicos_alimentacao_escolar_false_pncp_{sufixo}.csv.gz",
        "cache_consultas": PASTA_CACHE / sufixo / "consultas_servicos_pncp_vetorizadas.npy",
    }


# Colunas acrescentadas às do item, na ordem dos passos.
COLUNAS_RESULTADO = [
    "descricao_corrigida",
    "problema_codificacao",
    *COLUNAS_CONTRATACAO_ITENS,
    "codigo_catser_informado",
    "descricao_busca",
    *[f"termos_{grupo}_descricao" for grupo in GRUPOS_TERMOS],
    "contexto_alimentar",
    *[f"termos_{grupo}_objeto" for grupo in GRUPOS_TERMOS],
    "codigo_servico_catser",
    "nome_servico_catser",
    "grupo_proposto",
    "similaridade",
    "metodo_classificacao",
    "modelo_embedding",
    "limite_tokens_modelo",
    "limite_operacional",
    "limite_resgate",
    "data_referencia_catser",
    "janela_referencia_pncp",
]


# PASSO 1 · CONTRATAÇÕES ------------------------------------------------------

def preparar_contratacoes(caminho: Path) -> pd.DataFrame:
    """Lê as contratações e marca as que têm amparo legal do PNAE."""
    contratacoes = pd.read_csv(
        caminho, usecols=list(COLUNAS_CONTRATACOES), dtype=str
    ).rename(columns=COLUNAS_CONTRATACOES)
    contratacoes["chave_contratacao"] = (
        contratacoes["cnpj_orgao"]
        + "/" + contratacoes["ano_compra"]
        + "/" + contratacoes["sequencial_compra"]
    )
    if contratacoes["chave_contratacao"].duplicated().any():
        raise ValueError("O arquivo de contratações tem chaves repetidas.")

    contratacoes["servico_alimentacao_escolar"] = contratacoes["amparoLegal.codigo"].isin(
        CODIGOS_AMPARO_PNAE
    )
    return contratacoes.drop(columns=["cnpj_orgao", "ano_compra", "sequencial_compra"])


# PASSO 2 · SERVIÇOS COM A CONTRATAÇÃO ----------------------------------------

def preparar_servicos(itens: pd.DataFrame) -> pd.DataFrame:
    """Seleciona serviços com descrição sem alterar a descrição original."""
    servicos = itens.loc[itens["materialOuServico"].eq("S")].copy()
    correcao = servicos["descricao"].map(corrigir_codificacao)
    servicos["descricao_corrigida"] = correcao.str[0]
    servicos["problema_codificacao"] = correcao.str[1]
    # Descrições vazias não têm texto para comparar.
    return servicos.loc[servicos["descricao_corrigida"].str.strip().ne("")].copy()


def ligar_contratacoes(servicos: pd.DataFrame, contratacoes: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta a cada serviço as colunas da contratação a que pertence."""
    chave = servicos["endpoint"].str.extract(PADRAO_CHAVE_ENDPOINT)
    resultado = servicos.assign(
        chave_contratacao=chave[0] + "/" + chave[1] + "/" + chave[2]
    ).merge(
        contratacoes[["chave_contratacao", *COLUNAS_CONTRATACAO_ITENS]],
        on="chave_contratacao",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    sem_contratacao = resultado["_merge"].eq("left_only")
    if sem_contratacao.any():
        raise ValueError(
            f"{int(sem_contratacao.sum())} serviços não têm contratação correspondente."
        )
    return resultado.drop(columns=["chave_contratacao", "_merge"])


def marcar_codigo_catser(servicos: pd.DataFrame, referencia: pd.DataFrame) -> pd.DataFrame:
    """Marca os serviços que informam um dos 29 códigos CATSER da seleção."""
    resultado = servicos.copy()
    # A comparação é exata. Valores como "." viram vazio e "0001" vira 1.
    codigos = pd.to_numeric(resultado["catalogoCodigoItem"], errors="coerce")
    resultado["codigo_catser_informado"] = (
        resultado["catalogo.id"].eq(ID_CATALOGO_COMPRAS)
        & codigos.isin(set(referencia["codigo_servico"]))
    )
    return resultado


# PASSO 3 · BUSCA TEXTUAL NA DESCRIÇÃO ----------------------------------------

def carregar_padroes_termos(caminho: Path = CAMINHO_TERMOS) -> dict[str, re.Pattern]:
    """Valida o dicionário de termos e compila uma expressão por grupo."""
    dicionario = carregar_dicionario_termos(caminho, grupos=set(GRUPOS_TERMOS))
    ausentes = set(GRUPOS_TERMOS).difference(dicionario["grupo"])
    if ausentes:
        raise ValueError(f"Grupos sem termos no dicionário: {sorted(ausentes)}")
    return compilar_padroes_por_grupo(dicionario)


def buscar_termos(texto_busca: str, padroes: dict[str, re.Pattern]) -> dict[str, str]:
    """Apaga os trechos de exclusão e lista os termos encontrados em cada grupo."""
    texto = aplicar_exclusoes(texto_busca, padroes)
    termos = {
        grupo: encontrar_termos(texto, padroes, [grupo])
        for grupo in GRUPOS_TERMOS
        if grupo != "exclusao"
    }
    # Os trechos de exclusão são procurados no texto antes de serem apagados.
    termos["exclusao"] = encontrar_termos(texto_busca, padroes, ["exclusao"])
    return termos


def filtrar_contexto_alimentar(
    servicos: pd.DataFrame, padroes: dict[str, re.Pattern]
) -> pd.DataFrame:
    """Procura os termos na descrição e encerra os serviços sem contexto alimentar."""
    resultado = servicos.copy()
    resultado["descricao_busca"] = resultado["descricao_corrigida"].map(normalizar_busca)
    termos = pd.DataFrame(
        resultado["descricao_busca"].map(lambda texto: buscar_termos(texto, padroes)).tolist(),
        index=resultado.index,
    )
    for grupo in GRUPOS_TERMOS:
        resultado[f"termos_{grupo}_descricao"] = termos[grupo]

    resultado["contexto_alimentar"] = (
        termos[GRUPOS_CONTEXTO_ALIMENTAR].ne("").any(axis=1)
    )
    # Os serviços do amparo 138/139 e do vínculo direto pelo código já são True
    # e não passam pelo filtro. O método dos demais é definido na busca semântica.
    resultado["metodo_classificacao"] = pd.NA
    sem_contexto = (
        ~resultado["servico_alimentacao_escolar"]
        & ~resultado["codigo_catser_informado"]
        & ~resultado["contexto_alimentar"]
    )
    resultado.loc[sem_contexto, "metodo_classificacao"] = METODO_SEM_CONTEXTO
    return resultado


# PASSO 4 · BUSCA SEMÂNTICA E RESGATES ----------------------------------------

def parear_catser(
    servicos: pd.DataFrame,
    referencia: pd.DataFrame,
    modelo,
    nome_modelo: str,
    batch_size: int,
    caminho_cache_consultas: Path,
) -> pd.DataFrame:
    """Associa cada descrição ao serviço CATSER de referência mais semelhante."""
    # A referência é pequena e não precisa de cache.
    modelo.max_seq_length = LIMITE_TOKENS_CONSULTAS
    embeddings_referencia = modelo.encode(
        referencia["nome_servico"].tolist(),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # Descrições repetidas são vetorizadas uma vez, na ordem em que aparecem,
    # como nas versões anteriores. Assim o mesmo cache serve a todas.
    consultas = (
        servicos[["descricao_corrigida"]].drop_duplicates().reset_index(drop=True)
    )
    embeddings_consultas = obter_embeddings_consultas(
        consultas["descricao_corrigida"].tolist(),
        modelo,
        nome_modelo,
        caminho_cache_consultas,
        batch_size,
        embeddings_referencia.shape[1],
    )

    matriz_similaridade = embeddings_consultas @ embeddings_referencia.T
    posicoes = matriz_similaridade.argmax(axis=1)
    melhor = referencia.iloc[posicoes].reset_index(drop=True)
    consultas["codigo_servico_catser"] = melhor["codigo_servico"]
    consultas["nome_servico_catser"] = melhor["nome_servico"]
    consultas["grupo_proposto"] = melhor["grupo_proposto"]
    consultas["similaridade"] = matriz_similaridade[
        np.arange(len(consultas)), posicoes
    ].round(6)

    resultado = servicos.merge(
        consultas, on="descricao_corrigida", how="left", validate="many_to_one"
    )
    resultado["modelo_embedding"] = nome_modelo
    resultado["limite_tokens_modelo"] = LIMITE_TOKENS_CONSULTAS
    return resultado


def buscar_termos_objeto(
    servicos: pd.DataFrame, padroes: dict[str, re.Pattern]
) -> pd.DataFrame:
    """Aplica o dicionário ao objeto da compra, usado no resgate escolar."""
    resultado = servicos.copy()
    # Cada objeto é analisado uma vez, mesmo que a compra tenha vários itens.
    objetos = resultado["objetoCompra"].dropna().drop_duplicates()
    termos = pd.DataFrame(
        objetos.map(
            lambda texto: buscar_termos(normalizar_busca(corrigir_codificacao(texto)[0]), padroes)
        ).tolist(),
        index=objetos.values,
    )
    for grupo in GRUPOS_TERMOS:
        resultado[f"termos_{grupo}_objeto"] = (
            resultado["objetoCompra"].map(termos[grupo]).fillna("")
        )
    return resultado


def aplicar_regras_semanticas(
    servicos: pd.DataFrame,
    limite: float = LIMITE_PADRAO,
    limite_resgate: float = LIMITE_RESGATE_PADRAO,
) -> pd.DataFrame:
    """Decide os serviços que chegaram ao passo 4 pela similaridade e pelos resgates."""
    resultado = servicos.copy()
    similaridade = resultado["similaridade"]
    tem = {
        coluna: resultado[coluna].ne("")
        for coluna in [
            "termos_alimentar_descricao",
            "termos_escolar_forte_descricao",
            "termos_escolar_funcao_descricao",
            "termos_escolar_forte_objeto",
            "termos_escolar_funcao_objeto",
        ]
    }

    amparo = resultado["servico_alimentacao_escolar"]
    codigo = ~amparo & resultado["codigo_catser_informado"]
    # Os serviços sem contexto alimentar já foram encerrados no passo 3.
    filtrados = ~amparo & ~codigo & resultado["contexto_alimentar"]
    semantica = filtrados & similaridade.ge(limite)
    faixa = filtrados & similaridade.ge(limite_resgate) & similaridade.lt(limite)
    abaixo = filtrados & similaridade.lt(limite_resgate)

    escolar_descricao = (
        tem["termos_escolar_forte_descricao"] | tem["termos_escolar_funcao_descricao"]
    )
    escolar_objeto = (
        tem["termos_escolar_forte_objeto"] | tem["termos_escolar_funcao_objeto"]
    )
    resgate_escolar = faixa & tem["termos_alimentar_descricao"] & (
        escolar_descricao | escolar_objeto
    )
    resgate_funcao = (
        abaixo
        & tem["termos_alimentar_descricao"]
        & tem["termos_escolar_funcao_descricao"]
    )

    resultado["servico_alimentacao_escolar"] = (
        amparo | codigo | semantica | resgate_escolar | resgate_funcao
    )
    resultado["metodo_classificacao"] = np.select(
        [
            amparo,
            codigo,
            semantica,
            resgate_escolar,
            faixa,
            resgate_funcao,
            abaixo,
        ],
        [
            METODO_AMPARO,
            METODO_CODIGO,
            METODO_SEMANTICA,
            METODO_RESGATE_ESCOLAR,
            METODO_FAIXA_SEM_TERMOS,
            METODO_RESGATE_FUNCAO,
            METODO_ABAIXO_SEM_FUNCAO,
        ],
        default=resultado["metodo_classificacao"].fillna("").to_numpy(),
    )
    resultado["limite_operacional"] = limite
    resultado["limite_resgate"] = limite_resgate
    return resultado


# PASSO 5 · SAÍDAS ------------------------------------------------------------

def executar_classificacao(
    caminho_itens: Path,
    caminho_contratacoes: Path,
    caminho_true: Path,
    caminho_false: Path,
    caminho_cache_consultas: Path,
    janela: str,
    caminho_catser: Path = CAMINHO_CATSER,
    caminho_termos: Path = CAMINHO_TERMOS,
    caminho_referencia: Path = CAMINHO_REFERENCIA,
    nome_modelo: str = MODELO_PADRAO,
    limite: float = LIMITE_PADRAO,
    limite_resgate: float = LIMITE_RESGATE_PADRAO,
    batch_size: int = 128,
) -> dict[str, object]:
    """Executa os cinco passos e grava os CSVs de serviços True e False."""
    padroes = carregar_padroes_termos(caminho_termos)
    referencia = preparar_referencia(caminho_catser, caminho_referencia)
    itens = carregar_itens(caminho_itens)

    # Passos 1 a 3.
    contratacoes = preparar_contratacoes(caminho_contratacoes)
    servicos = ligar_contratacoes(preparar_servicos(itens), contratacoes)
    servicos = marcar_codigo_catser(servicos, referencia)
    servicos = filtrar_contexto_alimentar(servicos, padroes)

    # Passo 4.
    modelo = carregar_modelo(nome_modelo)
    resultado = parear_catser(
        servicos, referencia, modelo, nome_modelo, batch_size, caminho_cache_consultas
    )
    resultado = buscar_termos_objeto(resultado, padroes)
    resultado = aplicar_regras_semanticas(resultado, limite, limite_resgate)
    resultado["data_referencia_catser"] = DATA_REFERENCIA_CATSER
    resultado["janela_referencia_pncp"] = janela

    # Passo 5.
    colunas_saida = itens.columns.tolist() + COLUNAS_RESULTADO
    resultado = padronizar_textos_exportados(resultado[colunas_saida])
    verdadeiros = resultado.loc[resultado["servico_alimentacao_escolar"]]
    falsos = resultado.loc[~resultado["servico_alimentacao_escolar"]]
    caminho_true.parent.mkdir(parents=True, exist_ok=True)
    verdadeiros.to_csv(caminho_true, index=False, encoding="utf-8")
    falsos.to_csv(caminho_false, index=False, encoding="utf-8", compression="gzip")

    return {
        "itens": itens,
        "referencia": referencia,
        "resultado": resultado,
        "contagens": {
            "itens_recebidos": len(itens),
            "servicos": int(itens["materialOuServico"].eq("S").sum()),
            "servicos_com_descricao": len(resultado),
            "contratacoes_amparo_pnae": int(contratacoes["servico_alimentacao_escolar"].sum()),
            "servicos_codigo_catser": int(resultado["codigo_catser_informado"].sum()),
            "servicos_por_metodo": resultado["metodo_classificacao"].value_counts().to_dict(),
            "servicos_true": len(verdadeiros),
            "servicos_false": len(falsos),
        },
    }


# EXECUÇÃO POR LINHA DE COMANDO ----------------------------------------------

def criar_parser() -> argparse.ArgumentParser:
    """Define os argumentos da execução por linha de comando."""
    parser = argparse.ArgumentParser(
        description="Classifica serviços PNCP ligados à alimentação escolar."
    )
    parser.add_argument("--itens", type=Path, required=True, help="CSV de itens do lote PNCP")
    parser.add_argument(
        "--contratacoes", type=Path, required=True, help="CSV de contratações do mesmo lote"
    )
    parser.add_argument(
        "--sufixo",
        required=True,
        help="Identifica o lote nos nomes de saída e no cache (ex.: 2025-01_a_2025-01)",
    )
    parser.add_argument(
        "--janela", help="Período gravado em janela_referencia_pncp (padrão: o sufixo)"
    )
    parser.add_argument("--saida", type=Path, default=PASTA_OUTPUT, help="Pasta de saída")
    parser.add_argument("--catser", type=Path, default=CAMINHO_CATSER)
    parser.add_argument("--termos", type=Path, default=CAMINHO_TERMOS)
    parser.add_argument("--modelo", default=MODELO_PADRAO)
    parser.add_argument("--limite", type=float, default=LIMITE_PADRAO)
    parser.add_argument("--limite-resgate", type=float, default=LIMITE_RESGATE_PADRAO)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


def main() -> None:
    """Executa o classificador sobre o lote informado."""
    argumentos = criar_parser().parse_args()
    caminhos = caminhos_lote(argumentos.sufixo, argumentos.saida)
    resultado = executar_classificacao(
        caminho_itens=argumentos.itens,
        caminho_contratacoes=argumentos.contratacoes,
        caminho_true=caminhos["true"],
        caminho_false=caminhos["false"],
        caminho_cache_consultas=caminhos["cache_consultas"],
        janela=argumentos.janela or argumentos.sufixo,
        caminho_catser=argumentos.catser,
        caminho_termos=argumentos.termos,
        nome_modelo=argumentos.modelo,
        limite=argumentos.limite,
        limite_resgate=argumentos.limite_resgate,
        batch_size=argumentos.batch_size,
    )
    print(json.dumps(resultado["contagens"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
