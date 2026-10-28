"""Consolida os itens do PNCP de um período e separa materiais e serviços.

Lê os arquivos ``itens.csv`` das coletas quinzenais (mesma estrutura de pastas usada
em ``database_build.py``), junta todos os do período escolhido e grava:

- todos os itens, só os materiais e só os serviços;
- as mesmas três tabelas restritas aos itens de contratações com amparo legal de
  merenda, identificadas cruzando com os ``contratacoes.csv`` do mesmo período.

Exemplo:
    python src/tasks/modulo-construcao-da-base/src/itens_build.py --inicio 2025-01 --fim 2025-06
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from database_build import (
    AMPAROS_FILTRO,
    COLUNA_AMPARO,
    COLUNA_ID,
    ORIGEM_PADRAO,
    SAIDA_PADRAO,
    TAMANHO_CHUNK,
    ArquivoColeta,
    colunas_unificadas,
    filtrar_ultima_coleta,
    grupo_amparo,
    listar_arquivos,
    normalizar_amparo,
    parse_ano_mes,
    ultima_coleta_por_chave,
)

COLUNA_TIPO = "materialOuServico"
COLUNA_CONTROLE = "numeroControlePNCP"
# Um item é identificado pela compra (endpoint) e pelo número dentro dela.
COLUNAS_CHAVE_ITEM = ["endpoint", "numeroItem"]

# Endpoint do item: .../orgaos/{cnpj}/compras/{ano}/{sequencial}/itens
PADRAO_ENDPOINT = r"/orgaos/(?P<cnpj>\d+)/compras/(?P<ano>\d{4})/(?P<sequencial>\d+)/itens"


def carregar_amparos(arquivos: list[ArquivoColeta]) -> dict[str, str]:
    """Mapeia ``numeroControlePNCP`` -> código de amparo de todas as contratações do período."""
    amparos: dict[str, str] = {}
    for arquivo in arquivos:
        print(f"Lendo amparos de {arquivo.ano}/{arquivo.mes:02d} {arquivo.periodo}...")
        contratacoes = pd.read_csv(
            arquivo.caminho,
            dtype=str,
            keep_default_na=False,
            usecols=[COLUNA_ID, COLUNA_AMPARO],
        )
        codigos = normalizar_amparo(contratacoes[COLUNA_AMPARO])
        amparos.update(zip(contratacoes[COLUNA_ID], codigos))
    return amparos


def numero_controle(endpoints: pd.Series) -> pd.Series:
    """Monta o ``numeroControlePNCP`` da contratação a partir do endpoint do item."""
    partes = endpoints.str.extract(PADRAO_ENDPOINT)
    controle = partes["cnpj"] + "-1-" + partes["sequencial"].str.zfill(6) + "/" + partes["ano"]
    return controle.fillna("")


def caminhos_saida(pasta: Path, sufixo: str, recorte: str = "") -> dict[str, Path]:
    """Caminhos das tabelas de todos os itens, materiais e serviços de um recorte."""
    nomes = {"todos": "itens", "materiais": "itens_materiais", "servicos": "itens_servicos"}
    return {chave: pasta / f"{nome}{recorte}_{sufixo}.csv" for chave, nome in nomes.items()}


def gravar_por_tipo(tabela: pd.DataFrame, saidas: dict[str, Path], primeiro_bloco: bool) -> None:
    """Grava a tabela inteira e as partes de materiais (``M``) e de serviços (``S``)."""
    tipo = tabela[COLUNA_TIPO]
    partes = {
        "todos": tabela,
        "materiais": tabela[tipo.eq("M")],
        "servicos": tabela[tipo.eq("S")],
    }
    modo = "w" if primeiro_bloco else "a"
    for chave, parte in partes.items():
        parte.to_csv(saidas[chave], mode=modo, header=primeiro_bloco, index=False)


def construir_itens(
    arquivos: list[ArquivoColeta],
    amparos: dict[str, str],
    saidas: dict[str, Path],
    saidas_merenda: dict[str, Path],
    deduplicar: bool = False,
) -> None:
    """Grava as tabelas de itens do período, completas e filtradas por amparo, lendo em blocos."""
    colunas = colunas_unificadas(arquivos)
    colunas.insert(3, COLUNA_CONTROLE)
    colunas_merenda = [*colunas, COLUNA_AMPARO, "grupo_amparo"]
    ultima = ultima_coleta_por_chave(arquivos, COLUNAS_CHAVE_ITEM) if deduplicar else None

    total_itens = 0
    total_descartados = 0
    sem_contratacao = 0
    tipos: Counter[str] = Counter()
    tipos_merenda: Counter[str] = Counter()
    grupos_merenda: Counter[str] = Counter()
    primeiro_bloco = True

    for indice, arquivo in enumerate(arquivos):
        print(f"Lendo itens de {arquivo.ano}/{arquivo.mes:02d} {arquivo.periodo}...")
        blocos = pd.read_csv(
            arquivo.caminho,
            dtype=str,
            keep_default_na=False,
            chunksize=TAMANHO_CHUNK,
        )
        for bloco in blocos:
            bloco.insert(0, "coleta_ano", arquivo.ano)
            bloco.insert(1, "coleta_mes", arquivo.mes)
            bloco.insert(2, "coleta_periodo", arquivo.periodo)
            bloco.insert(3, COLUNA_CONTROLE, numero_controle(bloco["endpoint"]))
            bloco = bloco.reindex(columns=colunas, fill_value="")
            if ultima is not None:
                tamanho = len(bloco)
                bloco = filtrar_ultima_coleta(bloco, COLUNAS_CHAVE_ITEM, ultima, indice)
                total_descartados += tamanho - len(bloco)

            amparo =bloco[COLUNA_CONTROLE].map(amparos)
            merenda = bloco[amparo.isin(AMPAROS_FILTRO)].copy()
            merenda[COLUNA_AMPARO] = amparo[merenda.index]
            merenda["grupo_amparo"] = merenda[COLUNA_AMPARO].map(grupo_amparo)
            merenda = merenda.reindex(columns=colunas_merenda)

            gravar_por_tipo(bloco, saidas, primeiro_bloco)
            gravar_por_tipo(merenda, saidas_merenda, primeiro_bloco)

            total_itens += len(bloco)
            sem_contratacao += int(amparo.isna().sum())
            tipos.update(bloco[COLUNA_TIPO])
            tipos_merenda.update(merenda[COLUNA_TIPO])
            grupos_merenda.update(merenda["grupo_amparo"])
            primeiro_bloco = False

    print()
    print(f"Arquivos de itens lidos: {len(arquivos)}")
    print(f"Itens no consolidado: {total_itens:,}")
    if deduplicar:
        print(f"Repetições descartadas (mantida a coleta mais recente): {total_descartados:,}")
    print(f"  materiais (M): {tipos['M']:,}")
    print(f"  serviços (S):  {tipos['S']:,}")
    outros = {tipo: n for tipo, n in tipos.items() if tipo not in {"M", "S"}}
    if outros:
        print(f"  outros valores de {COLUNA_TIPO}: {outros}")
    print(f"Itens sem contratação correspondente no período: {sem_contratacao:,}")
    print(f"Itens de contratações com amparo de merenda: {sum(tipos_merenda.values()):,}")
    print(f"  materiais (M): {tipos_merenda['M']:,}")
    print(f"  serviços (S):  {tipos_merenda['S']:,}")
    for grupo, n in grupos_merenda.most_common():
        print(f"  grupo {grupo}: {n:,}")
    print()
    for caminho in [*saidas.values(), *saidas_merenda.values()]:
        print(caminho)


def main() -> None:
    """Lê os argumentos de linha de comando e executa a construção da base de itens."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inicio", type=parse_ano_mes, required=True, help="Mês inicial (AAAA-MM)")
    parser.add_argument("--fim", type=parse_ano_mes, required=True, help="Mês final (AAAA-MM)")
    parser.add_argument("--origem", type=Path, default=ORIGEM_PADRAO, help="Pasta tmp das coletas")
    parser.add_argument("--saida", type=Path, default=SAIDA_PADRAO, help="Pasta de saída")
    parser.add_argument(
        "--deduplicar",
        action="store_true",
        help="Mantém só a versão mais recente de cada item no período",
    )
    args = parser.parse_args()

    if args.inicio > args.fim:
        parser.error("--inicio deve ser anterior ou igual a --fim")

    arquivos_itens = listar_arquivos(args.origem, args.inicio, args.fim, "itens.csv")
    if not arquivos_itens:
        parser.error(f"Nenhum itens.csv encontrado em {args.origem} para o período")
    arquivos_contratacoes = listar_arquivos(args.origem, args.inicio, args.fim)

    sufixo = f"{args.inicio[0]}-{args.inicio[1]:02d}_a_{args.fim[0]}-{args.fim[1]:02d}"
    args.saida.mkdir(parents=True, exist_ok=True)
    construir_itens(
        arquivos_itens,
        carregar_amparos(arquivos_contratacoes),
        caminhos_saida(args.saida, sufixo),
        caminhos_saida(args.saida, sufixo, "_amparos_merenda"),
        args.deduplicar,
    )


if __name__ == "__main__":
    main()
