"""Consolida as contratações do PNCP de um período e filtra pelos amparos legais de merenda.

Lê os arquivos ``contratacoes.csv`` das coletas quinzenais baixadas do Drive
(estrutura ``Coletas - <ano>/<mês> - <Nome>/<período>/contratacoes.csv``),
junta todos os do período escolhido em um CSV único e gera um segundo CSV só
com as contratações cujos amparos legais interessam à alimentação escolar.

Exemplo:
    python src/tasks/modulo-construcao-da-base/database_build.py --inicio 2025-01 --fim 2025-06
"""

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ORIGEM_PADRAO = Path(
    r"C:\Users\gabri\TransparenciaBrasil\cesta-de-precos-pncp\tasks"
    r"\analise-unidades-compradoras\tmp"
)
SAIDA_PADRAO = Path(__file__).resolve().parents[1] / "output"

COLUNA_AMPARO = "data.amparoLegal.codigo"
COLUNA_ID = "data.numeroControlePNCP"

# Amparos específicos do PNAE (Lei 11.947/2009)
AMPAROS_PNAE = {"138", "139"}
# Outros amparos plausíveis para alimentos e serviços de merenda
AMPAROS_CANDIDATOS = {
    "1", "2", "26", "78", "54", "19", "20", "21", "38", "39", "47", "140", "142", "16",
}
AMPAROS_FILTRO = AMPAROS_PNAE | AMPAROS_CANDIDATOS

TAMANHO_CHUNK = 100_000


@dataclass(frozen=True)
class ArquivoColeta:
    """Um ``contratacoes.csv`` e o período de coleta a que pertence."""

    caminho: Path
    ano: int
    mes: int
    periodo: str


def parse_ano_mes(valor: str) -> tuple[int, int]:
    """Converte ``AAAA-MM`` em ``(ano, mes)``."""
    match = re.fullmatch(r"(\d{4})-(\d{1,2})", valor)
    if not match or not 1 <= int(match.group(2)) <= 12:
        raise argparse.ArgumentTypeError(f"Use o formato AAAA-MM, recebido: {valor!r}")
    return int(match.group(1)), int(match.group(2))


def listar_arquivos(
    origem: Path,
    inicio: tuple[int, int],
    fim: tuple[int, int],
    nome_arquivo: str = "contratacoes.csv",
) -> list[ArquivoColeta]:
    """Lista os ``nome_arquivo`` das coletas entre ``inicio`` e ``fim`` (inclusive)."""
    arquivos = []
    for caminho in origem.glob(f"Coletas - */*/*/{nome_arquivo}"):
        periodo_dir = caminho.parent
        mes_dir = periodo_dir.parent
        ano_dir = mes_dir.parent

        ano = re.search(r"(\d{4})$", ano_dir.name)
        mes = re.match(r"(\d{1,2})\s*-", mes_dir.name)
        if not ano or not mes:
            print(f"Ignorando pasta fora do padrão: {caminho.parent}")
            continue

        ano_mes = (int(ano.group(1)), int(mes.group(1)))
        if inicio <= ano_mes <= fim:
            arquivos.append(ArquivoColeta(caminho, *ano_mes, periodo_dir.name))

    return sorted(arquivos, key=lambda a: (a.ano, a.mes, a.periodo))


def colunas_unificadas(arquivos: list[ArquivoColeta]) -> list[str]:
    """Une as colunas de todos os arquivos, já que elas variam entre coletas."""
    colunas: dict[str, None] = {}
    for arquivo in arquivos:
        for coluna in pd.read_csv(arquivo.caminho, nrows=0).columns:
            colunas.setdefault(coluna, None)
    return ["coleta_ano", "coleta_mes", "coleta_periodo", *colunas]


def normalizar_amparo(codigos: pd.Series) -> pd.Series:
    """Padroniza o código do amparo como texto sem casas decimais (``"19.0"`` -> ``"19"``)."""
    return codigos.str.strip().str.replace(r"\.0+$", "", regex=True)


def grupo_amparo(codigo: str) -> str:
    """Classifica o código de amparo em PNAE ou candidato."""
    return "PNAE" if codigo in AMPAROS_PNAE else "candidato"


def construir_base(
    arquivos: list[ArquivoColeta], saida_consolidado: Path, saida_filtrado: Path
) -> None:
    """Grava o CSV consolidado e o CSV filtrado pelos amparos legais, lendo em blocos."""
    colunas = colunas_unificadas(arquivos)
    colunas_filtrado = [*colunas, "grupo_amparo"]

    total_linhas = 0
    total_filtradas = 0
    contagem_amparos: Counter[str] = Counter()
    ids_filtrados: Counter[str] = Counter()
    primeiro_bloco = True

    for arquivo in arquivos:
        print(f"Lendo {arquivo.ano}/{arquivo.mes:02d} {arquivo.periodo}...")
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
            bloco = bloco.reindex(columns=colunas, fill_value="")

            modo = "w" if primeiro_bloco else "a"
            bloco.to_csv(saida_consolidado, mode=modo, header=primeiro_bloco, index=False)

            codigos = normalizar_amparo(bloco[COLUNA_AMPARO])
            filtrado = bloco[codigos.isin(AMPAROS_FILTRO)].copy()
            filtrado[COLUNA_AMPARO] = codigos[filtrado.index]
            filtrado["grupo_amparo"] = filtrado[COLUNA_AMPARO].map(grupo_amparo)
            filtrado = filtrado.reindex(columns=colunas_filtrado)
            filtrado.to_csv(saida_filtrado, mode=modo, header=primeiro_bloco, index=False)

            total_linhas += len(bloco)
            total_filtradas += len(filtrado)
            contagem_amparos.update(filtrado[COLUNA_AMPARO])
            ids_filtrados.update(filtrado[COLUNA_ID])
            primeiro_bloco = False

    repetidos = sum(1 for n in ids_filtrados.values() if n > 1)

    print()
    print(f"Arquivos lidos: {len(arquivos)}")
    print(f"Contratações no consolidado: {total_linhas:,}")
    print(f"Contratações filtradas por amparo: {total_filtradas:,}")
    print(f"numeroControlePNCP repetidos entre coletas (filtrado): {repetidos:,}")
    print("Contratações por código de amparo:")
    for codigo, n in sorted(contagem_amparos.items(), key=lambda item: -item[1]):
        print(f"  {codigo:>4} ({grupo_amparo(codigo)}): {n:,}")
    print()
    print(f"Consolidado: {saida_consolidado}")
    print(f"Filtrado:    {saida_filtrado}")


def main() -> None:
    """Lê os argumentos de linha de comando e executa a construção da base."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inicio", type=parse_ano_mes, required=True, help="Mês inicial (AAAA-MM)")
    parser.add_argument("--fim", type=parse_ano_mes, required=True, help="Mês final (AAAA-MM)")
    parser.add_argument("--origem", type=Path, default=ORIGEM_PADRAO, help="Pasta tmp das coletas")
    parser.add_argument("--saida", type=Path, default=SAIDA_PADRAO, help="Pasta de saída")
    args = parser.parse_args()

    if args.inicio > args.fim:
        parser.error("--inicio deve ser anterior ou igual a --fim")

    arquivos = listar_arquivos(args.origem, args.inicio, args.fim)
    if not arquivos:
        parser.error(f"Nenhum contratacoes.csv encontrado em {args.origem} para o período")

    sufixo = f"{args.inicio[0]}-{args.inicio[1]:02d}_a_{args.fim[0]}-{args.fim[1]:02d}"
    args.saida.mkdir(parents=True, exist_ok=True)
    construir_base(
        arquivos,
        args.saida / f"contratacoes_{sufixo}.csv",
        args.saida / f"contratacoes_amparos_merenda_{sufixo}.csv",
    )


if __name__ == "__main__":
    main()
