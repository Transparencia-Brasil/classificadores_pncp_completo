"""Pipeline completo de classificação dos itens do PNCP.

Para cada mês do período, o pipeline constrói a base de contratações e de itens
(sem repetições entre as coletas quinzenais), classifica os materiais como
candidatos a alimentos (CATMAT), marca o contexto escolar desses alimentos pela
contratação e classifica os serviços ligados à alimentação escolar (CATSER). No
fim, junta os resultados dos meses em arquivos do período.

Cada etapa concluída deixa um marcador em ``output/progresso`` com a duração da
etapa em segundos. Uma nova execução pula as etapas já feitas, a menos que se use
``--refazer``. O resumo do período é gerado à parte por ``src/resumo.py``.

Exemplo:
    python -m src.main --inicio 2025-01 --fim 2025-06
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PASTA_TASKS = Path(__file__).resolve().parent / "tasks"
PASTA_BASE = PASTA_TASKS / "modulo-construcao-da-base"
PASTA_CLASSIFICACAO = PASTA_TASKS / "modulo-classificacao"

# Os módulos ficam em pastas com hífen, que não podem ser importadas como
# pacotes. Os scripts são executados como estão e só os caminhos são importados.
sys.path[:0] = [str(PASTA_BASE / "src"), str(PASTA_CLASSIFICACAO / "src")]

from classifica_alimentos_servicos import caminhos_lote as caminhos_servicos  # noqa: E402
from contexto_escolar_alimentos import caminhos_contexto  # noqa: E402
from database_build import ORIGEM_PADRAO, SAIDA_PADRAO, parse_ano_mes  # noqa: E402
from itens_build import caminhos_saida as caminhos_itens  # noqa: E402

SCRIPT_CONTRATACOES = PASTA_BASE / "src" / "database_build.py"
SCRIPT_ITENS = PASTA_BASE / "src" / "itens_build.py"
SCRIPT_ALIMENTOS = PASTA_CLASSIFICACAO / "src" / "classifica_alimentos.py"
SCRIPT_CONTEXTO_ALIMENTOS = PASTA_CLASSIFICACAO / "src" / "contexto_escolar_alimentos.py"
SCRIPT_SERVICOS = PASTA_CLASSIFICACAO / "src" / "classifica_alimentos_servicos.py"

PASTA_PERIODO = PASTA_CLASSIFICACAO / "output" / "periodo"
# Arquivos do período. Os alimentos candidatos ficam divididos pelo contexto
# escolar; dos serviços, o período guarda só os classificados como True.
ARQUIVOS_PERIODO = {
    "alimentos_true": "alimentos_contexto_escolar_true_pncp_{}.csv",
    "alimentos_false": "alimentos_contexto_escolar_false_pncp_{}.csv.gz",
    "servicos": "servicos_alimentacao_escolar_pncp_{}.csv",
}
PASTA_PROGRESSO = PASTA_CLASSIFICACAO / "output" / "progresso"

COLUNAS_CHAVE_ITEM = ["endpoint", "numeroItem"]


def meses_do_periodo(inicio: tuple[int, int], fim: tuple[int, int]) -> list[tuple[int, int]]:
    """Lista os meses ``(ano, mes)`` entre ``inicio`` e ``fim`` (inclusive)."""
    meses = []
    ano, mes = inicio
    while (ano, mes) <= fim:
        meses.append((ano, mes))
        ano, mes = (ano + 1, 1) if mes == 12 else (ano, mes + 1)
    return meses


def sufixo_periodo(inicio: tuple[int, int], fim: tuple[int, int]) -> str:
    """Sufixo dos arquivos, no mesmo formato da construção da base."""
    return f"{inicio[0]}-{inicio[1]:02d}_a_{fim[0]}-{fim[1]:02d}"


def marcador_etapa(sufixo: str, nome: str) -> Path:
    """Arquivo que marca a etapa como concluída e guarda sua duração em segundos."""
    return PASTA_PROGRESSO / f"{sufixo}_{nome}.ok"


def executar_etapa(nome: str, sufixo: str, comando: list[str], refazer: bool) -> bool:
    """Roda um script e registra a conclusão, ou pula se já foi concluído.

    Devolve ``True`` quando o script rodou nesta execução.
    """
    marcador = marcador_etapa(sufixo, nome)
    if marcador.exists() and not refazer:
        print(f"[{sufixo}] {nome}: já concluída, pulando")
        return False

    print(f"\n[{sufixo}] {nome}: {' '.join(comando)}", flush=True)
    inicio = time.perf_counter()
    subprocess.run([sys.executable, *comando], check=True)
    PASTA_PROGRESSO.mkdir(parents=True, exist_ok=True)
    marcador.write_text(f"{time.perf_counter() - inicio:.0f}", encoding="utf-8")
    return True


def processar_mes(mes: tuple[int, int], origem: Path, refazer: bool) -> str:
    """Constrói a base do mês e classifica materiais e serviços."""
    rotulo = f"{mes[0]}-{mes[1]:02d}"
    sufixo = sufixo_periodo(mes, mes)
    periodo = ["--inicio", rotulo, "--fim", rotulo, "--origem", str(origem), "--deduplicar"]
    itens = caminhos_itens(SAIDA_PADRAO, sufixo)
    contratacoes = str(SAIDA_PADRAO / f"contratacoes_{sufixo}.csv")

    executar_etapa("contratacoes", sufixo, [str(SCRIPT_CONTRATACOES), *periodo], refazer)
    executar_etapa("itens", sufixo, [str(SCRIPT_ITENS), *periodo], refazer)
    alimentos_refeitos = executar_etapa(
        "alimentos",
        sufixo,
        [str(SCRIPT_ALIMENTOS), "--itens", str(itens["materiais"]), "--sufixo", sufixo],
        refazer,
    )
    # Os arquivos com e sem contexto escolar saem do CSV de alimentos, por isso
    # o contexto é refeito sempre que os alimentos são refeitos. Mudanças só no
    # contexto (como o dicionário de termos) refazem apenas esta etapa.
    executar_etapa(
        "contexto_alimentos",
        sufixo,
        [str(SCRIPT_CONTEXTO_ALIMENTOS), "--sufixo", sufixo, "--contratacoes", contratacoes],
        refazer or alimentos_refeitos,
    )
    executar_etapa(
        "servicos",
        sufixo,
        [
            str(SCRIPT_SERVICOS),
            "--itens", str(itens["servicos"]),
            "--contratacoes", contratacoes,
            "--sufixo", sufixo,
        ],
        refazer,
    )
    return sufixo


def chave_item(tabela: pd.DataFrame) -> pd.Series:
    """Chave textual do item: compra e número do item."""
    return tabela["endpoint"] + "|" + tabela["numeroItem"]


def remover_versoes_antigas(
    tabelas: list[pd.DataFrame], chaves_base: list[set[str]]
) -> list[pd.DataFrame]:
    """Descarta itens que reaparecem na base de um mês posterior.

    ``tabelas`` e ``chaves_base`` estão em ordem cronológica. Um item classificado
    em janeiro e coletado de novo em fevereiro vale pela versão de fevereiro, seja
    ela candidata ou não, por isso a comparação usa a base completa de cada mês.
    """
    vistos_depois: set[str] = set()
    resultado = list(tabelas)
    for posicao in reversed(range(len(tabelas))):
        tabela = tabelas[posicao]
        resultado[posicao] = tabela[~chave_item(tabela).isin(vistos_depois)]
        vistos_depois |= chaves_base[posicao]
    return resultado


def juntar_periodo(sufixos: list[str], sufixo_total: str) -> dict[str, pd.DataFrame]:
    """Junta os resultados mensais em arquivos do período."""
    chaves_base = []
    for sufixo in sufixos:
        base = pd.read_csv(
            caminhos_itens(SAIDA_PADRAO, sufixo)["todos"],
            dtype=str,
            keep_default_na=False,
            usecols=COLUNAS_CHAVE_ITEM,
        )
        chaves_base.append(set(chave_item(base)))

    resultados = {
        "alimentos_true": [caminhos_contexto(sufixo)["true"] for sufixo in sufixos],
        "alimentos_false": [caminhos_contexto(sufixo)["false"] for sufixo in sufixos],
        "servicos": [caminhos_servicos(sufixo)["true"] for sufixo in sufixos],
    }
    destinos = {
        nome: PASTA_PERIODO / arquivo.format(sufixo_total)
        for nome, arquivo in ARQUIVOS_PERIODO.items()
    }

    PASTA_PERIODO.mkdir(parents=True, exist_ok=True)
    periodo = {}
    for nome, caminhos in resultados.items():
        tabelas = [
            pd.read_csv(caminho, dtype=str, keep_default_na=False) for caminho in caminhos
        ]
        tabelas = remover_versoes_antigas(tabelas, chaves_base)
        periodo[nome] = pd.concat(tabelas, ignore_index=True)
        periodo[nome].to_csv(destinos[nome], index=False, encoding="utf-8")
        print(f"{nome}: {len(periodo[nome]):,} itens -> {destinos[nome]}")
    return periodo


def main() -> None:
    """Lê os argumentos e executa o pipeline mês a mês."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inicio", type=parse_ano_mes, required=True, help="Mês inicial (AAAA-MM)")
    parser.add_argument("--fim", type=parse_ano_mes, required=True, help="Mês final (AAAA-MM)")
    parser.add_argument("--origem", type=Path, default=ORIGEM_PADRAO, help="Pasta tmp das coletas")
    parser.add_argument(
        "--refazer", action="store_true", help="Refaz as etapas já concluídas"
    )
    args = parser.parse_args()

    if args.inicio > args.fim:
        parser.error("--inicio deve ser anterior ou igual a --fim")

    sufixos = [
        processar_mes(mes, args.origem, args.refazer)
        for mes in meses_do_periodo(args.inicio, args.fim)
    ]

    print()
    periodo = juntar_periodo(sufixos, sufixo_periodo(args.inicio, args.fim))
    print("Alimentos candidatos por contexto escolar:")
    alimentos = pd.concat([periodo["alimentos_true"], periodo["alimentos_false"]])
    metodos = alimentos["metodo_contexto_escolar"].value_counts()
    for metodo, n in metodos.items():
        print(f"  {metodo}: {n:,}")
    print("Serviços por método de classificação:")
    metodos = periodo["servicos"]["metodo_classificacao"].value_counts()
    for metodo, n in metodos.items():
        print(f"  {metodo}: {n:,}")


if __name__ == "__main__":
    main()
