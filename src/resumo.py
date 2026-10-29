"""Resumo dos resultados do pipeline em um período.

Lê as bases de itens de cada mês, os arquivos do período gerados por
``src/main.py`` e os marcadores de progresso, e informa:

- quantos materiais e serviços foram analisados no período;
- quantos foram classificados e em qual grupo;
- quanto tempo cada etapa levou.

Itens que reaparecem em mais de um mês contam uma vez, como nos arquivos do
período. O resumo é impresso no terminal e gravado em
``output/periodo/resumo_<período>.xlsx``.

Exemplo:
    python src/resumo.py --inicio 2025-01 --fim 2025-03
"""

import argparse

import pandas as pd

from main import (
    COLUNAS_CHAVE_ITEM,
    PASTA_PERIODO,
    SAIDA_PADRAO,
    caminhos_itens,
    chave_item,
    marcador_etapa,
    meses_do_periodo,
    parse_ano_mes,
    sufixo_periodo,
)

ETAPAS = ["contratacoes", "itens", "alimentos", "servicos"]
# Coluna do grupo atribuído por cada classificador.
COLUNA_GRUPO = {"materiais": "grupo_taxonomico", "servicos": "grupo_proposto"}
ARQUIVO_PERIODO = {
    "materiais": "alimentos_candidatos_pncp_{}.csv",
    "servicos": "servicos_alimentacao_escolar_pncp_{}.csv",
}


def contar_analisados(sufixos: list[str], tipo: str) -> int:
    """Conta os itens distintos de um tipo (materiais ou serviços) no período."""
    chaves: set[str] = set()
    for sufixo in sufixos:
        base = pd.read_csv(
            caminhos_itens(SAIDA_PADRAO, sufixo)[tipo],
            dtype=str,
            keep_default_na=False,
            usecols=COLUNAS_CHAVE_ITEM,
        )
        chaves |= set(chave_item(base))
    return len(chaves)


def contar_por_grupo(sufixo_total: str, tipo: str) -> pd.DataFrame:
    """Conta os itens classificados no período por grupo."""
    coluna = COLUNA_GRUPO[tipo]
    classificados = pd.read_csv(
        PASTA_PERIODO / ARQUIVO_PERIODO[tipo].format(sufixo_total),
        dtype=str,
        keep_default_na=False,
        usecols=[coluna],
    )
    grupos = classificados[coluna].replace("", "(sem grupo)").value_counts()
    return pd.DataFrame(
        {
            "grupo": grupos.index,
            "itens": grupos.values,
            "percentual": (100 * grupos.values / grupos.sum()).round(1),
        }
    )


def ler_duracoes(sufixos: list[str]) -> pd.DataFrame:
    """Lê a duração de cada etapa, em segundos, dos marcadores de progresso.

    Marcadores gravados antes do registro da duração estão vazios; nesses casos
    a duração fica ausente.
    """
    linhas = []
    for sufixo in sufixos:
        for etapa in ETAPAS:
            marcador = marcador_etapa(sufixo, etapa)
            conteudo = marcador.read_text(encoding="utf-8").strip() if marcador.exists() else ""
            linhas.append(
                {
                    "mes": sufixo,
                    "etapa": etapa,
                    "concluida": marcador.exists(),
                    "segundos": float(conteudo) if conteudo else None,
                }
            )
    return pd.DataFrame(linhas)


def formatar_duracao(segundos: float) -> str:
    """Escreve a duração em horas, minutos e segundos."""
    horas, resto = divmod(int(segundos), 3600)
    minutos, segundos = divmod(resto, 60)
    if horas:
        return f"{horas} h {minutos:02d} min"
    if minutos:
        return f"{minutos} min {segundos:02d} s"
    return f"{segundos} s"


def montar_resumo(inicio: tuple[int, int], fim: tuple[int, int]) -> dict[str, pd.DataFrame]:
    """Monta as tabelas do resumo do período."""
    sufixos = [sufixo_periodo(mes, mes) for mes in meses_do_periodo(inicio, fim)]
    sufixo_total = sufixo_periodo(inicio, fim)

    indicadores = []
    grupos = {}
    for tipo in COLUNA_GRUPO:
        analisados = contar_analisados(sufixos, tipo)
        grupos[tipo] = contar_por_grupo(sufixo_total, tipo)
        classificados = int(grupos[tipo]["itens"].sum())
        indicadores += [
            {"indicador": f"{tipo} analisados", "valor": analisados},
            {"indicador": f"{tipo} classificados", "valor": classificados},
            {
                "indicador": f"{tipo} classificados (%)",
                "valor": round(100 * classificados / analisados, 2) if analisados else 0,
            },
        ]

    duracoes = ler_duracoes(sufixos)
    registradas = duracoes["segundos"].notna()
    indicadores += [
        {"indicador": "etapas concluídas", "valor": int(duracoes["concluida"].sum())},
        {"indicador": "etapas com duração registrada", "valor": int(registradas.sum())},
        {"indicador": "tempo de execução (s)", "valor": duracoes["segundos"].sum()},
    ]
    duracoes["duracao"] = duracoes["segundos"].map(
        lambda s: formatar_duracao(s) if pd.notna(s) else "não registrada"
    )

    return {
        "Resumo": pd.DataFrame(indicadores),
        "Materiais por grupo": grupos["materiais"],
        "Serviços por grupo": grupos["servicos"],
        "Tempo de execução": duracoes,
    }


def imprimir_resumo(tabelas: dict[str, pd.DataFrame], periodo: str) -> None:
    """Mostra o resumo no terminal."""
    resumo = tabelas["Resumo"].set_index("indicador")["valor"]
    duracoes = tabelas["Tempo de execução"]

    print(f"RESUMO DO PERÍODO {periodo}")
    for tipo, titulo in [("materiais", "Materiais"), ("servicos", "Serviços")]:
        analisados = int(resumo[f"{tipo} analisados"])
        classificados = int(resumo[f"{tipo} classificados"])
        print(f"\n{titulo} analisados: {analisados:,}")
        print(f"{titulo} classificados: {classificados:,} ({resumo[f'{tipo} classificados (%)']}%)")
        print("Por grupo:")
        nome_tabela = "Materiais por grupo" if tipo == "materiais" else "Serviços por grupo"
        for _, linha in tabelas[nome_tabela].iterrows():
            print(f"  {linha['grupo']}: {linha['itens']:,} ({linha['percentual']}%)")

    print("\nTempo de execução por etapa:")
    for _, linha in duracoes.iterrows():
        situacao = linha["duracao"] if linha["concluida"] else "não concluída"
        print(f"  {linha['mes']} {linha['etapa']}: {situacao}")
    registradas = int(resumo["etapas com duração registrada"])
    total = formatar_duracao(resumo["tempo de execução (s)"]) if registradas else "não registrado"
    print(f"Total: {total} ({registradas} de {len(duracoes)} etapas com duração registrada)")


def main() -> None:
    """Lê os argumentos, imprime o resumo e grava a planilha."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inicio", type=parse_ano_mes, required=True, help="Mês inicial (AAAA-MM)")
    parser.add_argument("--fim", type=parse_ano_mes, required=True, help="Mês final (AAAA-MM)")
    args = parser.parse_args()

    periodo = sufixo_periodo(args.inicio, args.fim)
    tabelas = montar_resumo(args.inicio, args.fim)
    imprimir_resumo(tabelas, periodo)

    destino = PASTA_PERIODO / f"resumo_{periodo}.xlsx"
    with pd.ExcelWriter(destino, engine="openpyxl") as escritor:
        for nome_aba, tabela in tabelas.items():
            tabela.to_excel(escritor, sheet_name=nome_aba, index=False)
    print(f"\nPlanilha: {destino}")


if __name__ == "__main__":
    main()
