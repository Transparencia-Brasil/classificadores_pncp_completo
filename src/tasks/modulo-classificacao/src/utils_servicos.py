"""Funções comuns aos classificadores de serviços ligados à alimentação escolar.

Reúne o que as versões original, melhoria e novo pipeline compartilham: a
leitura da seleção CATSER de referência, a vetorização das descrições com
cache, a busca de termos pelo dicionário de contexto e a padronização dos
textos exportados.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from classifica_alimentos import PASTA_REFERENCIAS


# REFERÊNCIA CATSER -----------------------------------------------------------

CAMINHO_CATSER = PASTA_REFERENCIAS / "catalogo_catser_2026-10-01.xlsx"
ABA_ALIMENTACAO = "Alimentação escolar"
# As três primeiras linhas da aba guardam título e observações.
LINHA_CABECALHO_CATSER = 3
DATA_REFERENCIA_CATSER = "2026-10-01"

# Nomes das colunas na aba do CATSER e nomes adotados nos resultados.
COLUNAS_CATSER = {
    "Grupo proposto": "grupo_proposto",
    "Código do serviço": "codigo_servico",
    "Nome do serviço": "nome_servico",
    "Status": "status",
    "Cód. grupo": "codigo_grupo_cpc",
    "Grupo": "grupo_cpc",
    "Cód. classe": "codigo_classe_cpc",
    "Classe": "classe_cpc",
}


def preparar_referencia(caminho_fonte: Path, caminho_saida: Path) -> pd.DataFrame:
    """Lê a seleção de serviços CATSER ligados à alimentação escolar."""
    referencia = pd.read_excel(
        caminho_fonte, sheet_name=ABA_ALIMENTACAO, header=LINHA_CABECALHO_CATSER
    )
    ausentes = set(COLUNAS_CATSER).difference(referencia.columns)
    if ausentes:
        raise ValueError(f"Colunas ausentes na aba do CATSER: {sorted(ausentes)}")

    referencia = referencia.loc[
        referencia["Código do serviço"].notna(), list(COLUNAS_CATSER)
    ].rename(columns=COLUNAS_CATSER)
    referencia["codigo_servico"] = pd.to_numeric(
        referencia["codigo_servico"], errors="raise"
    ).astype("int64")
    referencia["codigo_classe_cpc"] = pd.to_numeric(
        referencia["codigo_classe_cpc"], errors="coerce"
    ).astype("Int64")
    referencia = referencia.sort_values(
        ["grupo_proposto", "codigo_servico"], kind="stable"
    ).reset_index(drop=True)

    # A contagem fixa impede usar silenciosamente outra versão da seleção.
    if len(referencia) != 29 or referencia["codigo_servico"].nunique() != 29:
        raise ValueError("A seleção de alimentação escolar deveria ter 29 serviços únicos.")

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    referencia.to_csv(caminho_saida, index=False, encoding="utf-8")
    return referencia


# EMBEDDINGS ------------------------------------------------------------------

# Descrições de serviço costumam começar com fórmulas genéricas ("contratação
# de empresa especializada para..."). Com 32 tokens, o objeto do serviço
# frequentemente ficaria fora do texto lido pelo modelo.
LIMITE_TOKENS_CONSULTAS = 128


def obter_embeddings_consultas(
    textos: list[str],
    modelo,
    nome_modelo: str,
    caminho_cache: Path,
    batch_size: int,
    dimensao_embedding: int,
    tamanho_bloco: int = 512,
) -> np.ndarray:
    """Vetoriza as descrições em blocos e permite retomar após interrupções."""
    caminho_metadados = caminho_cache.with_suffix(".json")
    hash_textos = hashlib.sha256("\n".join(textos).encode("utf-8")).hexdigest()
    metadados_base = {
        "modelo": nome_modelo,
        "hash_textos": hash_textos,
        "quantidade_textos": len(textos),
        "dimensao_embedding": dimensao_embedding,
        "limite_tokens_modelo": LIMITE_TOKENS_CONSULTAS,
        "tamanho_bloco": tamanho_bloco,
    }

    def gravar_metadados(proximo_indice: int) -> None:
        temporario = caminho_metadados.with_suffix(".tmp")
        temporario.write_text(
            json.dumps(
                {**metadados_base, "proximo_indice": proximo_indice},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporario.replace(caminho_metadados)

    proximo_indice = 0
    cache_valido = False
    if caminho_cache.exists() and caminho_metadados.exists():
        try:
            metadados = json.loads(caminho_metadados.read_text(encoding="utf-8"))
            cache_valido = all(
                metadados.get(chave) == valor for chave, valor in metadados_base.items()
            )
            if cache_valido:
                proximo_indice = int(metadados.get("proximo_indice", 0))
        except (json.JSONDecodeError, OSError, ValueError):
            cache_valido = False

    caminho_cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache_valido:
        caminho_temporario = caminho_cache.with_suffix(".tmp.npy")
        embeddings = np.lib.format.open_memmap(
            caminho_temporario,
            mode="w+",
            dtype="float32",
            shape=(len(textos), dimensao_embedding),
        )
        embeddings[:] = np.nan
        embeddings.flush()
        del embeddings
        caminho_temporario.replace(caminho_cache)
        gravar_metadados(0)

    embeddings = np.lib.format.open_memmap(caminho_cache, mode="r+")
    modelo.max_seq_length = LIMITE_TOKENS_CONSULTAS
    for inicio in range(proximo_indice, len(textos), tamanho_bloco):
        fim = min(inicio + tamanho_bloco, len(textos))
        embeddings[inicio:fim] = modelo.encode(
            textos[inicio:fim],
            prompt_name="query",
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        # Cada bloco é persistido antes do avanço ser registrado nos metadados.
        embeddings.flush()
        gravar_metadados(fim)

    resultado = np.asarray(embeddings, dtype="float32").copy()
    del embeddings
    return resultado


# DICIONÁRIO DE TERMOS --------------------------------------------------------

COLUNAS_TERMOS = [
    "id",
    "grupo",
    "tipo",
    "padrao",
    "origem",
    "exemplo_positivo",
    "exemplo_negativo",
    "observacao",
]
TIPOS_TERMOS = {"radical", "expressao", "regex"}
# Uma célula de exemplo pode trazer mais de uma frase separada por " | ".
SEPARADOR_EXEMPLOS = " | "

# Números com separadores internos ("11.947", "11.947/2009") formam um único
# termo. As demais sequências de letras e números são separadas por espaço.
PADRAO_TERMOS_BUSCA = re.compile(r"\d+(?:[./]\d+)+|[a-z0-9]+")


def normalizar_busca(valor: object) -> str:
    """Normaliza o texto para busca de termos sem remover stopwords."""
    if pd.isna(valor):
        return ""

    texto = "".join(
        caractere
        for caractere in unicodedata.normalize("NFKD", str(valor).lower())
        if not unicodedata.combining(caractere)
    )
    # Pontuação, quebras de linha e espaços repetidos viram um único espaço.
    return " ".join(PADRAO_TERMOS_BUSCA.findall(texto))


def montar_regex_termo(tipo: str, padrao: str) -> str:
    """Converte um termo do dicionário na expressão regular aplicada ao texto."""
    if tipo == "radical":
        # Mesmo comportamento da lista original: o radical inicia uma palavra.
        return rf"\b{re.escape(padrao)}[a-z0-9]*"
    if tipo == "expressao":
        palavras = (re.escape(palavra) for palavra in padrao.split())
        return r"\b" + r"\s+".join(palavras) + r"\b"
    return padrao


def separar_exemplos(valor: object) -> list[str]:
    """Separa as frases de exemplo de uma célula do dicionário."""
    if pd.isna(valor) or not str(valor).strip():
        return []
    return [frase.strip() for frase in str(valor).split(SEPARADOR_EXEMPLOS)]


def carregar_dicionario_termos(caminho: Path, grupos: set[str]) -> pd.DataFrame:
    """Lê o dicionário de termos e interrompe a execução se houver erros."""
    dicionario = pd.read_csv(caminho, dtype=str, keep_default_na=False)
    ausentes = set(COLUNAS_TERMOS).difference(dicionario.columns)
    if ausentes:
        raise ValueError(f"Colunas ausentes no dicionário de termos: {sorted(ausentes)}")
    dicionario = dicionario[COLUNAS_TERMOS].copy()

    # Todos os problemas são reunidos para que a correção ocorra de uma vez.
    erros: list[str] = []
    duplicados = dicionario.loc[dicionario["id"].duplicated(), "id"].tolist()
    if duplicados:
        erros.append(f"ids repetidos: {duplicados}")
    if not dicionario["grupo"].eq("alimentar").any():
        erros.append("o grupo 'alimentar' não tem termos")

    regex_termos = []
    for termo in dicionario.itertuples(index=False):
        if termo.grupo not in grupos:
            erros.append(f"{termo.id}: grupo desconhecido '{termo.grupo}'")
        if termo.tipo not in TIPOS_TERMOS:
            erros.append(f"{termo.id}: tipo desconhecido '{termo.tipo}'")
        if not termo.padrao.strip():
            erros.append(f"{termo.id}: padrão vazio")
        # Radicais e expressões são escritos como ficam no texto de busca.
        if termo.tipo in {"radical", "expressao"} and termo.padrao != normalizar_busca(termo.padrao):
            erros.append(
                f"{termo.id}: '{termo.padrao}' deveria ser escrito como "
                f"'{normalizar_busca(termo.padrao)}'"
            )

        regex = montar_regex_termo(termo.tipo, termo.padrao)
        regex_termos.append(regex)
        try:
            compilado = re.compile(regex)
        except re.error as erro:
            erros.append(f"{termo.id}: expressão inválida ({erro})")
            continue

        positivos = separar_exemplos(termo.exemplo_positivo)
        if not positivos:
            erros.append(f"{termo.id}: falta exemplo positivo")
        for frase in positivos:
            if not compilado.search(normalizar_busca(frase)):
                erros.append(f"{termo.id}: não casa com o exemplo positivo '{frase}'")
        for frase in separar_exemplos(termo.exemplo_negativo):
            if compilado.search(normalizar_busca(frase)):
                erros.append(f"{termo.id}: casa com o exemplo negativo '{frase}'")

    if erros:
        raise ValueError("Dicionário de termos inválido:\n- " + "\n- ".join(erros))

    dicionario["regex"] = regex_termos
    return dicionario


def compilar_padroes_por_grupo(dicionario: pd.DataFrame) -> dict[str, re.Pattern]:
    """Reúne os termos de cada grupo em uma única expressão, na ordem do arquivo."""
    return {
        grupo: re.compile("|".join(f"(?:{regex})" for regex in termos["regex"]))
        for grupo, termos in dicionario.groupby("grupo", sort=False)
    }


def aplicar_exclusoes(texto: str, padroes: dict[str, re.Pattern]) -> str:
    """Apaga os trechos-armadilha e devolve o texto usado na busca dos grupos."""
    if "exclusao" not in padroes:
        return texto
    return " ".join(padroes["exclusao"].sub(" ", texto).split())


def encontrar_termos(
    texto: str, padroes: dict[str, re.Pattern], grupos: list[str]
) -> str:
    """Lista os trechos encontrados nos grupos indicados, na ordem do texto."""
    trechos = sorted(
        (m.start(), m.group(0))
        for grupo in grupos
        if grupo in padroes
        for m in padroes[grupo].finditer(texto)
    )
    return ", ".join(dict.fromkeys(trecho for _, trecho in trechos))


# EXPORTAÇÃO ------------------------------------------------------------------

def padronizar_textos_exportados(tabela: pd.DataFrame) -> pd.DataFrame:
    """Substitui travessões por hífens somente nos arquivos de resultado."""
    resultado = tabela.copy()
    colunas_texto = resultado.select_dtypes(include=["object", "string"]).columns
    for coluna in colunas_texto:
        # Colunas "object" podem conter só booleanos e vazios, sem texto,
        # e nesse caso o acessor .str falha.
        resultado[coluna] = resultado[coluna].map(
            lambda valor: valor.replace("—", "-") if isinstance(valor, str) else valor
        )
    return resultado
