"""Classifica itens do PNCP como candidatos a alimentos.

O fluxo primeiro identifica um único PDM no texto da compra. Em seguida,
compara a descrição do PNCP somente com os itens CATMAT desse PDM e conserva o
item mais semelhante. O resultado é exploratório: o limite de 0,5 seleciona
candidatos, mas não confirma sozinho que uma compra seja de alimento.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords


# O modelo transforma as descrições em vetores. A similaridade mínima de 0,5
# define quais pareamentos entram na base de alimentos candidatos.
MODELO_PADRAO = "Snowflake/snowflake-arctic-embed-l-v2.0"
LIMITE_PADRAO = 0.5

# As descrições CATMAT usam até 8.192 tokens. As consultas PNCP usam no máximo
# 32 tokens, e a descrição original permanece completa nos resultados.
LIMITE_TOKENS_DOCUMENTOS = 8192
LIMITE_TOKENS_MODELO = 32

# Os caminhos partem da pasta do módulo para evitar dependências de pastas
# pessoais e permitir a mesma execução em outras máquinas. O lote PNCP de
# entrada é sempre informado na execução.
PASTA_TAREFA = Path(__file__).resolve().parents[1]
# Referências copiadas das tarefas de taxonomia, CATSER e dicionário de termos
# do repositório alimentacao-escolar-analises.
PASTA_REFERENCIAS = PASTA_TAREFA / "referencias"
CAMINHO_CATALOGO_CONSOLIDADO = (
    PASTA_REFERENCIAS / "catalogo_consolidado_itens_catmat_grupo_89_v0_1.csv"
)
PASTA_OUTPUT = PASTA_TAREFA / "output"
PASTA_CACHE = PASTA_TAREFA / "cache"
# O catálogo é o mesmo para todos os lotes, então seus vetores ficam num cache
# único. As consultas mudam a cada lote e têm cache próprio (ver caminhos_lote).
CAMINHO_CLASSE_ALIMENTOS = PASTA_OUTPUT / "classe_alimentos_catmat_v0_1.csv"
CAMINHO_CACHE = PASTA_CACHE / "catalogo_alimentos_vetorizado.npz"


def caminhos_lote(sufixo: str, pasta_saida: Path = PASTA_OUTPUT) -> dict[str, Path]:
    """Caminhos dos resultados e do cache de consultas de um lote do PNCP."""
    return {
        "alimentos": pasta_saida / f"alimentos_candidatos_pncp_{sufixo}.csv",
        "diagnostico": pasta_saida / f"diagnostico_pareamentos_pncp_catmat_{sufixo}.csv.gz",
        "resumo_excel": pasta_saida / f"resumo_classificacao_alimentos_pncp_{sufixo}.xlsx",
        "cache_consultas": PASTA_CACHE / sufixo / "consultas_pncp_vetorizadas.npy",
    }


# Somente estes campos formam a referência alimentar usada no pareamento.
COLUNAS_CATALOGO_REFERENCIA = [
    "versao_taxonomia",
    "data_referencia_catmat",
    "codigo_pdm",
    "nome_pdm",
    "codigo_item",
    "descricao_item",
    "grupo_taxonomico",
    "subgrupo_taxonomico",
    "grau_processamento",
    "aplicabilidade_pnae",
    "sociobiodiversidade",
]

# Estas colunas documentam a origem de cada decisão acrescentada ao PNCP.
COLUNAS_RESULTADO = [
    "descricao_corrigida",
    "descricao_normalizada",
    "problema_codificacao",
    "codigo_pdm_detectado",
    "nome_pdm_detectado",
    "evidencia_pdm",
    "codigo_item_catmat",
    "descricao_item_catmat",
    "similaridade",
    "alimento_candidato",
    "grupo_taxonomico",
    "subgrupo_taxonomico",
    "grau_processamento",
    "aplicabilidade_pnae",
    "sociobiodiversidade",
    "metodo_classificacao",
    "modelo_embedding",
    "limite_tokens_modelo",
    "limite_operacional",
    "versao_taxonomia",
    "data_referencia_catmat",
    "janela_referencia_pncp",
]


# VALIDAÇÃO DOS ARQUIVOS E PREPARAÇÃO DOS TEXTOS ------------------------------

def calcular_sha256(caminho: Path) -> str:
    """Calcula o hash de um arquivo sem carregá-lo inteiro na memória."""
    resumo = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def carregar_stopwords() -> set[str]:
    """Carrega as stopwords em português e baixa o recurso se necessário."""
    try:
        return set(stopwords.words("portuguese"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("portuguese"))


def corrigir_codificacao(valor: object) -> tuple[str, str]:
    """Corrige mojibake recuperável e registra problemas que permanecem."""
    if pd.isna(valor):
        return "", "descricao ausente"

    texto = str(valor)
    corrigido = texto
    # A correção ocorre em uma cópia. A descrição original continua preservada
    # para auditoria nos arquivos de resultado.
    if "Ã" in texto or "Â" in texto:
        for codificacao in ("latin1", "cp1252"):
            try:
                candidato = texto.encode(codificacao).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            marcas_antes = texto.count("Ã") + texto.count("Â")
            marcas_depois = candidato.count("Ã") + candidato.count("Â")
            if marcas_depois < marcas_antes:
                corrigido = candidato
                break

    if "�" in corrigido:
        return corrigido, "caractere irrecuperavel"
    if corrigido != texto:
        return corrigido, "acentos corrigidos"
    return corrigido, "sem problema detectado"


def normalizar_texto(valor: object, palavras_vazias: set[str]) -> str:
    """Normaliza o texto preservando números e unidades relevantes."""
    if pd.isna(valor):
        return ""

    texto = "".join(
        caractere
        for caractere in unicodedata.normalize("NFKD", str(valor).lower())
        if not unicodedata.combining(caractere)
    )
    # Números permanecem porque podem representar concentração, peso ou volume.
    termos = re.findall(r"[a-z0-9]+", texto)
    return " ".join(termo for termo in termos if termo not in palavras_vazias)


def preparar_catalogo(caminho_fonte: Path, caminho_saida: Path) -> pd.DataFrame:
    """Seleciona os itens classificados e candidatos ao PNAE."""
    catalogo = pd.read_csv(caminho_fonte, low_memory=False)

    # Indeterminados e itens fora do escopo não entram como referência positiva.
    catalogo = catalogo.loc[
        catalogo["situacao_classificacao"].eq("Classificado")
        & catalogo["aplicabilidade_pnae"].eq("candidata"),
        COLUNAS_CATALOGO_REFERENCIA,
    ].copy()

    catalogo["codigo_pdm"] = pd.to_numeric(
        catalogo["codigo_pdm"], errors="raise"
    ).astype("int64")
    catalogo["codigo_item"] = pd.to_numeric(
        catalogo["codigo_item"], errors="raise"
    ).astype("int64")
    catalogo = catalogo.sort_values(
        ["codigo_pdm", "codigo_item"], kind="stable"
    ).reset_index(drop=True)

    # A contagem fixa impede usar silenciosamente outra versão da taxonomia.
    if len(catalogo) != 4047 or catalogo["codigo_item"].nunique() != 4047:
        raise ValueError("O catálogo de referência deveria conter 4.047 itens únicos.")

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    catalogo.to_csv(caminho_saida, index=False, encoding="utf-8")
    return catalogo


def carregar_itens(caminho: Path) -> pd.DataFrame:
    """Lê e valida a estrutura mínima do lote PNCP."""
    itens = pd.read_csv(caminho, low_memory=False, encoding="utf-8")
    obrigatorias = {"endpoint", "numeroItem", "descricao", "materialOuServico"}
    ausentes = obrigatorias.difference(itens.columns)
    if ausentes:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(ausentes)}")
    if itens.duplicated(["endpoint", "numeroItem"]).any():
        raise ValueError("A chave endpoint + numeroItem possui duplicações.")
    return itens


def preparar_materiais(
    itens: pd.DataFrame, palavras_vazias: set[str]
) -> pd.DataFrame:
    """Seleciona materiais e cria campos derivados sem alterar a descrição original."""

    # Serviços ficam registrados no resumo, mas não seguem para o catálogo de
    # referência, que contém somente materiais.
    materiais = itens.loc[itens["materialOuServico"].eq("M")].copy()
    correcao = materiais["descricao"].map(corrigir_codificacao)
    materiais["descricao_corrigida"] = correcao.str[0]
    materiais["problema_codificacao"] = correcao.str[1]
    materiais["descricao_normalizada"] = materiais["descricao_corrigida"].map(
        lambda texto: normalizar_texto(texto, palavras_vazias)
    )
    return materiais


def padronizar_textos_exportados(tabela: pd.DataFrame) -> pd.DataFrame:
    """Substitui travessões por hífens somente nos arquivos de resultado."""
    resultado = tabela.copy()
    colunas_texto = resultado.select_dtypes(include=["object", "string"]).columns
    for coluna in colunas_texto:
        resultado[coluna] = resultado[coluna].str.replace("\u2014", "-", regex=False)
    return resultado


# DETECÇÃO TEXTUAL DO PDM -----------------------------------------------------

def criar_indice_pdms(
    catalogo: pd.DataFrame, palavras_vazias: set[str]
) -> tuple[dict[str, set[int]], dict[int, str]]:
    """Mapeia cada termo dos nomes de PDM aos códigos correspondentes."""
    pdms = catalogo[["codigo_pdm", "nome_pdm"]].drop_duplicates()
    indice: dict[str, set[int]] = {}
    nomes: dict[int, str] = {}

    for linha in pdms.itertuples(index=False):
        codigo = int(linha.codigo_pdm)
        nomes[codigo] = str(linha.nome_pdm)
        nome_normalizado = normalizar_texto(linha.nome_pdm, palavras_vazias)
        for termo in nome_normalizado.split():
            indice.setdefault(termo, set()).add(codigo)
    return indice, nomes


def detectar_pdm(
    descricao: str, indice: dict[str, set[int]]
) -> tuple[int | None, str]:
    """Retorna um PDM apenas quando a interseção dos termos é única."""
    candidatos: set[int] = set()
    evidencias: list[str] = []

    for termo in descricao.split():
        if termo not in indice:
            continue
        candidatos = (
            set(indice[termo])
            if not candidatos
            else candidatos.intersection(indice[termo])
        )
        evidencias.append(termo)
        # O primeiro resultado único encerra a busca. Casos ambíguos não são
        # enviados aos embeddings para evitar comparar famílias diferentes.
        if len(candidatos) == 1:
            return next(iter(candidatos)), ", ".join(evidencias)
    if len(candidatos) == 1:
        return next(iter(candidatos)), ", ".join(evidencias)
    return None, ", ".join(evidencias)


def detectar_pdms(
    materiais: pd.DataFrame,
    catalogo: pd.DataFrame,
    palavras_vazias: set[str],
) -> pd.DataFrame:
    """Aplica a pré-filtragem textual e preserva somente PDMs únicos."""
    indice, nomes = criar_indice_pdms(catalogo, palavras_vazias)
    deteccoes = materiais["descricao_normalizada"].map(
        lambda descricao: detectar_pdm(descricao, indice)
    )
    materiais["codigo_pdm_detectado"] = pd.to_numeric(
        deteccoes.str[0], errors="coerce"
    ).astype("Int64")
    materiais["evidencia_pdm"] = deteccoes.str[1]
    materiais["nome_pdm_detectado"] = materiais["codigo_pdm_detectado"].map(nomes)
    return materiais.loc[materiais["codigo_pdm_detectado"].notna()].copy()


def texto_catalogo_para_modelo(descricao: object) -> str:
    """Retira o código inicial da descrição CATMAT antes da vetorização."""

    # O código identifica o item, mas não possui significado semântico.
    return re.sub(r"^\s*\d+\s*-\s*", "", str(descricao)).strip()


# MODELO E CACHES DE EMBEDDINGS ----------------------------------------------

def carregar_modelo(nome_modelo: str):
    """Carrega o modelo somente quando a vetorização é iniciada."""
    from sentence_transformers import SentenceTransformer

    # CPU foi escolhida porque é o recurso disponível e torna a execução local
    # independente de uma placa gráfica.
    modelo = SentenceTransformer(nome_modelo, device="cpu")
    modelo.max_seq_length = LIMITE_TOKENS_DOCUMENTOS
    return modelo


def obter_embeddings_catalogo(
    catalogo: pd.DataFrame,
    modelo,
    nome_modelo: str,
    caminho_catalogo: Path,
    caminho_cache: Path,
    batch_size: int,
) -> np.ndarray:
    """Carrega ou recria o cache validado dos embeddings CATMAT."""
    caminho_metadados = caminho_cache.with_suffix(".json")
    hash_catalogo = calcular_sha256(caminho_catalogo)
    metadados_esperados = {
        "modelo": nome_modelo,
        "sha256_catalogo": hash_catalogo,
        "quantidade_itens": len(catalogo),
        "versao_taxonomia": str(catalogo["versao_taxonomia"].iloc[0]),
        "limite_tokens_documentos": LIMITE_TOKENS_DOCUMENTOS,
    }

    # O cache só é reutilizado quando modelo, arquivo e versão são os mesmos.
    if caminho_cache.exists() and caminho_metadados.exists():
        metadados = json.loads(caminho_metadados.read_text(encoding="utf-8"))
        if metadados == metadados_esperados:
            with np.load(caminho_cache) as dados:
                embeddings = dados["embeddings"]
            if len(embeddings) == len(catalogo):
                return embeddings.astype("float32")

    textos = catalogo["descricao_item"].map(texto_catalogo_para_modelo).tolist()
    modelo.max_seq_length = LIMITE_TOKENS_DOCUMENTOS
    embeddings = modelo.encode(
        textos,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        # Vetores normalizados permitem calcular cosseno por produto matricial.
        normalize_embeddings=True,
    ).astype("float32")

    caminho_cache.parent.mkdir(parents=True, exist_ok=True)
    # A gravação temporária evita substituir um cache válido por arquivo parcial.
    caminho_cache_temporario = caminho_cache.with_suffix(".tmp")
    with caminho_cache_temporario.open("wb") as arquivo:
        np.savez_compressed(arquivo, embeddings=embeddings)
    caminho_cache_temporario.replace(caminho_cache)

    caminho_metadados_temporario = caminho_metadados.with_suffix(".tmp")
    caminho_metadados_temporario.write_text(
        json.dumps(metadados_esperados, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    caminho_metadados_temporario.replace(caminho_metadados)
    return embeddings


def obter_embeddings_consultas(
    consultas: pd.DataFrame,
    modelo,
    nome_modelo: str,
    caminho_cache: Path,
    batch_size: int,
    dimensao_embedding: int,
    tamanho_bloco: int = 512,
) -> np.ndarray:
    """Vetoriza consultas em blocos e permite retomar após interrupções."""
    caminho_metadados = caminho_cache.with_suffix(".json")
    hash_consultas = hashlib.sha256(
        pd.util.hash_pandas_object(
            consultas[["descricao_corrigida", "codigo_pdm_detectado"]], index=False
        ).values.tobytes()
    ).hexdigest()
    # A assinatura vincula o checkpoint ao texto, PDM, modelo e limite de tokens.
    metadados_base = {
        "modelo": nome_modelo,
        "hash_consultas": hash_consultas,
        "quantidade_consultas": len(consultas),
        "dimensao_embedding": dimensao_embedding,
        "limite_tokens_modelo": LIMITE_TOKENS_MODELO,
        "tamanho_bloco": tamanho_bloco,
    }

    proximo_indice = 0
    cache_valido = False
    if caminho_cache.exists() and caminho_metadados.exists():
        try:
            metadados = json.loads(caminho_metadados.read_text(encoding="utf-8"))
            cache_valido = all(
                metadados.get(chave) == valor
                for chave, valor in metadados_base.items()
            )
            if cache_valido:
                proximo_indice = int(metadados.get("proximo_indice", 0))
        except (json.JSONDecodeError, OSError, ValueError):
            cache_valido = False

    caminho_cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache_valido:
        # O arquivo é criado uma única vez e preenchido gradualmente por blocos.
        caminho_temporario = caminho_cache.with_suffix(".tmp.npy")
        embeddings = np.lib.format.open_memmap(
            caminho_temporario,
            mode="w+",
            dtype="float32",
            shape=(len(consultas), dimensao_embedding),
        )
        embeddings[:] = np.nan
        embeddings.flush()
        del embeddings
        caminho_temporario.replace(caminho_cache)
        metadados = {**metadados_base, "proximo_indice": 0}
        caminho_metadados_temporario = caminho_metadados.with_suffix(".tmp")
        caminho_metadados_temporario.write_text(
            json.dumps(metadados, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        caminho_metadados_temporario.replace(caminho_metadados)

    embeddings = np.lib.format.open_memmap(caminho_cache, mode="r+")
    # As consultas usam no máximo 32 tokens. A descrição original exportada não
    # é alterada por esse parâmetro do modelo.
    modelo.max_seq_length = LIMITE_TOKENS_MODELO
    for inicio in range(proximo_indice, len(consultas), tamanho_bloco):
        fim = min(inicio + tamanho_bloco, len(consultas))
        bloco = modelo.encode(
            consultas["descricao_corrigida"].iloc[inicio:fim].tolist(),
            prompt_name="query",
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        embeddings[inicio:fim] = bloco
        # Cada bloco é persistido antes do avanço ser registrado nos metadados.
        embeddings.flush()

        metadados = {**metadados_base, "proximo_indice": fim}
        caminho_temporario = caminho_metadados.with_suffix(".tmp")
        caminho_temporario.write_text(
            json.dumps(metadados, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        caminho_temporario.replace(caminho_metadados)

    resultado = np.asarray(embeddings, dtype="float32").copy()
    del embeddings
    return resultado


# PAREAMENTO PNCP E CATMAT ----------------------------------------------------

def encontrar_itens_similares(
    candidatos: pd.DataFrame,
    catalogo: pd.DataFrame,
    embeddings_catalogo: np.ndarray,
    modelo,
    nome_modelo: str,
    limite: float,
    batch_size: int,
    caminho_cache_consultas: Path,
    janela: str,
) -> pd.DataFrame:
    """Compara cada descrição apenas aos itens CATMAT do PDM detectado."""

    # Descrições repetidas no mesmo PDM são vetorizadas uma vez e depois
    # devolvidas a todas as ocorrências por meio da chave abaixo.
    chaves = ["descricao_corrigida", "codigo_pdm_detectado"]
    consultas = candidatos[chaves].drop_duplicates().reset_index(drop=True)
    embeddings_consultas = obter_embeddings_consultas(
        consultas,
        modelo,
        nome_modelo,
        caminho_cache_consultas,
        batch_size,
        embeddings_catalogo.shape[1],
    )

    melhor_indice = np.full(len(consultas), -1, dtype="int64")
    melhor_similaridade = np.full(len(consultas), np.nan, dtype="float32")
    codigos_catalogo = catalogo["codigo_pdm"].to_numpy()

    # A busca fica dentro do PDM para impedir que categorias CATMAT diferentes
    # concorram entre si apenas por possuírem palavras parecidas.
    for codigo_pdm, indices_consultas in consultas.groupby(
        "codigo_pdm_detectado", sort=True
    ).groups.items():
        indices_referencia = np.flatnonzero(codigos_catalogo == int(codigo_pdm))
        if len(indices_referencia) == 0:
            continue
        indices_consultas = np.asarray(list(indices_consultas), dtype="int64")
        matriz_similaridade = (
            embeddings_consultas[indices_consultas]
            @ embeddings_catalogo[indices_referencia].T
        )
        # Apenas o item CATMAT de maior similaridade é mantido para cada consulta.
        posicoes = matriz_similaridade.argmax(axis=1)
        melhor_indice[indices_consultas] = indices_referencia[posicoes]
        melhor_similaridade[indices_consultas] = matriz_similaridade[
            np.arange(len(indices_consultas)), posicoes
        ]

    consultas["indice_catalogo"] = melhor_indice
    consultas["similaridade"] = melhor_similaridade
    consultas = consultas.loc[consultas["indice_catalogo"].ge(0)].copy()

    referencia = catalogo.loc[
        consultas["indice_catalogo"], COLUNAS_CATALOGO_REFERENCIA
    ].reset_index(drop=True)
    consultas = consultas.reset_index(drop=True)
    consultas["codigo_item_catmat"] = referencia["codigo_item"].astype("int64")
    consultas["descricao_item_catmat"] = referencia["descricao_item"]
    for coluna in [
        "grupo_taxonomico",
        "subgrupo_taxonomico",
        "grau_processamento",
        "aplicabilidade_pnae",
        "sociobiodiversidade",
        "versao_taxonomia",
        "data_referencia_catmat",
    ]:
        consultas[coluna] = referencia[coluna]

    consultas["similaridade"] = consultas["similaridade"].round(6)
    # O limite produz candidatos para estudo, não uma confirmação nutricional.
    consultas["alimento_candidato"] = consultas["similaridade"].ge(limite)
    consultas["metodo_classificacao"] = "deteccao de PDM e embedding"
    consultas["modelo_embedding"] = nome_modelo
    consultas["limite_tokens_modelo"] = LIMITE_TOKENS_MODELO
    consultas["limite_operacional"] = limite
    consultas["janela_referencia_pncp"] = janela
    consultas = consultas.drop(columns="indice_catalogo")

    resultado = candidatos.merge(consultas, on=chaves, how="inner", validate="many_to_one")
    return resultado


def executar_classificacao(
    caminho_itens: Path,
    caminho_alimentos: Path,
    caminho_diagnostico: Path,
    caminho_cache_consultas: Path,
    janela: str,
    caminho_catalogo_consolidado: Path = CAMINHO_CATALOGO_CONSOLIDADO,
    caminho_classe_alimentos: Path = CAMINHO_CLASSE_ALIMENTOS,
    caminho_cache: Path = CAMINHO_CACHE,
    nome_modelo: str = MODELO_PADRAO,
    limite: float = LIMITE_PADRAO,
    batch_size: int = 128,
) -> dict[str, object]:
    """Executa o fluxo completo e devolve dados e contagens para o notebook."""

    # A ordem explicita as duas etapas do método: filtro textual por PDM e,
    # somente depois, comparação semântica dentro desse PDM.
    palavras_vazias = carregar_stopwords()
    catalogo = preparar_catalogo(caminho_catalogo_consolidado, caminho_classe_alimentos)
    itens = carregar_itens(caminho_itens)
    materiais = preparar_materiais(itens, palavras_vazias)
    candidatos = detectar_pdms(materiais, catalogo, palavras_vazias)

    modelo = carregar_modelo(nome_modelo)
    embeddings_catalogo = obter_embeddings_catalogo(
        catalogo,
        modelo,
        nome_modelo,
        caminho_classe_alimentos,
        caminho_cache,
        batch_size,
    )
    diagnostico = encontrar_itens_similares(
        candidatos,
        catalogo,
        embeddings_catalogo,
        modelo,
        nome_modelo,
        limite,
        batch_size,
        caminho_cache_consultas,
        janela,
    )

    colunas_originais = itens.columns.tolist()
    colunas_saida = colunas_originais + COLUNAS_RESULTADO
    # O diagnóstico preserva todos os pareamentos. A base de alimentos guarda
    # somente os registros que alcançaram o limite operacional.
    diagnostico = diagnostico[colunas_saida].copy()
    alimentos = diagnostico.loc[diagnostico["alimento_candidato"]].copy()

    diagnostico = padronizar_textos_exportados(diagnostico)
    alimentos = padronizar_textos_exportados(alimentos)

    caminho_diagnostico.parent.mkdir(parents=True, exist_ok=True)
    diagnostico.to_csv(
        caminho_diagnostico,
        index=False,
        encoding="utf-8",
        compression="gzip",
    )
    alimentos.to_csv(caminho_alimentos, index=False, encoding="utf-8")

    resumo = {
        "itens": itens,
        "materiais": materiais,
        "catalogo": catalogo,
        "diagnostico": diagnostico,
        "alimentos": alimentos,
        "contagens": {
            "itens_recebidos": len(itens),
            "materiais": len(materiais),
            "servicos_excluidos": int((~itens["materialOuServico"].eq("M")).sum()),
            "candidatos_com_pdm": len(candidatos),
            "alimentos_candidatos": len(alimentos),
            "descricoes_com_caractere_irrecuperavel": int(
                materiais["problema_codificacao"].eq("caractere irrecuperavel").sum()
            ),
        },
    }
    return resumo


# TABELAS DE LEITURA E ARQUIVOS DE SAIDA -------------------------------------

def criar_tabelas_resumo(
    itens: pd.DataFrame,
    catalogo: pd.DataFrame,
    diagnostico: pd.DataFrame,
    alimentos: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Organiza as tabelas de leitura usadas no notebook e no Excel."""
    materiais = itens.loc[itens["materialOuServico"].eq("M")]
    quantidade_sem_pdm = len(materiais) - len(diagnostico)
    quantidade_rejeitada = len(diagnostico) - len(alimentos)

    fontes = pd.DataFrame(
        [
            ["Itens PNCP", "Janela de atualização de 26/04/2026 a 30/04/2026"],
            ["CATMAT", "Base da API do CATMAT coletada em 21/08/2026"],
            ["Taxonomia", "Versão preliminar v0.1 do grupo 89"],
            ["Modelo", MODELO_PADRAO],
            ["Limite do texto no modelo", "32 tokens, com a descrição original preservada"],
            ["Limite operacional", "Similaridade mínima de 0,5"],
            [
                "Interpretação",
                "O resultado identifica candidatos e não confirma adequação ao PNAE",
            ],
        ],
        columns=["Fonte ou parâmetro", "Descrição"],
    )

    resumo_lote = pd.DataFrame(
        [
            ["Itens recebidos", len(itens)],
            ["Colunas da entrada", len(itens.columns)],
            ["Endpoints únicos", itens["endpoint"].nunique()],
            ["Descrições únicas", itens["descricao"].nunique(dropna=True)],
            ["Materiais", len(materiais)],
            ["Serviços excluídos", int((~itens["materialOuServico"].eq("M")).sum())],
            ["Itens no catálogo de referência", len(catalogo)],
            ["PDMs no catálogo de referência", catalogo["codigo_pdm"].nunique()],
            ["Materiais com PDM detectado", len(diagnostico)],
            ["Alimentos candidatos", len(alimentos)],
        ],
        columns=["Indicador", "Quantidade"],
    )

    etapas = pd.DataFrame(
        [
            [1, "Entrada", len(itens), "Todos os itens recebidos"],
            [2, "Materiais", len(materiais), "Serviços não seguem para o classificador"],
            [
                3,
                "PDM único detectado",
                len(diagnostico),
                "Pré-filtro textual por nome do PDM",
            ],
            [
                4,
                "Similaridade igual ou superior a 0,5",
                len(alimentos),
                "Resultado tratado como alimento candidato",
            ],
            [
                0,
                "Sem PDM único",
                quantidade_sem_pdm,
                "Material sem associação textual única",
            ],
            [
                0,
                "Abaixo do limite 0,5",
                quantidade_rejeitada,
                "Pareamento preservado somente no diagnóstico",
            ],
        ],
        columns=["Etapa", "Resultado", "Quantidade", "Interpretação"],
    )

    faixas = pd.DataFrame(
        [
            [limite, int(diagnostico["similaridade"].ge(limite).sum())]
            for limite in [0.4, 0.5, 0.6, 0.7]
        ],
        columns=["Limite de similaridade", "Quantidade de candidatos"],
    )
    faixas["Percentual dos pareamentos"] = (
        faixas["Quantidade de candidatos"].div(len(diagnostico)).mul(100).round(2)
    )

    grupos = (
        alimentos.groupby(["grupo_taxonomico", "subgrupo_taxonomico"], dropna=False)
        .agg(
            quantidade_itens=("numeroItem", "size"),
            descricoes_unicas=("descricao", "nunique"),
            similaridade_media=("similaridade", "mean"),
        )
        .reset_index()
        .sort_values(["quantidade_itens", "grupo_taxonomico"], ascending=[False, True])
    )
    grupos["similaridade_media"] = grupos["similaridade_media"].round(4)
    grupos.columns = [
        "Grupo taxonômico",
        "Subgrupo taxonômico",
        "Quantidade de itens",
        "Descrições únicas",
        "Similaridade média",
    ]

    colunas_exemplos = [
        "endpoint",
        "numeroItem",
        "descricao",
        "nome_pdm_detectado",
        "descricao_item_catmat",
        "grupo_taxonomico",
        "subgrupo_taxonomico",
        "similaridade",
        "alimento_candidato",
    ]
    partes_exemplos: list[pd.DataFrame] = []
    selecionados: set[tuple[str, object]] = set()
    # Os exemplos são escolhidos por critérios fixos, sem avaliação manual ou
    # sorteio, para tornar a planilha reproduzível.
    estratos = [
        ("Maiores similaridades", diagnostico.sort_values("similaridade", ascending=False)),
        (
            "Próximos ao limite",
            diagnostico.assign(
                distancia_limite=(diagnostico["similaridade"] - LIMITE_PADRAO).abs()
            ).sort_values("distancia_limite"),
        ),
        (
            "Maiores abaixo do limite",
            diagnostico.loc[diagnostico["similaridade"].lt(LIMITE_PADRAO)].sort_values(
                "similaridade", ascending=False
            ),
        ),
    ]
    for nome_estrato, dados_estrato in estratos:
        linhas = []
        for indice, linha in dados_estrato.iterrows():
            chave = (str(linha["endpoint"]), linha["numeroItem"])
            if chave in selecionados:
                continue
            selecionados.add(chave)
            linhas.append(indice)
            if len(linhas) == 20:
                break
        parte = diagnostico.loc[linhas, colunas_exemplos].copy()
        parte.insert(0, "Estrato", nome_estrato)
        partes_exemplos.append(parte)
    exemplos = pd.concat(partes_exemplos, ignore_index=True)
    exemplos.columns = [
        "Estrato",
        "Endpoint",
        "Número do item",
        "Descrição PNCP",
        "PDM detectado",
        "Descrição CATMAT associada",
        "Grupo taxonômico",
        "Subgrupo taxonômico",
        "Similaridade",
        "Alimento candidato",
    ]

    descricao_texto = itens["descricao"].fillna("").astype(str)
    codigos_catalogo = pd.to_numeric(itens["catalogoCodigoItem"], errors="coerce")
    codigos_referencia = set(catalogo["codigo_item"].astype(int))
    problemas = pd.DataFrame(
        [
            ["Descrição ausente", int(itens["descricao"].isna().sum()), "Não permite classificação textual"],
            [
                "Caractere irrecuperável",
                int(descricao_texto.str.contains("�", regex=False).sum()),
                "O texto original é preservado e a ocorrência é registrada",
            ],
            [
                "Codificação recuperável",
                int(materiais["descricao"].map(corrigir_codificacao).str[1].eq("acentos corrigidos").sum()),
                "A correção é aplicada somente à cópia usada na classificação",
            ],
            [
                "Código de catálogo preenchido",
                int(codigos_catalogo.notna().sum()),
                "O campo possui baixa cobertura neste lote",
            ],
            [
                "Código presente na referência alimentar",
                int(codigos_catalogo.isin(codigos_referencia).sum()),
                "Nenhum vínculo direto foi encontrado neste lote",
            ],
        ],
        columns=["Problema ou verificação", "Quantidade", "Tratamento"],
    )

    definicoes = {
        "descricao_corrigida": "Cópia da descrição usada para reparar problemas recuperáveis de codificação.",
        "descricao_normalizada": "Texto sem acentos e stopwords usado na detecção do PDM.",
        "problema_codificacao": "Situação encontrada na preparação do texto.",
        "codigo_pdm_detectado": "Código do único PDM identificado no pré-filtro textual.",
        "nome_pdm_detectado": "Nome do PDM identificado.",
        "evidencia_pdm": "Termos que participaram da detecção do PDM.",
        "codigo_item_catmat": "Código do item CATMAT com maior similaridade dentro do PDM.",
        "descricao_item_catmat": "Descrição do item CATMAT associado.",
        "similaridade": "Similaridade do cosseno entre os embeddings, de -1 a 1.",
        "alimento_candidato": "Indica similaridade igual ou superior ao limite operacional.",
        "grupo_taxonomico": "Grupo alimentar herdado do item CATMAT associado.",
        "subgrupo_taxonomico": "Subgrupo alimentar herdado do item CATMAT associado.",
        "grau_processamento": "Grau de processamento registrado na taxonomia v0.1.",
        "aplicabilidade_pnae": "Aplicabilidade registrada no catálogo de referência.",
        "sociobiodiversidade": "Correspondência textual registrada no catálogo de referência.",
        "metodo_classificacao": "Etapas usadas para produzir o pareamento.",
        "modelo_embedding": "Modelo usado para transformar textos em vetores.",
        "limite_tokens_modelo": "Quantidade máxima de tokens usada pelo modelo de embeddings.",
        "limite_operacional": "Valor mínimo adotado para formar a base de candidatos.",
        "versao_taxonomia": "Versão da taxonomia usada no processamento.",
        "data_referencia_catmat": "Data da coleta da base da API do CATMAT.",
        "janela_referencia_pncp": "Período de atualização das contratações do lote PNCP.",
    }
    dicionario = pd.DataFrame(
        [(campo, definicoes[campo]) for campo in COLUNAS_RESULTADO],
        columns=["Campo", "Definição"],
    )

    return {
        "Fonte e metodologia": fontes,
        "Resumo do lote": resumo_lote,
        "Etapas do filtro": etapas,
        "Faixas de similaridade": faixas,
        "Grupos e subgrupos": grupos,
        "Exemplos automáticos": exemplos,
        "Problemas de qualidade": problemas,
        "Dicionário": dicionario,
    }


def gerar_planilha_resumo(
    itens: pd.DataFrame,
    catalogo: pd.DataFrame,
    diagnostico: pd.DataFrame,
    alimentos: pd.DataFrame,
    caminho_saida: Path,
) -> dict[str, pd.DataFrame]:
    """Gera a planilha de leitura com filtros, cabeçalhos e larguras estáveis."""
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    tabelas = criar_tabelas_resumo(itens, catalogo, diagnostico, alimentos)
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    # O Excel também é montado em arquivo temporário para evitar saída incompleta.
    caminho_temporario = caminho_saida.with_suffix(".tmp.xlsx")
    with pd.ExcelWriter(caminho_temporario, engine="openpyxl") as escritor:
        for nome_aba, tabela in tabelas.items():
            tabela.to_excel(escritor, sheet_name=nome_aba, index=False)

    workbook = load_workbook(caminho_temporario)
    preenchimento = PatternFill("solid", fgColor="1F4E78")
    for planilha in workbook.worksheets:
        planilha.freeze_panes = "A2"
        planilha.auto_filter.ref = planilha.dimensions
        for celula in planilha[1]:
            celula.fill = preenchimento
            celula.font = Font(color="FFFFFF", bold=True)
            celula.alignment = Alignment(vertical="center", wrap_text=True)
        for coluna in planilha.columns:
            valores = [str(celula.value or "") for celula in coluna[:200]]
            largura = min(max(max(map(len, valores)) + 2, 12), 55)
            planilha.column_dimensions[get_column_letter(coluna[0].column)].width = largura
        for linha in planilha.iter_rows(min_row=2):
            for celula in linha:
                celula.alignment = Alignment(vertical="top", wrap_text=True)
    workbook.save(caminho_temporario)
    caminho_temporario.replace(caminho_saida)
    return tabelas


# EXECUÇÃO POR LINHA DE COMANDO ----------------------------------------------

def criar_parser() -> argparse.ArgumentParser:
    """Define os argumentos da execução por linha de comando."""
    parser = argparse.ArgumentParser(
        description="Classifica itens PNCP como candidatos a alimentos."
    )
    parser.add_argument("--itens", type=Path, required=True, help="CSV de itens do lote PNCP")
    parser.add_argument(
        "--sufixo",
        required=True,
        help="Identifica o lote nos nomes de saída e no cache (ex.: 2025-01_a_2025-01)",
    )
    parser.add_argument(
        "--janela", help="Período gravado em janela_referencia_pncp (padrão: o sufixo)"
    )
    parser.add_argument("--saida", type=Path, default=PASTA_OUTPUT, help="Pasta de saída")
    parser.add_argument(
        "--catalogo-consolidado", type=Path, default=CAMINHO_CATALOGO_CONSOLIDADO
    )
    parser.add_argument("--modelo", default=MODELO_PADRAO)
    parser.add_argument("--limite", type=float, default=LIMITE_PADRAO)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


def main() -> None:
    """Executa o classificador sobre o lote informado."""
    argumentos = criar_parser().parse_args()
    caminhos = caminhos_lote(argumentos.sufixo, argumentos.saida)
    resultado = executar_classificacao(
        caminho_itens=argumentos.itens,
        caminho_alimentos=caminhos["alimentos"],
        caminho_diagnostico=caminhos["diagnostico"],
        caminho_cache_consultas=caminhos["cache_consultas"],
        janela=argumentos.janela or argumentos.sufixo,
        caminho_catalogo_consolidado=argumentos.catalogo_consolidado,
        nome_modelo=argumentos.modelo,
        limite=argumentos.limite,
        batch_size=argumentos.batch_size,
    )
    gerar_planilha_resumo(
        resultado["itens"],
        resultado["catalogo"],
        resultado["diagnostico"],
        resultado["alimentos"],
        caminhos["resumo_excel"],
    )
    print(json.dumps(resultado["contagens"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
