"""Testes da etapa que marca o contexto escolar dos alimentos candidatos."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import src.main  # noqa: F401  (põe as pastas dos módulos no sys.path)
from classifica_alimentos import PASTA_SAIDA_MATERIAIS
from contexto_escolar_alimentos import (
    COLUNAS_CONTEXTO,
    COLUNAS_CONTRATACAO_ALIMENTOS,
    METODO_TERMO_FORTE,
    METODO_TERMO_FRACO,
    caminhos_contexto,
    carregar_contratacoes,
    executar,
    ligar_contratacoes,
    marcar_contexto_escolar,
)
from classifica_alimentos_servicos import (
    METODO_AMPARO,
    METODO_SEM_CONTEXTO_ESCOLAR,
    carregar_padroes_termos,
)

PADROES = carregar_padroes_termos()


def endpoint(sequencial: int, ano: int = 2025) -> str:
    """Endpoint de itens no formato do PNCP, com o órgão 123."""
    return f"https://pncp.gov.br/api/pncp/v1/orgaos/123/compras/{ano}/{sequencial}/itens"


def alimentos(*endpoints: str) -> pd.DataFrame:
    """Alimentos candidatos mínimos, com as colunas em volta da posição das novas."""
    return pd.DataFrame(
        {
            "endpoint": list(endpoints),
            "descricao": "Arroz beneficiado",
            "problema_codificacao": "sem problema detectado",
            "similaridade": "0.8",
        }
    )


def contratacoes(*linhas: tuple[int, str, str]) -> pd.DataFrame:
    """Contratações já preparadas: (sequencial, objeto, código do amparo)."""
    tabela = pd.DataFrame(
        [
            {
                "chave_contratacao": f"123/2025/{sequencial}",
                "objetoCompra": objeto,
                "amparoLegal.codigo": amparo,
            }
            for sequencial, objeto, amparo in linhas
        ]
    )
    for coluna in COLUNAS_CONTRATACAO_ALIMENTOS.values():
        if coluna not in tabela:
            tabela[coluna] = "x"
    return tabela[["chave_contratacao", *COLUNAS_CONTRATACAO_ALIMENTOS.values()]]


def marcar(objeto: str, amparo: str = "1") -> pd.Series:
    """Marca o contexto de um único alimento e devolve a linha resultante."""
    ligados = ligar_contratacoes(alimentos(endpoint(1)), contratacoes((1, objeto, amparo)))
    return marcar_contexto_escolar(ligados, PADROES).iloc[0]


class TestLigarContratacoes(unittest.TestCase):
    def test_acrescenta_as_colunas_depois_de_problema_codificacao(self) -> None:
        resultado = ligar_contratacoes(
            alimentos(endpoint(1)), contratacoes((1, "Compra de arroz", "1"))
        )
        novas = list(COLUNAS_CONTRATACAO_ALIMENTOS.values())
        self.assertEqual(
            resultado.columns.tolist(),
            ["endpoint", "descricao", "problema_codificacao", *novas, "similaridade"],
        )
        self.assertEqual(resultado.loc[0, "objetoCompra"], "Compra de arroz")

    def test_mantem_alimento_sem_contratacao_com_colunas_vazias(self) -> None:
        # O ano inválido (20214) não casa com nenhuma contratação.
        resultado = ligar_contratacoes(
            alimentos(endpoint(1), endpoint(1, ano=20214)),
            contratacoes((1, "Compra de arroz", "1")),
        )
        self.assertEqual(len(resultado), 2)
        self.assertEqual(resultado.loc[1, "objetoCompra"], "")
        self.assertEqual(resultado.loc[1, "amparoLegal.codigo"], "")


class TestMarcarContextoEscolar(unittest.TestCase):
    def test_amparo_pnae_basta_mesmo_com_objeto_generico(self) -> None:
        linha = marcar("Aquisição de gêneros alimentícios", amparo="138")
        self.assertTrue(linha["contexto_escolar"])
        self.assertEqual(linha["metodo_contexto_escolar"], METODO_AMPARO)

    def test_amparo_tem_prioridade_sobre_os_termos(self) -> None:
        linha = marcar("Gêneros para a merenda escolar", amparo="139")
        self.assertEqual(linha["metodo_contexto_escolar"], METODO_AMPARO)

    def test_termo_forte_no_objeto(self) -> None:
        linha = marcar("Aquisição de gêneros para a merenda escolar da rede municipal")
        self.assertTrue(linha["contexto_escolar"])
        self.assertEqual(linha["metodo_contexto_escolar"], METODO_TERMO_FORTE)

    def test_termo_fraco_sem_outro_contexto(self) -> None:
        linha = marcar("Gêneros alimentícios para a Secretaria Municipal de Educação")
        self.assertTrue(linha["contexto_escolar"])
        self.assertEqual(linha["metodo_contexto_escolar"], METODO_TERMO_FRACO)

    def test_termo_fraco_com_outro_contexto_nao_vale(self) -> None:
        linha = marcar("Alimentação nos eventos da Secretaria de Educação")
        self.assertFalse(linha["contexto_escolar"])
        self.assertEqual(linha["metodo_contexto_escolar"], METODO_SEM_CONTEXTO_ESCOLAR)
        self.assertNotEqual(linha["termos_outro_contexto_objeto"], "")

    def test_agricultura_familiar_sozinha_e_so_indicio(self) -> None:
        # Também aparece no PAA (ex.: compras do Exército pela Lei 14.628/2023).
        linha = marcar("Aquisição de gêneros da agricultura familiar para o 4º GAC")
        self.assertFalse(linha["contexto_escolar"])
        self.assertEqual(linha["termos_escolar_indicio_objeto"], "agricultura familiar")
        self.assertEqual(linha["termos_escolar_fraco_objeto"], "")

    def test_agricultura_familiar_com_termo_escolar(self) -> None:
        linha = marcar("Gêneros da agricultura familiar para a alimentação escolar")
        self.assertTrue(linha["contexto_escolar"])
        self.assertEqual(linha["metodo_contexto_escolar"], METODO_TERMO_FORTE)

    def test_sem_termo_escolar(self) -> None:
        linha = marcar("Gêneros alimentícios para a Câmara Municipal")
        self.assertFalse(linha["contexto_escolar"])
        self.assertEqual(linha["metodo_contexto_escolar"], METODO_SEM_CONTEXTO_ESCOLAR)

    def test_colunas_da_marcacao_depois_das_da_contratacao(self) -> None:
        ligados = ligar_contratacoes(alimentos(endpoint(1)), contratacoes((1, "Arroz", "1")))
        colunas = marcar_contexto_escolar(ligados, PADROES).columns.tolist()
        ultima = list(COLUNAS_CONTRATACAO_ALIMENTOS.values())[-1]
        inicio = colunas.index(ultima) + 1
        self.assertEqual(colunas[inicio:inicio + len(COLUNAS_CONTEXTO)], COLUNAS_CONTEXTO)
        self.assertEqual(colunas[-1], "similaridade")


class TestExecutar(unittest.TestCase):
    def test_divide_em_true_e_false_sem_alterar_a_entrada(self) -> None:
        with tempfile.TemporaryDirectory() as pasta:
            pasta = Path(pasta)
            entrada = pasta / "alimentos_candidatos_pncp_teste.csv"
            alimentos(endpoint(1), endpoint(2), endpoint(3)).to_csv(entrada, index=False)
            original = entrada.read_bytes()
            pd.DataFrame(
                {
                    "data.orgaoEntidade.cnpj": ["123"] * 3,
                    "data.anoCompra": ["2025"] * 3,
                    "data.sequencialCompra": ["1", "2", "3"],
                    "data.objetoCompra": [
                        "Gêneros para a merenda escolar",
                        "Gêneros para o Hospital Municipal",
                        "Gêneros da agricultura familiar",
                    ],
                    "data.amparoLegal.codigo": ["1", "1", "138"],
                    **{coluna: ["x"] * 3 for coluna in COLUNAS_CONTRATACAO_ALIMENTOS
                       if coluna not in {"data.objetoCompra", "data.amparoLegal.codigo"}},
                }
            ).to_csv(pasta / "contratacoes.csv", index=False)

            destinos = caminhos_contexto("teste", pasta)
            executar(entrada, pasta / "contratacoes.csv", destinos)

            ler = lambda caminho: pd.read_csv(caminho, dtype=str, keep_default_na=False)
            verdadeiros, falsos = ler(destinos["true"]), ler(destinos["false"])
            self.assertEqual(entrada.read_bytes(), original)
            self.assertEqual(verdadeiros["endpoint"].tolist(), [endpoint(1), endpoint(3)])
            self.assertEqual(falsos["endpoint"].tolist(), [endpoint(2)])
            self.assertEqual(set(verdadeiros["contexto_escolar"]), {"True"})
            self.assertEqual(set(falsos["contexto_escolar"]), {"False"})


class TestCaminhos(unittest.TestCase):
    def test_true_em_csv_e_false_compactado_na_pasta_de_materiais(self) -> None:
        caminhos = caminhos_contexto("2025-01_a_2025-01")
        self.assertEqual(
            caminhos["true"],
            PASTA_SAIDA_MATERIAIS / "alimentos_contexto_escolar_true_pncp_2025-01_a_2025-01.csv",
        )
        self.assertEqual(
            caminhos["false"],
            PASTA_SAIDA_MATERIAIS
            / "alimentos_contexto_escolar_false_pncp_2025-01_a_2025-01.csv.gz",
        )

    def test_carregar_contratacoes_rejeita_chave_repetida(self) -> None:
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "contratacoes.csv"
            pd.DataFrame(
                {coluna: ["1", "1"] for coluna in [
                    "data.orgaoEntidade.cnpj", "data.anoCompra", "data.sequencialCompra",
                    *COLUNAS_CONTRATACAO_ALIMENTOS,
                ]}
            ).to_csv(caminho, index=False)
            with self.assertRaises(ValueError):
                carregar_contratacoes(caminho)


if __name__ == "__main__":
    unittest.main()
