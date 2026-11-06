"""Testes das funções do pipeline que não dependem dos dados do PNCP."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import src.main as pipeline
from src.main import meses_do_periodo, remover_versoes_antigas, sufixo_periodo


def itens(*chaves: tuple[str, str]) -> pd.DataFrame:
    """Tabela mínima de itens com as colunas da chave."""
    return pd.DataFrame(chaves, columns=["endpoint", "numeroItem"])


class TestMesesDoPeriodo(unittest.TestCase):
    def test_atravessa_a_virada_do_ano(self) -> None:
        self.assertEqual(
            meses_do_periodo((2025, 11), (2026, 2)),
            [(2025, 11), (2025, 12), (2026, 1), (2026, 2)],
        )

    def test_um_unico_mes(self) -> None:
        self.assertEqual(meses_do_periodo((2025, 1), (2025, 1)), [(2025, 1)])

    def test_sufixo(self) -> None:
        self.assertEqual(sufixo_periodo((2025, 1), (2025, 6)), "2025-01_a_2025-06")


class TestRemoverVersoesAntigas(unittest.TestCase):
    def test_mantem_a_versao_do_mes_mais_recente(self) -> None:
        janeiro = itens(("a", "1"), ("b", "1"))
        fevereiro = itens(("a", "1"))
        resultado = remover_versoes_antigas(
            [janeiro, fevereiro], [{"a|1", "b|1"}, {"a|1"}]
        )
        self.assertEqual(resultado[0]["endpoint"].tolist(), ["b"])
        self.assertEqual(resultado[1]["endpoint"].tolist(), ["a"])

    def test_descarta_candidato_que_deixou_de_ser_candidato(self) -> None:
        # O item "a|1" foi candidato em janeiro, mas reapareceu na base de
        # fevereiro sem ser candidato: a versão de janeiro não vale mais.
        janeiro = itens(("a", "1"))
        fevereiro = itens()
        resultado = remover_versoes_antigas([janeiro, fevereiro], [{"a|1"}, {"a|1"}])
        self.assertTrue(resultado[0].empty)


class TestEtapasDoMes(unittest.TestCase):
    """Quais scripts rodam em cada situação dos marcadores, sem rodá-los de fato."""

    def setUp(self) -> None:
        pasta = tempfile.TemporaryDirectory()
        self.addCleanup(pasta.cleanup)
        self.progresso = Path(pasta.name)
        self.rodados: list[str] = []
        for substituto in [
            mock.patch.object(pipeline, "PASTA_PROGRESSO", self.progresso),
            mock.patch.object(pipeline.subprocess, "run", self.registrar),
            mock.patch("builtins.print"),
        ]:
            substituto.start()
            self.addCleanup(substituto.stop)

    def registrar(self, comando: list[str], check: bool) -> None:
        self.rodados.append(Path(comando[1]).stem)

    def rodar(self, refazer: bool = False) -> list[str]:
        self.rodados.clear()
        pipeline.processar_mes((2025, 1), Path("origem"), refazer)
        return list(self.rodados)

    def apagar(self, etapa: str) -> None:
        pipeline.marcador_etapa("2025-01_a_2025-01", etapa).unlink()

    def test_ordem_das_etapas(self) -> None:
        self.assertEqual(
            self.rodar(),
            [
                "database_build",
                "itens_build",
                "classifica_alimentos",
                "contexto_escolar_alimentos",
                "classifica_alimentos_servicos",
            ],
        )
        self.assertEqual(self.rodar(), [])

    def test_contexto_refeito_sozinho_sem_embeddings(self) -> None:
        self.rodar()
        self.apagar("contexto_alimentos")
        self.assertEqual(self.rodar(), ["contexto_escolar_alimentos"])

    def test_contexto_refeito_quando_os_alimentos_sao_refeitos(self) -> None:
        self.rodar()
        self.apagar("alimentos")
        self.assertEqual(
            self.rodar(), ["classifica_alimentos", "contexto_escolar_alimentos"]
        )


if __name__ == "__main__":
    unittest.main()
