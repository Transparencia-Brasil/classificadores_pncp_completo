"""Testes das funções do pipeline que não dependem dos dados do PNCP."""

import unittest

import pandas as pd

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


if __name__ == "__main__":
    unittest.main()
