"""Testes de que as features NÃO vazam informação do futuro.

Rodar:  python tests/test_features.py   (ou: pytest)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from features import FEATURES_MODELO, carregar_partidas, construir_features  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def test_resultado_do_jogo_nao_altera_suas_proprias_features():
    base = carregar_partidas(DATA_DIR)
    alvo = base[base['gols_mandante'].notna()].index[300]  # um jogo no meio da base
    original = construir_features(base).loc[alvo, FEATURES_MODELO].astype(float)

    alterado = base.copy()
    alterado.loc[alvo, ['gols_mandante', 'gols_visitante']] = [7.0, 0.0]  # placar absurdo
    novo = construir_features(alterado).loc[alvo, FEATURES_MODELO].astype(float)

    assert np.allclose(original.values, novo.values), 'features do jogo mudaram com o próprio resultado (vazamento!)'


def test_jogos_anteriores_nao_dependem_de_jogos_posteriores():
    base = carregar_partidas(DATA_DIR)
    alvo = base[base['gols_mandante'].notna()].index[300]
    antes = construir_features(base).loc[:alvo, FEATURES_MODELO].astype(float)

    alterado = base.copy()
    alterado.loc[alvo + 1:, ['gols_mandante', 'gols_visitante']] = [9.0, 9.0]  # reescreve o futuro
    depois = construir_features(alterado).loc[:alvo, FEATURES_MODELO].astype(float)

    assert np.allclose(antes.values, depois.values), 'features do passado mudaram ao alterar o futuro (vazamento!)'


def test_sem_valores_ausentes_nas_features():
    partidas = construir_features(carregar_partidas(DATA_DIR))
    assert not partidas[FEATURES_MODELO].isna().any().any()


if __name__ == '__main__':
    for nome, fn in list(globals().items()):
        if nome.startswith('test_'):
            fn()
            print('OK ', nome)
