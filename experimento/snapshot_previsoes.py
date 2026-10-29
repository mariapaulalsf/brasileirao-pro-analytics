"""Salva as previsões do modelo para todos os jogos de 2026, para comparar antes e depois de uma mudança.

Uso (na raiz do projeto):
    python experimento/snapshot_previsoes.py > experimento/antes_previsoes.txt
    ... faz a mudança ...
    python experimento/snapshot_previsoes.py > experimento/depois_previsoes.txt
    diff experimento/antes_previsoes.txt experimento/depois_previsoes.txt
Sem nenhuma linha de diferença = as previsões continuam idênticas.
"""
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, 'src'))

from features import carregar_partidas, construir_features  # noqa: E402
from modelo import ALVOS, gols_esperados, placar_mais_provavel, probabilidades_1x2, treinar  # noqa: E402

partidas = construir_features(carregar_partidas(os.path.join(RAIZ, 'data')))
treino = partidas[partidas['temporada'] == 2025].dropna(subset=ALVOS)
alvo = partidas[partidas['temporada'] == 2026].reset_index(drop=True)
modelo = treinar(treino)
lam = gols_esperados(modelo, alvo)
for (_, jogo), (lc, lf) in zip(alvo.iterrows(), lam):
    pm, pe, pv = probabilidades_1x2(lc, lf)
    gm, gv = placar_mais_provavel(lc, lf)
    print(f"{jogo['Date']} {jogo['Home']} x {jogo['Away']} | placar {gm}-{gv} | P(casa/emp/fora) {pm:.4f} {pe:.4f} {pv:.4f}")