"""Avalia o modelo com validação temporal e compara com baselines.

Uso:  python src/avaliar_modelo.py

Protocolo (sem vazamento):
  - Features calculadas só com jogos anteriores a cada partida (src/features.py).
  - Treino: temporada 2025. Teste: jogos de 2026 já realizados.
  - Hiperparâmetro (alpha) escolhido por validação cruzada temporal (TimeSeriesSplit) DENTRO do treino.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURES_MODELO, carregar_partidas, construir_features  # noqa: E402
from modelo import ALVOS, criar_modelo, gols_esperados, probabilidades_1x2, resultado_1x2  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
ROTULOS = ['Mandante', 'Empate', 'Visitante']


def matriz_probs(lam):
    return np.array([probabilidades_1x2(a, b) for a, b in lam])


def main():
    partidas = construir_features(carregar_partidas(DATA_DIR))
    jogos = partidas.dropna(subset=ALVOS).reset_index(drop=True)
    jogos['res'] = resultado_1x2(jogos['gols_mandante'], jogos['gols_visitante'])
    treino = jogos[jogos['temporada'] == 2025].reset_index(drop=True)
    teste = jogos[jogos['temporada'] == 2026].reset_index(drop=True)
    print(f'Treino: {len(treino)} jogos (2025) | Teste: {len(teste)} jogos (2026 já realizados)\n')

    prior = np.array([(treino['res'] == k).mean() for k in range(3)])
    print('Escolha de alpha por validação cruzada temporal (log loss, menor é melhor):')
    melhor_alpha, melhor_ll = None, np.inf
    for alpha in (0.1, 0.3, 1.0, 3.0, 10.0):
        lls = []
        for a, b in TimeSeriesSplit(5).split(treino):
            m = criar_modelo(alpha).fit(treino.loc[a, FEATURES_MODELO], treino.loc[a, ALVOS])
            p = matriz_probs(gols_esperados(m, treino.loc[b]))
            lls.append(log_loss(treino.loc[b, 'res'], p, labels=[0, 1, 2]))
        print(f'  alpha={alpha:<4} log loss = {np.mean(lls):.4f}')
        if np.mean(lls) < melhor_ll:
            melhor_alpha, melhor_ll = alpha, np.mean(lls)
    print(f'  -> alpha escolhido: {melhor_alpha}\n')

    modelo = criar_modelo(melhor_alpha).fit(treino[FEATURES_MODELO], treino[ALVOS])
    lam = gols_esperados(modelo, teste)
    probs = matriz_probs(lam)
    pred = probs.argmax(axis=1)
    real = teste['res'].values

    ll_modelo = log_loss(real, probs, labels=[0, 1, 2])
    ll_prior = log_loss(real, np.tile(prior, (len(teste), 1)), labels=[0, 1, 2])
    mae_m = np.abs(lam - teste[ALVOS].values).mean(axis=0)
    mae_b = np.abs(treino[ALVOS].mean().values - teste[ALVOS].values).mean(axis=0)

    print('RESULTADOS NO TESTE (2026)')
    print(f'  Acurácia 1X2        modelo: {(pred == real).mean() * 100:.1f}%  | sempre mandante: {(real == 0).mean() * 100:.1f}%')
    print(f'  Log loss 1X2        modelo: {ll_modelo:.4f} | frequências do treino: {ll_prior:.4f}')
    print(f'  MAE gols mandante   modelo: {mae_m[0]:.3f} | média do treino: {mae_b[0]:.3f}')
    print(f'  MAE gols visitante  modelo: {mae_m[1]:.3f} | média do treino: {mae_b[1]:.3f}')
    print(f'  Previsões por classe: ' + ', '.join(f'{r}={int((pred == k).sum())}' for k, r in enumerate(ROTULOS)))
    print('\n  Matriz de confusão (linhas = real, colunas = previsto):')
    cm = pd.DataFrame(confusion_matrix(real, pred, labels=[0, 1, 2]), index=ROTULOS, columns=ROTULOS)
    print(cm.to_string())


if __name__ == '__main__':
    main()
