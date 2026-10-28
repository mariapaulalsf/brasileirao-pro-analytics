"""Modelo de previsão de gols (Poisson) e conversão em probabilidades 1X2."""
import numpy as np
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES_MODELO

ALVOS = ['gols_mandante', 'gols_visitante']
MAX_GOLS = 10
ALPHA = 0.3  # regularização L2, escolhida por validação cruzada temporal (ver avaliar_modelo.py)


def criar_modelo(alpha=ALPHA):
    return MultiOutputRegressor(
        make_pipeline(StandardScaler(), PoissonRegressor(alpha=alpha, max_iter=500))
    )


def treinar(df_treino, alpha=ALPHA):
    modelo = criar_modelo(alpha)
    modelo.fit(df_treino[FEATURES_MODELO], df_treino[ALVOS])
    return modelo


def gols_esperados(modelo, df):
    """Gols esperados (mandante, visitante) para cada partida de `df`."""
    return np.clip(modelo.predict(df[FEATURES_MODELO]), 0.05, None)


def matriz_placares(lambda_casa, lambda_fora):
    """P(placar i x j) assumindo gols independentes com distribuição de Poisson."""
    g = np.arange(MAX_GOLS)
    m = np.outer(poisson.pmf(g, lambda_casa), poisson.pmf(g, lambda_fora))
    return m / m.sum()


def probabilidades_1x2(lambda_casa, lambda_fora):
    """Retorna (P mandante vence, P empate, P visitante vence)."""
    m = matriz_placares(lambda_casa, lambda_fora)
    return float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum())


def placar_mais_provavel(lambda_casa, lambda_fora):
    """Placar mais provável DENTRO do resultado mais provável (coerente com o 1X2)."""
    m = matriz_placares(lambda_casa, lambda_fora)
    p_casa, p_emp, p_fora = probabilidades_1x2(lambda_casa, lambda_fora)
    i, j = np.indices(m.shape)
    mascara = [i > j, i == j, i < j][int(np.argmax([p_casa, p_emp, p_fora]))]
    alvo = np.where(mascara, m, -1)
    gm, gv = np.unravel_index(np.argmax(alvo), alvo.shape)
    return int(gm), int(gv)


def resultado_1x2(gols_mandante, gols_visitante):
    """0 = mandante, 1 = empate, 2 = visitante."""
    return np.where(gols_mandante > gols_visitante, 0, np.where(gols_mandante == gols_visitante, 1, 2))
