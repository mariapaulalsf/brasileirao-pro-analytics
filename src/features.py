"""Features pré-jogo para o Brasileirão Pro Analytics.

Princípio central: toda feature de uma partida é calculada SOMENTE com jogos
que aconteceram antes dela. Assim, o modelo nunca "vê" o resultado que está
tentando prever (sem vazamento de informação / data leakage).

Features (todas relativas: mandante vs. visitante):
- elo_diff:  diferença de rating Elo (com vantagem de mando incluída)
- delta_gf:  média de gols marcados nos últimos N jogos (mandante − visitante)
- delta_ga:  média de gols sofridos nos últimos N jogos (mandante − visitante)
- delta_ppg: pontos por jogo nos últimos N jogos (mandante − visitante)
- xg_casa:   ataque do mandante vs. defesa do visitante (gols esperados, proxy)
- xg_fora:   ataque do visitante vs. defesa do mandante (gols esperados, proxy)
"""
import os
import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd

MAPEAMENTO = {
    'Athletico–PR': 'Athletico-PR', 'Athletico PR': 'Athletico-PR',
    'Atlético Mineiro': 'Atlético-MG', 'Atletico MG': 'Atlético-MG',
    'RB Bragantino': 'Bragantino', 'Botafogo–RJ': 'Botafogo', 'Botafogo RJ': 'Botafogo',
    'Vasco da Gama': 'Vasco', 'Sao Paulo': 'São Paulo',
}

FEATURES_MODELO = ['elo_diff', 'delta_gf', 'delta_ga', 'delta_ppg', 'xg_casa', 'xg_fora']

# Hiperparâmetros do cálculo de features (fixos, não ajustados no conjunto de teste)
JANELA = 5            # últimos N jogos de cada time
PESO_PRIOR = 2        # "jogos fictícios" com a média da liga (estabiliza times sem histórico)
ELO_INICIAL = 1500.0
ELO_K = 20.0
ELO_MANDO = 60.0      # vantagem de jogar em casa, em pontos de Elo
ELO_REGRESSAO = 0.25  # na virada de temporada, aproxima o rating da média


def carregar_partidas(data_dir, temporadas=(2025, 2026)):
    """Lê partidas_<ano>.csv, padroniza nomes, extrai gols e ordena no tempo.

    Partidas ainda não disputadas ficam com gols NaN (servem para previsão).
    """
    quadros = []
    for ano in temporadas:
        df = pd.read_csv(os.path.join(data_dir, f'partidas_{ano}.csv'))
        df = df.dropna(subset=['Home', 'Away', 'Date']).copy()  # remove linhas separadoras
        df['Home'] = df['Home'].replace(MAPEAMENTO)
        df['Away'] = df['Away'].replace(MAPEAMENTO)
        gols = df['Score'].astype(str).str.extract(r'(\d+)\D+(\d+)')
        df['gols_mandante'] = gols[0].astype(float)
        df['gols_visitante'] = gols[1].astype(float)
        df['temporada'] = ano
        quadros.append(df)
    partidas = pd.concat(quadros, ignore_index=True)
    partidas['_quando'] = pd.to_datetime(partidas['Date'] + ' ' + partidas['Time'].fillna('00:00'), errors='coerce')
    return partidas.sort_values(['_quando', 'Home']).reset_index(drop=True)


def _media_com_prior(valores, prior):
    return (sum(valores) + prior * PESO_PRIOR) / (len(valores) + PESO_PRIOR)


def construir_features(partidas):
    """Devolve `partidas` com as colunas de FEATURES_MODELO (sem vazamento).

    Percorre os jogos em ordem cronológica; para cada um, calcula as features
    com o estado atual dos times e só depois (se o jogo já foi disputado)
    atualiza esse estado com o resultado.
    """
    elo = defaultdict(lambda: ELO_INICIAL)
    historico = defaultdict(lambda: deque(maxlen=JANELA))  # (gols_pro, gols_contra, pontos)
    gols_liga, jogos_liga = 0.0, 0
    temporada_atual = None
    linhas = []

    for _, p in partidas.iterrows():
        if p['temporada'] != temporada_atual:
            if temporada_atual is not None:
                for t in list(elo):
                    elo[t] = ELO_INICIAL + (1 - ELO_REGRESSAO) * (elo[t] - ELO_INICIAL)
            temporada_atual = p['temporada']

        casa, fora = p['Home'], p['Away']
        prior = (gols_liga / (2 * jogos_liga)) if jogos_liga else 1.2  # gols por time por jogo
        hc, hf = historico[casa], historico[fora]
        gf_c = _media_com_prior([x[0] for x in hc], prior)
        ga_c = _media_com_prior([x[1] for x in hc], prior)
        gf_f = _media_com_prior([x[0] for x in hf], prior)
        ga_f = _media_com_prior([x[1] for x in hf], prior)
        ppg_c = _media_com_prior([x[2] for x in hc], 1.4)
        ppg_f = _media_com_prior([x[2] for x in hf], 1.4)

        linhas.append({
            'elo_diff': elo[casa] + ELO_MANDO - elo[fora],
            'delta_gf': gf_c - gf_f,
            'delta_ga': ga_c - ga_f,
            'delta_ppg': ppg_c - ppg_f,
            'xg_casa': (gf_c + ga_f) / 2,
            'xg_fora': (gf_f + ga_c) / 2,
        })

        # --- só agora o resultado (se existir) entra no estado dos times ---
        gm, gv = p['gols_mandante'], p['gols_visitante']
        if pd.notna(gm) and pd.notna(gv):
            esperado = 1 / (1 + 10 ** (-(elo[casa] + ELO_MANDO - elo[fora]) / 400))
            real = 1.0 if gm > gv else (0.5 if gm == gv else 0.0)
            ajuste = ELO_K * (1 + math.log(1 + abs(gm - gv))) * (real - esperado)
            elo[casa] += ajuste
            elo[fora] -= ajuste
            pts_c = 3 if gm > gv else (1 if gm == gv else 0)
            pts_f = 3 if gv > gm else (1 if gm == gv else 0)
            hc.append((gm, gv, pts_c))
            hf.append((gv, gm, pts_f))
            gols_liga += gm + gv
            jogos_liga += 1

    return pd.concat([partidas.reset_index(drop=True), pd.DataFrame(linhas)], axis=1)
