# ⚽ Brasileirão Pro Analytics

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Poisson%20Regression-orange)
![Pandas](https://img.shields.io/badge/Pandas-data-150458)

Aplicação em Python que **prevê placares e probabilidades de resultado (vitória / empate / derrota) de partidas do Campeonato Brasileiro Série A** usando apenas informação disponível *antes* de cada jogo, e monta uma **classificação simulada** com essas previsões. A interface é um dashboard em Streamlit.

> Projeto acadêmico em evolução. A seção [Resultados e limitações](#-resultados-e-limitações) descreve com transparência o que o modelo entrega hoje e o que ainda não faz.

---

## 🎯 O que o sistema faz

- **Previsão por rodada:** placar mais provável, **probabilidades de 1X2** (mandante / empate / visitante) e, quando o jogo já aconteceu, o resultado real para comparação.
- **Indicadores do modelo por partida:** diferença de rating Elo, gols esperados e forma recente de cada time.
- **Classificação simulada:** tabela construída acumulando os pontos das *previsões* do modelo até a rodada escolhida.
- **Métricas de avaliação na própria interface:** acurácia e *log loss* nas partidas de 2026 já realizadas, ao lado da baseline.

## 🧠 Como funciona

```text
partidas_AAAA.csv → ordem cronológica → features pré-jogo (Elo + forma recente)
      → Regressão de Poisson (gols do mandante e do visitante)
      → distribuição de placares → P(1X2), placar mais provável → dashboard
```

1. **Features sem vazamento (`src/features.py`).** As partidas são percorridas em ordem cronológica. Para cada jogo, as features usam **somente jogos anteriores**; só depois o resultado atualiza o estado dos times. São 6 features:
   `elo_diff` (diferença de rating Elo, incluindo vantagem de mando), `delta_gf`, `delta_ga` e `delta_ppg` (diferença de gols marcados, gols sofridos e pontos por jogo nos últimos 5 jogos) e `xg_casa` / `xg_fora` (gols esperados combinando ataque de um time com defesa do outro). Times sem histórico recebem a média da liga como valor inicial.
2. **Modelo (`src/modelo.py`).** `PoissonRegressor` (regularização L2, com padronização) dentro de um `MultiOutputRegressor`, prevendo os gols esperados de cada lado. Com os dois gols esperados, calcula-se a matriz de placares (gols independentes, distribuição de Poisson), de onde saem as **probabilidades 1X2** e o **placar mais provável dentro do resultado mais provável**.
3. **Avaliação (`src/avaliar_modelo.py`).** Treino em 2025, teste nos jogos de 2026 já realizados. O hiperparâmetro (`alpha`) é escolhido por **validação cruzada temporal** dentro do treino, nunca no conjunto de teste.
4. **Testes (`tests/test_features.py`).** Verificam automaticamente que o resultado de um jogo não altera as próprias features e que alterar o futuro não muda as features do passado.

### Por que Poisson e não Gradient Boosting?

A primeira versão usava `GradientBoostingRegressor` com estatísticas de elenco da temporada inteira, o que **vazava informação do futuro** (os acumulados de 2026 incluíam os próprios jogos que o modelo previa). Ao recalcular as features só com o passado, o Gradient Boosting ficou **pior que a baseline** na validação temporal (log loss 1,053 contra 1,042), indicando sobreajuste com apenas 380 jogos de treino. O modelo de Poisson regularizado ficou melhor (1,010) e ainda entrega probabilidades reais, no lugar de uma barra apenas ilustrativa.

## 📊 Resultados e limitações

Reproduza com `python src/avaliar_modelo.py` (treino: 380 jogos de 2025; teste: 177 jogos de 2026).

| Métrica (teste 2026) | Modelo | Baseline |
|---|---|---|
| Acurácia 1X2 | 48,0% | 49,2% (sempre vitória do mandante) |
| **Log loss 1X2** (menor é melhor) | **1,020** | 1,045 (frequências do treino) |
| Erro médio absoluto, gols do mandante | 0,883 | 0,908 (média do treino) |
| Erro médio absoluto, gols do visitante | 0,739 | 0,696 (média do treino) |

Matriz de confusão (linhas = real, colunas = previsto):

| | Mandante | Empate | Visitante |
|---|---|---|---|
| **Mandante** | 81 | 0 | 6 |
| **Empate** | 43 | 0 | 4 |
| **Visitante** | 39 | 0 | 4 |

**Leitura honesta dos resultados:**

- O modelo é **melhor que a baseline em qualidade das probabilidades** (log loss menor), mas **não supera a baseline em acurácia**: ele acerta o resultado em 48,0% dos jogos, abaixo dos 49,2% de "sempre apostar no mandante". Prever futebol com dados públicos simples é difícil, e uma acurácia próxima da baseline é um resultado comum nesse tipo de problema.
- **O modelo nunca prevê empate** como resultado mais provável (a probabilidade de empate fica entre 10% e 29% e nunca é a maior das três). Ele ainda atribui probabilidade de empate a cada jogo; só não a transforma em previsão de classe.
- O erro de gols do visitante é **pior que prever a média** (0,739 contra 0,696).
- **Amostra pequena:** 177 jogos de teste e 380 de treino geram bastante incerteza; diferenças de 1 a 2 pontos percentuais não são estatisticamente conclusivas.
- **Rodadas futuras** usam as features do último jogo disputado, então quanto mais distante a rodada, mais defasada a previsão.
- As bases de elencos (`estatisticas-*.csv`) e de jogadores presentes em `data/` **não são mais usadas pelo modelo atual**, pois eram acumulados da temporada inteira (fonte do vazamento). Seguem no repositório para uso futuro.

## 🗺️ Próximos passos

- [ ] **Atualizar a base e suportar a temporada 2027:** tornar a temporada um parâmetro (ex.: `--ano 2027` ou seletor no app) em vez de arquivos com nomes fixos, e incluir os jogos já disputados e atualizações de mercado.
- [ ] **Coleta automática de dados:** script que baixe e atualize os CSVs de partidas, respeitando os termos de uso de cada fonte (FBref, Kaggle) e com limite de requisições. Se a raspagem não for permitida, usar download manual documentado ou fontes com licença aberta.
- [ ] Prever empates melhor (ex.: modelo de Poisson bivariado ou classificador ordinal) e calibrar as probabilidades.
- [ ] Reintroduzir estatísticas de elenco e de jogadores **calculadas apenas com jogos anteriores** (ex.: posse e chutes móveis).
- [ ] Modo "análise para comissão técnica": análise do adversário e, depois, simulação de escalação com dados por jogador.
- [ ] Publicar o modelo no Hugging Face com *model card*.

## 🚀 Como executar

Pré-requisitos: Python 3.10+ e Git. **Não é necessário banco de dados**: a aplicação lê os CSVs da pasta `data/`.

```bash
git clone https://github.com/mariapaulalsf/brasileirao-pro-analytics.git

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run src/app.py         # abre em http://localhost:8501
```

Avaliação do modelo: `python src/avaliar_modelo.py` · Testes: `python tests/test_features.py`

## 📁 Estrutura

```text
├── data/
│   ├── partidas_2025.csv        # jogos de 2025 (treino)
│   ├── partidas_2026.csv        # jogos de 2026 (teste / simulação)
│   └── (demais CSVs de elencos e jogadores; não usados pelo modelo atual)
├── src/
│   ├── app.py                   # dashboard Streamlit
│   ├── features.py              # features pré-jogo (Elo, forma recente), sem vazamento
│   ├── modelo.py                # regressão de Poisson e probabilidades 1X2
│   └── avaliar_modelo.py        # validação temporal e comparação com baselines
├── tests/
│   └── test_features.py         # testes anti-vazamento
├── requirements.txt
└── README.md
```

## 🛠️ Tecnologias

Python · Streamlit · Pandas · NumPy · SciPy · scikit-learn

## 📚 Fontes dos dados

- Estatísticas de elencos e partidas: [FBref – Série A](https://fbref.com/en/comps/24/Serie-A-Stats)
- Histórico de partidas: [Kaggle – Campeonato Brasileiro de Futebol](https://www.kaggle.com/datasets/adaoduque/campeonato-brasileiro-de-futebol)

Os dados pertencem às respectivas fontes; consulte os termos de uso antes de redistribuir ou coletar de forma automatizada.

## 👩‍💻 Autora

Maria Paula Lacerda · [LinkedIn](https://www.linkedin.com/in/maria-paula-lacerda-5306812b5) · [GitHub](https://github.com/mariapaulalsf)
