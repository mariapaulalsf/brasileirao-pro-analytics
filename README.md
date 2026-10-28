# ⚽ Brasileirão Pro Analytics

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Gradient%20Boosting-orange)
![Pandas](https://img.shields.io/badge/Pandas-data-150458)

Aplicação em Python que **prevê placares e resultados (1X2) de partidas do Campeonato Brasileiro Série A** a partir de estatísticas coletivas dos elencos, e monta uma **classificação simulada** com essas previsões. A interface é um dashboard em Streamlit.

> Projeto acadêmico (disciplina de Inteligência Artificial Aplicada a Dados) em evolução. A seção [Resultados e limitações](#-resultados-e-limitações) descreve com transparência o que o modelo entrega hoje e o que ainda precisa melhorar.

---

## O que o sistema faz

- **Previsão por rodada:** placar previsto, barra de probabilidade (mandante / empate / visitante) e, quando o jogo já aconteceu, o resultado real ao lado para comparação.
- **Comparativo tático pré-jogo:** posse de bola, gols, desarmes certos, chutes no alvo sofridos e % de defesas do goleiro, lado a lado.
- **Classificação simulada:** tabela construída acumulando os pontos das *previsões* do modelo até a rodada escolhida.
- **Acurácia 1X2:** percentual de acerto do resultado (vitória/empate/derrota) nas partidas de 2026 já realizadas.

## Como funciona

```text
CSVs (data/)  →  limpeza e padronização de nomes  →  features por time
      →  diferença (delta) mandante − visitante  →  Gradient Boosting (2 saídas)
      →  gols previstos  →  placar, resultado 1X2 e tabela simulada (Streamlit)
```

1. **Dados:** partidas de 2025 (treino) e 2026 (teste/simulação) e estatísticas agregadas por elenco, lidas de arquivos CSV com Pandas. Nomes de clubes são padronizados (ex.: `Atlético Mineiro` → `Atlético-MG`).
2. **Features:** para cada time, `Poss` (posse), `Gls` e `Ast` (gols e assistências), `Save%` e `SoTA` (defesas e chutes no alvo sofridos pelo goleiro), `TklW` (desarmes certos) e `CrdY` (cartões amarelos). O modelo recebe a **diferença (delta)** entre mandante e visitante de cada uma, ou seja, 7 features.
3. **Modelo:** `GradientBoostingRegressor` dentro de um `MultiOutputRegressor`, prevendo ao mesmo tempo os gols do mandante e do visitante (`n_estimators=300`, `learning_rate=0.01`, `max_depth=3`, `subsample=0.8`, `random_state=42`).
4. **Do gol previsto ao resultado:** se a diferença prevista (mandante − visitante) for maior que `0.35`, o resultado é vitória do mandante; menor que `-0.35`, vitória do visitante; no intervalo entre os dois, empate. A barra de probabilidade do dashboard é uma **representação visual derivada dessa diferença**, não uma probabilidade calibrada.

## Resultados e limitações

Números obtidos executando o pipeline atual com os dados deste repositório (treino: 380 jogos de 2025; teste: 177 jogos de 2026 já realizados). Reproduza com `python src/avaliar_modelo.py`.

| Métrica (teste 2026) | Modelo | Baseline |
|---|---|---|
| Acurácia 1X2 (limiar 0,35) | **49,2%** | 49,2% (sempre prever vitória do mandante) |
| Erro médio absoluto, gols do mandante | 0,90 | 0,91 (média do treino) |
| Erro médio absoluto, gols do visitante | 0,76 | 0,70 (média do treino) |

**Leitura honesta:** hoje o modelo **empata com a baseline mais simples** (sempre apostar no mandante). Ele prevê empates e vitórias do mandante, mas quase nunca vitórias do visitante (12 de 177 previsões). Os motivos conhecidos, que são também o roteiro de melhoria:

- **Vazamento de informação (data leakage):** as estatísticas de elenco são acumulados da temporada 2026 (cerca de 18 jogos por time) e são usadas tanto para treinar com as partidas de 2025 quanto para prever as partidas de 2026 que compõem esses mesmos acumulados. A acurácia medida, portanto, não representa um uso real, em que só se conhece o que aconteceu *antes* do jogo.
- **Features estáticas:** cada time tem um único conjunto de estatísticas para todas as rodadas. Não há forma recente, média móvel de gols nem mando de campo como feature.
- **Avaliação simples:** não há validação temporal (treinar no passado, testar no futuro), métricas por classe nem matriz de confusão.
- **Limiar de empate fixo (`0.35`)** escolhido manualmente, sem ajuste em validação.
- **Dados parciais:** o arquivo de 2026 contém 177 jogos realizados; a temporada tem 380.

## Roadmap

- [ ] Reconstruir as features usando **somente informação anterior ao jogo** (médias móveis dos últimos N jogos, mando de campo) e avaliar com **split temporal**.
- [ ] Comparar com baselines e com um classificador 1X2 (probabilidades reais, `log loss`, matriz de confusão).
- [ ] Atualizar a base com os jogos já disputados e parametrizar a temporada para suportar **2027**.
- [ ] Automatizar a coleta de dados, respeitando os termos de uso das fontes.
- [ ] Modo "análise para comissão técnica": análise do adversário e simulação de escalação.
- [ ] Publicar o modelo treinado no Hugging Face com *model card*.

## Como executar

Pré-requisitos: Python 3.10+ e Git. **Não é necessário banco de dados**: a aplicação lê os CSVs da pasta `data/`.

```bash
git clone https://github.com/mariapaulalsf/Brasileir-o-Predictor-AI-2026.git
cd Brasileir-o-Predictor-AI-2026

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run src/app.py         # abre em http://localhost:8501
```

Para recalcular as métricas da seção anterior: `python src/avaliar_modelo.py`.

## Estrutura

```text
├── data/
│   ├── partidas_2025.csv                    # jogos de 2025 (treino)
│   ├── partidas_2026.csv                    # jogos de 2026 (teste/simulação)
│   ├── estatisticas-padrao-do-elenco.csv    # posse, gols, assistências
│   ├── goleiros-do-elenco.csv               # defesas e chutes sofridos
│   ├── estatisticas-diversas-do-elenco.csv  # desarmes, cartões
│   └── (demais CSVs de jogadores e histórico; ainda não usados pelo app)
├── src/
│   ├── app.py                               # dashboard Streamlit + pipeline de ML
│   └── avaliar_modelo.py                    # reproduz as métricas deste README
├── requirements.txt
└── README.md
```

## Tecnologias

Python · Streamlit · Pandas · NumPy · scikit-learn

## Fontes dos dados

- Estatísticas de elencos e partidas: [FBref – Série A](https://fbref.com/en/comps/24/Serie-A-Stats)
- Histórico de partidas: [Kaggle – Campeonato Brasileiro de Futebol](https://www.kaggle.com/datasets/adaoduque/campeonato-brasileiro-de-futebol)

Os dados pertencem às respectivas fontes; consulte os termos de uso antes de redistribuir ou coletar de forma automatizada.

## Autora

Maria Paula Lacerda · [LinkedIn](https://www.linkedin.com/in/maria-paula-lacerda-5306812b5) · [GitHub](https://github.com/mariapaulalsf)
