# Plan: Arquitetura de Implementação

## 1. Estratégia Técnica
Utilizaremos injeção de dependência simples passando os anos como números inteiros (`int`) para as funções de leitura e processamento. Construiremos os caminhos dos arquivos usando *f-strings* do Python.

## 2. Modificações por Arquivo

- **`src/features.py`**:
  Refatorar as funções de carregamento (como `carregar_partidas` ou `carregar_dados`) para aceitarem `ano_treino` e `ano_teste` como argumentos. Substituir as chamadas fixas de `read_csv` para utilizarem `f'partidas_{ano_treino}.csv'`.

- **`src/app.py`**:
  Definir variáveis globais de configuração no topo do arquivo (ex: `ANO_TREINO = 2025` e `ANO_TESTE = 2026`). O dashboard passará essas constantes ao chamar as funções importadas de `features.py`.

- **`src/avaliar_modelo.py`**:
  Implementar a leitura de argumentos por linha de comando utilizando a biblioteca padrão `sys` (ex: `sys.argv[1]`). Caso nenhum argumento seja passado no terminal, o script fará um *fallback* seguro para os anos padrão (2025 e 2026).