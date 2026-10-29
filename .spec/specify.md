# Specify: Parametrização das Temporadas

## 1. O Problema
Atualmente, as temporadas de treino (2025) e teste (2026) estão fixas (*hardcoded*) como strings de texto dentro dos arquivos principais do projeto. Isso impede que o sistema seja escalado para a temporada de 2027 sem exigir alterações manuais no código-fonte em múltiplos lugares.

## 2. O Objetivo
A aplicação deve ser capaz de receber o ano de treino e o ano de teste/simulação como parâmetros dinâmicos. Essa alteração deve garantir que o sistema carregue os arquivos CSV corretos (ex: `partidas_2027.csv`) apenas alterando variáveis de configuração ou argumentos de linha de comando.

## 3. Critérios de Aceite (Success Criteria)
- Os testes em `tests/test_features.py` devem continuar passando sem falhas.
- Executar o script de avaliação com os parâmetros de 2025 e 2026 deve produzir exatamente os mesmos resultados anteriores: Acurácia de 48,0% e Log Loss de 1,020.
- O painel Streamlit deve renderizar normalmente com os mesmos dados de antes, sem quebrar o layout.
- Nenhuma *string* de ano (como '2025' ou '2026') deve sobrar solta no meio da lógica de carregamento de dados.