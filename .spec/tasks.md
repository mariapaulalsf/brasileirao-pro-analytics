# Tasks: Plano de Execução

- [ ] **Task 1:** Modificar as assinaturas e o corpo das funções de ingestão no `src/features.py` (como `carregar_partidas` ou `carregar_dados`) para aceitarem `ano_treino: int` e `ano_teste: int`, utilizando f-strings para resolver o caminho dos arquivos (ex: `f'partidas_{ano_treino}.csv'`).
- [ ] **Task 2:** Adicionar as constantes de configuração `ANO_TREINO = 2025` e `ANO_TESTE = 2026` no escopo global do `src/app.py`. Alterar as chamadas de inicialização de dados para passar essas constantes como parâmetros.
- [ ] **Task 3:** Atualizar `src/avaliar_modelo.py` para capturar `sys.argv[1]` (treino) e `sys.argv[2]` (teste) de forma segura, com fallback automático para 2025 e 2026 caso o script seja rodado sem argumentos.