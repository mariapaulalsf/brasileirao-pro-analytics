# Constitution

## Core Principles
1. **Linguagem e Versão**: Todo o código desenvolvido deve ser compatível com Python 3.10+.
2. **Integridade de Dados (Zero Data Leakage)**: Todas as features preditivas devem ser baseadas estritamente em eventos históricos anteriores à rodada que está sendo prevista. É terminantemente proibido o uso de consolidados futuros ou médias da temporada inteira.
3. **Testes Obrigatórios**: Novos métodos, lógicas de features ou alterações arquiteturais devem ser validados por meio de testes automatizados localizados no diretório `/tests`.
4. **Manutenção**: O arquivo `README.md` deve refletir de forma transparente o estado atual da arquitetura e as limitações do modelo.