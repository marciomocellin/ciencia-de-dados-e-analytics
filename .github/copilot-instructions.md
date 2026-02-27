# Instrucoes para Copilot - MVP da Especializacao

## Contexto do repositorio
- Este repositorio contem scripts Python e arquivos de apoio das disciplinas da pos-graduacao em Ciencia de Dados e Analytics da PUC-Rio.
- Ha exemplos em varias areas: analise exploratoria, machine learning, deep learning, visualizacao e engenharia de dados.
- Os materiais estao distribuidos por pastas de disciplinas e pastas especificas de MVP.

## Objetivo principal
- Ajudar a criar um MVP (produto viavel minimo) para a conclusao do curso, reutilizando e adaptando materiais existentes.
- Priorizar clareza, reprodutibilidade e uma narrativa curta do problema ate o resultado.
- Antes de gerar arquivos ou codigo, alinhar e registrar o tema e o dataset escolhidos.

## Escopo esperado do MVP
- Um problema claro, uma base de dados definida e um pipeline simples, completo e executavel.
- Entregaveis minimos:
  - 1 script Python principal do MVP com células `# %%`, contendo introducao, dados, modelagem, avaliacao e conclusao.
  - 1 README explicando objetivo, como rodar e fontes de dados.
  - Artefatos gerados (modelos, figuras, tabelas) salvos em pastas organizadas.

## Pastas de referencia
- Reaproveite exemplos de:
  - mvp-analise-de-dados-e-boas-praticas/
  - mvp-machine-learning-e-analytics/
  - mvp-engenharia-de-dados/
  - analise-exploratoria-pre-processamento-de-dados/
  - machine-learning/
  - visualizacao-de-informacao/
- Use datasets locais apenas se estiverem no repositorio. Caso precise de dados externos, documente o link e instrucoes de download.

## Padrao de estrutura para o MVP
- Crie uma nova pasta MVP no nivel raiz, por exemplo: mvp-final/
- Estrutura sugerida:
  - mvp-final/
    - README.md
    - mvp.py (script principal com células `# %%`)
    - data/
      - raw/
      - processed/
    - models/
    - figures/
    - src/

## Regras para o Copilot ao gerar codigo
- Use scripts Python com células `# %%` como formato principal, isolando funcoes reutilizaveis em scripts dentro de src/ quando fizer sentido.
- Evite dependencias desnecessarias; use apenas bibliotecas comuns do ecossistema Python.
- Evite caminhos absolutos. Use caminhos relativos baseados na pasta do MVP.
- Garanta que as celulas (secoes entre `# %%`) executem do inicio ao fim, sem estados ocultos.
- Inclua pequenas explicacoes com comentarios apenas quando necessario para entendimento.

## Regras de documentacao
- Sempre explicar:
  - Qual problema esta sendo resolvido.
  - Qual dataset foi usado e sua fonte.
  - Como executar o script Python.
  - Quais metricas foram usadas para avaliar o resultado.
- Use um tom objetivo e conciso.

## Regras para dados
- Se os dados nao estiverem no repositorio, nao inclua arquivos grandes.
- Forneca instrucoes para baixar e colocar em data/raw/.
- Sempre salvar resultados intermediarios em data/processed/.

## Qualidade minima
- Inclua validacoes simples (tamanho do dataset, valores ausentes, distribuicoes basicas).
- Use baseline simples e, se possivel, uma pequena melhoria.
- Evite overfitting com split treino/teste.

## O que evitar
- Nao criar MVPs muito amplos ou com muitas etapas.
- Nao copiar notebooks inteiros sem adaptacao ao objetivo final.
- Nao usar dados sem referencia ou licenca clara.

## Saidas esperadas do Copilot
- Plano curto do MVP antes de gerar codigo.
- Confirmacao do tema e do dataset antes de criar arquivos iniciais.
- Sugestao de scripts/arquivos do repositorio que podem ser reaproveitados.
- Scripts Python com `# %%` alinhados com a estrutura padrao do MVP.
