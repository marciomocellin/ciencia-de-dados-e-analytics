Códigos e arquivos de apoio das disciplinas da especialização online **Pós-Graduação em Ciência de Dados e Analytics**, do Departamento de Informática da PUC-Rio.

Coordenação:
* **Prof. Hélio Côrtes Vieira Lopes** (*lopes@inf.puc-rio.br*)
* **Prof. Tatiana Escovedo** (*tatiana@inf.puc-rio.br*)

Mais informações em: https://especializacao.ccec.puc-rio.br/especializacao/ciencia-de-dados-e-analytics

Dataset: Taxas dos Títulos Ofertados pelo Tesouro Direto - https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/resource/796d2059-14e9-44e3-80c9-2d9e30b405c1/download/precotaxatesourodireto.csv

Este arquivo contém listagem de preços e taxas dos títulos ofertados pelo Tesouro Direto. A lista é diária.

# Você deverá trabalhar desde a definição do problema até a etapa de pré-processamento de dados, conforme esquema visto na disciplina Análise exploratória e pré-processamento de dados. 
- Produza um notebook no Google Colab, Dataset: Taxas dos Títulos Ofertados pelo Tesouro Direto, com as características a seguir:
- O notebook servirá como relatório, descrevendo textualmente (utilizando as células de texto) o contexto do problema e as operações com os dados (veja a checklist sugerida abaixo).
- Utilize a linguagem Python e bibliotecas que considera apropriadas para abordar o problema.
- Crie o notebook seguindo as boas práticas de codificação vistas no curso. 
- O notebook deve seguir a estrutura proposta neste template (lembre-se de copiar pro seu Drive e editar, este link leva para uma versão de leitura apenas). 

# Observações: 
- O dataset pode ser qualquer um a sua escolha, desde que não sejam os datasets vistos na disciplina Análise exploratória e pré-processamento de Dados
- Durante a sua análise exploratória, no momento de utilizar gráficos, podem ser utilizadas as bibliotecas Python vistas na disciplina Análise exploratória e pré-processamento de dados, as ferramentas vistas na disciplina Visualização de informaçãoou outras a sua escolha. Se forem utilizadas outras ferramentas para a construção dos gráficos, que não as bibliotecas Python, você deverá adicionar no notebook uma figura com cada gráfico produzido.

# Requisitos e composição da nota: 
- (1,0 pt) Execução sem erros:o notebook deve poder ser executado pelo professor do início ao fim sem erros no Google Colab.
- (2,0 pts) Documentação consistente:utilize blocos de texto que expliquem textualmente cada etapa e cada decisão do seu código, contando uma história completa e compreensível, do início ao fim.
- (1,0 pt) Código limpo:seu código deve estar legível e organizado. Devem ser utilizadas as boas práticas de codificação vistas nas disciplinas Programação orientada a objetose Engenharia de software para ciência de dados, mas não é necessário que você crie classes no seu código.
- (2,0 pts) Análise de dados:após cada gráfico, você deverá escrever um parágrafo resumindo os principais achados, analisando os resultados e levantando eventuais pontos de atenção.
- (2,0 pts) Checklist:você deverá responder às perguntas (aplicáveis ao seu dataset) da checklist fornecida, utilizando-a como guia para o desenvolvimento do trabalho.
- (2,0 pts) Capricho e qualidade do trabalho como um todo. 

# Checklist sugerida:
## Definição do problema 
**Objetivo:** entender e descrever claramente o problema que está sendo resolvido. 

- Qual é a descrição do problema?
- Este é um problema de aprendizado supervisionado ou não supervisionado?
- Que premissas ou hipóteses você tem sobre o problema?
- Que restrições ou condições foram impostas para selecionar os dados?
- Defina cada um dos atributos do dataset. 

## Análise de dados 
**Objetivo:** entender a informação disponível. 

### Estatísticas descritivas: 

- Quantos atributos e instâncias existem?
- Quais são os tipos de dados dos atributos?
- Verifique as primeiras linhas do dataset. Algo chama a atenção?
- Há valores faltantes, discrepantes ou inconsistentes?
- Faça um resumo estatístico dos atributos com valor numérico (mínimo, máximo, mediana, moda, média, desvio-padrão e número de valores ausentes). O que você percebe?

### Visualizações: 

- Verifique a distribuição de cada atributo. O que você percebe? Dica: esta etapa pode dar ideias sobre a necessidade de transformações na etapa de preparação de dados (por exemplo, converter atributos de um tipo para outro, realizar operações de discretização, normalização, padronização, etc.).
- Se for um problema de classificação, verifique a distribuição de frequência das classes. O que você percebe? Dica: esta etapa pode indicar a possível necessidade futura de balanceamento de classes.
- Analise os atributos individualmente ou de forma combinada, usando os gráficos mais apropriados. 

## Pré-processamento de dados
**Objetivo:** realizar operações de limpeza, tratamento e preparação dos dados. 

- Verifique quais operações de pré-processamento podem ser interessantes para o seu problema e salve visões diferentes do seu dataset (por exemplo, normalização, padronização, discretização e one-hot-encoding).
- Trate (removendo ou substituindo) os valores faltantes (se existentes).
- Realize outras transformações de dados porventura necessárias.
- Explique, passo a passo, as operações realizadas, justificando cada uma delas.
- Se julgar necessário, utilizando os dados pré-processados, volte na etapa de análise exploratória e verifique se surge algum insight diferente após as operações realizadas.
