# MVP Final - Análise de Dados da Receita Federal (CNPJ)

## Objetivo

Este projeto apresenta uma análise completa dos dados públicos da Receita Federal sobre empresas brasileiras (base CNPJ), combinando:

1. **Análise Exploratória e Visualização**: Compreender a distribuição de empresas por setor, porte, região e situação cadastral
2. **Previsão de Encerramento**: Modelo de machine learning para prever empresas com risco de encerramento
3. **Rede de Relacionamentos**: Análise de grafos para mapear conexões entre sócios e identificar grupos empresariais

## Dados

**Fonte**: [Dados Públicos CNPJ - Receita Federal](https://arquivos.receitafederal.gov.br/index.php/s/gn672Ad4CF8N6TK?dir=/Dados/Cadastros/CNPJ)

**Data de referência**: Janeiro de 2026

**Arquivos utilizados**:
- `Empresas*.zip`: Informações cadastrais das empresas (CNPJ, razão social, natureza jurídica, porte)
- `Estabelecimentos*.zip`: Dados dos estabelecimentos (situação cadastral, CNAE, endereço)
- `Socios*.zip`: Relação de sócios e administradores
- `Cnaes.zip`: Classificação Nacional de Atividades Econômicas
- `Municipios.zip`: Tabela de municípios brasileiros
- Demais arquivos auxiliares (Naturezas, Qualificações, etc.)

## Como Executar

### Pré-requisitos

Python 3.8+ com as seguintes bibliotecas:

```bash
pip install pandas numpy matplotlib seaborn plotly
pip install scikit-learn xgboost
pip install networkx python-louvain
pip install jupyter
```

### Passos

1. **Preparar os dados**:
   - Os arquivos ZIP devem estar em `data/raw/2026-01/`
   - O notebook fará a extração e processamento automaticamente

2. **Executar o notebook**:
   ```bash
   jupyter notebook notebooks/mvp.ipynb
   ```

3. **Resultados**:
   - Dados processados: `data/processed/`
   - Figuras: `figures/`
   - Modelos treinados: `models/`

## Estrutura do Projeto

```
mvp-final/
├── README.md                 # Este arquivo
├── data/
│   ├── raw/                  # Dados originais (ZIPs da Receita)
│   │   └── 2026-01/
│   └── processed/            # Dados processados
├── notebooks/
│   └── mvp.ipynb            # Notebook principal
├── src/                      # Scripts auxiliares
├── models/                   # Modelos salvos
└── figures/                  # Gráficos gerados
```

## Metodologia

### 1. Análise Exploratória
- Carregamento e limpeza dos dados
- Estatísticas descritivas
- Visualizações: distribuição por CNAE, porte, situação cadastral, distribuição geográfica
- Análise temporal de aberturas e encerramentos

### 2. Previsão de Encerramento
- **Problema**: Classificação binária (empresa ativa vs inativa/baixada)
- **Features**: CNAE, porte, natureza jurídica, idade da empresa, município
- **Modelos**: 
  - Baseline: Logistic Regression
  - Avançado: Random Forest / XGBoost
- **Métricas**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Validação**: Split treino/teste (80/20)

### 3. Rede de Relacionamentos
- Construção de grafo bipartido: sócios ↔ empresas
- Projeção: rede de sócios conectados por empresas em comum
- Métricas de rede: grau, centralidade, coeficiente de clustering
- Detecção de comunidades (algoritmo Louvain)
- Visualização interativa

## Resultados Esperados

- **Insights descritivos** sobre o perfil das empresas brasileiras
- **Modelo preditivo** com acurácia > 75% para identificar empresas em risco de encerramento
- **Mapa de relacionamentos** revelando estruturas de grupos empresariais e sócios com múltiplas participações

## Autor

MVP desenvolvido como trabalho de conclusão da Pós-Graduação em Ciência de Dados e Analytics - PUC-Rio

## Licença

Os dados são públicos e fornecidos pela Receita Federal do Brasil.
