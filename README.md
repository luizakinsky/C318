# C318 - Projeto
## Classificação de Sentimentos em Avaliações de Clientes - Filmes

### 1. Definição do Projeto e Objetivos
*	Objetivo: Classificar o sentimento das avaliações dos clientes para filmes (positivo, negativo ou neutro).
*	Aplicação Prática: Entender melhor as opiniões dos clientes, melhorar produtos ou serviços com base no feedback.

### 2. Coleta de Dados
*	Dataset: [IMDB Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download).
*	Formato dos Dados: reviews | sentiments

### 3. Exploração e Limpeza dos Dados
* Análise Exploratória: 
  - Verificações Iniciais: Verificar informações gerais e valores ausentes;
  - Estudo dos Dados: Entender o equilíbrio das classes, analisar a extensão das avaliações e visualizar algumas avaliações para entender padrões.
* Limpeza dos Dados:
  - Verificar e remover duplicatas;
  - Remover caracteres especiais, links e números irrelevantes;
  - Converter o texto para minúsculas para padronização;
  - Remover pontuação;
  - Tokenizar o texto;
  - Remover palavras comuns (stop words) que não carregam significado relevante para o sentimento, como “o”, “e”, “um”;
  - Reconstruir o texto limpo.

### 4. Pré-processamento de Texto
*	Tokenização e Vetorização com TF-IDF: Transformar os textos em uma matriz numérica baseada na importância das palavras (TF-IDF);
*	Divisão do Dataset: Separar os dados em conjuntos de treinamento (80%) e teste (20%) para avaliar o desempenho do modelo.

### 5. Seleção e Treinamento do Modelo
*	Escolha de Algoritmo: Para um projeto inicial, modelos como Naive Bayes, Random Forest ou Support Vector Machine (SVM) são boas opções.
*	Treinamento: Alimentar o modelo com o conjunto de treinamento e ajustar os hiperparâmetros (valores que melhoram a performance do modelo).
*	Validação: Utilizar validação cruzada para avaliar o modelo de maneira mais robusta e evitar overfitting (ajuste excessivo aos dados de treino).

### 6. Avaliação do Modelo
*	Métricas: Avaliar a precisão, revocação, F1-Score e matriz de confusão para entender o desempenho.
*	Interpretação: Verificar onde o modelo comete mais erros (ex.: confunde avaliações neutras com positivas) e ajuste os hiperparâmetros se necessário.

### 7. Implementação do Modelo
*	Testes com o Conjunto de Teste: Testar o modelo no conjunto de teste e observar os resultados.
*	Salvamento do Modelo: Utilizar bibliotecas como joblib ou pickle para salvar o modelo treinado e poder reutilizá-lo.

