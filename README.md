# C318 - Projeto
## Classificação de Sentimentos em Avaliações de Clientes - Filmes

### 1. Definição do Projeto e Objetivos
*	Objetivo: Classificar o sentimento das avaliações dos clientes para filmes (positivo, negativo ou neutro).
*	Aplicação Prática: Entender melhor as opiniões dos clientes, melhorar produtos ou serviços com base no feedback.

### 2. Coleta de Dados
*	Dataset: Utilizar um dataset público, como o [IMDB Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).
*	Formato dos Dados: 

### 3. Exploração e Limpeza dos Dados
* Análise Exploratória: Observe a distribuição dos sentimentos no dataset e veja se há um equilíbrio entre classes positivas, negativas e neutras.
*	Limpeza dos Dados:
-	Remover caracteres especiais, links e números irrelevantes.
-	Converter o texto para minúsculas para padronização.
-	Remover palavras comuns (stop words) que não carregam significado relevante para o sentimento, como “o”, “e”, “um”.

### 4. Pré-processamento de Texto
*	Tokenização: Dividir cada texto em palavras (tokens).
*	Vetorização: Usar o método TF-IDF ou Bag of Words para converter o texto em uma matriz numérica. Alternativamente, técnicas como Word2Vec ou Embedding podem melhorar a representação semântica do texto.
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
