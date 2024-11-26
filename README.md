# C318 - Projeto
## Classificação de Sentimentos em Avaliações de Clientes - Filmes

### 1. Definição do Projeto e Objetivos
*  Objetivo: Classificar o sentimento das avaliações dos clientes para filmes (positivo, negativo ou neutro).
*  Aplicação Prática: Entender melhor as opiniões dos clientes, melhorar produtos ou serviços com base no feedback.

### 2. Coleta de Dados
*  Dataset: [IMDB Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download).
*  Formato dos Dados: reviews | sentiments

### 3. Exploração e Limpeza dos Dados
*   Análise Exploratória: 
    - Verificações Iniciais: Verificar informações gerais e valores ausentes;
    - Estudo dos Dados: Entender o equilíbrio das classes, analisar a extensão das avaliações e visualizar algumas avaliações para entender padrões.
*   Limpeza dos Dados:
    - Verificar e remover duplicatas;
    - Remover caracteres especiais, links e números irrelevantes;
    - Converter o texto para minúsculas para padronização;
    - Remover pontuação;
    - Tokenizar o texto;
    - Remover palavras comuns (stop words) que não carregam significado relevante para o sentimento, como “o”, “e”, “um”;
    - Reconstruir o texto limpo.

### 4. Pré-processamento de Texto
*  Tokenização e Vetorização com TF-IDF: Transformar os textos em uma matriz numérica baseada na importância das palavras (TF-IDF);
*  Divisão do Dataset: Separar os dados em conjuntos de treinamento (80%) e teste (20%) para avaliar o desempenho do modelo.

### 5. Seleção e Treinamento do Modelo
*   Escolha de Algoritmo: Random Forest
*   Treinamento: Fazer previsões no conjunto de teste.
*   Ajuste de Hiperparâmetros: Valores que melhoram a performance do modelo.
    - Grid Search.
*	Validação: Utilizar validação cruzada para avaliar o modelo de maneira mais robusta e evitar overfitting (ajuste excessivo aos dados de treino).

### 6. Avaliação do Modelo
*  Métricas: Avaliar a precisão, revocação, F1-Score e matriz de confusão para entender o desempenho.
    Explicação das Métricas:
	- Precisão (Precision): Percentual de predições positivas corretas em relação ao total de predições positivas feitas;
	- Revocação (Recall): Percentual de predições positivas corretas em relação ao total de valores positivos reais;
	- F1-Score: Média harmônica entre precisão e revocação (importante para conjuntos de dados desbalanceados);
	- Acurácia: Percentual geral de classificações corretas.
*  Matriz de Confusão: Verificar onde o modelo comete mais erros.
    Interpretação:
	- Verdadeiros Positivos (VP): Número de avaliações positivas previstas corretamente;
	- Verdadeiros Negativos (VN): Número de avaliações negativas previstas corretamente;
	- Falsos Positivos (FP): Número de avaliações negativas previstas como positivas (erro tipo I);
	- Falsos Negativos (FN): Número de avaliações positivas previstas como negativas (erro tipo II).
*   Interpretação dos Erros:
    1.	Desequilíbrio em Classes:
        -	Se houver muitos erros nas classes menos representadas, avaliar usar balanceamento no modelo (class_weight='balanced' no Random Forest).
    2.	Confusão entre Classes:
        -	Se o modelo confundir avaliações neutras com positivas, é um indicativo de que os dados ou a representação (TF-IDF) não estão capturando a diferença entre elas.
*   Ajuste de Hiperparâmetros: Após identificar erros comuns, ajustar os hiperparâmetros do modelo. Por exemplo:
	- Aumentar o número de estimadores (n_estimators) para melhorar a estabilidade;
	- Ajustar a profundidade máxima (max_depth) para evitar overfitting;
	- Usar validação cruzada para encontrar os melhores parâmetros.


### 7. Implementação do Modelo
*	Testes com o Conjunto de Teste: Testar o modelo no conjunto de teste e observar os resultados.
*	Salvamento do Modelo: Utilizar a biblioteca joblib para salvar o modelo treinado e poder reutilizá-lo.
