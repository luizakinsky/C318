# Análise do dataset
import pandas as pd
# Plotar gráficos
import matplotlib.pyplot as plt
# Limpeza dos dados
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
# Pré-processamento de texto
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Carregando o dataset
db = pd.read_csv('/Users/luizakinsky/Documents/Material de aula INATEL/2024.2/C318/C318/IMDB Dataset.csv', delimiter=',')

# --------------------- ANÁLISE EXPLORATÓRIA ---------------------
# VERIFICAÇÕES INICIAIS
# Verificar informações gerais
#print(db.info())
# Verificar valores ausentes
#print(db.isnull().sum())

# ESTUDO DOS DADOS
# Entender o equilíbrio das classes
# Distribuição dos sentimentos - 50%/50%
# sentiment_counts = db['sentiment'].value_counts()
# print(sentiment_counts)
# # Visualizar como gráfico
# plt.figure(figsize=(8, 6))
# sentiment_counts.plot(kind='bar', color=['blue', 'orange'])
# plt.title('Distribuição dos Sentimentos')
# plt.xlabel('Sentimentos')
# plt.ylabel('Quantidade')
# plt.xticks(rotation=0)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()

# Comprimento dos textos
# Analisar a extensão das avaliações
# Adicionar uma coluna com o comprimento dos textos
# db['review_length'] = db['review'].apply(len)
# # Estatísticas básicas do comprimento dos textos
# print(db['review_length'].describe())
# # Histograma do comprimento dos textos
# plt.figure(figsize=(10, 6))
# plt.hist(db['review_length'], bins=50, color='purple', alpha=0.7, edgecolor='black')
# plt.title('Distribuição do Comprimento das Avaliações')
# plt.xlabel('Comprimento do Texto')
# plt.ylabel('Frequência')
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()

# Visualizar algumas avaliações para entender padrões
# Avaliações positivas
# print("Avaliações Positivas:")
# print(db[db['sentiment'] == 'positive']['review'].head(3))
# # Avaliações negativas
# print("Avaliações Negativas:")
# print(db[db['sentiment'] == 'negative']['review'].head(3))


# --------------------- LIMPEZA DOS DADOS ---------------------
# Verificar duplicatas
#print("Número de duplicatas:", db.duplicated().sum())
# Remover duplicatas (se necessário)
db = db.drop_duplicates()

# Obter stopwords do sklearn
stop_words = ENGLISH_STOP_WORDS

def clean_text_alternative(text):
    # Remover caracteres especiais, links e números
    text = re.sub(r'http\S+|www\S+|[^a-zA-Z\s]', '', text)
    
    # Converter para minúsculas
    text = text.lower()
    
    # Remover pontuação
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenizar (dividir o texto em palavras)
    words = text.split()
    
    # Remover stopwords
    words = [word for word in words if word not in stop_words]
    
    # Reconstruir o texto limpo
    cleaned_text = ' '.join(words)
    return cleaned_text

# Aplicar a função de limpeza
db['cleaned_review'] = db['review'].apply(clean_text_alternative)
# Verificar os resultados
#print(db[['review', 'cleaned_review']].head())


# --------------------- PRÉ-PROCESSAMENTO ---------------------
# Tokenização e Vetorização com TF-IDF
# Transformar os textos em uma matriz numérica baseada na importância das palavras (TF-IDF)
# Inicializar o vetor TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
# Ajustar o vetor TF-IDF e transformar os textos limpos
X = tfidf_vectorizer.fit_transform(db['cleaned_review'])
# Mostrar as primeiras 5 palavras mais frequentes
#print("Palavras mais frequentes:", tfidf_vectorizer.get_feature_names_out()[:5])

# Divisão do Dataset em conjuntos de treinamento e teste
# Variável independente (X) é a matriz TF-IDF
# Variável dependente (y) é a coluna 'sentiment' (convertida para números)
y = db['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
# Divisão dos dados (80% treinamento, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Mostrar os tamanhos dos conjuntos
#print("Tamanho do conjunto de treinamento:", X_train.shape)
#print("Tamanho do conjunto de teste:", X_test.shape)

# --------------------- TREINAMENTO DO MODELO ---------------------
# Inicializar o modelo Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# Treinar o modelo com os dados de treinamento
rf_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = rf_model.predict(X_test)

# AJUSTE DE HIPERPARÂMETROS
# Definir o grid de hiperparâmetros
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Configurar a busca com validação cruzada
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,  # 5 folds na validação cruzada
    scoring='accuracy',
    n_jobs=-1,  # Paralelizar para acelerar
    verbose=2
)
# Realizar a busca
grid_search.fit(X_train, y_train)
# Exibir os melhores parâmetros encontrados
#print("Melhores Hiperparâmetros:", grid_search.best_params_)
# Melhor modelo ajustado
best_rf_model = grid_search.best_estimator_

# VALIDAÇÃO
# Realizar validação cruzada com 5 folds
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
# Resultados da validação cruzada
#print("Acurácias em cada fold:", cv_scores)
#print("Acurácia média:", cv_scores.mean())


# --------------------- AVALIAÇÃO DO MODELO ---------------------
# Relatório de classificação
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

# MATRIZ DE CONFUSÃO
# Gerar matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualizar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Matriz de Confusão')
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.show()
