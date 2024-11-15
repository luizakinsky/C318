# Análise do dataset
import pandas as pd
# Plotar gráficos
import matplotlib.pyplot as plt
# Limpeza dos dados
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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
print("Número de duplicatas:", db.duplicated().sum())
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
print(db[['review', 'cleaned_review']].head())