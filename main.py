import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import pipeline, AutoModel, AutoTokenizer
from bertopic import BERTopic
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import umap
import networkx as nx

# Загрузка данных
data = pd.read_csv('synthetic_customer_feedback.csv')
data = data[['review_id', 'review_text']]

#
nltk.download('stopwords')
nltk.download('wordnet')

# Подготовка текста
stop_words = set(stopwords.words('russian'))
lemmatizer = WordNetLemmatizer()


# Создание набора стоп-слов
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words)
    return text


data['cleaned_text'] = data['review_text'].apply(preprocess_text)

# Анализ тональности
sentiment_analyzer = pipeline('sentiment-analysis', model='cointegrated/rubert-tiny2')
data['sentiment'] = data['cleaned_text'].apply(lambda x: sentiment_analyzer(x)[0]['label'])

model_name = 'cointegrated/rubert-tiny2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


# Получение эмбеддингов
def get_embeddings(texts):
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state[:, 0, :]
    return embeddings.numpy()


embeddings = get_embeddings(data['cleaned_text'].to_list())

if embeddings.size == 0:
    raise ValueError('Ошибка: массив эмбеддингов пуст. Проверьте процесс генерации эмбеддингов.')

topic_model = BERTopic(language='russian')
topics, probabilities = topic_model.fit_transform(data['cleaned_text'])

data['topic'] = topics
data['topic_probability'] = probabilities

sns.countplot(data=data, x='sentiment')
plt.title('Распределение тональности отзывов')
plt.xlabel('Тональность')
plt.ylabel('Количество')
plt.show()

for topic in set(data['topic']):
    topic_text = ' '.join(data[data['topic'] == topic]['cleaned_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(topic_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Облако слов для темы {topic}')
    plt.show()
# try:
#     topic_model.visualize_topics()
# except ValueError as e:
#     print(f'Ошибка при визуализации тем: {e}')

# topic_model.visualize_topics()

topic_sentiment = data.groupby(['topic', 'sentiment']).size().unstack()
topic_sentiment.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Тональность по темам')
plt.xlabel('Тема')
plt.ylabel('Количество отзывов')
plt.show()

correlation_matrix = data.pivot_table(index='topic', columns='sentiment', aggfunc='size', fill_value=0)
sns.heatmap(correlation_matrix, annot=True, cmap='viridis')
plt.title('Корреляция между темами и тональностью')
plt.show()

pca = PCA(n_components=2)
pca_embeddings = pca.fit_transform(embeddings)

umap = umap.UMAP(n_neighbors=5, n_components=2, metric='cosine')
umap_embeddings = umap.fit_transform(embeddings)

G = nx.Graph()
unique_topics = data['topic'].unique()

for i, topic in enumerate(unique_topics):
    G.add_node(i, pos=umap_embeddings[i], label=f'Тема {topic}')

for i in range(len(unique_topics)):
    for j in range(i + 1, len(unique_topics)):
        distance = np.linalg.norm(umap_embeddings[i] - umap_embeddings[j])
        if distance < 1.5:
            G.add_edge(i, j, weight=1.0 / distance)

pos = {i: umap_embeddings[i] for i in range(len(unique_topics))}
nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold')
plt.title('Граф взаимосвязей между темами')
plt.show()

plt.figure(figsize=(10, 8))
sns.scatterplot(x=umap_embeddings[:, 0], y=umap_embeddings[:, 1],
                hue=data['topic'], palette='Spectral', legend='full')
plt.title('UMAP кластеризация тем')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend(title='Темы')
plt.show()
