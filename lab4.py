from builtins import print, range, len

import numpy as np
import pandas as pd
import statsmodels.tsa.api as smt
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def find_optimal_k(data, max_clusters=10):
    distortions = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)

    second_derivatives = [distortions[i] - 2 * distortions[i - 1] + distortions[i - 2] for i in
                          range(2, len(distortions))]

    elbow_index = np.argmax(second_derivatives) + 2  # Смещение на 2, так как мы начали считать с 2

    return elbow_index

def kmeans_custom(X, k, max_iters=100):
    centroids = X[np.random.choice(range(len(X)), size=k, replace=False)]

    for _ in range(max_iters):
        labels = np.argmin(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2), axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    return centroids, labels



data = pd.read_csv('D:\\Глеб\\Учёба\\4КУРС\\MO\\Wholesale customers data.csv')

channels_regions = data[['Channel', 'Region']]
data = data.drop(['Channel', 'Region'], axis=1)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Построение модели кластеризации методом k-средних
k = find_optimal_k(scaled_data)
print(k)
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
kmeans.fit(scaled_data)

# Получение меток кластеров
cluster_labels = kmeans.labels_

# Добавление меток кластеров к оригинальным данным
data_with_clusters = data.copy()
data_with_clusters['Cluster'] = cluster_labels

# Соединяем данные о каналах и регионах с метками кластеров
data_with_clusters = pd.concat([channels_regions, data_with_clusters], axis=1)

# Вывод результатов
print(data_with_clusters.head())

# Дополнительный анализ каждого кластера
for i in range(k):
    print(f"Cluster {i + 1}:")
    print(data_with_clusters[data_with_clusters['Cluster'] == i].describe())


def plot_elbow_method(data, max_clusters=10):
    distortions = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
    # plt.plot(range(1, max_clusters + 1), distortions, marker='o')
    plt.scatter(range(1, max_clusters + 1), distortions, marker='o', color='b')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')
    plt.show()


plot_elbow_method(scaled_data)
# Получение меток кластеров
cluster_labels = kmeans.labels_

# Получение центроидов кластеров
centroids = kmeans.cluster_centers_

# Получение количества вхождений в каждый кластер
cluster_counts = np.bincount(cluster_labels)

# Вывод центроидов и количества вхождений в кластеры
for i in range(k):
    print(f"Cluster {i + 1}:")
    print("Centroid:", centroids[i])
    print("Number of entries:", cluster_counts[i])

print("1 кластер клиентов, которые покупают большие объемы продуктов для повседневного использования")
print(
    "2 Этот кластер может отличаться высоким потреблением свежих продуктов (Fresh) и замороженных продуктов (Frozen), что может указывать на рестораны или кафе, специализирующиеся на приготовлении свежих блюд")
print(
    "3 Этот кластер может быть характеризован высокими значениями переменных, связанных с молочными продуктами (Milk) и Delicassen (деликатесы), что может указывать на кластер клиентов, предпочитающих высококачественные или специализированные продукты, такие как кафе или магазины деликатесов")

print('Из наборов можно получить сегментацию клиентов и прогнозирвоание спроса')

# Реализация кластеризации методом k-средних без использования scikit-learn
centroids_custom, labels_custom = kmeans_custom(scaled_data, k=3)

# Вывод результатов кластеризации без scikit-learn
print("Centroids (Custom):", centroids_custom)
print("Labels (Custom):", labels_custom)