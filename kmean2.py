import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def load_data():
    """
    Carrega o dataset wine.csv e retorna os dados normalizados.
    """
    data = pd.read_csv('wine.csv')
    X = data.iloc[:, 1:].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X

def find_num_clusters(X):
    """
    Encontra o número ideal de clusters usando o método Silhouette para KMeans e KMedoids.
    """
    scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmedoids = KMedoids(n_clusters=k, random_state=42)
        kmeans_score = silhouette_score(X, kmeans.fit_predict(X))
        kmedoids_score = silhouette_score(X, kmedoids.fit_predict(X))
        score = max(kmeans_score, kmedoids_score)
        scores.append(score)
    return scores.index(max(scores)) + 2

def plot_clusters(X, labels, title):
    """
    Plota os clusters no espaço de duas dimensões.
    """
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.title(title)
    plt.show()

def main():
    X = load_data()

    # Encontrar o número ideal de clusters usando o método Silhouette
    k = find_num_clusters(X)

    # Executar os algoritmos de clusterização
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmedoids = KMedoids(n_clusters=k, random_state=42)
    dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X)

    # Plotar os resultados
    plot_clusters(X, kmeans.fit_predict(X), "KMeans")
    plot_clusters(X, kmedoids.fit_predict(X), "KMedoids")
    plot_clusters(X, dbscan.labels_, "DBSCAN")

if __name__ == "__main__":
    main()
