from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

class FindNumClusters:
    def __init__(self, data):
        self.data = data
        self.inertias, self.optimal_k_elbow = self.byElbow()
        self.scores, self.optimal_k_silhouette = self.bySilhouette()

    def byElbow(self):
        # Calcula o inertia para cada valor de k
        inertias = []
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k).fit(self.data)
            kmedoids = KMedoids(n_clusters=k).fit(self.data)
            kmeans_inertia = kmeans.inertia_
            kmedoids_inertia = kmedoids.inertia_
            # Adiciona o menor inertia entre os dois métodos
            inertias.append(min(kmeans_inertia, kmedoids_inertia))
        # Retorna o valor de k que minimiza o inertia
        return inertias, inertias.index(min(inertias)) + 2

    def bySilhouette(self):
        scores = []
        # Calcula o silhouette score para cada valor de k
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k)
            kmedoids = KMedoids(n_clusters=k)
            kmeans_score = silhouette_score(self.data, kmeans.fit_predict(self.data))
            kmedoids_score = silhouette_score(self.data, kmedoids.fit_predict(self.data))
            # Adiciona o maior silhouette score entre os dois métodos
            score = max(kmeans_score, kmedoids_score)
            scores.append(score)
        # Retorna o valor de k que maximiza o silhouette score
        return scores, scores.index(max(scores)) + 2
    
    def plotElbow(self):
        import matplotlib.pyplot as plt
        plt.plot(range(2, 11), self.inertias)
        plt.title(f'Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertias')
        plt.show()
    
    def plotSilhouette(self):
        import matplotlib.pyplot as plt
        plt.plot(range(2, 11), self.scores)
        plt.title(f'Silhouette Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.show()