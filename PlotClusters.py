from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.cluster import DBSCAN


class PlotClusters:
    def __init__(self, data, optimal_k, title):
        self.data = data
        self.optimal_k = optimal_k
        self.title = title
        self.labelsKmeans = self.calcKMeans()
        self.labelsKmedoids = self.calcKMedoids()
        self.distances = self.calcEPS()
        self.labelsDBSCAN = self.calcDBSCAN()


    def calcKMeans(self):
        kmeans = KMeans(n_clusters=self.optimal_k).fit(self.data)
        label = kmeans.fit_predict(self.data)
        return label


    def calcKMedoids(self):
        kmedoids = KMedoids(n_clusters=self.optimal_k).fit(self.data)
        label = kmedoids.fit_predict(self.data)
        return label

    def calcEPS(self):
        nbrs = NearestNeighbors(n_neighbors=2).fit(self.data)
        distances, _ = nbrs.kneighbors(self.data)
        distances = np.sort(distances, axis=0)
        distances = distances[:,1]
        plt.plot(distances)
        plt.title(f'EPS {self.title}')
        plt.show()
        return distances

    def calcDBSCAN(self):
        dbscan = DBSCAN(eps=self.distances[self.optimal_k], min_samples=self.data.shape[1] + 1).fit(self.data)
        return dbscan.labels_

    def plotByKMeans(self):
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labelsKmeans)
        plt.title(f'K-Means Clustering - {self.title}')
        plt.show()

    def plotByKMedoids(self):
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labelsKmedoids)
        plt.title(f'K-Medoids Clustering - {self.title}')
        plt.show()
    
    def plotByDBSCAN(self):
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labelsDBSCAN)
        plt.title(f'DBSCAN Clustering - {self.title}')
        plt.show()

    