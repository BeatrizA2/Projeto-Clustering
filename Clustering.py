import numpy as np
from sklearn.preprocessing import StandardScaler
import FindNumClusters as fnc
import PlotClusters as pc



def load_data():
    """
    Carrega o dataset wine.csv e retorna os dados normalizados.
    """
    
    X = np.loadtxt('wine.data', delimiter=',')
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X


def main():
    # Carrega os dados e os normaliza
    X = load_data()

    # Encontra o número ótimo de clusters
    findNumClusters = fnc.FindNumClusters(X)

    # Plota o gráfico do método do cotovelo e do metódo do Silhouette
    findNumClusters.plotElbow()
    findNumClusters.plotSilhouette()

    # Imprime o número ótimo de clusters
    print(f'Optimal number of clusters by Elbow Method: {findNumClusters.optimal_k_elbow}')
    print(f'Optimal number of clusters by Silhouette Method: {findNumClusters.optimal_k_silhouette}')

    # Plota os gráficos dos clusters para os métodos K-Means, K-Medoids e DBSCAN
    plotClustersElbow = pc.PlotClusters(X, findNumClusters.optimal_k_elbow, "Elbow Method")
    plotClustersElbow.plotByKMeans()
    plotClustersElbow.plotByKMedoids()
    plotClustersElbow.plotByDBSCAN()

    # Plota os gráficos dos clusters para os métodos K-Means, K-Medoids e DBSCAN
    plotClustersSilhouette = pc.PlotClusters(X, findNumClusters.optimal_k_silhouette, "Silhouette Method")
    plotClustersSilhouette.plotByKMeans()
    plotClustersSilhouette.plotByKMedoids()
    plotClustersSilhouette.plotByDBSCAN()


    

if __name__ == "__main__":
    main()
