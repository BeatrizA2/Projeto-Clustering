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
    X = load_data()
    findNumClusters = fnc.FindNumClusters(X)
    findNumClusters.plotElbow()
    findNumClusters.plotSilhouette()

    print(f'Optimal number of clusters by Elbow Method: {findNumClusters.optimal_k_elbow}')
    print(f'Optimal number of clusters by Silhouette Method: {findNumClusters.optimal_k_silhouette}')

    plotClustersElbow = pc.PlotClusters(X, findNumClusters.optimal_k_elbow, "Elbow Method")
    plotClustersElbow.plotByKMeans()
    plotClustersElbow.plotByKMedoids()
    plotClustersElbow.plotByDBSCAN()

    plotClustersSilhouette = pc.PlotClusters(X, findNumClusters.optimal_k_silhouette, "Silhouette Method")
    plotClustersSilhouette.plotByKMeans()
    plotClustersSilhouette.plotByKMedoids()
    plotClustersSilhouette.plotByDBSCAN()


    

if __name__ == "__main__":
    main()
