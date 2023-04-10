from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

#alterando o diretorio
os.chdir("C:\\Users\\CAP. PERNA\\OneDrive\\Documentos\\UFPE\\SI\\Projeto-Clustering")

# carregando os dados
data = pd.read_csv("wine.data", header=None)

# normalizando os dados
scaler = StandardScaler()
data_norm = scaler.fit_transform(data)

# executando o K-means para diferentes valores de k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data_norm)
    sse.append(kmeans.inertia_)

# plotando o gráfico SSC x k
import matplotlib.pyplot as plt

plt.plot(range(1, 11), sse)
plt.xlabel('Número de clusters')
plt.ylabel('Soma dos quadrados dentro do cluster')
plt.show()

# executando o K-means para o valor escolhido de k
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(data_norm)

# imprimindo os rótulos dos clusters
print(clusters)
