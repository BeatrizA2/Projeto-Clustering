from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

# load the data into a pandas dataframe
data = pd.read_csv('wine.csv')

# extract the features into a separate dataframe
X = data.iloc[:, :-1]

# standardize the data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# create the covariance matrix
cov_mat = np.cov(X_std.T)

# calculate the eigenvalues and eigenvectors
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# sort the eigenvectors by descending eigenvalues
sorted_idx = eig_vals.argsort()[::-1]
eig_vals_sorted = eig_vals[sorted_idx]
eig_vecs_sorted = eig_vecs[:, sorted_idx]

# select the top 2 eigenvectors
k = 2
eig_vecs_subset = eig_vecs_sorted[:, :k]

# transform the data
X_reduced = X_std.dot(eig_vecs_subset)

# print the explained variance ratio of the top 2 components
pca = PCA(n_components=k)
pca.fit(X_std)
print(pca.explained_variance_ratio_)
