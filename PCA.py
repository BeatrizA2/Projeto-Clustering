from sklearn.preprocessing import StandardScaler
import numpy as np

class createPCA:
    def __init__(self, data, k):
        self.df = data
        self.standardize()
        self.covariance()
        self.eigen()
        self.sort()
        self.select(k)
        self.transform()

    def standardize(self):
        scaler = StandardScaler()
        self.X_std = scaler.fit_transform(self.df)
    
    def covariance(self):
        self.cov_mat = np.cov(self.X_std.T)
    
    def eigen(self):
        self.eig_vals, self.eig_vecs = np.linalg.eig(self.cov_mat)

    def sort(self):
        self.sorted_idx = self.eig_vals.argsort()[::-1]
        self.eig_vals_sorted = self.eig_vals[self.sorted_idx]
        self.eig_vecs_sorted = self.eig_vecs[:, self.sorted_idx]
    
    def select(self, k):
        self.eig_vecs_subset = self.eig_vecs_sorted[:, :k]

    def transform(self):
        self.X_reduced = self.X_std.dot(self.eig_vecs_subset)
