import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the dataset
df = pd.read_csv('./iris_data.csv')

# Drop the target variable
df = df.drop('Targets', axis=1)

# -----------------------------
# EM Algorithm (Gaussian Mixture)
model = GaussianMixture(n_components=3)
model.fit(df)
labels_em = model.predict(df)

# -----------------------------
# k-Means Algorithm
kmeans = KMeans(n_clusters=3)
kmeans.fit(df)
labels_kmeans = kmeans.labels_

# -----------------------------
# Silhouette Scores
silhouette_em = silhouette_score(df, labels_em)
silhouette_kmeans = silhouette_score(df, labels_kmeans)

# -----------------------------
# Output results
print("EM Algorithm:")
print("Silhouette Score:", silhouette_em)
print("\nk-Means Algorithm:")
print("Silhouette Score:", silhouette_kmeans)
