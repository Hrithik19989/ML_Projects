import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

df = pd.read_csv("Mall_Customers.csv")
X = StandardScaler().fit_transform(df[["Annual Income (k$)", "Spending Score (1-100)"]])

for name, model in [("KMeans", KMeans(n_clusters=5, random_state=42)),
                     ("Agglomerative", AgglomerativeClustering(n_clusters=5))]:
    labels = model.fit_predict(X)
    print(f"{name} → Silhouette Score: {silhouette_score(X, labels):.3f}")

# DBSCAN (no n_clusters needed)
labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X)
print(f"DBSCAN → Clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}")