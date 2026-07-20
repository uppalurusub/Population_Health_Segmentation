"""
Population Health Patient Segmentation
Agglomerative Hierarchical Clustering
Agglomerative Clustering (Bottom-Up)
"""

import pandas as pd
import joblib

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv(
    "data/patient_population_health.csv"
)

print("Dataset Shape:", df.shape)

# ==========================================
# Preprocessing
# ==========================================

df_model = df.copy()

encoder = LabelEncoder()

df_model["gender"] = encoder.fit_transform(
    df_model["gender"]
)

df_model.drop(
    columns=["patient_id"],
    inplace=True
)

# ==========================================
# Feature Matrix
# ==========================================

X = df_model.copy()

# ==========================================
# Scaling
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================
# Agglomerative Clustering
# ==========================================

model = AgglomerativeClustering(
    n_clusters=8,
    metric="euclidean",
    linkage="ward"
)

clusters = model.fit_predict(
    X_scaled
)

# ==========================================
# Evaluation
# ==========================================

silhouette = silhouette_score(
    X_scaled,
    clusters
)

davies = davies_bouldin_score(
    X_scaled,
    clusters
)

calinski = calinski_harabasz_score(
    X_scaled,
    clusters
)

print("\nModel Evaluation")
print("=" * 50)

print(
    f"Silhouette Score: "
    f"{silhouette:.4f}"
)

print(
    f"Davies-Bouldin Score: "
    f"{davies:.4f}"
)

print(
    f"Calinski-Harabasz Score: "
    f"{calinski:.4f}"
)

# ==========================================
# Add Cluster Labels
# ==========================================

df["cluster"] = clusters


#Cluster Profiling
cluster_profile = df.groupby(
    "cluster"
).agg({
    "age":"mean",
    "bmi":"mean",
    "diabetes":"mean",
    "hypertension":"mean",
    "copd":"mean",
    "mental_health":"mean",
    "visits_per_year":"mean",
    "er_visits":"mean",
    "admissions":"mean",
    "total_cost":"mean"
})

print(cluster_profile)

#Segment Mapping
segment_mapping = {

    0: "Healthy Young Adults",

    1: "Preventive Care Users",

    2: "Chronic Disease Patients",

    3: "High-Cost Multi-Morbidity Patients",

    4: "Frequent Emergency Visitors",

    5: "Elderly Frail Patients",

    6: "Mental Health Patients",

    7: "Palliative Care Patients"
}

df["segment_name"] = (
    df["cluster"]
    .map(segment_mapping)
)

#Dendrogram Visualization
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import (
    dendrogram,
    linkage
)

# Use a sample for visualization
sample = X_scaled[:500]

linked = linkage(
    sample,
    method="ward"
)

plt.figure(
    figsize=(15,8)
)

dendrogram(
    linked,
    truncate_mode="level",
    p=5
)

plt.title(
    "Patient Hierarchical Clustering Dendrogram"
)

plt.xlabel(
    "Patient Groups"
)

plt.ylabel(
    "Distance"
)

plt.show()

#Finding the Optimal Number of Clusters
scores = []

for k in range(2,15):

    model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"
    )

    labels = model.fit_predict(
        X_scaled
    )

    score = silhouette_score(
        X_scaled,
        labels
    )

    scores.append(score)

    print(
        f"K={k} "
        f"Silhouette={score:.4f}"
    )

import matplotlib.pyplot as plt

k_values = list(range(2,15))

plt.figure(figsize=(10,6))

plt.plot(
    k_values,
    scores,
    marker="o"
)

plt.title(
    "Agglomerative Clustering - Silhouette Scores"
)

plt.xlabel(
    "Number of Clusters"
)

plt.ylabel(
    "Silhouette Score"
)

plt.grid(True)

plt.show()


#PCA Visualization
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)

plt.figure(
    figsize=(10,6)
)

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters
)

plt.title(
    "Patient Population Segments"
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.colorbar()

plt.show()


# ==========================================
# Save Results
# ==========================================

df.to_csv(
    "data/agglomerative_clusters.csv",
    index=False
)

joblib.dump(
    clusters,
    "saved_models/agglomerative.pkl"
)

print(
    "\nClustered data saved successfully."
)