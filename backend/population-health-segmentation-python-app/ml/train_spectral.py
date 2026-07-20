"""
Population Health Patient Segmentation
Spectral Clustering
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.cluster import SpectralClustering
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

print(
    f"Dataset Shape: {df.shape}"
)

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
# Spectral Clustering
# ==========================================

model = SpectralClustering(

    n_clusters=8,

    affinity="nearest_neighbors",

    n_neighbors=15,

    assign_labels="kmeans",

    random_state=42
)

labels = model.fit_predict(
    X_scaled
)

df["cluster"] = labels

# ==========================================
# Evaluation
# ==========================================

silhouette = silhouette_score(
    X_scaled,
    labels
)

davies = davies_bouldin_score(
    X_scaled,
    labels
)

calinski = calinski_harabasz_score(
    X_scaled,
    labels
)

print("\nModel Evaluation")

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

#Finding Best Number of Clusters
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

results = []

for k in range(2,15):

    model = SpectralClustering(

        n_clusters=k,

        affinity="nearest_neighbors",

        random_state=42
    )

    labels = model.fit_predict(
        X_scaled
    )

    score = silhouette_score(
        X_scaled,
        labels
    )

    results.append(score)

    print(
        f"K={k} "
        f"Silhouette={score:.4f}"
    )

#Silhouette Score Visualization
import matplotlib.pyplot as plt

k_values = list(range(2,15))

plt.figure(figsize=(10,6))

plt.plot(
    k_values,
    results,
    marker="o"
)

plt.title(
    "Spectral Clustering Silhouette Scores"
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

plt.figure(figsize=(12,8))

scatter = plt.scatter(

    X_pca[:,0],

    X_pca[:,1],

    c=labels,

    cmap="tab10",

    alpha=0.7
)

plt.title(
    "Spectral Clustering Patient Segments"
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.colorbar(scatter)

plt.show()

#Segment Mapping
segment_mapping = {

    0:"Healthy Young Adults",

    1:"Preventive Care Users",

    2:"Chronic Disease Patients",

    3:"High-Cost Multi-Morbidity Patients",

    4:"Frequent Emergency Visitors",

    5:"Elderly Frail Patients",

    6:"Mental Health Patients",

    7:"Palliative Care Patients"
}

df["segment_name"] = (
    df["cluster"]
    .map(segment_mapping)
)



# ==========================================
# Save Results
# ==========================================

df.to_csv(
    "data/spectral_clusters.csv",
    index=False
)



joblib.dump(
    model,
    "saved_models/spectral.pkl"
)


print(
    "\nSpectral Clustering Complete"
)