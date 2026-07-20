"""
Population Health Patient Segmentation
DBSCAN Clustering
"""

import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
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
# Scaling
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    df_model
)

# ==========================================
# DBSCAN
# ==========================================

dbscan = DBSCAN(
    eps=1.2,
    min_samples=25,
    metric="euclidean"
)

labels = dbscan.fit_predict(
    X_scaled
)

df["cluster"] = labels

# ==========================================
# Cluster Summary
# ==========================================

print(
    "\nNumber of Clusters:",
    len(set(labels)) -
    (1 if -1 in labels else 0)
)

print(
    "Noise Patients:",
    list(labels).count(-1)
)

#Evaluation
mask = labels != -1

X_eval = X_scaled[mask]

labels_eval = labels[mask]

if len(set(labels_eval)) > 1:

    silhouette = silhouette_score(
        X_eval,
        labels_eval
    )

    davies = davies_bouldin_score(
        X_eval,
        labels_eval
    )

    calinski = calinski_harabasz_score(
        X_eval,
        labels_eval
    )

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

#Cluster Distribution
cluster_counts = (
    pd.Series(labels)
    .value_counts()
    .sort_index()
)

print(cluster_counts)


#Cluster Profiling
profile = df.groupby(
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

print(profile)

#Visualize Clusters using PCA
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

    cmap="tab20"

)

plt.title(
    "DBSCAN Patient Segmentation"
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.colorbar(scatter)

plt.show()

#Finding Best eps Value
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np

neighbors = NearestNeighbors(
    n_neighbors=10
)

neighbors_fit = neighbors.fit(
    X_scaled
)

distances, indices = (
    neighbors_fit.kneighbors(
        X_scaled
    )
)

distances = np.sort(
    distances[:,9]
)

plt.figure(figsize=(10,6))

plt.plot(distances)

plt.title(
    "K-Distance Graph"
)

plt.xlabel("Patients")
plt.ylabel("Distance")

plt.show()

#segment_mapping
segment_mapping = {

    -1: "Outlier Patients",

     0: "Healthy Young Adults",

     1: "Preventive Care Users",

     2: "Chronic Disease Patients",

     3: "High-Cost Multi-Morbidity",

     4: "Frequent ER Visitors",

     5: "Elderly Frail Patients"
}

df["segment_name"] = (
    df["cluster"]
    .map(segment_mapping)
)

# =====================================================
# Save Model
# =====================================================
import joblib
joblib.dump(
    dbscan,
    "saved_models/dbscan.pkl"
)

print("\nModel Saved Successfully")