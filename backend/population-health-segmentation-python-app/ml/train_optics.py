"""
Population Health Patient Segmentation
OPTICS Clustering
"""

import pandas as pd
import numpy as np

from sklearn.cluster import OPTICS
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
# Load Data
# ==========================================

df = pd.read_csv(
    "data/patient_population_health.csv"
)

print("Dataset Shape:", df.shape)

# ==========================================
# Encode Gender
# ==========================================

df_model = df.copy()

encoder = LabelEncoder()

df_model["gender"] = encoder.fit_transform(
    df_model["gender"]
)

# ==========================================
# Drop Patient ID
# ==========================================

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
# OPTICS Model
# ==========================================

model = OPTICS(

    min_samples=50,

    metric="euclidean",

    cluster_method="xi",

    xi=0.05,

    min_cluster_size=0.03
)

labels = model.fit_predict(
    X_scaled
)

df["cluster"] = labels

# ==========================================
# Cluster Summary
# ==========================================

n_clusters = len(
    set(labels)
) - (1 if -1 in labels else 0)

n_noise = np.sum(
    labels == -1
)

print(
    f"Clusters Found: {n_clusters}"
)

print(
    f"Noise Patients: {n_noise}"
)

#Cluster Evaluation
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
        f"Silhouette Score: {silhouette:.4f}"
    )

    print(
        f"Davies-Bouldin Score: {davies:.4f}"
    )

    print(
        f"Calinski-Harabasz Score: {calinski:.4f}"
    )

#Cluster Distribution
cluster_distribution = (
    pd.Series(labels)
    .value_counts()
    .sort_index()
)

print("cluster_distribution: ",cluster_distribution)

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

#Reachability Plot
import matplotlib.pyplot as plt

space = np.arange(
    len(X_scaled)
)

reachability = (
    model.reachability_
)

labels = model.labels_

plt.figure(
    figsize=(15,6)
)

plt.plot(
    space,
    reachability
)

plt.title(
    "OPTICS Reachability Plot"
)

plt.xlabel(
    "Patients"
)

plt.ylabel(
    "Reachability Distance"
)

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
    figsize=(12,8)
)

scatter = plt.scatter(

    X_pca[:,0],
    X_pca[:,1],

    c=labels,

    cmap="tab20"
)

plt.title(
    "OPTICS Patient Segmentation"
)

plt.xlabel("PC1")

plt.ylabel("PC2")

plt.colorbar(scatter)

plt.show()

#Save Results
df.to_csv(
    "saved_models/optics_clusters.csv",
    index=False
)
# =====================================================
# Save Model
# =====================================================
import joblib
joblib.dump(
    model,
    "saved_models/optics.pkl"
)

print("\nModel Saved Successfully")