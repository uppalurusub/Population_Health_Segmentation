import os
import mlflow

print("MLFLOW_TRACKING_URI ENV:",
      os.environ.get("MLFLOW_TRACKING_URI"))

print("Tracking URI Before:",
      mlflow.get_tracking_uri())

import joblib
import skops.io as sio
import mlflow
import mlflow.sklearn
import pandas as pd
import logging
import numpy as np

logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("mlflow").setLevel(logging.CRITICAL)
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="mlflow",
    message=".*Saving scikit-learn models in the pickle or cloudpickle format.*"
    )
warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="LightGBM"
)


import pandas as pd
import numpy as np
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

from utils.preprocess import DataPreprocessor

df = pd.read_csv("data/patient_population_health.csv")

dp = DataPreprocessor()

X_scaled = dp.fit_transform(df)

# =====================================================
# Elbow Method
# =====================================================

print("\nFinding Optimal Clusters...")

wcss = []

k_values = range(1, 16)

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)

print("WCSS Values")

# ==========================================
# Plot Elbow Curve
# ==========================================
from matplotlib import pyplot as plt

plt.figure(figsize=(10, 6))

plt.plot(
    k_values,
    wcss,
    marker="o",
    linewidth=2
)

plt.title(
    "Elbow Method for Optimal Number of Clusters"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "WCSS (Within Cluster Sum of Squares)"
)

#plt.xticks(k)

plt.grid(True)

plt.show()

for i, val in enumerate(wcss, start=2):

    print(f"K={i}: {val:.2f}")

# =====================================================
# Train Final Model
# =====================================================

N_CLUSTERS = 8

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init=20
)

clusters = kmeans.fit_predict(
    X_scaled
)

# =====================================================
# Evaluation
# =====================================================

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

print("="*50)

print(
    f"Silhouette Score: "
    f"{silhouette:.4f}"
)

print(
    f"Davies Bouldin Score: "
    f"{davies:.4f}"
)

print(
    f"Calinski Harabasz Score: "
    f"{calinski:.4f}"
)

# =====================================================
# Save Cluster Labels
# =====================================================

df["cluster"] = clusters

# =====================================================
# Cluster Profiling
# =====================================================

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

print("\nCluster Profile")

print(cluster_profile)

# =====================================================
# Segment Mapping
# =====================================================

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

df["segment_name"] = df[
    "cluster"
].map(segment_mapping)

# =====================================================
# Save Results
# =====================================================

df.to_csv(
    "data/patient_segments.csv",
    index=False
)

cluster_profile.to_csv(
    "data/cluster_profile.csv"
)

# =====================================================
# Save Model
# =====================================================

joblib.dump(
    kmeans,
    "saved_models/kmeans.pkl"
)

print("\nModel Saved Successfully")

