"""
Population Health Patient Segmentation
BIRCH Clustering
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.cluster import Birch

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# ======================================
# Load Dataset
# ======================================

df = pd.read_csv(
    "data/patient_population_health.csv"
)

print(
    f"Dataset Shape: {df.shape}"
)

# ======================================
# Preprocessing
# ======================================

df_model = df.copy()

encoder = LabelEncoder()

df_model["gender"] = encoder.fit_transform(
    df_model["gender"]
)

df_model.drop(
    columns=["patient_id"],
    inplace=True
)

# ======================================
# Scaling
# ======================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    df_model
)

# ======================================
# Train BIRCH
# ======================================

birch = Birch(

    threshold=0.5,

    branching_factor=50,

    n_clusters=8
)

labels = birch.fit_predict(
    X_scaled
)

df["cluster"] = labels

print(
    f"Clusters Found: "
    f"{len(np.unique(labels))}"
)

#Evaluation
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

print()

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
cluster_distribution = (

    pd.Series(labels)

    .value_counts()

    .sort_index()
)

print(
    cluster_distribution
)

#Cluster Profiling
cluster_profile = (

    df.groupby("cluster")

    .agg({

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

)

print(cluster_profile)

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

#Save Model
joblib.dump(
    birch,
    "saved_models/birch.pkl"
)



print(
    "Model Saved Successfully"
)