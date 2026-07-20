import pandas as pd

from ml.divisive_clustering import (
    DivisiveClustering
)

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
# ==========================
# Load Dataset
# ==========================

df = pd.read_csv(
    "data/patient_population_health.csv"
)

# ==========================
# Encode Gender
# ==========================

encoder = LabelEncoder()

df["gender"] = encoder.fit_transform(
    df["gender"]
)

# ==========================
# Drop Patient ID
# ==========================

df.drop(
    columns=["patient_id"],
    inplace=True
)

# ==========================
# Scaling
# ==========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    df
)

# ==========================
# Divisive Clustering
# ==========================

model = DivisiveClustering(
    n_clusters=8
)

labels = model.fit_predict(
    X_scaled
)

print(
    "Clusters:",
    len(set(labels))
)

df["cluster"] = labels

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

print(
    "Silhouette:",
    round(silhouette,4)
)

print(
    "Davies Bouldin:",
    round(davies,4)
)

print(
    "Calinski Harabasz:",
    round(calinski,4)
)

#Cluster Profiling
df["cluster"] = labels

profile = df.groupby(
    "cluster"
).mean(
    numeric_only=True
)

print(profile)

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

#PCA Visualization
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)

plt.figure(figsize=(10,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=labels
)

plt.title(
    "Divisive Clustering"
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.colorbar()

plt.show()

# =====================================================
# Save Model
# =====================================================
import joblib
joblib.dump(
    model,
    "saved_models/divisive.pkl"
)

print("\nModel Saved Successfully")