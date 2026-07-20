"""
Patient Segmentation Implementation
"""

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)
import hdbscan

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

import os



class PatientSegmentationImpl:

    def __init__(self):

        self.data_path = (
            "../data/patient_population_health.csv"
        )

        os.makedirs("best_saved_models",exist_ok=True)

        self.model_path = (
            "best_saved_models/phs_model.pkl" 
        )

        self.scaler_path = (
            "best_saved_models/scaler.pkl"
        )

        self.encoder_path = (
            "best_saved_models/encoder.pkl"
        )

        self.segment_mapping = {

            -1: "Outlier / Unclassified Patient",

            0: "Healthy Young Adults",

            1: "Preventive Care Users",

            2: "Chronic Disease Patients",

            3: "High-Cost Multi-Morbidity Patients",

            4: "Frequent Emergency Visitors",

            5: "Elderly Frail Patients",

            6: "Mental Health Patients",

            7: "Palliative Care Patients"
        }

    
    # ==================================================
    # Load Dataset
    # ==================================================

    def load_data(self):

        return pd.read_csv(
            self.data_path
        )

    # ==================================================
    # Preprocess Data
    # ==================================================

    def preprocess_data(
        self,
        df
    ):

        df_model = df.copy()

        encoder = LabelEncoder()
        

        df_model["gender"] = (
            encoder.fit_transform(
                df_model["gender"]
            )
        )

        if "patient_id" in df_model.columns:

            df_model.drop(
                columns=["patient_id"],
                inplace=True
            )

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(
            df_model
        )

        
        joblib.dump(
            encoder,
            self.encoder_path
        )

        joblib.dump(
            scaler,
            self.scaler_path
        )

        return X_scaled

    # ==================================================
    # Train Model
    # ==================================================

    def train_model(self):

        df = self.load_data()

        X_scaled = self.preprocess_data(
            df
        )

        model = KMeans(

            n_clusters=8,

            random_state=42,

            n_init=20
        )

        labels = model.fit_predict(
            X_scaled
        )

        joblib.dump(
            model,
            self.model_path
        )

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

        return {

            "algorithm": "KMeans",

            "n_clusters": 8,

            "silhouette_score":
            round(silhouette, 4),

            "davies_bouldin_score":
            round(davies, 4),

            "calinski_harabasz_score":
            round(calinski, 4)
        }

    # ==================================================
    # Predict Patient Segment
    # ==================================================
    def predict(self, patient_data):

        model = joblib.load(self.model_path)
        scaler = joblib.load(self.scaler_path)
        encoder = joblib.load(self.encoder_path)

        df = pd.DataFrame([patient_data])

        df["gender"] = encoder.transform(df["gender"])

        X_scaled = scaler.transform(df)

        labels, strengths = hdbscan.approximate_predict(
            model,
            X_scaled
        )

        cluster = int(labels[0])

        return {
            "cluster": cluster,
            "segment": self.segment_mapping.get(
                cluster,
                "Unknown"
            ),
            "confidence": float(strengths[0])
        }
    

    

    # ==================================================
    # Evaluation
    # ==================================================

    def get_evaluation(self):

        df = self.load_data()

        model = joblib.load(
            self.model_path
        )

        encoder = joblib.load(
            self.encoder_path
        )

        scaler = joblib.load(
            self.scaler_path
        )

        df_model = df.copy()

        df_model["gender"] = (
            encoder.transform(
                df_model["gender"]
            )
        )

        df_model.drop(
            columns=["patient_id"],
            inplace=True
        )

        X_scaled = scaler.transform(
            df_model
        )

        #labels = model.labels_

        labels = model.predict(
            X_scaled
        )

        return {

            "silhouette_score":
            round(
                silhouette_score(
                    X_scaled,
                    labels
                ),
                4
            ),

            "davies_bouldin_score":
            round(
                davies_bouldin_score(
                    X_scaled,
                    labels
                ),
                4
            ),

            "calinski_harabasz_score":
            round(
                calinski_harabasz_score(
                    X_scaled,
                    labels
                ),
                4
            )
        }

    # ==================================================
    # Cluster Distribution
    # ==================================================

    def cluster_distribution(self):

        df = self.load_data()

        model = joblib.load(
            self.model_path
        )

        encoder = joblib.load(
            self.encoder_path
        )

        scaler = joblib.load(
            self.scaler_path
        )

        df_model = df.copy()

        df_model["gender"] = (
            encoder.transform(
                df_model["gender"]
            )
        )

        df_model.drop(
            columns=["patient_id"],
            inplace=True
        )

        X_scaled = scaler.transform(
            df_model
        )

        #labels = model.labels_
        labels = model.predict(
            X_scaled
        )

        distribution = (

            pd.Series(labels)

            .value_counts()

            .sort_index()

            .to_dict()
        )

        return distribution

    # ==================================================
    # Cluster Profile
    # ==================================================

    def cluster_profile(self):

        df = self.load_data()

        model = joblib.load(
            self.model_path
        )

        encoder = joblib.load(
            self.encoder_path
        )

        scaler = joblib.load(
            self.scaler_path
        )

        df_model = df.copy()

        df_model["gender"] = (
            encoder.transform(
                df_model["gender"]
            )
        )

        df_model.drop(
            columns=["patient_id"],
            inplace=True
        )

        X_scaled = scaler.transform(
            df_model
        )

        #labels = model.labels_

        labels = model.predict(
            X_scaled
        )

        df["cluster"] = labels

        profile = (

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

            .round(2)

            .to_dict(
                orient="index"
            )
        )

        return profile