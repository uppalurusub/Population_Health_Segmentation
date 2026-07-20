"""
Patient Segmentation Service
"""

from implementations.patient_segmentation_impl import (
    PatientSegmentationImpl
)

from implementations.mlflow_inference_tracker import (
    track_prediction_request
)

import time

class PatientSegmentationService:

    def __init__(self):

        self.impl = PatientSegmentationImpl()

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

    # ==========================================
    # Train Model
    # ==========================================

    def train_model(self):

        result = self.impl.train_model()

        return result

    # ==========================================
    # Predict Segment
    # ==========================================

    def predict_segment(
        self,
        patient_data: dict
    ):

        start_time = time.time()

        prediction = self.impl.predict(
            patient_data
        )

        execution_time = (
            time.time() - start_time
        )

        cluster = int(
            prediction["cluster"]
        )

        print("cluster: ",cluster)
        

        prediction_response = {

            "cluster": cluster,

            "segment_name":
            self.segment_mapping.get(
                cluster,
                "Unknown Segment"
            ),

            "confidence":
            prediction.get(
                "confidence",
                None
            )
        }

        track_prediction_request(
            request_data=patient_data,
            prediction_result=prediction_response,
            execution_time=execution_time
        )

        return prediction_response

    # ==========================================
    # Evaluation
    # ==========================================

    def get_evaluation(self):

        

        prediction_response = self.impl.get_evaluation()
        
        return prediction_response
        #return self.impl.get_evaluation()

    # ==========================================
    # Cluster Distribution
    # ==========================================

    def cluster_distribution(self):

        return self.impl.cluster_distribution()

    # ==========================================
    # Cluster Profile
    # ==========================================

    def cluster_profile(self):

        return self.impl.cluster_profile()

    # ==========================================
    # Segment Summary
    # ==========================================

    def segment_summary(self):

        distribution = (
            self.impl.cluster_distribution()
        )

        summary = []

        for cluster, count in (
            distribution.items()
        ):

            summary.append({

                "cluster": cluster,

                "segment_name":
                self.segment_mapping.get(
                    int(cluster),
                    "Unknown Segment"
                ),

                "patient_count":
                count
            })

        return summary