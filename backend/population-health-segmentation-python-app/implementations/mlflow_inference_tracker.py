import uuid
import mlflow
from datetime import datetime

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

EXPERIMENT_NAME = "Population_Health_Inference"

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_experiment(
    EXPERIMENT_NAME
)


def track_prediction_request(
    request_data: dict,
    prediction_result: dict,
    execution_time: float,
    model_name: str = "HDBSCAN"
):

    with mlflow.start_run(
        nested=True,
        run_name="patient_segmentation_prediction"
    ) as run:

        request_id = str(
            uuid.uuid4()
        )

        mlflow.log_param(
            "request_id",
            request_id
        )

        mlflow.log_param(
            "model_name",
            model_name
        )

        # ==========================
        # Input Features
        # ==========================

        for key, value in request_data.items():

            mlflow.log_param(
                key,
                value
            )

        # ==========================
        # Prediction
        # ==========================

        mlflow.log_param(
            "cluster",
            prediction_result["cluster"]
        )

        mlflow.log_param(
            "segment_name",
            prediction_result["segment_name"]
        )

        mlflow.log_metric(
            "confidence",
            float(
                prediction_result.get(
                    "confidence",
                    0
                )
            )
        )

        mlflow.log_metric(
            "execution_time_ms",
            execution_time
        )

        # ==========================
        # Tags
        # ==========================

        mlflow.set_tag(
            "application",
            "Population Health Segmentation"
        )

        mlflow.set_tag(
            "timestamp",
            str(datetime.utcnow())
        )

        # ==========================
        # Request
        # ==========================

        mlflow.log_dict(
            request_data,
            "request.json"
        )

        # ==========================
        # Response
        # ==========================

        mlflow.log_dict(
            prediction_result,
            "response.json"
        )

        # -------------------------------------------------
        # Print Tracking Details
        # -------------------------------------------------

        print("\n" + "=" * 60)

        print(
            "MLflow Tracking Completed Successfully"
        )

        run_id = run.info.run_id
        print(f"Request ID : {request_id}")

        print(f"Run ID     : {run_id}")

        print(
            f"Run URL    : "
            f"http://127.0.0.1:5000/#/experiments/"
            f"{EXPERIMENT_NAME}/runs/{run_id}"
        )

        print("=" * 60)