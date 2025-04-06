"""Evaluation script for measuring root mean squared error (RMSE) with MLflow,
   including model registry functionality and metadata tracking, without using DMatrix."""

import sys
import subprocess
import os
import json
import logging
import pathlib
import pickle
import tarfile
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

subprocess.check_call([
    sys.executable, "-m", "pip", "install", 
    "mlflow==2.16.2",
    "sagemaker-mlflow",
])

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.info("Starting evaluation.")
    model_path = "/opt/ml/processing/model/model.tar.gz"

    # Retrieve MLflow configuration from environment variables
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
    tracking_server_arn = os.getenv("MLFLOW_TRACKING_SERVER_ARN")
    run_id = os.getenv("MLFLOW_RUN_ID")  # if needed elsewhere
    model_package_group_name = os.getenv("MODEL_PACKAGE_GROUP_NAME") 

    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri(tracking_server_arn)
    mlflow.set_experiment(experiment_name)

    # Create a timestamp for naming the run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize MLflow client for model registry operations
    client = MlflowClient()

    # Check if the registered model exists; if not, create it with tags
    try:
        client.get_registered_model(model_package_group_name)
        logger.info(f"Registered model '{model_package_group_name}' already exists.")
    except Exception as e:
        client.create_registered_model(model_package_group_name, tags={"framework": "xgboost"})
        logger.info(f"Created new registered model '{model_package_group_name}' with tag framework=xgboost.")

    # Start a new MLflow run
    with mlflow.start_run(run_id=run_id) as parent_run:
        with mlflow.start_run(run_name="ModelEvaluation", nested=True) as run:
            # Extract the model from the tar file
            with tarfile.open(model_path) as tar:
                tar.extractall(path="")

            logger.info("Loading xgboost model.")
            with open("xgboost-model", "rb") as f:
                model = pickle.load(f)

            logger.info("Reading test data.")
            test_path = "/opt/ml/processing/test/test.csv"
            df = pd.read_csv(test_path, header=None)
            y_test = df.iloc[:, 0]
            X_test = df.iloc[:, 1:]

            logger.info("Performing predictions on test data.")
            # Use the native predict method from the scikit-learn API
            predictions = model.predict(X_test)

            logger.info("Calculating root mean squared error (RMSE).")
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            rmse_std = np.std(y_test - predictions)

            # Log evaluation metrics and additional metadata tags to MLflow
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("rmse_std", rmse_std)
            mlflow.set_tag("model_framework", "xgboost")
            mlflow.set_tag("evaluation_timestamp", timestamp)

            # Log model hyperparameters
            hyperparameters = model.get_params()
            mlflow.log_params(hyperparameters)

            # Log the XGBoost model and register it under the experiment name in MLflow Model Registry.
            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                registered_model_name=model_package_group_name,
            )

            # Prepare the evaluation report
            report_dict = {
                "regression_metrics": {
                    "rmse": {
                        "value": rmse,
                        "standard_deviation": rmse_std
                    }
                }
            }

            output_dir = "/opt/ml/processing/evaluation"
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

            evaluation_path = os.path.join(output_dir, "evaluation.json")
            with open(evaluation_path, "w") as f:
                f.write(json.dumps(report_dict))

            logger.info("Evaluation report written with RMSE: %f", rmse)
