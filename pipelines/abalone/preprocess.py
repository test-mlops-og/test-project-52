"""Feature engineers the abalone dataset."""

import sys
import subprocess

subprocess.check_call([
    sys.executable, "-m", "pip", "install", 
    "mlflow==2.16.2",
    "sagemaker-mlflow==0.1.0",
])

import argparse
import logging
import os
import pathlib
import requests
import tempfile
import time
import json

import boto3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import mlflow

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


# Since we get a headerless CSV file we specify the column names here.
feature_columns_names = [
    "sex",
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
]
label_column = "rings"

feature_columns_dtype = {
    "sex": str,
    "length": np.float64,
    "diameter": np.float64,
    "height": np.float64,
    "whole_weight": np.float64,
    "shucked_weight": np.float64,
    "viscera_weight": np.float64,
    "shell_weight": np.float64,
}
label_column_dtype = {"rings": np.float64}


def merge_two_dicts(x, y):
    """Merges two dicts, returning a new copy."""
    z = x.copy()
    z.update(y)
    return z


if __name__ == "__main__":

    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--tracking-server-arn", type=str, required=True)
    args = parser.parse_args()
    input_data = args.input_data
    experiment_name = args.experiment_name
    tracking_server_arn = args.tracking_server_arn
    mlflow.set_tracking_uri(tracking_server_arn)
    mlflow.set_experiment(experiment_name)
    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{base_dir}/parent_run_id").mkdir(parents=True, exist_ok=True)

    # Create the parent run once and log all processing within it.
    with mlflow.start_run() as parent_run:
        parent_run_id = parent_run.info.run_id
        with open(f"{base_dir}/parent_run_id/parent_run_id.txt", "w") as f:
            json.dump({"parent_run_id": parent_run_id}, f)
        with mlflow.start_run(run_name="DataPreprocessing", nested=True):

            bucket = input_data.split("/")[2]
            key = "/".join(input_data.split("/")[3:])

            logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
            fn = f"{base_dir}/data/abalone-dataset.csv"
            s3 = boto3.resource("s3")
            s3.Bucket(bucket).download_file(key, fn)

            logger.debug("Reading downloaded data.")
            df = pd.read_csv(
                fn,
                header=None,
                names=feature_columns_names + [label_column],
                dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype),
            )
            os.unlink(fn)
            # logo data versioning
            mlflow.log_input(
                mlflow.data.from_pandas(df, input_data, targets=label_column),
                context="DataPreprocessing",
            )
            logger.debug("Defining transformers.")
            numeric_features = list(feature_columns_names)
            numeric_features.remove("sex")
            numeric_transformer = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
            )

            categorical_features = ["sex"]
            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            preprocess = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_features),
                    ("cat", categorical_transformer, categorical_features),
                ]
            )

            logger.info("Applying transforms.")
            y = df.pop("rings")
            X_pre = preprocess.fit_transform(df)
            y_pre = y.to_numpy().reshape(len(y), 1)

            X = np.concatenate((y_pre, X_pre), axis=1)

            # Log preprocessed data summary
            mlflow.log_metric("num_rows", len(X))
            mlflow.log_metric("num_features", X.shape[1])

            logger.info("Splitting %d rows of data into train, validation, test datasets.", len(X))
            np.random.shuffle(X)
            train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

            logger.info("Writing out datasets to %s.", base_dir)
            pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
            pd.DataFrame(validation).to_csv(
                f"{base_dir}/validation/validation.csv", header=False, index=False
            )
            pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)

            # Log dataset versioning
            mlflow.log_param("data_split", {"train": 0.7, "validation": 0.15, "test": 0.15})
            mlflow.log_param("transformations", {
                "numeric_imputer": "median",
                "categorical_imputer": "constant_missing",
                "scaler": "StandardScaler",
                "onehot_encoder": "ignore_unknown"
            })