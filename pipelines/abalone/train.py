"""Trains an XGBoost model using provided hyperparameters without using DMatrix."""

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
import pandas as pd
import pickle as pkl
import xgboost as xgb
import mlflow
import mlflow.xgboost

from sklearn.metrics import mean_squared_error, r2_score

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.info("Starting training script.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.0, help="L1 regularization term")
    parser.add_argument("--lambda", type=float, dest="lambda_param", default=1.0, help="L2 regularization term")
    parser.add_argument("--num_round", type=int, required=True, help="Number of boosting rounds (n_estimators)")
    parser.add_argument("--max_depth", type=int, required=True, help="Maximum tree depth")
    parser.add_argument("--eta", type=float, required=True, help="Learning rate")
    parser.add_argument("--gamma", type=float, required=True, help="Minimum loss reduction")
    parser.add_argument("--min_child_weight", type=float, required=True, help="Minimum sum of child weights")
    parser.add_argument("--subsample", type=float, required=True, help="Subsample ratio")
    parser.add_argument("--objective", type=str, required=True, help="Objective function")

    args = parser.parse_args()

    # Load data
    train_df = pd.read_csv(os.path.join("/opt/ml/input/data/train/train.csv"), header=None)
    validation_df = pd.read_csv(os.path.join("/opt/ml/input/data/validation/validation.csv"), header=None)

    y_train = train_df.iloc[:, 0]
    X_train = train_df.iloc[:, 1:]
    y_validation = validation_df.iloc[:, 0]
    X_validation = validation_df.iloc[:, 1:]

    logger.info("Data loaded. Starting training.")

    # Configure MLflow
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
    tracking_server_arn = os.getenv("MLFLOW_TRACKING_SERVER_ARN")
    run_id = os.getenv("MLFLOW_RUN_ID")

    mlflow.set_tracking_uri(tracking_server_arn)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        with mlflow.start_run(run_name="ModelTraining", nested=True) as training_run:
            mlflow.xgboost.autolog(
                log_input_examples=True,
                log_model_signatures=True,
                log_models=True,
                log_datasets=True,
                model_format="xgb",
            )

            # Create and train the XGBoost regressor using scikit-learn API
            model = xgb.XGBRegressor(
                objective=args.objective,
                max_depth=args.max_depth,
                learning_rate=args.eta,
                gamma=args.gamma,
                min_child_weight=args.min_child_weight,
                subsample=args.subsample,
                reg_lambda=args.lambda_param,
                reg_alpha=args.alpha,
                n_estimators=args.num_round,
                eval_metric="rmse"
            )

            # Fit the model and use validation data for evaluation
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_validation, y_validation)],
                verbose=True
            )

            # Optionally, you could calculate and log additional metrics
            y_pred = model.predict(X_validation)
            rmse = mean_squared_error(y_validation, y_pred, squared=False)
            r2 = r2_score(y_validation, y_pred)
            logger.info(f"Validation RMSE: {rmse:.4f}")
            logger.info(f"Validation R2: {r2:.4f}")

            # Save the trained model
            model_location = os.path.join("/opt/ml/model", "xgboost-model")
            with open(model_location, "wb") as f:
                pkl.dump(model, f)
            logger.info("Stored trained model at {}".format(model_location))

            logger.info("Model training complete.")
