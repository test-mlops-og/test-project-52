import sys
import os
import time
import tarfile
import json
import argparse
import numpy as np
import boto3
import mlflow
import xgboost as xgb
from mlflow import MlflowClient

def convert_to_csv(data: np.ndarray) -> str:
    """
    Convert a numpy array to a CSV string.
    """
    return ",".join(map(str, data.flatten().tolist()))

def update_existing_endpoint(args, new_model_version, sklearn_input):
    """
    Updates an existing SageMaker endpoint with a new single model version.
    This function creates a new model and endpoint configuration, then updates the
    specified endpoint to use the new configuration.
    """
    print(f"Updating endpoint '{args.endpoint_name}' with model version '{new_model_version.version}'...")
    timestamp = int(time.time())
    new_model_name = f"{args.model_package_group_name}-v{new_model_version.version}-update-{timestamp}"

    # Initialize SageMaker and S3 clients for the specified region
    sagemaker_client = boto3.client("sagemaker", region_name=args.region)
    s3 = boto3.client("s3", region_name=args.region)
    
    # Load the new model from MLflow and save it locally
    model_uri = new_model_version.source
    model = mlflow.xgboost.load_model(model_uri)
    output_dir = f"model_version_update_{new_model_version.version}"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "xgboost-model.bst")
    model.get_booster().save_model(model_path)
    
    # Package the model into a tar.gz file
    model_tar_file = f"model_v{new_model_version.version}_update.tar.gz"
    with tarfile.open(model_tar_file, "w:gz") as tar:
        tar.add(model_path, arcname="xgboost-model.bst")
    
    # Upload the packaged model to S3
    s3_prefix = f"{args.model_package_group_name}/update/v{new_model_version.version}"
    s3_uri = f"s3://{args.default_bucket}/{s3_prefix}/{model_tar_file}"
    s3.upload_file(model_tar_file, args.default_bucket, f"{s3_prefix}/{model_tar_file}")
    print(f"Uploaded updated model artifact to: {s3_uri}")
    
    # Create a new model in SageMaker
    sagemaker_client.create_model(
        ModelName=new_model_name,
        ExecutionRoleArn=args.role,
        PrimaryContainer={
            "Image": "341280168497.dkr.ecr.ca-central-1.amazonaws.com/sagemaker-xgboost:1.7-1",
            "ModelDataUrl": s3_uri
        }
    )
    
    # Create a new endpoint configuration with a single production variant (all traffic)
    new_endpoint_config_name = f"{args.model_package_group_name}-update-config-{timestamp}"
    production_variant = [{
         "VariantName": "AllTraffic",
         "ModelName": new_model_name,
         "InstanceType": args.instance_type,
         "InitialInstanceCount": 1,
         "InitialVariantWeight": 1
    }]
    
    sagemaker_client.create_endpoint_config(
        EndpointConfigName=new_endpoint_config_name,
        ProductionVariants=production_variant,
        DataCaptureConfig={
            "EnableCapture": True,
            "InitialSamplingPercentage": 100,
            "DestinationS3Uri": f"s3://{args.default_bucket}/{args.endpoint_name}/data-capture",
            "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
            "CaptureContentTypeHeader": {
                "CsvContentTypes": ["text/csv"],
                "JsonContentTypes": ["application/json"]
            }
        }
    )
    
    # Update the existing endpoint with the new configuration
    sagemaker_client.update_endpoint(
        EndpointName=args.endpoint_name,
        EndpointConfigName=new_endpoint_config_name
    )
    print(f"Endpoint '{args.endpoint_name}' is being updated with model version '{new_model_version.version}'.")
    
    # Test the updated endpoint with sample input
    runtime = boto3.client("sagemaker-runtime", region_name=args.region)
    csv_payload = convert_to_csv(sklearn_input)
    response = runtime.invoke_endpoint(
        EndpointName=args.endpoint_name,
        ContentType="text/csv",
        Body=csv_payload.encode("utf-8")
    )
    print(f"Updated endpoint inference result: {response['Body'].read().decode('utf-8')}")

def main():
    parser = argparse.ArgumentParser(
        description="Update an existing SageMaker endpoint with a new single model version."
    )
    parser.add_argument("--region", type=str, required=True, help="AWS region (e.g., ca-central-1)")
    parser.add_argument("--role", type=str, required=True, help="IAM role ARN for SageMaker")
    parser.add_argument("--tracking_server_arn", type=str, required=True, help="MLflow tracking server ARN")
    parser.add_argument("--default_bucket", type=str, required=True, help="Default S3 bucket name")
    parser.add_argument("--model_package_group_name", type=str, required=True, help="Model package group name")
    parser.add_argument("--endpoint_name", type=str, required=True, help="Existing endpoint name to update")
    parser.add_argument("--instance_type", type=str, default="ml.m5.xlarge", help="Instance type for the endpoint")
    parser.add_argument("--version", type=str, help="Optional model version to update the endpoint with. If not provided, the latest version is used.")
    args = parser.parse_args()
    
    # Set up MLflow tracking using the provided tracking server ARN
    mlflow.set_tracking_uri(args.tracking_server_arn)
    client = MlflowClient()
    
    # Retrieve model versions for the given model package group name
    versions = client.search_model_versions(f"name='{args.model_package_group_name}'")
    if not versions:
        print("No model versions found for update.")
        sys.exit(1)
    
    # If a version is provided, try to find that version; otherwise use the latest version
    if args.version:
        filtered_versions = [v for v in versions if v.version == args.version]
        if not filtered_versions:
            print(f"Specified model version {args.version} not found.")
            sys.exit(1)
        new_model_version = filtered_versions[0]
    else:
        new_model_version = sorted(versions, key=lambda x: int(x.version))[-1]
    
    # Create a sample NumPy array input for inference testing
    sklearn_input = np.array([
        -0.6161975481284616,
        -0.6840938444613145,
        -0.4666533461032037,
        -0.6857462206781514,
        -0.5333397337170492,
        -0.6669056147788445,
        -0.8537557997803805,
        0,
        1,
        0
    ]).reshape(1, -1)
    
    update_existing_endpoint(args, new_model_version, sklearn_input)

if __name__ == '__main__':
    main()
