import sys
import os
import time
import tarfile
import json
import numpy as np
import yaml
import boto3
import mlflow
import xgboost as xgb
from mlflow import MlflowClient
from sagemaker.serve import SchemaBuilder
from sagemaker.model_monitor import DefaultModelMonitor

# Glue job boilerplate imports
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from pyspark.context import SparkContext
from awsglue.job import Job

from datetime import datetime

# Parse Glue job arguments (e.g., JOB_NAME and CONFIG_PATH)
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'config_path'])
sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

def load_config(config_path):
    """
    Loads configuration from a local file or from S3 if the path starts with 's3://'
    """
    if config_path.startswith("s3://"):
        # Parse bucket and key from the S3 URI
        s3_parts = config_path.replace("s3://", "").split("/", 1)
        bucket = s3_parts[0]
        key = s3_parts[1]
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        config_str = obj['Body'].read().decode("utf-8")
        config = yaml.safe_load(config_str)
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    return config

def enable_data_capture_config(endpoint_name, bucket, region):
    """
    Enables data capture for an existing SageMaker endpoint.
    Captured data is stored in the given S3 bucket under a dedicated prefix.
    """
    sagemaker_client = boto3.client("sagemaker", region_name=region)
    capture_config = {
        "EnableCapture": True,
        "InitialSamplingPercentage": 100,
        "DestinationS3Uri": f"s3://{bucket}/{endpoint_name}/data-capture",
        "CaptureOptions": [
            {"CaptureMode": "Input"},
            {"CaptureMode": "Output"}
        ],
        "CaptureContentTypeHeader": {
            "CsvContentTypes": ["text/csv"],
            "JsonContentTypes": ["application/json"]
        }
    }
    sagemaker_client.update_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_name,
        DataCaptureConfig=capture_config
    )
    print(f"Data capture enabled for endpoint {endpoint_name} at s3://{bucket}/{endpoint_name}/data-capture")

def setup_monitoring(endpoint_name, config, role, instance_type):
    """
    Sets up a SageMaker Model Monitor schedule for data drift and model performance monitoring.
    Uses training data (or a specified baseline dataset) to generate baseline statistics.
    """
    monitoring_config = config.get("monitoring", {})
    if not monitoring_config.get("enabled", False):
        print("Monitoring not enabled. Skipping monitor setup.")
        return

    baseline_data_path = monitoring_config.get("baseline_data_path", config.get("train_data_path"))
    output_bucket = monitoring_config.get("output_bucket", config.get("default_bucket"))
    schedule_expression = monitoring_config.get("schedule_expression", "cron(0 12 * * ? *)")
    
    monitor = DefaultModelMonitor(
         role=role,
         instance_count=1,
         instance_type=instance_type,
         volume_size_in_gb=20,
         max_runtime_in_seconds=3600
    )
    
    print("Generating baseline statistics using training data as baseline...")
    monitor.suggest_baseline(
         baseline_dataset=baseline_data_path,
         dataset_format={"csv": {"header": False}},
         output_s3_uri=f"s3://{output_bucket}/monitoring/baseline"
    )
    
    monitor.create_monitoring_schedule(
         endpoint_input=endpoint_name,
         schedule_cron_expression=schedule_expression,
         monitor_schedule_name=f"{endpoint_name}-monitoring-schedule"
    )
    print(f"Monitoring schedule created for endpoint: {endpoint_name}")

def convert_to_csv(data: np.ndarray) -> str:
    """
    Convert a numpy array to a CSV string.
    """
    return ",".join(map(str, data.flatten().tolist()))

def create_or_update_endpoint(sagemaker_client, endpoint_name, endpoint_config_name):
    """
    Checks whether an endpoint exists. If it exists, update the endpoint with the new configuration.
    Otherwise, create a new endpoint.
    """
    try:
        # Try to retrieve the endpoint description.
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        print(f"Endpoint {endpoint_name} exists. Updating endpoint configuration to {endpoint_config_name}...")
        sagemaker_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
    except sagemaker_client.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code in ["ValidationException", "ResourceNotFoundException", "NotFound"]:
            print(f"Endpoint {endpoint_name} does not exist. Creating new endpoint with configuration {endpoint_config_name}...")
            sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
        else:
            raise

def create_model_if_not_exists(sagemaker_client, model_name, role, s3_uri, image):
    """
    Checks whether a SageMaker model exists. If not, create it.
    Since models are immutable once created, if a model with the name exists, we reuse it.
    """
    try:
        sagemaker_client.describe_model(ModelName=model_name)
        print(f"Model {model_name} already exists, skipping creation.")
    except sagemaker_client.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code in ["ValidationException", "ResourceNotFoundException", "NotFound"]:
            sagemaker_client.create_model(
                ModelName=model_name,
                ExecutionRoleArn=role,
                PrimaryContainer={
                    "Image": image,
                    "ModelDataUrl": s3_uri
                }
            )
            print(f"Created SageMaker model: {model_name}")
        else:
            raise

def deploy_single_model(config, env, sklearn_schema_builder, model_version, instance_type, sklearn_input, configured_endpoint_name):
    """
    Deploy a single model to a SageMaker endpoint using the configured endpoint name,
    following a similar method to deploy_multi_variant and deploy_shadow_variant.
    """
    print(f"Deploying Single Model for environment: {env}")
    region = config.get("region")
    bucket = config.get("default_bucket")
    role = config.get("role")
    model_package_group_name = config['model_package_group_name']
    image = "341280168497.dkr.ecr.ca-central-1.amazonaws.com/sagemaker-xgboost:1.7-1"

    # Use the configured endpoint name from the YAML configuration.
    endpoint_name = configured_endpoint_name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    endpoint_config_name = f"{endpoint_name}-single-{env}-config-{timestamp}"

    # Load the model using mlflow
    model_uri = model_version.source
    model = mlflow.xgboost.load_model(model_uri)
    
    # Save the model artifact locally
    output_dir = f"model_version_{model_version.version}"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "xgboost-model.bst")
    model.get_booster().save_model(model_path)
    
    # Package the model artifact
    model_tar_file = f"model_v{model_version.version}.tar.gz"
    with tarfile.open(model_tar_file, "w:gz") as tar:
        tar.add(model_path, arcname="xgboost-model.bst")
    
    # Upload the model artifact to S3
    s3 = boto3.client('s3')
    s3_prefix = f"{model_package_group_name}/{env}/v{model_version.version}"
    s3_uri = f"s3://{bucket}/{s3_prefix}/{model_tar_file}"
    s3.upload_file(model_tar_file, bucket, f"{s3_prefix}/{model_tar_file}")
    print(f"Uploaded model artifact to: {s3_uri}")

    # Create a unique model name for the deployment
    model_name = f"{model_package_group_name}-v{model_version.version}-{env}"

    # Create the SageMaker model if it does not exist.
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    create_model_if_not_exists(sagemaker_client, model_name, role, s3_uri, image)

    # Create the endpoint configuration with a single production variant
    production_variant = {
        "VariantName": "AllTraffic",
        "ModelName": model_name,
        "InstanceType": instance_type,
        "InitialInstanceCount": 1,
        "InitialVariantWeight": 1
    }
    sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[production_variant],
        DataCaptureConfig={
            "EnableCapture": True,
            "InitialSamplingPercentage": 100,
            "DestinationS3Uri": f"s3://{bucket}/{endpoint_name}/data-capture",
            "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
            "CaptureContentTypeHeader": {
                "CsvContentTypes": ["text/csv"],
                "JsonContentTypes": ["application/json"]
            }
        }
    )
    print(f"Created endpoint configuration: {endpoint_config_name}")

    # Create or update the endpoint using the helper function
    create_or_update_endpoint(sagemaker_client, endpoint_name, endpoint_config_name)
    print(f"Deploying endpoint: {endpoint_name}")

    # Wait for the endpoint to become InService
    elapsed_time = 0
    timeout = 600
    while True:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]
        print(f"Endpoint status: {status}")
        if status == "InService":
            print(f"Endpoint {endpoint_name} is InService.")
            break
        elif status == "Failed":
            raise Exception(f"Endpoint creation failed: {response['FailureReason']}")
        time.sleep(30)
        elapsed_time += 30
        print(f"Elapsed time: {elapsed_time}/{timeout} seconds")
        if elapsed_time > timeout:
            raise TimeoutError(f"Endpoint {endpoint_name} did not reach 'InService' status within {timeout} seconds.")

    # Test inference on the deployed endpoint
    runtime = boto3.client("sagemaker-runtime")
    csv_payload = ",".join(map(str, sklearn_input.flatten().tolist()))
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="text/csv",
        Body=csv_payload.encode("utf-8")
    )
    result = response['Body'].read().decode("utf-8")
    print(f"Endpoint {endpoint_name} inference result: {result}")

    # Setup monitoring if enabled
    setup_monitoring(endpoint_name, config, role, instance_type)
    print(f"Deployed Single Model endpoint for {env} environment.")

def deploy_multi_variant(config, env, sklearn_schema_builder, versions, instance_type, sklearn_input):
    """
    Deploy multiple model variants to a single SageMaker endpoint.
    """
    s3 = boto3.client('s3')
    region = config.get("region")
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    print(f"Deploying Multi-Variant for environment: {env}")
    weights = config.get("deployment", {}).get("variant_weights", [1/len(versions)] * len(versions))
    bucket = config.get("default_bucket")
    model_variants = []
    role = config.get("role")
    model_package_group_name = config['model_package_group_name']
    image = "341280168497.dkr.ecr.ca-central-1.amazonaws.com/sagemaker-xgboost:1.7-1"
    timestamp = int(time.time())
    
    # For multi-variant mode, generate a new endpoint name since a shadow or single mode name is not used.
    endpoint_config_name = f"{model_package_group_name}-multi-config-{env}-{timestamp}"
    endpoint_name = f"{model_package_group_name}-multi-{env}-{timestamp}"
    variant_names = [f"Modelv{version.version}" for version in versions]

    for i, version in enumerate(versions):
        print(f"Processing model version: {version.version}")
        model_uri = version.source
        model = mlflow.xgboost.load_model(model_uri)
        output_dir = f"model_version_{version.version}"
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "xgboost-model.bst")
        model.get_booster().save_model(model_path)
        model_tar_file = f"model_v{version.version}.tar.gz"
        with tarfile.open(model_tar_file, "w:gz") as tar:
            tar.add(model_path, arcname="xgboost-model.bst")
        s3_prefix = f"{model_package_group_name}/{env}/v{version.version}"
        s3_uri = f"s3://{bucket}/{s3_prefix}/{model_tar_file}"
        s3.upload_file(model_tar_file, bucket, f"{s3_prefix}/{model_tar_file}")
        print(f"Uploaded model artifact to: {s3_uri}")
        
        model_name = f"{model_package_group_name}-v{version.version}-{env}-{timestamp}"
        create_model_if_not_exists(sagemaker_client, model_name, role, s3_uri, image)
        model_variants.append({
            "VariantName": f"Modelv{version.version}",
            "ModelName": model_name,
            "InstanceType": instance_type,
            "InitialInstanceCount": 1,
            "InitialVariantWeight": weights[i]
        })

    sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=model_variants,
        DataCaptureConfig={
            "EnableCapture": True,
            "InitialSamplingPercentage": 100,
            "DestinationS3Uri": f"s3://{bucket}/{endpoint_name}/data-capture",
            "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
            "CaptureContentTypeHeader": {
                "CsvContentTypes": ["text/csv"],
                "JsonContentTypes": ["application/json"]
            }
        }
    )

    print("Deploying multi-variant endpoint...")
    create_or_update_endpoint(sagemaker_client, endpoint_name, endpoint_config_name)

    print(f"Waiting for endpoint {endpoint_name} to be InService...")
    elapsed_time = 0
    timeout = 600
    while True:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]
        print(status)
        if status == "InService":
            print(f"Endpoint {endpoint_name} is InService.")
            break
        elif status == "Failed":
            print(f"Endpoint creation failed: {response['FailureReason']}")
            raise Exception("Endpoint creation failed.")
        time.sleep(30)
        elapsed_time += 30
        print(f"Elapsed time: {elapsed_time}/{timeout} seconds...")
        if elapsed_time > timeout:
            raise TimeoutError(f"Endpoint {endpoint_name} did not reach 'InService' status within {timeout} seconds.")

    print(f"Deployed Multi-Variant endpoint: {endpoint_name}")
    print("Testing inference...")
    runtime = boto3.client("sagemaker-runtime")
    csv_payload = convert_to_csv(sklearn_input)
    for variant in variant_names:
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="text/csv",
            TargetVariant=variant,
            Body=csv_payload.encode("utf-8")
        )
        print(f"Model {variant} inference result: {response['Body'].read().decode('utf-8')}")
    setup_monitoring(endpoint_name, config, role, instance_type)
    print(f"Deployed Multi-Variant endpoint for {env} environment.")

def deploy_shadow_variant(config, env, sklearn_schema_builder, versions, instance_type, sklearn_input, configured_endpoint_name):
    """
    Deploys a shadow variant endpoint by designating one model as the primary variant 
    and another as a shadow variant, using the configured endpoint name.
    """
    s3 = boto3.client('s3')
    region = config.get("region")
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    print(f"Deploying Shadow Variant for environment: {env}")
    bucket = config.get("default_bucket")
    model_variants = []
    role = config.get("role")
    model_package_group_name = config['model_package_group_name']
    image = "341280168497.dkr.ecr.ca-central-1.amazonaws.com/sagemaker-xgboost:1.7-1"
    
    # Use the endpoint name defined in the YAML configuration.
    endpoint_name = configured_endpoint_name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    endpoint_config_name = f"{endpoint_name}-shadow-{env}-config-{timestamp}"

    for i, version in enumerate(versions):
        print(f"Processing model version: {version.version}")
        model_uri = version.source
        model = mlflow.xgboost.load_model(model_uri)
        output_dir = f"model_version_{version.version}"
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "xgboost-model.bst")
        model.get_booster().save_model(model_path)
        model_tar_file = f"model_v{version.version}.tar.gz"
        with tarfile.open(model_tar_file, "w:gz") as tar:
            tar.add(model_path, arcname="xgboost-model.bst")
        s3_prefix = f"{model_package_group_name}/{env}/v{version.version}"
        s3_uri = f"s3://{bucket}/{s3_prefix}/{model_tar_file}"
        s3.upload_file(model_tar_file, bucket, f"{s3_prefix}/{model_tar_file}")
        print(f"Uploaded model artifact to: {s3_uri}")
        
        model_name = f"{model_package_group_name}-v{version.version}-{env}"
        create_model_if_not_exists(sagemaker_client, model_name, role, s3_uri, image)
        model_variants.append({
            "VariantName": f"Modelv{version.version}",
            "ModelName": model_name,
            "InstanceType": instance_type,
            "InitialInstanceCount": 1,
            "InitialVariantWeight": 1
        })

    # In this shadow-variant setup, we assume that the first variant is the shadow and the second is primary.
    primary_variant = [model_variants[1]]
    shadow_variant = [model_variants[0]]

    sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=primary_variant,
        ShadowProductionVariants=shadow_variant,
        DataCaptureConfig={
            "EnableCapture": True,
            "InitialSamplingPercentage": 100,
            "DestinationS3Uri": f"s3://{bucket}/{endpoint_name}/data-capture",
            "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
            "CaptureContentTypeHeader": {
                "CsvContentTypes": ["text/csv"],
                "JsonContentTypes": ["application/json"]
            }
        }
    )

    print("Deploying shadow variant endpoint...")
    create_or_update_endpoint(sagemaker_client, endpoint_name, endpoint_config_name)

    print(f"Waiting for endpoint {endpoint_name} to be InService...")
    elapsed_time = 0
    timeout = 600
    while True:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]
        print(status)
        if status == "InService":
            print(f"Endpoint {endpoint_name} is InService.")
            break
        elif status == "Failed":
            print(f"Endpoint creation failed: {response['FailureReason']}")
            raise Exception("Endpoint creation failed.")
        time.sleep(30)
        elapsed_time += 30
        print(f"Elapsed time: {elapsed_time}/{timeout} seconds...")
        if elapsed_time > timeout:
            raise TimeoutError(f"Endpoint {endpoint_name} did not reach 'InService' status within {timeout} seconds.")

    print(f"Deployed Shadow Variant endpoint: {endpoint_name}")
    print("Testing inference...")
    runtime = boto3.client("sagemaker-runtime")
    csv_payload = convert_to_csv(sklearn_input)
    response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="text/csv",
            Body=csv_payload.encode("utf-8")
    )
    print(f"Endpoint {endpoint_name} inference result: {response['Body'].read().decode('utf-8')}")
    setup_monitoring(endpoint_name, config, role, instance_type)
    print(f"Deployed Shadow Variant endpoint for {env} environment.")

def main():
    # Load configuration file (local or S3)
    config_file = args.get("config_path")
    config = load_config(config_file)

    tracking_server_arn = config.get("tracking_server_arn")
    mlflow.set_tracking_uri(tracking_server_arn)
    client = MlflowClient()

    deployment_config = config.get("deployment", {})
    environments = deployment_config.get("environments", [])
    instance_type = deployment_config.get("instance_type", "ml.m5.xlarge")
    mode = deployment_config.get("mode", "single")
    # Get the configured endpoint name from the YAML file
    configured_endpoint_name = deployment_config.get("endpoint_name")
    # Get the version numbers from the YAML file (e.g., [3]) and filter the returned versions accordingly.
    config_versions = deployment_config.get("versions", [])
    versions = client.search_model_versions(f"name='{config['model_package_group_name']}'")
    if not versions:
        raise ValueError("No model versions found.")
    
    selected_versions = [v for v in versions if int(v.version) in config_versions]
    if not selected_versions:
        raise ValueError("No model versions matching the deploy config versions were found.")

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
    sklearn_output = 1

    for env in environments:
        sklearn_schema_builder = SchemaBuilder(
            sample_input=sklearn_input,
            sample_output=sklearn_output,
        )
        if mode == "single":
            deploy_single_model(config, env, sklearn_schema_builder, selected_versions[0], instance_type, sklearn_input, configured_endpoint_name)
        elif mode == "multi-variant":
            variant_weights = deployment_config.get("variant_weights")
            if len(selected_versions) < len(variant_weights):
                raise ValueError("Insufficient model versions for the specified variant weights.")
            deploy_multi_variant(config, env, sklearn_schema_builder, selected_versions[:len(variant_weights)], instance_type, sklearn_input)
        elif mode == "shadow-variant":
            # For shadow deployment, use the first two matching versions and the configured endpoint name.
            deploy_shadow_variant(config, env, sklearn_schema_builder, selected_versions[:2], instance_type, sklearn_input, configured_endpoint_name)
        else:
            print(f"Invalid mode: {mode}. Please use 'single', 'multi-variant', or 'shadow-variant'.")

if __name__ == '__main__':
    main()
    job.commit()
