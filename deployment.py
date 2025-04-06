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
from sagemaker.serve import SchemaBuilder, ModelBuilder
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.model_monitor import DefaultModelMonitor

# Glue job boilerplate imports
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from pyspark.context import SparkContext
from awsglue.job import Job

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

def deploy_single_model(config, env, sklearn_schema_builder, model_version, instance_type, sklearn_input):
    """
    Deploy a single model to a SageMaker endpoint.
    """
    print(f"Deploying Single Model for environment: {env}")

    model_name = f"{config['model_package_group_name']}-v{model_version.version}-{env}"
    role = config.get("role")
    region = config.get("region")
    bucket = config.get("default_bucket")

    model_builder = ModelBuilder(
        name=model_name,
        mode=Mode.SAGEMAKER_ENDPOINT,
        schema_builder=sklearn_schema_builder,
        role_arn=role,
        model_metadata={"MLFLOW_MODEL_PATH": model_version.source},
    )

    built_model = model_builder.build()

    predictor = built_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        data_capture_config={
            "EnableCapture": True,
            "InitialSamplingPercentage": 100,
            "DestinationS3Uri": f"s3://{bucket}/{model_name}/data-capture",
            "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}]
        }
    )

    print(f"Deployed Single Model endpoint: {model_name}")
    result = predictor.predict(sklearn_input)
    print(result)

    setup_monitoring(model_name, config, role, instance_type)
    return predictor

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
    timestamp = int(time.time())
    endpoint_config_name = f"{model_package_group_name}-multi-config-{env}-{timestamp}"
    endpoint_name = f"{model_package_group_name}-multi-{env}-{timestamp}"
    variant_names = [f"Modelv{version.version}" for version in versions]

    model_names = []
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
        model_names.append(model_name)
        sagemaker_client.create_model(
            ModelName=model_name,
            ExecutionRoleArn=role,
            PrimaryContainer={
                "Image": "341280168497.dkr.ecr.ca-central-1.amazonaws.com/sagemaker-xgboost:1.7-1",
                "ModelDataUrl": s3_uri
            }
        )
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
    sagemaker_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )

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

def deploy_shadow_variant(config, env, sklearn_schema_builder, versions, instance_type, sklearn_input):
    """
    Deploys a shadow variant endpoint by designating one model as the primary variant 
    and another as a shadow variant.
    """
    s3 = boto3.client('s3')
    region = config.get("region")
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    print(f"Deploying Shadow Variant for environment: {env}")
    bucket = config.get("default_bucket")
    model_variants = []
    role = config.get("role")
    model_package_group_name = config['model_package_group_name']
    timestamp = int(time.time())
    endpoint_config_name = f"{model_package_group_name}-shadow-config-{env}-{timestamp}"
    endpoint_name = f"{model_package_group_name}-shadow-{env}-{timestamp}"
    variant_names = [f"Modelv{version.version}" for version in versions]

    model_names = []
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
        model_names.append(model_name)
        sagemaker_client.create_model(
            ModelName=model_name,
            ExecutionRoleArn=role,
            PrimaryContainer={
                "Image": "341280168497.dkr.ecr.ca-central-1.amazonaws.com/sagemaker-xgboost:1.7-1",
                "ModelDataUrl": s3_uri
            }
        )
        model_variants.append({
            "VariantName": f"Modelv{version.version}",
            "ModelName": model_name,
            "InstanceType": instance_type,
            "InitialInstanceCount": 1,
            "InitialVariantWeight": 1
        })

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
    sagemaker_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )

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
    time.sleep(300)
    shadow_variant_name = model_variants[0]['VariantName']
    endpoint_capture_prefix = f"{endpoint_name}/data-capture/{endpoint_name}/{shadow_variant_name}"
    result = s3.list_objects(Bucket=bucket, Prefix=endpoint_capture_prefix)
    if "Contents" in result:
        shadow_var_capture_files = [capture_file.get("Key") for capture_file in result.get("Contents")]
        def get_obj_body(obj_key):
            return s3.get_object(Bucket=bucket, Key=obj_key).get('Body').read().decode("utf-8")
        shadow_var_capture_file = get_obj_body(shadow_var_capture_files[-1])
        try:
            capture_json = json.loads(shadow_var_capture_file.split('\n')[0])
            print("Shadow Variant Capture Data:")
            print(json.dumps(capture_json, indent=2))
        except Exception as e:
            print("Error parsing shadow capture file:", e)
    else:
        print("No shadow capture files found at the specified S3 prefix.")
    setup_monitoring(endpoint_name, config, role, instance_type)
    print(f"Deployed Shadow Variant endpoint for {env} environment.")

def main():
    # The configuration file is loaded from the S3 URI passed in as CONFIG_PATH
    config_file = args.get("config_path")
    config = load_config(config_file)

    tracking_server_arn = config.get("tracking_server_arn")
    mlflow.set_tracking_uri(tracking_server_arn)
    client = MlflowClient()

    deployment_config = config.get("deployment", {})
    environments = deployment_config.get("environments", [])
    instance_type = deployment_config.get("instance_type", "ml.m5.xlarge")
    mode = deployment_config.get("mode", "single")

    versions = client.search_model_versions(f"name='{config['model_package_group_name']}'")
    if not versions:
        raise ValueError("No model versions found.")

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
            deploy_single_model(config, env, sklearn_schema_builder, versions[0], instance_type, sklearn_input)
        elif mode == "multi-variant":
            variant_weights = deployment_config.get("variant_weights")
            if len(versions) < len(variant_weights):
                raise ValueError("Insufficient model versions for the specified variant weights.")
            deploy_multi_variant(config, env, sklearn_schema_builder, versions[:len(variant_weights)], instance_type, sklearn_input)
        elif mode == "shadow-variant":
            deploy_shadow_variant(config, env, sklearn_schema_builder, versions[:2], instance_type, sklearn_input)
        else:
            print(f"Invalid mode: {mode}. Please use 'single', 'multi-variant', or 'shadow-variant'.")

if __name__ == '__main__':
    main()
    job.commit()
