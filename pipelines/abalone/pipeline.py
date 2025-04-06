"""Example workflow pipeline script for abalone pipeline.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""

# Standard library imports
import os
import pathlib as pl
from datetime import datetime
import yaml
import time

# AWS imports
import boto3
import json
# SageMaker imports
import sagemaker
import sagemaker.session
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.tuner import IntegerParameter, ContinuousParameter, HyperparameterTuner
from sagemaker.xgboost import XGBoostPredictor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.functions import JsonGet, Join
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline_experiment_config import PipelineExperimentConfig
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig, TuningStep
from sagemaker.workflow.step_collections import RegisterModel
# Other specific imports
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.function_step import step
from sagemaker.metadata_properties import MetadataProperties


BASE_DIR = pl.Path(__file__).resolve().parent

def is_file_empty(file_path):
    """Check if the given file is empty."""
    if not os.path.exists(file_path):
        return True
    elif os.stat(file_path).st_size == 0:
        return True
    return False

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_name=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.describe_project(ProjectName=sagemaker_project_name)
        sagemaker_project_arn = response["ProjectArn"]
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_name=None,
    role=None,
    default_bucket=None,
    model_package_group_name="AbalonePackageGroup",
    pipeline_name="AbalonePipeline",
    base_job_prefix="Abalone",
    train_data_path="s3://sagemaker-servicecatalog-seedcode-ca-central-1/dataset/abalone-dataset.csv",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
    experiment_name="Abalone-pipelines-experiment",
    tracking_server_arn=None,
    max_jobs=5,
    max_parallel_jobs=5,
    strategy='Random',
    hyperparameters=None,
    use_sg_model_registry=False,

):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    print(f"region: {region}, role: {role}, default_bucket: {default_bucket}")
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=train_data_path,
    )
    # Preprocessing step: Include the parent run id as a property file output.
    parent_run_property = PropertyFile(
        name="ParentRunIdProperty",
        output_name="parent_run_id",  # Must match the ProcessingOutput defined in the preprocessing job.
        path="parent_run_id.txt",
    )
    # processing step for feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-abalone-preprocess",
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = sklearn_processor.run(
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
            ProcessingOutput(output_name="parent_run_id", source="/opt/ml/processing/parent_run_id"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        arguments=["--input-data",
                   input_data,
                   "--experiment-name",
                   experiment_name,
                   "--tracking-server-arn",
                   tracking_server_arn,
                  ],
    )
    step_process = ProcessingStep(
        name=f"{base_job_prefix}PreprocessData",
        step_args=step_args,
        property_files=[parent_run_property],
    )

    # Use JsonGet to extract the parent run id from the preprocessing property file.
    parent_run_id = JsonGet(
        step_name=step_process.name,
        property_file=parent_run_property,
        json_path="parent_run_id"  # This extracts the content of the file.
    )
    # Set default hyperparameter configuration if none is provided
    if hyperparameters is None:
        hyperparameters = {
            "alpha": {"min": 0.1, "max": 10.0, "scaling_type": "Logarithmic"},
            "lambda": {"min": 0.1, "max": 10.0, "scaling_type": "Logarithmic"},
            "num_round": {"min": 10, "max": 1000, "scaling_type": "Linear"},
            "max_depth": {"min": 3, "max": 10, "scaling_type": "Linear"},
            "eta": {"min": 0.01, "max": 0.3, "scaling_type": "Logarithmic"},
            "gamma": {"min": 0.0, "max": 5.0, "scaling_type": "Linear"},
            "min_child_weight": {"min": 1, "max": 10, "scaling_type": "Linear"},
            "subsample": {"min": 0.5, "max": 1.0, "scaling_type": "Linear"},
        }

    # training step for generating model artifacts
    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}"
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.7-1",
        py_version="py3",
        instance_type=training_instance_type,
    )
    xgb_train = Estimator(
        entry_point=str(BASE_DIR.joinpath("train.py"))
        if not is_file_empty(str(BASE_DIR.joinpath("train.py")))
        else None,
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/abalone-train",
        sagemaker_session=pipeline_session,
        role=role,
        environment={
            "MLFLOW_EXPERIMENT_NAME": experiment_name,  # Experiment Name
            "MLFLOW_TRACKING_SERVER_ARN": tracking_server_arn,  # Tracking Server ARN
            "MLFLOW_RUN_ID": parent_run_id,  # Run ID
        }
    )

    xgb_train.set_hyperparameters(
        objective="reg:squarederror",  # Define the object metric for the training job
    )

    # Tuning Step
    objective_metric_name = "validation:rmse"
    hyperparameter_ranges = {
        "alpha": ContinuousParameter(
            hyperparameters["alpha"]["min"],
            hyperparameters["alpha"]["max"],
            scaling_type=hyperparameters["alpha"]["scaling_type"],
        ),
        "lambda": ContinuousParameter(
            hyperparameters["lambda"]["min"],
            hyperparameters["lambda"]["max"],
            scaling_type=hyperparameters["lambda"]["scaling_type"],
        ),
        "num_round": IntegerParameter(
            hyperparameters["num_round"]["min"],
            hyperparameters["num_round"]["max"],
            scaling_type=hyperparameters["num_round"]["scaling_type"],
        ),
        "max_depth": IntegerParameter(
            hyperparameters["max_depth"]["min"],
            hyperparameters["max_depth"]["max"],
            scaling_type=hyperparameters["max_depth"]["scaling_type"],
        ),
        "eta": ContinuousParameter(
            hyperparameters["eta"]["min"],
            hyperparameters["eta"]["max"],
            scaling_type=hyperparameters["eta"]["scaling_type"],
        ),
        "gamma": ContinuousParameter(
            hyperparameters["gamma"]["min"],
            hyperparameters["gamma"]["max"],
            scaling_type=hyperparameters["gamma"]["scaling_type"],
        ),
        "min_child_weight": IntegerParameter(
            hyperparameters["min_child_weight"]["min"],
            hyperparameters["min_child_weight"]["max"],
            scaling_type=hyperparameters["min_child_weight"]["scaling_type"],
        ),
        "subsample": ContinuousParameter(
            hyperparameters["subsample"]["min"],
            hyperparameters["subsample"]["max"],
            scaling_type=hyperparameters["subsample"]["scaling_type"],
        ),
    }

    metric_definitions = [
        {"Name": objective_metric_name, "Regex": "Validation RMSE: ([0-9\\.]+)"}
    ]

    tuner_log = HyperparameterTuner(
        xgb_train,
        objective_metric_name,
        hyperparameter_ranges,
        metric_definitions,
        max_jobs=max_jobs,
        max_parallel_jobs=max_parallel_jobs,
        strategy=strategy,
        objective_type="Minimize",
        early_stopping_type="Auto",
    )

    hpo_args = tuner_log.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        }
    )

    step_tuning = TuningStep(
        name=f"{base_job_prefix}ModelTuning",
        step_args=hpo_args,
    )

    best_model = Model(
        image_uri=image_uri,
        model_data=step_tuning.get_top_model_s3_uri(
            top_k=0, s3_bucket=pipeline_session.default_bucket(), prefix=base_job_prefix
        ),
        predictor_cls=XGBoostPredictor,
        sagemaker_session=pipeline_session,
        role=role,
    )

    step_create_first = ModelStep(
        name=f"{base_job_prefix}CreateBestModel",
        step_args=best_model.create(instance_type="ml.m5.xlarge"),
    )
    # processing step for evaluation
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-abalone-eval",
        sagemaker_session=pipeline_session,
        role=role,
        env={
            "MLFLOW_EXPERIMENT_NAME": experiment_name,  # Experiment Name
            "MLFLOW_TRACKING_SERVER_ARN": tracking_server_arn,  # Tracking Server ARN
            "MLFLOW_RUN_ID": parent_run_id,  # Run ID
            "MODEL_PACKAGE_GROUP_NAME": model_package_group_name
        },
    )
    step_args = script_eval.run(
        inputs=[
            ProcessingInput(
                source=step_tuning.get_top_model_s3_uri(
                    top_k=0,
                    s3_bucket=pipeline_session.default_bucket(),
                    prefix=base_job_prefix,
                ),
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
    )
    evaluation_report = PropertyFile(
        name="AbaloneEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name=f"{base_job_prefix}EvaluateModel",
        step_args=step_args,
        property_files=[evaluation_report],
    )

    # register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )

    step_args = best_model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    step_register = ModelStep(
        name=f"{base_job_prefix}RegisterModel",
        step_args=step_args,
    )

    # condition step for evaluating model quality and branching execution
    cond_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metrics.rmse.value"
        ),
        right=6.0,
    )
    step_cond = ConditionStep(
        name=f"{base_job_prefix}CheckRMSEEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )
    steps=[step_process, step_tuning, step_eval]
    if use_sg_model_registry:
        steps.append(step_cond)
    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
        ],
        steps=steps,
        sagemaker_session=pipeline_session,
    )
    return pipeline
