"""Example workflow pipeline script for abalone pipeline.

                                               . -RegisterModel
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os
import boto3
import sagemaker
import sagemaker.session

from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo

from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.model import Model  
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    CreateModelStep,
    CacheConfig,
    TuningStep
)

from sagemaker.predictor import RealTimePredictor
from sagemaker.tuner import (
    ContinuousParameter,
    HyperparameterTuner,
    WarmStartConfig,
    WarmStartTypes,
)

from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import CreateModelStep


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_session(region, default_bucket):
    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
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
        sagemaker_project_arn=None,
        role=None,
        default_bucket=None,
        model_package_group_name="cv-metastasis2",
        pipeline_name="cv-metastasis2",
        base_job_prefix="cv-metastasis2",
    ):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    # setting up role and session
    boto3.setup_default_session(region_name=region)
    sagemaker_session = get_session(region, default_bucket)
    bucket = sagemaker.Session().default_bucket()
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    
    s3 = boto3.client('s3')
    s3.put_object(Bucket=bucket, Key=(model_package_group_name+'/'))
    
    # parameters for preprocessing
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.4xlarge")
    
    # replace this with an url to your custom container: https://docs.aws.amazon.com/sagemaker/latest/dg/processing-container-run-scripts.html
    preprocessing_image_uri = '<account-id>.dkr.ecr.<region>.amazonaws.com/sagemaker-preprocessing-container:latest'
    
    # parameters for training
    training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.xlarge")
    training_image = sagemaker.image_uris.retrieve("image-classification", region)
    
    # parameters for model registration
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
    
    ## Data processing step    
    split_data_step_processor = ScriptProcessor(
        image_uri=preprocessing_image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=2,
        base_job_name=f"s3://{bucket}/{base_job_prefix}/script-skin-preprocess",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    
    split_data_step = ProcessingStep(
        name="SplitData",
        processor=split_data_step_processor,
        outputs=[
            sagemaker.processing.ProcessingOutput(
                output_name="train_data", source="/opt/ml/processing/output/data/train", 
                destination=f"s3://{bucket}/{base_job_prefix}/data/train/train.rec"
            ),
            sagemaker.processing.ProcessingOutput(
                output_name="val_data", source="/opt/ml/processing/output/data/val",
                destination=f"s3://{bucket}/{base_job_prefix}/data/val/val.rec"
            ),
            sagemaker.processing.ProcessingOutput(
                output_name="test_data", source="/opt/ml/processing/output/data/test",
                destination=f"s3://{bucket}/{base_job_prefix}/data/test/test.rec"
            ),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        job_arguments=["--region", region, "--prefix", base_job_prefix]
    )

    ## Training step for generating model artifacts
    train_step_inputs = {
        "train": sagemaker.inputs.TrainingInput(
            s3_data=split_data_step.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
            content_type="application/x-recordio",
            s3_data_type="S3Prefix",
            input_mode="Pipe",
        ),
        "validation": sagemaker.inputs.TrainingInput(
            s3_data=split_data_step.properties.ProcessingOutputConfig.Outputs["val_data"].S3Output.S3Uri,
            content_type="application/x-recordio",
            s3_data_type="S3Prefix",
            input_mode="Pipe",
        )
    }
    
    
    hyperparameters = {
        "num_layers": 18,
        "use_pretrained_model": 1,
        "augmentation_type": 'crop_color_transform',
        "image_shape": '3,224,224', 
        "num_classes": 7,
        "num_training_samples": 29311, 
        "mini_batch_size": 8,
        "epochs": 5, 
        "learning_rate": 0.00001,
        "precision_dtype": 'float32'
    }

    estimator_config = {
        "hyperparameters": hyperparameters,
        "image_uri": training_image,
        "role": role,
        "instance_count": 1,
        "instance_type": "ml.p3.2xlarge",
        "volume_size": 100,
        "max_run": 360000,
        "output_path": f"s3://{bucket}/{base_job_prefix}/training_jobs",
    }
    
    image_classifier = sagemaker.estimator.Estimator(**estimator_config)
    train_step = TrainingStep(name="TrainModel", estimator=image_classifier, inputs=train_step_inputs)
    
    ## Evaluation step: compute Precision, Recall and F1 score
    script_eval = ScriptProcessor(
        image_uri=preprocessing_image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"s3://{bucket}/{base_job_prefix}/script-skin-eval",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    evaluation_report = PropertyFile(
        name="SkinEvalReport",
        output_name="evaluation",
        path="evaluation.json",
    )
        
    model_artifact = train_step.properties.ModelArtifacts.S3ModelArtifacts
    eval_step = ProcessingStep(
        name="Eval",
        processor=script_eval,
        inputs=[
            ProcessingInput(
                source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=split_data_step.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri,
                destination="/opt/ml/processing/output/data/test", 
                s3_input_mode="Pipe",
            ),
            ProcessingInput(
                source=split_data_step.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
                destination="/opt/ml/processing/output/data/train", 
                s3_input_mode="Pipe",
            ),
        ],
        outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
        code=os.path.join(BASE_DIR, "evaluate.py"),
        property_files=[evaluation_report],
        job_arguments=["--region", region, "--role", role, "--modelartifact", model_artifact, "--prefix", base_job_prefix]
    )
    
    
    ## Model registration step
    register_step = RegisterModel(
        name="RegisterModel",
        estimator=image_classifier,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["image/jpeg"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=base_job_prefix,
        approval_status=model_approval_status,
    )
    
    ## Condition step: register model only if F1 score is above 0.6
    cond_lte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=eval_step.name,
            property_file=evaluation_report,
            json_path="classification_metrics.f1_score.value"
        ),
        right=0.6
    )

    cond_step = ConditionStep(
        name="SkinF1ScoreCond",
        conditions=[cond_lte],
        if_steps=[register_step],
        else_steps=[]
    )
    
    ## Pipeline definition
    pipeline_name = f"{base_job_prefix}-pipeline"
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status],
        steps=[split_data_step, train_step, eval_step, cond_step], 
        sagemaker_session=sagemaker_session
    )
    
    return pipeline
