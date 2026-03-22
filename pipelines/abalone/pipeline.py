import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.estimator import Estimator


def get_pipeline(region, role, default_bucket, pipeline_name, base_job_prefix, **kwargs):
    sagemaker_session = sagemaker.session.Session(default_bucket=default_bucket)

    training_image = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.5-1"
    )

    estimator = Estimator(
        image_uri=training_image,
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        output_path=f"s3://{default_bucket}/{base_job_prefix}/output",
        sagemaker_session=sagemaker_session,
    )

    step_train = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={}
    )

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_train],
        sagemaker_session=sagemaker_session
    )

    return pipeline