"""S3 Interaction Utilities"""

import logging
import os
from io import BytesIO
from typing import Optional

import boto3
import joblib
import psutil
from botocore.exceptions import ClientError
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


env = os.environ.get("ENVIRONMENT", "development")
BUCKET_NAME = os.environ.get("ML_MODEL_BUCKET", f"uk-national-forecaster-models-{env}")
MODEL = "v4"


def build_object_name(forecast_horizon_hour: int) -> str:
    """Helper function to create s3 object name, based on model forecast horizon

    Args:
        forecast_horizon_hour (int): Hour

    Returns:
        str: Object name
    """
    return f"xgboost_model_forecast_hour_{forecast_horizon_hour}"


def create_s3_client(access_key_id: Optional[str] = None, secret_key: Optional[str] = None):
    """Creates boto3 s3 client with optional keys.

    If keys are not specified, they are taken from .aws config file
    Args:
        access_key_id (Optional[str], optional): AWS access key. Defaults to None.
        secret_key (Optional[str], optional): AWS secret key. Defaults to None.

    Returns:
        _type_: s3 client
    """
    if access_key_id is None:
        return boto3.client("s3")
    else:
        return boto3.client("s3", aws_access_key_id=access_key_id, aws_secret_access_key=secret_key)


def load_model(s3_client, object_name: str, bucket_name: str = BUCKET_NAME) -> XGBRegressor:
    """Downloads XGBRegressor model from s3 storage.

    Args:
        s3_client (_type_): s3 interaction client (boto3.client.S3)
        object_name (str): Key for model
        bucket_name (str, optional): Defaults to DEV_BUCKET_NAME.

    Returns:
        XGBRegressor: NationalBoost model for specific forecast horizon
    """

    object_name = f"{MODEL}/{object_name}"
    logger.debug(f"Going to load model from {bucket_name=} {object_name=}")

    with BytesIO() as f:
        s3_client.download_fileobj(Bucket=bucket_name, Key=object_name, Fileobj=f)
        f.seek(0)
        model = joblib.load(f)

    assert isinstance(model, XGBRegressor)

    process = psutil.Process(os.getpid())
    logger.debug(f"Memory is {process.memory_info().rss / 10 ** 6} MB")

    return model


def save_model(
    s3_client,
    object_name: str,
    model: XGBRegressor,
    bucket_name: str = BUCKET_NAME,
    overwrite_current: bool = False,
) -> bool:
    """Saves an XGBRegressor model to s3

    Args:
        s3_client (_type_): s3 interaction client (boto3.client.S3)
        object_name (str): Key to save model onto
        model (XGBRegressor): model for storage
        bucket_name (str, optional): Defaults to DEV_BUCKET_NAME.
        overwrite_current (bool, optional): Defaults to False.

    Returns:
        bool: outcome of saving, True equates to success.
    """

    object_name = f"{MODEL}/{object_name}"

    bucket_contents = s3_client.list_objects(Bucket=bucket_name)
    keys = (
        [_file["Key"] for _file in bucket_contents["Contents"]]
        if "Contents" in bucket_contents
        else list()
    )

    if object_name in keys and not overwrite_current:
        logging.info(f"Found object with name {object_name}, will not overwrite.")
        return True

    try:
        with BytesIO() as f:
            joblib.dump(model, f)
            f.seek(0)
            s3_client.upload_fileobj(Bucket=bucket_name, Key=object_name, Fileobj=f)
    except ClientError as e:
        logging.error(e)
        return False
    return True
