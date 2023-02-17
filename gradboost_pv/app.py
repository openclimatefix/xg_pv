"""Script to simulate data read, model inference and prediction write"""
import pathlib
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import click
from xgboost import XGBRegressor

import gradboost_pv
from gradboost_pv.inference.data_feeds import ProductionDataFeed
from gradboost_pv.inference.models import Hour, NationalBoostInferenceModel, NationalPVModelConfig
from gradboost_pv.inference.run import MockDatabaseConnection, NationalBoostModelInference
from gradboost_pv.models.s3 import build_object_name, create_s3_client, load_model
from gradboost_pv.models.utils import load_nwp_coordinates
from gradboost_pv.utils.logger import getLogger

DEFAULT_PATH_TO_MOCK_DATABASE = (
    Path(gradboost_pv.__file__).parents[1] / "data" / "mock_inference_database.pickle"
)

logger = getLogger(__name__)


@click.command()
@click.option(
    "--path_to_model_config",
    type=click.Path(path_type=pathlib.Path),
    help="Path to NationalBoost model config yaml.",
)
@click.option(
    "--path_to_datafeed_config",
    type=click.Path(path_type=pathlib.Path),
    help="Path to Production Datafeed config yaml.",
)
@click.option(
    "--write_to_database",
    is_flag=True,
    default=False,
    help="Set this flag to actually write the results to the database."
    "By default we only print to stdout using mock local database.",
)
@click.option(
    "--s3_access_key",
    type=str,
    default=None,
    help="Optional AWS s3 Access Key.",
)
@click.option(
    "--s3_secret_key",
    type=str,
    default=None,
    help="Optional AWS s3 Secret Key.",
)
def main(
    path_to_model_config: Path,
    path_to_datafeed_config: Path,
    write_to_database: bool = False,
    s3_access_key: Optional[str] = None,
    s3_secret_key: Optional[str] = None,
):
    """Entry point for inference script"""
    if s3_access_key is None or s3_secret_key is None:
        logger.debug("Creating s3 client with default env.var keys.")
        client = create_s3_client()
    else:
        logger.debug(f"Creating s3 client with specified keys: {s3_access_key}/{s3_secret_key}.")
        client = create_s3_client(s3_access_key, s3_secret_key)

    def model_loader_by_hour(hour: Hour) -> XGBRegressor:
        """Get a model by forecast hour using client"""
        return load_model(client, build_object_name(hour))

    # load in our national pv model
    x, y = load_nwp_coordinates()
    model_config = NationalPVModelConfig.load_from_yaml(path_to_model_config)
    model = NationalBoostInferenceModel(model_config, model_loader_by_hour, x, y)
    model.initialise()
    logger.debug("Intitialised model")

    data_feed = ProductionDataFeed(path_to_datafeed_config)
    logger.debug("Defined production feed.")

    if not write_to_database:
        # create a mock database to write to
        logger.debug("Not writing to database, storing in local mock database")
        database_conn = MockDatabaseConnection(
            DEFAULT_PATH_TO_MOCK_DATABASE, overwrite_database=True
        )

    else:
        # create connection to actual database TODO
        pass

    inference_pipeline = NationalBoostModelInference(model, data_feed, database_conn)
    inference_pipeline.run()
    logger.debug("Model inference complete")

    if not write_to_database:
        # print predictions to console
        database_conn = MockDatabaseConnection(
            DEFAULT_PATH_TO_MOCK_DATABASE, overwrite_database=False
        )
        database_conn.connect()
        print(database_conn.database.data)


if __name__ == "__main__":
    main()
