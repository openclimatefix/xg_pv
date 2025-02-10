"""Model inference pipeline"""

import logging
from pathlib import Path
from typing import Dict

import pandas as pd
from torch.utils.data import IterDataPipe, functional_datapipe

from gradboost_pv.inference.models import NationalBoostInferenceModel, Prediction
from gradboost_pv.utils.logger import getLogger
from gradboost_pv.utils.typing import Hour

logger = logging.getLogger(__name__)


def process_predictions_to_pandas(predictions: Dict[Hour, Prediction]) -> pd.DataFrame:
    """Method that translates the prediction object of our models into a DataFrame

    Args:
        predictions (Dict[Hour, Prediction]): Output of model, predictions at each hourly horizon.

    Returns:
        pd.DataFrame: DataFrame of the forecasts, for ease of injection into database.
    """
    return pd.concat(
        {
            hour: pd.DataFrame.from_dict(
                {
                    "datetime_of_model_inference_utc": pred.datetime_of_model_inference_utc,
                    "datetime_of_target_utc": pred.datetime_of_target_utc,
                    "forecast_mw": pred.forecast_mw,
                    "forecast_mw_plevel_10": pred.forecast_mw_plevel_10,
                    "forecast_mw_plevel_90": pred.forecast_mw_plevel_90,
                },
                orient="index",
            ).T
            for hour, pred in predictions.items()
        }
    ).set_index("datetime_of_model_inference_utc")


class MockDatabase:
    """Draft Database for prototyping, storing results in a pandas DataFrame."""

    def __init__(self, path_to_database: Path, overwrite_current: bool = True) -> None:
        """Setup for mock database

        Args:
            path_to_database (Path): Path to write local database to
            overwrite_current (bool, optional): Defaults to True.
        """
        self.path_to_database = path_to_database
        self.overwrite = overwrite_current

        self.data = None
        self.logger = getLogger("mock_database")

    def create_new_database(self) -> pd.DataFrame:
        """Creates a mock database

        Returns:
            pd.DataFrame: mock db
        """
        return pd.DataFrame(
            columns=[
                "datetime_of_target_utc",
                "forecast_kw",
            ]
        )

    def load(self) -> None:
        """Loads mock database from file or creates a new one."""
        if self.overwrite:
            self.data = self.create_new_database()
        else:
            try:
                self.data = pd.read_pickle(self.path_to_database)
            except FileNotFoundError:
                # could not find database, initialising a new one
                self.data = self.create_new_database()

    def write(self, predictions: Dict[Hour, Prediction]) -> None:
        """Writes to mock database

        Args:
            predictions (Dict[Hour, Prediction]): output of NationalBoost model.
        """
        assert self.data is not None

        self.data = pd.concat([self.data, process_predictions_to_pandas(predictions)], axis=0)

    def disconnect(self):
        """Save mock database locally"""
        assert self.data is not None

        self.data.to_pickle(self.path_to_database)


class MockDatabaseConnection:
    """Class for creating connection to mock local database"""

    def __init__(self, path_to_mock_database: Path, overwrite_database: bool = False) -> None:
        """Pre-connection setup to local mock database

        Args:
            path_to_mock_database (Path): path to database
            overwrite_database (bool, optional): Defaults to False.
        """
        self.path_to_database = path_to_mock_database
        self.overwrite_database = overwrite_database
        self.logger = getLogger("mock_database_connection")

        self.connected = False
        self.database = None

    def connect(self):
        """Creates connection to mock database"""
        self.database = MockDatabase(self.path_to_database, self.overwrite_database)
        self.database.load()
        self.connected = True
        logger.debug("Initialised connection to database")

    def disconnect(self):
        """Disconnects from mock database, saving locally"""
        assert self.database is not None
        self.database.disconnect()
        self.database = None
        self.connected = False
        logger.debug("Closed connection to database")

    def __enter__(self) -> MockDatabase:
        """Context manager returning direct access to the database

        Returns:
            MockDatabase
        """
        if not self.connected:
            self.connect()

        assert self.database is not None
        return self.database

    def __exit__(self, type, value, traceback) -> None:
        """Disconnect and save mock database upon exiting"""
        self.disconnect()
        return


@functional_datapipe("nationalboost_model_inference")
class NationalBoostModelInference(IterDataPipe):
    """Simple model inference pipeline"""

    def __init__(
        self,
        model: NationalBoostInferenceModel,
        data_feed: IterDataPipe,
        database_connection: MockDatabaseConnection,
    ) -> None:
        """Initalization of model pipeline.

        Args:
            model (NationalBoostInferenceModel): Model
            data_feed (IterDataPipe): Data feed
            database_connection (MockDatabaseConnection): conn to database
        """
        self.model = model
        self.data_feed = data_feed
        self.database_connection = database_connection

    def run(self):
        """Runs the inference pipeline, exhausting all data in the datafeed."""
        with self.database_connection as conn:
            for data in self.data_feed:
                logger.debug(f"Running model for data {data}")
                prediction: Dict[Hour, Prediction] = self.model(data)
                conn.write(prediction)

                # TODO would it be simpiler here just collect the results into a
                #  dataframe and returning it
                # Then we dont have to worry about pickling or saving the results.
                # Or have an option to save the results to a database
