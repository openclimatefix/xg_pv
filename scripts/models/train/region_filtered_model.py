"""Model training script"""

import logging
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Dict

import numpy as np
import xarray as xr

from gradboost_pv.models.region_filtered import (
    DEFAULT_VARIABLES_FOR_PROCESSING,
    build_datasets_from_local,
    load_all_variable_slices,
)
from gradboost_pv.models.s3 import build_object_name, create_s3_client, save_model
from gradboost_pv.models.training import (
    DEFFAULT_HYPARAM_CONFIG,
    ExperimentSummary,
    plot_feature_importances,
    plot_loss_metrics,
    run_experiment,
)
from gradboost_pv.models.utils import DEFAULT_DIRECTORY_TO_PROCESSED_NWP, GSP_FPATH
from gradboost_pv.utils.logger import getLogger
from gradboost_pv.utils.typing import Hour

logger = getLogger(__name__)

formatString = "[%(levelname)s][%(asctime)s] : %(message)s"  # specify a format string
logLevel = logging.INFO  # specify standard log level
logging.basicConfig(format=formatString, level=logLevel, datefmt="%Y-%m-%d %I:%M:%S")


def parse_args():
    """Parse command line arguments"""
    parser = ArgumentParser(description="Script for training Region-masked model.")
    parser.add_argument(
        "--path_to_processed_nwp",
        type=Path,
        default=DEFAULT_DIRECTORY_TO_PROCESSED_NWP,
        help="Directory to load Processed NWP data.",
    )
    parser.add_argument(
        "--model_nwp_variables",
        nargs="+",
        default=DEFAULT_VARIABLES_FOR_PROCESSING,
        help="Numerical Weather Prediction Variables for model selection.",
    )
    parser.add_argument("--display_plots", default=False, action=BooleanOptionalAction)
    parser.add_argument("--save_to_s3", default=False, action=BooleanOptionalAction)
    parser.add_argument(
        "--s3_access_key", type=str, required=False, default=None, help="s3 API Access Key"
    )
    parser.add_argument(
        "--s3_secret_key", type=str, required=False, default=None, help="s3 API Secret Key"
    )
    parser.add_argument("--s3_overwrite_current", default=False, action=BooleanOptionalAction)
    args = parser.parse_args()
    return args


def load_gsp() -> xr.Dataset:
    """TODO - Decide to remove GCP usage?"""
    return xr.open_zarr(GSP_FPATH).isel(gsp_id=0)


def main(path_to_processed_nwp: Path, nwp_variables: list[str]) -> Dict[Hour, ExperimentSummary]:
    """Training NationalBoost model for all forecast horizons

    Args:
        path_to_processed_nwp (str): path to locally preprocessed NWP data
        nwp_variables (list[str]): variables to select for design matrix creation

    Returns:
        Dict[int, ExperimentSummary]: Results of training for each forecast horizon
    """
    gsp_data = load_gsp()

    results = dict()

    for forecast_horizon_hour in range(41):
        print(forecast_horizon_hour)
        # independently fit an XGBoost model for each forecast horizon
        processed_nwp = load_all_variable_slices(
            forecast_horizon_hour,
            nwp_variables,
            directory=path_to_processed_nwp,
            years=[2016, 2017, 2018, 2019, 2020, 2021, 2022],
        )
        X, y = build_datasets_from_local(
            processed_nwp, gsp_data, np.timedelta64(forecast_horizon_hour, "h")
        )
        training_results = run_experiment(
            X,
            y,
            DEFFAULT_HYPARAM_CONFIG,
            forecast_hour=forecast_horizon_hour,
            save_errors_locally=True,
            errors_local_save_file=f"results/backtest_test_horizon_{forecast_horizon_hour}",
        )

        results[forecast_horizon_hour] = training_results

        logger.info(f"Trained model for {forecast_horizon_hour} hour forecast. {training_results=}")

    return results


if __name__ == "__main__":
    args = parse_args()
    training_results = main(args.path_to_processed_nwp, args.model_nwp_variables)

    if args.display_plots:
        plot_feature_importances(training_results)
        plot_loss_metrics(training_results)

    if args.save_to_s3:
        if args.s3_access_key is None or args.s3_secret_key is None:
            client = create_s3_client()
        else:
            client = create_s3_client(args.s3_access_key, args.s3_secret_key)

        for forecast_hour, results in training_results.items():
            object_name = build_object_name(forecast_hour)
            save_result = save_model(
                client, object_name, results.model, overwrite_current=args.s3_overwrite_current
            )
            logger.error(f"Saved xgboost model with filename {object_name}.")
