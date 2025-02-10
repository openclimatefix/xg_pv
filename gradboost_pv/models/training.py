"""Functions for model training"""

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_pinball_loss,
)
from xgboost import XGBRegressor

ALPHA = np.array([0.1, 0.5, 0.9])

DEFFAULT_HYPARAM_CONFIG = {
    "objective": "reg:quantileerror",
    "booster": "gbtree",
    "colsample_bylevel": 1,
    "colsample_bynode": 1,
    "colsample_bytree": 0.85,
    "early_stopping_rounds": None,
    "gamma": 0,
    "gpu_id": -1,
    "grow_policy": "depthwise",
    "importance_type": None,
    "interaction_constraints": "",
    "learning_rate": 0.005,
    "max_bin": 256,
    "max_cat_threshold": 64,
    "max_depth": 80,
    "max_leaves": 0,
    "min_child_weight": 5,
    "n_estimators": 1250,
    "n_jobs": -1,
    "num_parallel_tree": 1,
    "predictor": "auto",
    "random_state": 0,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "sampling_method": "uniform",
    "scale_pos_weight": 1,
    "subsample": 0.65,
    "tree_method": "hist",
    "quantile_alpha": ALPHA,
    "validate_parameters": 1,
    "verbosity": 1,
}


@dataclass
class ExperimentSummary:
    """Object for storing basic model train/test results"""

    pinball_train_loss: float
    pinball_test_loss: float
    pinball_train_10_percentile_loss: float
    pinball_test_10_percentile_loss: float
    pinball_train_90_percentile_loss: float
    pinball_test_90_percentile_loss: float
    mae_train_loss: float
    mae_test_loss: float
    model: XGBRegressor


class EnhancedJSONEncoder(json.JSONEncoder):
    """
    Class for json dumping a dataclass object to file.

    Stolen from:
    https://stackoverflow.com/questions/51286748/
    make-the-python-json-encoder-support-pythons-new-dataclasses
    """

    def default(self, o):
        """Default encoding"""
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def run_experiment(
    X: pd.DataFrame,
    y: pd.DataFrame,
    booster_hyperparam_config: dict = DEFFAULT_HYPARAM_CONFIG,
    save_errors_locally: bool = False,
    errors_local_save_file: Optional[Union[Path, str]] = None,
    forecast_hour: int = 0,
) -> ExperimentSummary:
    """Trains and tests XGBoost Regression model.

    Args:
        X (pd.DataFrame): X - Features
        y (pd.DataFrame): y - target, normalised PV data
        booster_hyperparam_config (dict, optional): Model hyperparams,
        defaults to DEFFAULT_HYPARAM_CONFIG.
        save_errors_locally (bool, optional): Defaults to False.
        errors_local_save_file (Optional[Union[Path, str]], optional): Defaults to None.
        forecast_hour (int, optional): Defaults to 0

    Returns:
        ExperimentSummary: Object storing some basic fit/evalutation stats + model.
    """

    if save_errors_locally:
        assert errors_local_save_file is not None

    maes = []
    pinball_losses = []
    pinball_losses_10_percentile = []
    pinball_losses_90_percentile = []
    # Cross validate based on the year
    for year in range(2016, 2023):
        print(year)
        # use 2020 as training period and 2021 as test, train on 2022 as well
        X_train, y_train = (
            X.loc[(X.index < f"{year}-01-01") | (X.index > f"{year}-12-31 23:30:00")],
            y.loc[(y.index < f"{year}-01-01") | (y.index > f"{year}-12-31 23:30:00")],
        )
        X_test, y_test = (
            X.loc[(X.index >= f"{year}-01-01") & (X.index <= f"{year}-12-31 23:30:00")],
            y.loc[(y.index >= f"{year}-01-01") & (y.index <= f"{year}-12-31 23:30:00")],
        )
        # Select only the rows in test where the time of day is between 10am and 2pm
        # X_test = X_test.loc[(X_test.index.hour >= 10) & (X_test.index.hour <= 14)]
        # y_test = y_test.loc[(y_test.index.hour >= 10) & (y_test.index.hour <= 14)]
        X_train = X_train.dropna()
        X_test = X_test.dropna()
        y_train = y_train.fillna(0.0)
        y_test = y_test.fillna(0.0)
        model = XGBRegressor(**booster_hyperparam_config)
        model.fit(X_train, y_train)

        y_pred_test, y_pred_train = model.predict(X_test), model.predict(X_train)

        train_pinballs = []
        test_pinballs = []
        for idx, alpha in enumerate(ALPHA):
            y_pred_train_alpha = y_pred_train[:, idx]
            y_pred_test_alpha = y_pred_test[:, idx]
            train_pinball, test_pinball = mean_pinball_loss(
                y_train, y_pred_train_alpha, alpha=alpha
            ), mean_pinball_loss(y_test, y_pred_test_alpha, alpha=alpha)
            train_pinballs.append(train_pinball)
            test_pinballs.append(test_pinball)

        non_night_percentiles = []
        for idx, alpha in enumerate(ALPHA):
            y_pred_test_alpha = y_pred_test[:, idx]
            non_night_percentiles.append(
                np.sum(
                    (y_test["target"].values < y_pred_test_alpha) & (y_test["target"].values > 0.01)
                )
            )
        # Get percentage of total test data that is below each percentile
        non_night_percentiles = np.array(non_night_percentiles) / len(
            y_test[y_test["target"].values > 0.01]["target"].values
        )
        pinball_losses.append(non_night_percentiles[1])
        pinball_losses_10_percentile.append(non_night_percentiles[0])
        pinball_losses_90_percentile.append(non_night_percentiles[2])

        y_pred_train = y_pred_train[:, 1]
        y_pred_test_50 = y_pred_test[:, 1]
        y_pred_test_10 = y_pred_test[:, 0]
        y_pred_test_90 = y_pred_test[:, 2]
        train_mae, test_mae = mean_absolute_error(y_train, y_pred_train), mean_absolute_error(
            y_test, y_pred_test_50
        )
        # print(f"Median test MAE: {np.round(test_mae, 5)}")
        maes.append(test_mae)

        if save_errors_locally:
            errors_test = pd.DataFrame(
                data=np.concatenate(
                    [
                        y_test.values,
                        y_pred_test_50.reshape(-1, 1),
                        y_pred_test_10.reshape(-1, 1),
                        y_pred_test_90.reshape(-1, 1),
                        (y_test.values - y_pred_test_50.reshape(-1, 1)) ** 2,
                        np.abs(y_test.values - y_pred_test_50.reshape(-1, 1)),
                    ],
                    axis=1,
                ),
                columns=["y", "pred", "p10", "p90", "test_mse", "test_mae"],
                index=y_test.index,
            )
            errors_train = pd.DataFrame(
                data=np.concatenate(
                    [
                        (y_train.values - y_pred_train.reshape(-1, 1)) ** 2,
                        np.abs(y_train.values - y_pred_train.reshape(-1, 1)),
                    ],
                    axis=1,
                ),
                columns=["train_mse", "train_mae"],
                index=y_train.index,
            )

            errors = pd.concat([errors_train, errors_test], axis=1)

            errors_local_save_file_year = f"{errors_local_save_file}_year_{year}.csv"

            print(errors_local_save_file_year)

            errors.to_csv(errors_local_save_file_year)

    return ExperimentSummary(
        train_pinballs[1],
        test_pinballs[1],
        train_pinballs[0],
        test_pinballs[0],
        train_pinballs[2],
        test_pinballs[2],
        train_mae,
        test_mae,
        model,  # just save the last trained model for nwp
    )


def plot_loss_metrics(results_by_step: dict[int, ExperimentSummary]):
    """Convenience function for plotting loss metrics over forecast horizons"""
    title_mapping = {
        "MAE Median Train": lambda x: x.mae_train_loss,
        "MAE Median Test": lambda x: x.mae_test_loss,
    }
    title_mapping2 = {
        "0.5 Train": lambda x: x.pinball_train_loss,
        "0.5 Test": lambda x: x.pinball_test_loss,
        "0.1 Train": lambda x: x.pinball_train_10_percentile_loss,
        "0.1 Test": lambda x: x.pinball_test_10_percentile_loss,
        "0.9 Train": lambda x: x.pinball_train_90_percentile_loss,
        "0.9 Test": lambda x: x.pinball_test_90_percentile_loss,
    }

    fig, axes = plt.subplots(2, 2, figsize=(20, 20))

    for idx, title in enumerate(title_mapping.keys()):
        # Put each metric on a different row
        col = idx
        data = pd.Series({step: title_mapping[title](r) for step, r in results_by_step.items()})
        axes[0][col].scatter(data.index, data.values)
        axes[0][col].set_title(title)
        axes[0][col].set_xlabel("Forecast Horizon (Hours from init_time_utc)")
    for idx, title in enumerate(title_mapping2.keys()):
        if "Test" in title:
            continue
        # Put each metric on a different row
        data = pd.Series({step: title_mapping2[title](r) for step, r in results_by_step.items()})
        axes[1][0].scatter(data.index, data.values, label=title.split(" ")[0].strip())
        # Add title to legend for axis

    axes[1][0].set_title("Pinball Losses Train")
    axes[1][0].set_xlabel("Forecast Horizon (Hours from init_time_utc)")
    axes[1][0].legend()
    for idx, title in enumerate(title_mapping2.keys()):
        if "Train" in title:
            continue
        # Put each metric on a different row
        col = idx
        data = pd.Series({step: title_mapping2[title](r) for step, r in results_by_step.items()})
        axes[1][1].scatter(data.index, data.values, label=title.split(" ")[0].strip())
    axes[1][1].set_title("Pinball Losses Test")
    axes[1][1].set_xlabel("Forecast Horizon (Hours from init_time_utc)")
    axes[1][1].legend()

    plt.show()


def plot_feature_importances(
    results_by_step: dict[int, ExperimentSummary], forecast_horizons=[1, 12, 24, 34]
):
    """Convenience function for plotting feature importances over several forecast horizons"""
    fig, axes = plt.subplots(2, len(forecast_horizons), figsize=(28, 18))

    for param_idx, param in enumerate(["weight", "gain"]):
        for idx, fh in enumerate(forecast_horizons):
            data = pd.DataFrame.from_dict(
                results_by_step[fh].model.get_booster().get_score(importance_type=param),
                orient="index",
            ).sort_values(by=0, ascending=False)
            axes[param_idx][idx].bar(range(len(data)), data.values.flatten())
            axes[param_idx][idx].set_xticks(range(len(data)))
            axes[param_idx][idx].set_xticklabels(data.index, rotation=90)
            axes[param_idx][idx].set_title(f"Feat. Importance: {param}, Forecast-Horizon: {fh}")

    plt.show()
