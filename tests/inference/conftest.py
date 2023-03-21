"""Generate s3 client, and model configs for testing"""
from pathlib import Path

import pytest
import xarray as xr
from xgboost import XGBRegressor

from gradboost_pv.inference.models import NationalPVModelConfig
from gradboost_pv.models.s3 import build_object_name, create_s3_client, load_model
from tests.inference.test_app import PATH_TO_SAMPLE_NWP, PATH_TO_SAMPLE_GSP

MODEL_FORECAST_HORIZON_HOUR = 5
PATH_TO_TEST_MODEL_CONFIG = Path(__file__).parents[1] / "configs" / "test_model_config.yaml"


@pytest.fixture
def mock_model():
    """Loads a model at example forecast horizon"""

    model = XGBRegressor()

    # 33 features
    xtrain = [[1] * 33]
    ytrain = [1]

    model.fit(xtrain, ytrain)

    return model


@pytest.fixture
def model_config() -> NationalPVModelConfig:
    """Loads test model configuration from yaml file."""
    return NationalPVModelConfig.load_from_yaml(PATH_TO_TEST_MODEL_CONFIG)


@pytest.fixture
def model_config_path() -> Path:
    """configuration yaml file."""
    return PATH_TO_TEST_MODEL_CONFIG


@pytest.fixture
def sample_prod_nwp_data() -> xr.Dataset:
    """Loads a sample NWP [x, y] observation from file.

    This data looks like the (processed) NWP data from prod aws.
    Returns:
        xr.Dataset: NWP Observation.
    """
    print(PATH_TO_SAMPLE_NWP)
    nwp = xr.open_zarr(PATH_TO_SAMPLE_NWP)
    return nwp


@pytest.fixture
def sample_prod_gsp_data() -> xr.Dataset:
    """Loads a sample GSP observation from file.

    This data looks like the prod GSP data read from database.
    Returns:
        xr.Dataset: GSP Observations.
    """
    print(PATH_TO_SAMPLE_GSP)
    gsp = xr.open_zarr(PATH_TO_SAMPLE_GSP)
    return gsp