"""Datafeeds for model inference"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np
import pandas as pd
import psutil
import pytz
import s3fs
import xarray as xr
from ocf_datapipes.config.load import load_yaml_configuration
from ocf_datapipes.config.model import Configuration
from ocf_datapipes.load import OpenGSPFromDatabase
from torch.utils.data import IterDataPipe, functional_datapipe

from gradboost_pv.models.utils import load_nwp_coordinates

logger = logging.getLogger(__name__)


# TODO - fix interpolation of data points and move to datapipes?
@functional_datapipe("production_open_nwp_netcdf")
class ProductionOpenNWPNetcdfIterDataPipe(IterDataPipe):
    """Datapipe for accessing the latest netcdf NWP file from s3."""

    def __init__(
        self,
        s3_path_to_data: Path,
        nwp_channels: Optional[list[str]] = None,
        s3_access_key: Optional[str] = None,
        s3_ssecret_key: Optional[str] = None,
    ) -> None:
        """Initalise Datapipe with s3 path and optional s3 keys.

        Args:
            s3_path_to_data (Path): path to s3 file, does not require s3:// prefix
            nwp_channels (Optional[list[str]], optional): Optional list of NWP channels to load.
                Defaults to None.
            s3_access_key (Optional[str], optional): Optional access key. Defaults to None.
            s3_ssecret_key (Optional[str], optional): Optional secret key. Defaults to None.
        """
        self.s3_path_to_data = s3_path_to_data
        if s3_access_key is not None and s3_ssecret_key is not None:
            self.fs = s3fs.S3FileSystem(key=s3_access_key, secret=s3_ssecret_key)
        else:
            self.fs = s3fs.S3FileSystem()

        self.nwp_channels = nwp_channels

    def _process_nwp_from_netcdf(self, nwp: xr.Dataset) -> xr.Dataset:
        """Processes the NWP data in the same fashion as 'open_nwp' method

        Processes netcdf file to the same NWP format as datapipes processing.

        Args:
            nwp (xr.Dataset): NWP data loaded from prod s3

        Returns:
            xr.Dataset: Processed NWP data, with only the latest time slice.
        """
        nwp = nwp.transpose("init_time", "step", "variable", "y", "x")
        nwp = nwp.rename(
            {"init_time": "init_time_utc", "variable": "channel", "y": "y_osgb", "x": "x_osgb"}
        )

        # select most recent time point
        nwp = nwp.isel(init_time_utc=-1)

        # TODO - create a fix for this
        # prod data is y: 633, x: 449, training data was y: 704, x: 548, interpolate onto
        # the old coordinates.
        process = psutil.Process(os.getpid())
        logger.debug(f"Memory is {process.memory_info().rss / 10 ** 6} MB")
        x_coords, y_coords = load_nwp_coordinates()

        # there is some bug with extrapolation/interpolation - see below
        # https://github.com/pydata/xarray/discussions/6189
        # nwp = nwp.interp(
        #     x_osgb=x_coords,
        #     y_osgb=y_coords,
        #     method="linear",
        #     kwargs={"fill_value": "extrapolation"},  # TODO - fix this!
        # )

        # quickest workaround, not production ready!!!!!
        nwp = nwp.reindex({"x_osgb": x_coords, "y_osgb": y_coords}, method="nearest")

        return nwp

    def __iter__(self) -> Iterator[xr.Dataset]:
        """Yields the latest NWP data from s3."""
        while True:
            # load zarr file

            logger.debug(f"Loading NWP data: {self.s3_path_to_data}")

            nwp = xr.open_dataset(
                self.s3_path_to_data,
                engine="zarr",
                consolidated=True,
                mode="r",
                chunks=None,  # Reading Satellite Zarr benefits from setting chunks='auto'
                # (see issue #456) but 'auto' massively slows down reading NWPs.
            )

            if self.nwp_channels is not None:
                logger.info(
                    f"Selecting NWP channels: {self.nwp_channels} " f"from {nwp.variable.values}"
                )
                nwp = nwp.sel(variable=self.nwp_channels)

            nwp = self._process_nwp_from_netcdf(nwp)
            yield nwp


# TODO - move to datapipes? or remove other function from datapipes
def xgnational_production(configuration_filename: Union[Path, str]) -> dict:
    """
    Create the National XG Boost  using a configuration

    Args:
        configuration_filename: Name of the configuration
    Returns:
        dictionary of 'nwp' and 'gsp' containing xarray for both
    """

    configuration: Configuration = load_yaml_configuration(filename=configuration_filename)

    nwp_datapipe = ProductionOpenNWPNetcdfIterDataPipe(
        configuration.input_data.nwp["ukv"].nwp_zarr_path,
        nwp_channels=list(configuration.input_data.nwp["ukv"].nwp_channels),
    )
    gsp_datapipe = OpenGSPFromDatabase(
        history_minutes=configuration.input_data.gsp.history_minutes,
        interpolate_minutes=configuration.input_data.gsp.live_interpolate_minutes,
        load_extra_minutes=configuration.input_data.gsp.live_load_extra_minutes,
        national_only=True,
    )

    nwp_xr = next(iter(nwp_datapipe))
    gsp_xr = next(iter(gsp_datapipe))

    return {"nwp": nwp_xr, "gsp": gsp_xr.to_dataset()}


@dataclass
class DataInput:
    """Input representation from Database read."""

    nwp: xr.Dataset
    gsp: xr.Dataset
    forecast_intitation_datetime_utc: np.datetime64


@functional_datapipe("mock_datafeed")
class MockDataFeed(IterDataPipe):
    """Mock Data Feed to simulate reading from a database of NWP and GSP values"""

    def __init__(self, nwp_data: xr.Dataset, national_gsp_data: xr.Dataset) -> None:
        """Setup the mock data feed, pre-initalisation

        Args:
            nwp_data (xr.Dataset): Dataset of NWP variables
            national_gsp_data (xr.Dataset): Dataset of GSP PV data
        """
        self.nwp = nwp_data
        self.national_gsp_data = national_gsp_data

    def initialise(self):
        """Create a data feed."""
        self.data_feed = self.create_data_feed(self.nwp, self.national_gsp_data)

    def create_data_feed(
        self,
        nwp: xr.Dataset,
        gsp: xr.Dataset,
        gsp_added_history: np.timedelta64 = np.timedelta64(24, "h"),
    ) -> list[DataInput]:
        """Create a tuple of individual time-slices for coupled nwp and gsp.

        Since the nwp and gsp are sampled at different frequencies, we realign
        them so that at the tick of one data, we also have the most recent data
        of the other source at hand. This is meant to simulate iteratively
        reading the most recent elements a database at given intervals.

        Args:
            nwp (xr.Dataset): NWP dataset with "init_time" np.datetime64 coords
            gsp (xr.Dataset): GSP PV data with "datetime_gmt" np.datetime64 coords
            gsp_added_history (np.timedelta64): Instead of one obs per timestamp, supply
            a rolling history of pv data.
        """

        tseries_nwp = nwp.coords["init_time"].values
        tseries_gsp = gsp.coords["datetime_gmt"].values
        assert tseries_nwp[0] == tseries_gsp[0], "Datasets must start at the same point in time"

        tseries = np.sort(
            np.unique(
                np.concatenate(
                    (
                        tseries_gsp[tseries_gsp >= tseries_gsp[0] + gsp_added_history],
                        tseries_nwp[tseries_nwp >= tseries_nwp[0] + gsp_added_history],
                    ),
                    axis=0,
                )
            )
        )

        return [
            DataInput(
                nwp.sel(init_time=np.datetime64(t, "ns"), method="ffill"),
                gsp.sel(
                    datetime_gmt=slice(
                        np.datetime64(t, "ns") - gsp_added_history,
                        np.datetime64(t, "ns"),
                    )
                ),
                forecast_intitation_datetime_utc=t,
            )
            for t in tseries
        ]

    def __iter__(self) -> Iterator[DataInput]:
        """Iteratively supply data from data feed.

        Yields:
            Iterator[DataInput]: Data, as read from a database
        """
        assert self.data_feed is not None
        for datapoint in self.data_feed:
            yield datapoint


@functional_datapipe("production_datafeed")
class ProductionDataFeed(IterDataPipe):
    """DataPipe reading NWP and GSP values from ocf_datapipes function"""

    def __init__(self, path_to_configuration_file: Union[str, Path]) -> None:
        """Datafeed."""
        self.path_to_configuration_file = path_to_configuration_file

    @classmethod
    def get_inference_time(cls) -> np.datetime64:
        """Round a UTC datetime to the nearest 1/2 hour upwards and minus 1 hour.

        E.g. 09:17 -> 08:30, 15:47 -> 15:00

        This is because the first model prediction is for one hours time,
        so we should move the inference time to 1 hour before rounded up now

        Returns:
            np.datetime64: resultant datetime in np.datetime64 format.
        """
        now = datetime.now(pytz.UTC)

        # round up to nearest 30 minutes
        dt = (
            now
            + (datetime.min.replace(tzinfo=pytz.UTC) - now) % timedelta(minutes=30)
            - timedelta(hours=1)
        )

        return np.datetime64(dt)

    def __iter__(self) -> Iterator[DataInput]:
        """Returns a single observation of NWP data and 24 hours of GSP data.

        Yields:
            Iterator[DataInput]: Input data for model feature generation.
        """
        logger.debug("Getting Data")

        data = xgnational_production(self.path_to_configuration_file)
        yield from self.post_process(data)

    def post_process(self, data):
        """Post process data"""
        inference_time = self.get_inference_time()
        logger.debug(f"{inference_time=}")
        logger.debug(f"{data['nwp'].init_time_utc.values=}")
        logger.debug(data["gsp"])
        logger.debug(data["nwp"])
        logger.debug("The following times should be the same, so will adjust if not")
        logger.debug(f"{inference_time=}")
        logger.debug(f"{data['nwp'].init_time_utc.values=}")

        # need to adjust NWP values so they are the
        # - have the same 'nwp_init_time_utc' as 'inference_time'
        # and the steps are in hours
        nwp_init_time_utc = data["nwp"].init_time_utc.values
        # TODO hack to stop any interpolating, make sure delta is an hour int
        delta = pd.to_datetime(inference_time).replace(minute=0) - nwp_init_time_utc

        logger.debug(f"Need to move NWP data forward {delta}")
        old_steps = data["nwp"].step.values
        logger.debug(f"Old steps are {old_steps}")
        new_step = pd.to_timedelta(data["nwp"].step - delta)
        data["nwp"].coords["step"] = new_step
        non_negative_steps = [step for step in new_step if step >= timedelta(minutes=0)]
        logger.debug(f"Removing negative step values {non_negative_steps}")
        data["nwp"] = data["nwp"].sel(step=slice(non_negative_steps[0], non_negative_steps[-1]))
        process = psutil.Process(os.getpid())
        logger.debug(f"Memory is {process.memory_info().rss / 10 ** 6} MB")
        logger.debug("Load data into memory")  # This takes ~3 mins

        # load data into memory
        data["nwp"] = self.load(data["nwp"])

        # change to new step and resample to 1 hour
        logger.debug("Resampling to 1 hour")
        process = psutil.Process(os.getpid())
        logger.debug(f"Memory is {process.memory_info().rss / 10 ** 6} MB")
        data["nwp"] = data["nwp"].resample(step="1H").mean()
        data["nwp"].init_time_utc.values = inference_time
        logger.debug(f'Final steps are {data["nwp"].step.values}')

        yield DataInput(
            nwp=data["nwp"], gsp=data["gsp"], forecast_intitation_datetime_utc=inference_time
        )

    def load(self, data: xr.Dataset):
        """Load the data into memory"""
        # By loading by step by step, this seems to keep the memory lower,
        # roughly this reduces it from 9 GB to 3 GB
        # The old command was data["nwp"].load()
        steps = []
        step_values = data.step.values
        for i in range(len(step_values)):
            logger.debug(f"Loading step {step_values[i]}. {i} out of {len(step_values)}")
            step = data.sel(step=step_values[i])
            step.load()
            steps.append(step)
        data_xr = xr.concat(steps, "step")

        return data_xr
