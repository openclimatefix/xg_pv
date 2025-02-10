"""Process NWP data with pretrained model"""

from math import ceil, floor
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe

from gradboost_pv.models.utils import DEFAULT_DIRECTORY_TO_PROCESSED_NWP, NWP_VARIABLE_NUM

AUTO_REGRESSION_TARGET_LAG = np.timedelta64(1, "h")  # to avoid look ahead bias
AUTO_REGRESSION_COVARIATE_LAG = AUTO_REGRESSION_TARGET_LAG + np.timedelta64(1, "h")
PRETRAINED_OUTPUT_DIMS = 1_000


def build_local_save_path(
    forecast_horizon_step: int, directory: Path = DEFAULT_DIRECTORY_TO_PROCESSED_NWP
) -> Path:
    """Builds filepath based on the forecast horizon

    Args:
        directory (Path): Path to data. Defaults to DEFAULT_DIRECTORY_TO_PROCESSED_NWP.
        forecast_horizon_step (int): Forecast step slice of processed NWP data

    Returns:
        Path: Filepath
    """
    return directory / f"pretrained_nwp_processed_step_{forecast_horizon_step}.pickle"


@functional_datapipe("process_nwp_pretrained")
class ProcessNWPPretrainedIterDataPipe(IterDataPipe):
    """Datapipe for applying pretrained CNN to NWP data."""

    def __init__(
        self,
        base_nwp_datapipe: IterDataPipe,
        pretrained_model: Callable[[torch.Tensor], torch.Tensor],
        step: int,
        batch_size: int = 50,
        interpolate: bool = False,
        interpolation_timepoints: Optional[Iterable[np.datetime64]] = None,
    ) -> None:
        """Initialise NWP Datapipe

        Args:
            base_nwp_datapipe (IterDataPipe): Datapipe of NWP data
            pretrained_model (Callable[[torch.Tensor], torch.Tensor]): CNN model
            step (int): index of forecast horizon, for slicing
            batch_size (int, optional): Defaults to 50.
            interpolate (bool, optional): Whether to interpolate init_time on
            interpolation_timepoints, defaults to False.
            interpolation_timepoints (Optional[ Iterable[np.datetime64] ], optional): for init_time
        """

        if interpolate:
            assert interpolation_timepoints is not None, "Must provide points for interpolation."
        self.source_datapipe = base_nwp_datapipe
        self.step = step
        self.pretrained_model = pretrained_model
        self.batch_size = batch_size
        self.interpolate = interpolate
        self.interpolation_timepoints = interpolation_timepoints

    def process_for_pretrained_input(self, nwp_batch: xr.DataArray) -> torch.Tensor:
        """Convert NWP zarr data to torch tensors

        Args:
            nwp_batch (xr.DataArray): xarray data loaded from zarr.

        Returns:
            torch.Tensor: NWP as torch tensors for pretrained model
        """
        nwp_batch = nwp_batch.to_numpy()
        n = nwp_batch.shape[0]  # batch size for all but the last batch
        nwp_batch = nwp_batch.reshape(NWP_VARIABLE_NUM * n, 64, 68)
        nwp_batch = np.tile(nwp_batch, (3, 1, 1, 1)).reshape(NWP_VARIABLE_NUM * n, 3, 64, 68)
        nwp_batch = torch.from_numpy(np.nan_to_num(nwp_batch))
        return nwp_batch

    def downsample_pretrained_output(self, model_output: torch.Tensor) -> np.ndarray:
        """Downsample the output of the pretrained model.

        e.g CNN gives 1_000 x batch_size output, we wish to simplify this further,
        downsampling it further to 5 x batch_size.

        Args:
            model_output (torch.Tensor): output of pretrained model

        Returns:
            np.ndarray: downsampled output as np.array
        """
        with torch.no_grad():
            output = torch.softmax(model_output, 1)
            output = np.split(output, range(200, PRETRAINED_OUTPUT_DIMS, 200), axis=1)
            n = torch.linalg.norm(output[0], axis=1).shape[0]
            dsampled_output = torch.concat(
                [torch.linalg.norm(x, axis=1).reshape(1, n) for x in output]
            ).T
        return dsampled_output.numpy()

    def __iter__(self) -> Iterator[Tuple[pd.DatetimeIndex, np.ndarray]]:
        """Iteratively: NWP data -> Pretrained CNN -> Downsampled output

        Yields:
            Iterator[Tuple[pd.DatetimeIndex, np.ndarray]]
        """
        for nwp in self.source_datapipe:
            nwp = nwp.isel(step=self.step)  # select the horizon we want
            if self.interpolate:
                nwp = nwp.interp(init_time_utc=self.interpolation_timepoints, method="linear")
            for batch_idx in range(1, ceil(len(nwp.init_time_utc) / self.batch_size) + 1):
                batch = nwp.isel(
                    init_time_utc=slice(
                        self.batch_size * (batch_idx - 1), self.batch_size * batch_idx
                    )
                )
                batch = batch.coarsen(
                    dim={
                        "x_osgb": floor(548 / 64),
                        "y_osgb": floor(704 / 64),
                    },
                    boundary="trim",
                ).mean()
                time_slice = batch.coords["init_time_utc"].values
                with torch.no_grad():
                    batch = self.process_for_pretrained_input(batch)
                    output = self.pretrained_model(batch)
                    output = self.downsample_pretrained_output(output)
                yield (time_slice, output)
