"""Preprocess NWP data using geospatial mask"""

import itertools
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Iterable, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import xarray as xr
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from gradboost_pv.models.utils import DEFAULT_DIRECTORY_TO_PROCESSED_NWP

logger = logging.getLogger(__name__)

ESO_GEO_JSON_URL = (
    "https://data.nationalgrideso.com/backend/dataset/2810092e-d4b2-472f-b955-d8bea01f9ec0/"
    "resource/08534dae-5408-4e31-8639-b579c8f1c50b/download/gsp_regions_20220314.geojson"
)

# processing takes quite a long time, so take a subset for now.
DEFAULT_VARIABLES_FOR_PROCESSING = [
    "dswrf",
    # "hcct",
    "lcc",
    "t",
    # "sde",
    "wdir10",
]

MAX_FORECAST_HORIZON = 36


def build_local_save_path(
    forecast_horizon_step: int,
    variable: str,
    year: int,
    directory: Path = DEFAULT_DIRECTORY_TO_PROCESSED_NWP,
) -> Tuple[Path, Path]:
    """Paths to inner and outer masked NWP data for specific year/variable/forecast horizon

    Args:
        forecast_horizon_step (int): Forecast step index
        variable (str): NWP variable
        year (int): Year of processed data
        directory (Path, optional): Directory to data.
        Defaults to DEFAULT_DIRECTORY_TO_PROCESSED_NWP.

    Returns:
        Tuple[Path, Path]: Paths to respective datasets
    """

    if forecast_horizon_step > MAX_FORECAST_HORIZON:
        logger.debug(
            f"Forecast horizon step {forecast_horizon_step} exceeds maximum of "
            f"{MAX_FORECAST_HORIZON}, so reducing to {MAX_FORECAST_HORIZON}"
        )
        forecast_horizon_step = MAX_FORECAST_HORIZON

    return (
        directory
        / str(year)
        / f"uk_region_inner_variable_{variable}_step_{forecast_horizon_step}.pickle",
        directory
        / str(year)
        / f"uk_region_outer_variable_{variable}_step_{forecast_horizon_step}.pickle",
    )


def query_eso_geojson() -> gpd.GeoDataFrame:
    """Query National grid ESO for spatial structure data for UK GSPs

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing UK-region information
    """
    with requests.get(ESO_GEO_JSON_URL) as response:
        shape_gpd = gpd.read_file(response.text)
    return shape_gpd


def process_eso_uk_multipolygon(uk_shape: gpd.GeoDataFrame) -> MultiPolygon:
    """Processes the response from `query_eso_geojson` into a MultiPolygon object

    Args:
        uk_shape (gpd.GeoDataFrame): Input from National Grid ESO

    Returns:
        MultiPolygon: Object representing the UK-region.
    """
    concat_poly = unary_union(uk_shape["geometry"].values)

    return MultiPolygon(Polygon(p.exterior) for p in concat_poly.geoms)


def generate_polygon_mask(
    coordinates_x: Iterable[int], coordinates_y: Iterable[int], polygon: MultiPolygon
) -> np.ndarray:
    """Multiprocessed wrapper function to check if lists of coordinates lie within a polygon.

    Args:
        coordinates_x (Iterable[int]): x-coordinates of points of interest in OSGB
        coordinates_y (Iterable[int]): y-coordinates of points of interest in OSGB
        polygon (MultiPolygon): polygon to infer if points of interest lie within.

    Returns:
        np.ndarray: 2-D array where each (x_i, y_i) value signifies if the point (x_i, y_i) belong
        to the polygon.
    """
    coords = list(map(lambda x: Point(x[0], x[1]), itertools.product(coordinates_x, coordinates_y)))

    # create a mask for belonging to UK region or not
    mask = check_points_in_multipolygon_multiprocessed(coords, polygon)
    mask = mask.reshape(len(coordinates_x), len(coordinates_y)).T
    mask = mask.astype(float)
    mask[mask == 0] = np.nan

    return mask


def check_point_in_multipolygon(point: Point, polygon: Union[MultiPolygon, Polygon]) -> bool:
    """Check if a point exists in a polygon

    Args:
        point (Point): Point of interest
        polygon (Union[MultiPolygon, Polygon]): Polygon to check

    Returns:
        bool: True if Point is in the Polygon
    """
    return polygon.contains(point)


def check_points_in_multipolygon_multiprocessed(
    points: Iterable[Point],
    polygon: Union[MultiPolygon, Polygon],
    num_processes: int = 3,
) -> np.ndarray:
    """Multiprocessed wrapper for checking points within a polygon

    Args:
        points (Iterable[Point]): collection of Points
        polygon (Union[MultiPolygon, Polygon]): polygon to check
        num_processes (int, optional): Defaults to 3.

    Returns:
        np.ndarray: _description_
    """
    items = [(point, polygon) for point in points]
    results = list()
    with mp.Pool(num_processes) as pool:
        for result in pool.starmap(check_point_in_multipolygon, items):
            results.append(result)
    return np.asarray(results)


def _process_nwp(
    nwp_slice: xr.Dataset, mask: xr.DataArray, x_coord: str = "x", y_coord: str = "y"
) -> Tuple[xr.Dataset, xr.Dataset]:
    """Processing logic for region masked downsampling

    Args:
        nwp_slice (xr.Dataset): slice of NWP data
        mask (xr.DataArray): geospatial mask of nan/non-nan values
        x_coord (str, optional): coordinate name of x dimension in NWP dataset. Defaults to "x".
        y_coord (str, optional): coordinate name of y dimension in NWP dataset. Defaults to "y".

    Returns:
        Tuple[xr.Dataset, xr.Dataset]: _description_
    """
    uk_region = xr.where(~mask.isnull(), nwp_slice, np.nan).mean(dim=[x_coord, y_coord])
    outer_region = xr.where(mask.isnull(), nwp_slice, np.nan).mean(dim=[x_coord, y_coord])

    return uk_region, outer_region


class NWPUKRegionMaskedDatasetBuilder:
    """Class for iteratively processing NWP data."""

    def __init__(self, nwp: xr.Dataset, evaluation_timepoints: pd.DatetimeIndex) -> None:
        """Initalise dataset builder.

        Args:
            nwp (xr.Dataset): NWP xarray dataset [variable, step, init_time, x, y]
            evaluation_timepoints (pd.DatetimeIndex): datetime points to interpolate onto
        """
        self.nwp = nwp
        self.eval_timepoints = evaluation_timepoints
        self.mask = self.load_mask()

    def load_mask(self) -> xr.DataArray:
        """Loads UK region mask from National Grid ESO

        Returns:
            xr.DataArray: UK-region mask, on NWP (x,y) coords
        """
        uk_polygon = query_eso_geojson()
        uk_polygon = process_eso_uk_multipolygon(uk_polygon)
        mask = generate_polygon_mask(self.nwp.coords["x"], self.nwp.coords["y"], uk_polygon)

        # convert numpy array to xarray mask for a 1 variable, 1 step times series of (x,y) coords
        mask = xr.DataArray(
            np.tile(mask.T, (len(self.nwp.coords["init_time"]), 1, 1)),
            dims=["init_time", "x", "y"],
        )
        return mask

    def build_region_masked_covariates(
        self,
        variable: str,
        step: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process NWP sliced by variable and forecast step

        Args:
            variable (str): variable to slice
            step (int): forecast horizon index to slice

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: inner and outer downsampled data
        """
        _nwp = self.nwp.isel(step=step).sel(variable=variable)

        uk_region, outer_region = _process_nwp(_nwp, self.mask)

        # interpolate both to the common GSP time points
        uk_region = (
            uk_region.interp(init_time=self.eval_timepoints, method="linear").to_array().as_numpy()
        )
        outer_region = (
            outer_region.interp(init_time=self.eval_timepoints, method="linear")
            .to_array()
            .as_numpy()
        )

        # cast to dataframe
        uk_region = pd.DataFrame(
            index=self.eval_timepoints, columns=[f"{variable}_within"], data=uk_region
        )
        outer_region = pd.DataFrame(
            index=self.eval_timepoints, columns=[f"{variable}_outer"], data=outer_region
        )

        return (uk_region, outer_region)
