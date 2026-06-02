import xarray as xr
import re
import pandas as pd
import numpy as np
from scipy import stats
import time
from datetime import datetime as dt


def extract_year_mls(filename):
    """Extract a year from a filename ending '_YYYY.nc'."""
    # Look for the pattern '_YYYY.nc'
    match = re.search(r"_(\d{4})\.nc$", filename)

    return int(match.group(1)) if match else float("inf")


def monthly_anomalies(data: xr.DataArray) -> xr.DataArray:
    """Calculate monthly anomalies."""
    # Group data by month and calculate the monthly means (climatology)
    monthly_climatology = data.groupby("time.month").mean("time", skipna=True)

    # Calculate anomalies by subtracting the monthly climatology
    anomalies = data.groupby("time.month") - monthly_climatology

    return anomalies.rename("monthly_anomalies")


def process_dataset(base_in, filename, var_name):
    """Extract, scale, and averages climate dataset variables.

    Averages data at 10 hPa over the tropics (10S-10N).
    """
    ds = (
        xr.open_dataset(f"{base_in}{filename}")
        .sel(lvl=10, lat=slice(-10, 10))
        .mean(dim="lat")
    )
    df = pd.DataFrame(
        {name: (ds[var].data * factor) for var, (name, factor) in var_name.items()}
    )
    # Convert time to datetime format
    df["time"] = pd.to_datetime(ds.time.data)
    df.set_index("time", inplace=True)

    return df


def align_and_dropna_same_rows(df_dict, keys, how="intersection"):
    """
    Make all DataFrames in `keys` share the same index AND the same rows
    (drop any row that has NaN in ANY df).
    `how="intersection"` keeps only timestamps present in all dfs.
    """
    keys = [k for k in keys if k in df_dict]
    if len(keys) == 0:
        raise ValueError("No keys found to align.")

    # 1) align time index across all keys
    if how == "intersection":
        common_time = df_dict[keys[0]].index
        for k in keys[1:]:
            common_time = common_time.intersection(df_dict[k].index)
    else:
        raise ValueError("Only how='intersection' is implemented.")

    for k in keys:
        df_dict[k] = df_dict[k].loc[common_time].sort_index()

    # 2) build one shared validity mask (no NaNs anywhere)
    valid = np.ones(len(common_time), dtype=bool)
    for k in keys:
        valid &= ~df_dict[k].isna().any(axis=1)

    common_time_valid = common_time[valid]
    for k in keys:
        df_dict[k] = df_dict[k].loc[common_time_valid]

    return df_dict, keys, common_time_valid


def detrending(timeseries):
    """Remove a linear trend from a time series, ignoring NaN values.

    Fit a linear regression to the non-NaN values of the input data,
    compute the trend over the full index, and subtracts it from the
    original series.
    """
    timeseries = np.asarray(timeseries)

    # Create a mask for non-NaN values
    mask = ~np.isnan(timeseries)

    # Indices for non-NaN values
    indices = np.arange(len(timeseries))[mask]

    # Non-NaN values of the time series
    clean_timeseries = timeseries[mask]

    # Perform linear regression on the clean data
    reg = stats.linregress(indices, clean_timeseries)

    # Calculate the trend
    trend = reg.intercept + reg.slope * np.arange(len(timeseries))

    # Subtract the trend from the original time series
    detrended_timeseries = timeseries - trend

    return detrended_timeseries


def detrend_df(df, time_col="time"):
    """
    Remove a linear trend from all numeric columns of a DataFrame.

    Converts the time column to fractional years since the first timestamp,
    fits a linear regression to each numeric column ,
    and subtracts the trend from the original values.
    """
    df_out = df.copy()
    df_out[time_col] = pd.to_datetime(df_out[time_col])

    # time in fractional years since start
    t = (df_out[time_col] - df_out[time_col].iloc[0]).dt.total_seconds()
    t = t.values / (365.25 * 24 * 3600)

    numeric_cols = df_out.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        y = df_out[col].values
        mask = np.isfinite(y)

        if mask.sum() < 2:
            continue

        slope, intercept = np.polyfit(t[mask], y[mask], 1)
        trend = intercept + slope * t

        df_out[col] = y - trend

    return df_out


def load_qbo_proxies(
    file_path,
    start=None,
    end=None,
    target_index=None,
    levels=(10, 30),
    standardize=True,
    return_shear=False,
    var_name="u",
    level_dim="pressure",
    lat_dim=None,
    lat_slice=None,
    lat_mean=False,
):
    """
    Load QBO winds (OBS or TOMCAT) and return selected levels as a pandas
    DataFrame, optionally averaged over latitude, aligned to a target index.
    The following works:
    for OBS (var=u, level_dim=pressure)
    for TOMCAT ( var=uwind_mm, level_dim=lvl).

    Parameters
    ----------
    file_path : str
        Path to NetCDF file.
    start, end : str/datetime, optional
        Time slice.
    target_index : pandas.DatetimeIndex, optional
        Reindex output to this index.
    levels : tuple
        Levels to extract, e.g. (10, 30).
    standardize : bool
        Z-score each level series (and shear if requested).
    return_shear : bool
        If True, add shear = level1 - level2.
    var_name : str
        Variable name in ds.
    level_dim : str
        Dimension/coordinate name for vertical level (e.g., "pressure" or "lvl").
    lat_dim : str or None
        Latitude dimension name (e.g., "lat"). If None, no lat selection/mean.
        lat_dim should be "lat" for TOMCAT; None for OBS
    lat_slice : (float,float) or None
        Latitude slice (min_lat, max_lat), e.g. (-10, 10).
    lat_mean : bool
        If True, average over latitude after slicing.
        lat_mean should be True for TOMCAT.

    Returns
    -------
    qbo_df : pandas.DataFrame
        Columns: qbo{level} for each requested level, and optionally "shear".
    """
    ds = xr.open_dataset(file_path)

    # time subset
    if start is not None and end is not None:
        ds = ds.sel(time=slice(start, end))

    # optional lat subset + mean
    if lat_dim is not None:
        if lat_slice is not None:
            ds = ds.sel({lat_dim: slice(lat_slice[0], lat_slice[1])})
        if lat_mean:
            ds = ds.mean(dim=lat_dim)

    data = {}
    for lev in levels:
        s = ds[var_name].sel({level_dim: lev}).to_pandas()

        if target_index is not None:
            s = s.reindex(target_index)

        if standardize:
            s = (s - s.mean()) / s.std(ddof=0)

        data[f"qbo{lev}"] = s

    qbo_df = pd.DataFrame(
        data, index=target_index if target_index is not None else None
    )

    if return_shear and len(levels) == 2:
        l1, l2 = levels
        shear = qbo_df[f"qbo{l1}"] - qbo_df[f"qbo{l2}"]

        if standardize:
            shear = (shear - shear.mean()) / shear.std(ddof=0)

        qbo_df["shear"] = shear

    return qbo_df


def toYearFraction(date):
    """
    Convert a datetime object to a fractional year
    (e.g. from mid-2023 to 2023.5).
    """

    def sinceEpoch(date):
        # returns seconds since epoch
        return time.mktime(date.timetuple())

    s = sinceEpoch

    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year + 1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed / yearDuration

    return date.year + fraction


def summarize_year_fraction(time_fin):
    """
    Convert a sequence of datetime objects to an array of fractional years.
    """
    date_upd = list()
    for i in range(0, len(time_fin)):
        # add each converted value into list
        date_upd.append(toYearFraction(time_fin[i]))
    # convert list into array
    year_fraction = np.asarray(date_upd)

    return year_fraction


def build_qbo_regime_masks_from_shear(shear: "pd.Series", th: float):
    """
    Returns masks for Tigramite where 0 means 'use/select this time step' and
    1 means 'mask it out'.

    Parameters
    ----------
    shear : pandas.Series
        QBO shear time series.
    th : float
        Threshold defining easterly/westerly regimes.
    """
    # use all points
    no_mask = np.zeros(len(shear), dtype=int)

    # easterly shear regime selected
    mask_qbo_east = np.where(shear <= -th, 0, 1)
    # westerly shear regime selected
    mask_qbo_west = np.where(shear >= th, 0, 1)

    return {
        "No_mask": no_mask,
        "east": mask_qbo_east,
        "west": mask_qbo_west,
    }
