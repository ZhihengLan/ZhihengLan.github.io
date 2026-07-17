#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name :  EC_processing_functions.py
   Description :  [Used to save functions used for calculating 30min flux from 10 hz EC measurements
                    and plots on website]
   Author :       Micromet
   Date :         6/2/2026 1:03 PM
   Project :      ZhihengLan
   Email: zhiheng.lan@wsu.edu
-------------------------------------------------
   Change Activity:
                   6/2/2026 - Created
-------------------------------------------------
"""
import netCDF4 as nc
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from matplotlib.pyplot import tight_layout
from pyparsing import lineStart
from scipy.stats import linregress
import matplotlib

import glob
import warnings
import sys
from datetime import datetime, timedelta
import socket
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import xarray as xr
import re

def despike_MADstd(data, threshold=3.5, time_window=30, freq=10, continuous_spike_bool=False, continuous_spike_num=3):
    """
    Despike 1-day 10-hz nc variable based on 30-mins MAD filtering

    Reference
    ---------
    Mauder, M., Cuntz, M., Drüe, C., Graf, A., Rebmann, C., Schmid, H. P., Schmidt, M., & Steinbrecher, R. (2013).
    A strategy for quality and uncertainty assessment of long-term eddy-covariance measurements. Agricultural and Forest Meteorology, 169, 122–135.
    https://doi.org/10.1016/j.agrformet.2012.09.006

    Parameters
    ----------
    data: np.ndarray
        Input 1D (or 2D) array of signal(s) containing potential spikes. Columns are different layer measurements.
        Input should have continuous time record (no missing rows).
    threshold: float, optional
        Standard deviation multiplier for despiking. Default is 3.5.
    time_window: int, optional
        Time window for calculate MAD (minutes). Default is 30.
    freq: int, optional
        Sampling frequency (Hz). Default is 10.
    continuous_spike_bool: bool, optional
        Whether continuous spike detection should be performed. Default is False.
    continuous_spike_num: int, optional
        Number of continuous spike detection should be performed. Default is 3 (spikes longer than 3 samples will be ignored).
        Useful when ramp structure occur frequently.

    Returns
    -------
    data_despiked: np.ndarray
        Output signal with spikes replaced by NaN.
    """
    # --- Input validation ---
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, np.newaxis]  # treat as column for uniformity
    elif data.ndim > 2:
        raise ValueError("Input 'data' must be 1D or 2D.")
    n_samples, n_cols = data.shape
    data_despiked = data.copy()
    spike = np.zeros_like(data_despiked, dtype=bool)  # True is spike
    # --- Block-wise MAD despiking ---
    time_block_start = np.arange(0, 24 * 60 * 60 * freq, time_window * 60 * freq).astype(int)
    time_block_end = time_block_start + time_window * 60 * freq
    for row_num in range(0, len(time_block_start)):
        start_idx = time_block_start[row_num]
        end_idx = time_block_end[row_num]
        data_block = data_despiked[start_idx:end_idx,:]
        MAD = np.nanmedian(abs(data_block - np.nanmedian(data_block,axis=0)),axis=0)  # use block median
        MAD_derived_std = 1.4826 * MAD  # assume normal distribution
        limit_up = np.nanmedian(data_block) + threshold * MAD_derived_std
        limit_dn = np.nanmedian(data_block) - threshold * MAD_derived_std
        # True is spike
        spike[start_idx:end_idx,:] = (data_block > limit_up) | (data_block < limit_dn)
    # --- Continuous spike filtering ---
    # keep only spikes less than 3 points, signal should be cleaned once by diagnostic value
    if continuous_spike_bool:
        max_len = continuous_spike_num
        for col_num in np.arange(0, n_cols):
            i = 0
            while i < n_samples:
                if spike[i, col_num]:
                    # start of a spike sequence
                    j = i
                    while j < n_samples and spike[j, col_num]:
                        j += 1
                    # j is first index after sequence
                    length = j - i
                    if length > max_len:
                        spike[i:j, col_num] = False  # remove long spike
                    i = j
                else:
                    i += 1
    # apply mask
    data_despiked[spike] = np.nan
    if data_despiked.shape[1] == 1:
        data_despiked = data_despiked[:, 0]
    return data_despiked

def double_rotation(ux, uy, uz):
    """
    Perform double rotation tilt correction on 3D wind components.

    Reference
    ---------
    Wilczak, J. M., Oncley, S. P., & Stage, S. A. (2001).
    Sonic Anemometer Tilt Correction Algorithms. Boundary-Layer Meteorology, 99(1), 127–150.
    https://doi.org/10.1023/A:1018966204465

    Parameters
    ----------
    ux, uy, uz : np.ndarray
        1D arrays of wind components (m/s), typically 30-min, 10-Hz data.
        Should be despiked.

    Returns
    -------
    ux_DR, uy_DR, uz_DR : np.ndarray
        Double-rotation corrected wind components (m/s), where:
        - ux_DR is along mean wind direction
        - uy_DR is crosswind (should have near-zero mean)
        - uz_DR is vertical (should have near-zero mean)
    """
    # --- Input validation ---
    ux = np.asarray(ux, dtype=float)
    uy = np.asarray(uy, dtype=float)
    uz = np.asarray(uz, dtype=float)
    if not (ux.shape == uy.shape == uz.shape):
        raise ValueError("ux, uy, uz must have the same shape")
    # --- Horizontal rotation (theta) ---
    theta = np.arctan2(np.nanmean(uy), np.nanmean(ux))
    ux1 = ux * np.cos(theta) + uy * np.sin(theta)
    uy1 = -ux * np.sin(theta) + uy * np.cos(theta)
    uz1 = uz
    # --- Vertical rotation (phi) ---
    phi = np.arctan2(np.nanmean(uz1), np.nanmean(ux1))
    ux_DR = ux1 * np.cos(phi) + uz1 * np.sin(phi)
    uy_DR = uy1
    uz_DR = -ux1 * np.sin(phi) + uz1 * np.cos(phi)
    return ux_DR, uy_DR, uz_DR

def rho_a_correct(T, rho_v, P):
    """
    Calculate air density based air temperature, water vapor density and air pressure

    Reference
    ---------
    Wallace, J. M., & Hobbs, P. V. (2006).
    Atmospheric science: An introductory survey (2nd ed). Elsevier Academic Press.

    Parameters
    ----------
    T: np.ndarray
        1D array of air temperature (K).
    rho_v: np.ndarray
        1D array of water vapor density (kg/m^3).
    P: np.ndarray
        1D array of air pressure (kPa).

    Returns
    ------
    rho_a: np.ndarray
        1D array of moist air density (kg/m^3).
    rho_d: np.ndarray
        1D array of dry air density (kg/m^3).
    """
    # --- Compute water vapor pressure (e) in kPa ---
    e = rho_v / 18.016 * 8.3145 * T  # unit: kPa
    # --- Compute virtual temperature (Tv) in K ---
    Tv = T / (1 - e / P * (1 - 0.622))  # ref p67 equation 3.16
    # --- Compute air density ---
    rho_a = P / (287 * Tv) * 1000  # unit: kg/m^3
    rho_d = rho_a - rho_v
    return rho_a, rho_d

def corerct_T_SND(Ts_30min, qc, rho_d):
    """
    Perform sonic temperature correction (SND) to timeseries data.

    Reference
    ---------
    Schotanus, P., Nieuwstadt, F. T. M., & De Bruin, H. A. R. (1983).
    Temperature measurement with a sonic anemometer and its application to heat and moisture fluxes. Boundary-Layer Meteorology, 26(1), 81–93.
    https://doi.org/10.1007/BF00164332

    Parameters
    ----------
    Ts_30min: np.ndarray
        1D array of sonic temperature (K).
    qc: np.ndarray
        1D array of corrected water vapor density (kg/m^3).
    rho_d: np.ndarray
        1D array of dry air density (kg/m^3).

    Returns
    -------
    Tc: np.ndarray
        1D array of corrected air temperature (K).
    """
    Ts_mean = np.nanmean(Ts_30min)
    Ts_prime = Ts_30min - Ts_mean
    rho_a = rho_d + qc  # unit kg/m^3
    # --- Correct mean value ---
    Tc_mean = Ts_mean / (1 + 0.51 * np.nanmean(qc / rho_a))
    # --- Correct prime vaule ---
    Tc_prime = Ts_prime - 0.51 * (qc / rho_a - np.nanmean(qc / rho_a)) * Tc_mean
    Tc = Tc_mean + Tc_prime  # unit: K
    return Tc


def correct_q_WPL(Tc, q_30min, rho_d):
    """
    Perform water vapor correction (WPL) to timeseries data.

    Reference
    ---------
    Webb, E. K., Pearman, G. I., & Leuning, R. (1980).
    Correction of flux measurements for density effects due to heat and water vapour transfer. Quarterly Journal of the Royal Meteorological Society, 106(447), 85–100.
    https://doi.org/10.1002/qj.49710644707

    Parameters
    ----------
    Tc: np.ndarray
        1D array of corrected air temperature (K).
    q_30min: np.ndarray
        1D array of non-corrected water vapor density (kg/m^3).
    rho_d: np.ndarray
        1D array of dry air density (kg/m^3).

    Returns
    -------
    qc: np.ndarray
        1D array of corrected water vapor density (kg/m^3).
    """
    Tc_mean = np.nanmean(Tc)
    Tc_prime = Tc - Tc_mean
    q_mean = np.nanmean(q_30min)
    q_prime = q_30min - q_mean
    qc_prime = q_prime + 1.61 * (q_mean / np.nanmean(rho_d)) * q_prime + (1 + 1.61 * (q_mean / np.nanmean(rho_d))) * (
                q_mean / Tc_mean) * Tc_prime
    qc = q_mean + qc_prime  # unit kg/m^3
    return qc

def correct_co2_WPL(Tc, qc, co2_30min, rho_d):
    """
    Perform CO2 correction (WPL) to timeseries data.

    Reference
    ---------
    Webb, E. K., Pearman, G. I., & Leuning, R. (1980).
    Correction of flux measurements for density effects due to heat and water vapour transfer. Quarterly Journal of the Royal Meteorological Society, 106(447), 85–100.
    https://doi.org/10.1002/qj.49710644707

    Parameters
    ----------
    Tc: np.ndarray
        1D array of corrected air temperature (K).
    qc: np.ndarray
        1D array of corrected water vapor density (kg/m^3).
    co2_30min: np.ndarray
        1D array of non-corrected CO2 density (kg/m^3).
    rho_d: np.ndarray
        1D array of dry air density (kg/m^3).

    Returns
    -------
    co2c: np.ndarray
        1D array of corrected CO2 density (kg/m^3).
    """
    Tc_mean = np.nanmean(Tc)
    Tc_prime = Tc - Tc_mean
    co2_mean = np.nanmean(co2_30min)
    co2_prime = co2_30min - co2_mean
    q_mean = np.nanmean(qc)
    q_prime = qc - q_mean
    co2c_prime = co2_prime + 1.61 * (co2_mean / np.nanmean(rho_d)) * q_prime + (
                1 + 1.61 * (q_mean / np.nanmean(rho_d))) * (co2_mean / Tc_mean) * Tc_prime
    co2c = co2_mean + co2c_prime  # unit kg/m^3
    return co2c


def cal_flux(uz_30min_doublerotate, Ts_correct, q_correct, co2_correct, rho_air):
    """
    Calculate sensible/latent/CO2 flux
    Parameters
    ----------
    uz_30min_doublerotate: np.ndarray
        1D array of vertical wind speed after double rotation (m/s).
    Ts_correct: np.ndarray
        1D array of corrected air temperature (K).
    q_correct: np.ndarray
        1D array of corrected water vapor density (kg/m^3).
    co2_correct: np.ndarray
        1D array of corrected CO2 density (kg/m^3).
    rho_air: np.ndarray
        1D array of moist air density (kg/m^3).

    Returns
    -------
    SHc: float
        Corrected sensible heat flux (W/m^2)
    LEc: float
        Corrected latent heat flux (W/m^2)
    Fcc: float
        Corrected CO2 flux (umol/m^2 s)
    """
    uz_mean = np.nanmean(uz_30min_doublerotate)
    uz_prime = uz_30min_doublerotate - uz_mean
    qc_mean = np.nanmean(q_correct)  # unit: kg/m^3
    qc_prime = q_correct - qc_mean
    rho_air_mean = np.nanmean(rho_air)
    Tc_mean = np.nanmean(Ts_correct)
    Tc_prime = Ts_correct - Tc_mean
    CPd = 1004.67  # J/(kg K)
    CP = CPd * (1 + 0.84 * qc_mean / rho_air_mean)  # q_mean (kg/m^3); rho_a (kg/m^3).

    wTc = np.nanmean(uz_prime * Tc_prime)
    SHc = wTc * CP * rho_air_mean  # corrected sensible heat flux

    LV = (2.501 - 0.00237 * (Tc_mean - 273.15)) * 1.0E+06  # unit J/kg, Tc_mean (K).
    wqc = np.nanmean(uz_prime * qc_prime)
    LEc = wqc * LV  # corrected latent heat flux, unit: W/m^2
    # eff_co2 = 22.72;% Convert the unit of CO2 flux from mg/(m^2 s) into vmol/(m^2 s)
    co2c_mean = np.nanmean(co2_correct)
    co2c_prime = co2_correct - co2c_mean
    wcc = np.nanmean(uz_prime * co2c_prime)
    Fcc = wcc * 1000 / 44 * 10 ** 6  # corrected co2 flux,convert unit to umol/m^2 s
    return SHc, LEc, Fcc

def cal_winddir(ux_mean, uy_mean, angle):
    """
    Calculate wind direction from sonic measurements
    Parameters
    ----------
    ux_mean: float
        30-min mean value x wind speed from sonic anemometer without rotation (m/s).
    uy_mean: float
        30-min mean value y wind speed from sonic anemometer without rotation (m/s).
    angle: float
        anemometer direction (true north, degree)
    Returns
    -------
    direction: float
    wind direction (degree, 0 is north)
    speed: float
    mean wind speed (m/s)
    """
    direction = 360 + angle - np.arctan2(uy_mean, ux_mean) * 180 / np.pi  # +360 to prevent negative angle input
    while direction > 360:
        direction = direction - 360
    speed = (ux_mean**2+uy_mean**2)**(1/2)
    return direction,speed

def cal_meanprime(data):
    """
    calculate mean and prime for data
    Parameters
    ----------
    data: np.ndarray
        1D array of input data
    Returns
    -------
    data_mean: np.ndarray
        block mean
    data_prime: np.ndarray
        fluctuation
    """
    data_mean = np.nanmean(data)
    data_prime = data - data_mean
    return data_mean, data_prime

def convert_to_nc(dat_in,dat_out_nc,dates):
    """
    Example for converting 1-day dat file to standard nc file
    Parameters
    ----------
    dat_in: string
        filepath for input .dat files
    dat_in: np.ndarray
        filepath for output .nc files
    dates: pandas datetime
        date range of processed data
    Returns
    -------
    """
    files = glob.glob(os.path.join(dat_in, '*_TS.DAT'))
    for date in dates:
        print('Start Processing... ', date)
        # format to 'yyyy_mm_dd'
        date_formated = date.strftime('%Y-%m-%d') # change the format if needed
        date_nc_output = date.strftime('%Y_%m_%d')
        #  Multiple files may exist per day due to short period failure of the equipments,
        #      reconnection produce new files.
        matching_files = [file for file in files if date_formated in os.path.basename(file)]

        # skip date without data.
        if len(matching_files) == 0:
            continue

        # Create formated dataframe, filled with nan.
        steps = 24 * 3600 * 10  # 24 h * 3600 s /h * 10Hz
        day_time = pd.date_range(start=date, periods=steps, freq='0.1s', inclusive='left')

        for idx, file in enumerate(matching_files):
            # print(idx)
            if idx == 0:
                print(f'Process... {date_formated}', f'{len(matching_files)}: {idx + 1}')
                df = pd.read_csv(file, delimiter=',', skiprows=[0, 2, 3], header=0, index_col=0, dtype='object')
                df.index = pd.to_datetime(df.index, format='mixed')
                df_day = pd.DataFrame(np.nan, dtype='object', index=day_time, columns=df.columns)
                # df_hour.index → Selects rows in df_day that match df_hour.
                try:
                    df_day.loc[df.index, :] = df
                except KeyError:
                    pass

                del df
            else:
                print(f'Process... {date_formated}', f'{len(matching_files)}: {idx + 1}')
                df = pd.read_csv(file, delimiter=',', skiprows=[0, 2, 3], header=0, index_col=0, dtype='object')
                df.index = pd.to_datetime(df.index, format='mixed')
                print(max(df.index))
                # check whether columns are same
                if len(df_day.columns) == len(df.columns):
                    try:
                        df_day.loc[df.index, :] = df
                    except KeyError:
                        pass
                else:
                    print('measurements changed')
                del df

        # convert to nc files
        time = df_day.index
        # 2. Detect layered columns using regex
        #    Match patterns like Ux_up, Ux_A, Ux_B, Ts_dn, co2_C, etc.
        # ---------------------------------------------
        layered = {}
        for col in df_day.columns:
            match = re.match(r"(.+)_([^_]+)$", col)
            if match:
                base, layer = match.groups()
                layered.setdefault(base, []).append((layer, col))

        # 3. Prepare data variables
        # ---------------------------------------------
        data_vars = {}
        # Variables that have layers
        for base, items in layered.items():
            items = sorted(items, key=lambda x: x[0])  # sort by layer name
            layer_names = [layer for layer, col in items]
            cols = [col for layer, col in items]

            arr = df_day[cols].to_numpy()
            data_vars[base] = (("time", "layer_" + base), arr)

        # Variables without layers (not include RECORD)
        for col in df_day.columns:
            if re.match(r".+_.+$", col):  # has underscore → already handled
                continue
            if col == 'RECORD':
                continue
            data_vars[col] = ("time", df_day[col].to_numpy())

        # ---------------------------------------------
        # 4. Coordinates
        # ---------------------------------------------
        # units and description
        prefix_units = {
            'Ux': 'm/s',
            'Uy': 'm/s',
            'Uz': 'm/s',
            'Ts': 'C',
            'co2': 'mg/m^3',
            'h2o': 'g/m^3',
            'Press_irga': 'kPa',
            'FW': 'C',
        }
        prefix_description = {
            'Ux': 'Sonic anemometer wind speed in x direction',
            'Uy': 'Sonic anemometer wind speed in y direction',
            'Uz': 'Sonic anemometer wind speed in z direction',
            'Ts': 'Sonic temperature',
            'co2': 'CO2 density',
            'h2o': 'Water vapor density',
            'Press_irga': 'Air pressure',
            'FW': 'FW temperature',
        }
        coords = {"time": time}
        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        for var in ds.data_vars:
            for prefix, unit in prefix_units.items():
                if var.startswith(prefix):
                    ds[var].attrs['units'] = unit
            for prefix, des in prefix_description.items():
                if var.startswith(prefix):
                    ds[var].attrs['longname'] = des

        for base, items in layered.items():
            layer_names = [layer for layer, col in sorted(items)]
            ds = ds.assign_coords({f"layer_{base}": layer_names})

        comp = dict(zlib=True, complevel=4, dtype='float32')
        encoding = {var: comp for var in ds.data_vars}
        ds.attrs.update({
            'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'created_by': socket.gethostname(),
            'description': 'RBR multi-layer 10Hz EC measurements, contact: zhiheng.lan@wsu.edu',
        })
        ds.to_netcdf(f"{dat_out_nc}RBR_{date_nc_output}_TS.nc", encoding=encoding)
