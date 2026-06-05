#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name :  300A_cal_from_nc.py
   Description :  [calculate 30min flux from 1day nc files]
   Author :       Micromet
   Date :         6/3/2026 3:01 PM
   Project :      ZhihengLan
   Email: zhiheng.lan@wsu.edu
-------------------------------------------------
   Change Activity:
                   6/3/2026 - Created
-------------------------------------------------
"""

import os
os.chdir(r'D:\pythoncode\ZhihengLan\lab423 website') # change here based on work directory
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import EC_processing_functions as ecf
import glob
import xarray as xr

def main():
    """Main execution block"""
    dat_in_nc = 'E:/Data/Hanford_300A/300A_nc/'
    # z = 3m
    angle = 297 # angle of csat at 300A site, degree
    dates = pd.date_range('2017-08-19', '2025-12-31')  # modify based on the dat_in
    outfile = dat_in_nc + "flux_results/" + f"300A_{dates[0].date()}_{dates[-1].date()}.xlsx"
    files = glob.glob(os.path.join(dat_in_nc, '*.nc'))
    # set output parameters and units
    flux_cols = [
        "SH", "SHc", "SHc_vp", "LE", "LEc", "Fc", "Fcc",
        "Bo", "TKE", "tau", "u_star", "T_star", "q_star", "L", "WS_ave", "Winddirc", "rho_a_mean",
        "ux_DR_mean", "uy_DR_mean", "uz_DR_mean", "qc_mean", "co2c_mean", "Press_mean", "Tc_mean",
        "Tc_vp_mean",
        "ux_DR_var", "uy_DR_var", "uz_DR_var", "qc_var", "co2c_var", "Press_var", "Tc_var", "Tc_vp_var",
        "cor_wu", "cor_wv", "cor_wqc", "cor_wco2c", "cor_wTc", "cor_wTc_vp",
        "wu", "wv", "wqc", "wco2c", "wTc", "wTc_vp",
        "diag_csat_mean","diag_csat_std","diag_irga_mean", "diag_irga_std", "nan_num_max"
    ]
    flux_units = {
        "SH": "W/m^2",
        "SHc": "W/m^2",
        "SHc_vp": "W/m^2",
        "LE": "W/m^2",
        "LEc": "W/m^2",
        "Fc": "umol/m^2/s",
        "Fcc": "umol/m^2/s",
        "Bo": "#",
        "TKE": "(m/s)^2",
        "tau": "kg/(m s^2)",
        "u_star": "m/s",
        "T_star": "K",
        "q_star": "g/m^3",
        "L": "m",
        "WS_ave": "m/s",
        "Winddirc": "degree",
        "rho_a_mean": "kg/m^3",
        "ux_DR_mean": "m/s",
        "uy_DR_mean": "m/s",
        "uz_DR_mean": "m/s",
        "qc_mean": "g/m^3",
        "co2c_mean": "mg/m^3",
        "Press_mean": "kPa",
        "Tc_mean": "K",
        "Tc_vp_mean": "K",
        "ux_DR_var": "(m/s)^2",
        "uy_DR_var": "(m/s)^2",
        "uz_DR_var": "(m/s)^2",
        "qc_var": "(g/m^3)^2",
        "co2c_var": "(mg/m^3)^2",
        "Press_var": "(kPa^3",
        "Tc_var": "K^2",
        "Tc_vp_var": "K^2",
        "cor_wu": "#",
        "cor_wv": "#",
        "cor_wqc": "#",
        "cor_wco2c": "#",
        "cor_wTc": "#",
        "cor_wTc_vp": "#",
        "wu": "(m/s)^2",
        "wv": "(m/s)^2",
        "wqc": "m/s*(g/m^3)",
        "wco2c": "m/s*(mg/m^3)",
        "wTc": "K*m/s",
        "wTc_vp": "K*m/s",
        "diag_irga_mean": "#",
        "diag_irga_std": "#",
        "nan_num_max": "#",
        "diag_csat_mean": "#",
        "diag_csat_std": "#",
    }
    # read nc files and calculate
    all_rows = []
    for date in dates:
        print('Start Processing... ', date)
        # format to 'yyyy_mm_dd'
        date_nc_output = date.strftime('%Y-%m-%d')
        matching_files = [file for file in files if date_nc_output in os.path.basename(file)]
        # skip date without data.
        if len(matching_files) == 0:
            continue
        for idx, file in enumerate(matching_files):
            ds = xr.open_dataset(file)
            # read variables
            # 300A only have 1 layer measurements
            ux_all = ds['Ux']
            uy_all = ds['Uy']
            uz_all = ds['Uz']
            Ts_all = ds['Ts']
            diag_csat_all = ds['diag_cast']
            co2_all = ds['co2']
            q_all = ds['h2o']
            Press_all = ds['Press']
            diag_irga_all = ds['diag_irga']

            # remove by threshold, adjust based on sites
            ux_all[ux_all > 30] = np.nan
            ux_all[ux_all < -30] = np.nan
            uy_all[uy_all > 30] = np.nan
            uy_all[uy_all < -30] = np.nan
            uz_all[uz_all > 10] = np.nan
            uz_all[uz_all < -10] = np.nan
            Ts_all[Ts_all > 50] = np.nan
            Ts_all[Ts_all < -20] = np.nan
            q_all[q_all > 60] = np.nan
            q_all[q_all < 0] = np.nan
            co2_all[co2_all < 400] = np.nan
            co2_all[co2_all > 1200] = np.nan
            Press_all[Press_all > 110] = np.nan
            Press_all[Press_all < 90] = np.nan
            # remove spikes
            ux_all_despiked = ecf.despike_MADstd(ux_all, time_window=30, threshold=7, continuous_spike_bool=True,
                                             continuous_spike_num=5)
            uy_all_despiked = ecf.despike_MADstd(uy_all, time_window=30, threshold=7, continuous_spike_bool=True,
                                             continuous_spike_num=5)
            uz_all_despiked = ecf.despike_MADstd(uz_all, time_window=30, threshold=7, continuous_spike_bool=True,
                                             continuous_spike_num=5)
            Ts_all_despiked = ecf.despike_MADstd(Ts_all, time_window=30, threshold=7, continuous_spike_bool=True,
                                             continuous_spike_num=5)
            q_all_despiked = ecf.despike_MADstd(q_all, time_window=30, threshold=7, continuous_spike_bool=True,
                                            continuous_spike_num=5)
            co2_all_despiked = ecf.despike_MADstd(co2_all, time_window=30, threshold=7, continuous_spike_bool=True,
                                              continuous_spike_num=5)

            # convert unit
            Ts_all_despiked = Ts_all_despiked + 273.15
            q_all_despiked = q_all_despiked / 1000
            co2_all_despiked = co2_all_despiked / (1000 ** 2)
            time_block_start_array = np.arange(0, 24 * 60 * 60 * 10, 30 * 60 * 10)
            time_block_end_array = time_block_start_array + 30 * 60 * 10
            for row_num in range(0, len(time_block_start_array)):
                ux = ux_all_despiked[time_block_start_array[row_num]:time_block_end_array[row_num]]
                uy = uy_all_despiked[time_block_start_array[row_num]:time_block_end_array[row_num]]
                uz = uz_all_despiked[time_block_start_array[row_num]:time_block_end_array[row_num]]
                Ts = Ts_all_despiked[time_block_start_array[row_num]:time_block_end_array[row_num]]
                q = q_all_despiked[time_block_start_array[row_num]:time_block_end_array[row_num]]
                co2 = co2_all_despiked[time_block_start_array[row_num]:time_block_end_array[row_num]]
                Press = np.array(Press_all[time_block_start_array[row_num]:time_block_end_array[row_num]])
                nan_num = np.max([sum(np.isnan(ux)), sum(np.isnan(uy)), sum(np.isnan(uz)), sum(np.isnan(Ts)),
                                  sum(np.isnan(q)), sum(np.isnan(co2)), sum(np.isnan(Press))])
                diag_csat = np.array(diag_csat_all[time_block_start_array[row_num]:time_block_end_array[row_num]])
                diag_irga = np.array(diag_irga_all[time_block_start_array[row_num]:time_block_end_array[row_num]])
                # break

                ux_DR, uy_DR, uz_DR = ecf.double_rotation(ux, uy, uz)

                # use only high frequency loop correction
                rho_a, rho_d = ecf.rho_a_correct(Ts, q, Press)
                Tc = ecf.corerct_T_SND(Ts, q, rho_d)
                qc = ecf.correct_q_WPL(Tc, q, rho_d)
                for loop in np.arange(0, 5):
                    rho_a, rho_d = ecf.rho_a_correct(Tc, qc, Press)
                    Tc = ecf.corerct_T_SND(Ts, qc, rho_d)
                    qc = ecf.correct_q_WPL(Tc, q, rho_d)
                rho_a, rho_d = ecf.rho_a_correct(Tc, qc, Press)
                co2c = ecf.correct_co2_WPL(Tc, qc, co2, rho_d)
                SHc, LEc, Fcc = ecf.cal_flux(uz_DR, Tc, qc, co2c, rho_a)
                SH, LE, Fc = ecf.cal_flux(uz_DR, Ts, q, co2, rho_a)

                e = qc / 18.016 * 8.3145 * Tc  # unit: kPa
                Tc_v = Tc / (1 - e / Press * (
                            1 - 0.622))  # virtural temp, ref: atmospheric science an introductory survey, P66
                Tc_vp = Tc_v * (Press / 100) ** 0.286
                SHc_vp, _, _ = ecf.cal_flux(uz_DR, Tc_vp, qc, co2c, rho_a)

                ux_DR_mean, ux_DR_prime = ecf.cal_meanprime(ux_DR)
                ux_DR_var = np.nanmean(ux_DR_prime * ux_DR_prime)
                uy_DR_mean, uy_DR_prime = ecf.cal_meanprime(uy_DR)
                uy_DR_var = np.nanmean(uy_DR_prime * uy_DR_prime)
                uz_DR_mean, uz_DR_prime = ecf.cal_meanprime(uz_DR)
                uz_DR_var = np.nanmean(uz_DR_prime * uz_DR_prime)

                qc_mean, qc_prime = ecf.cal_meanprime(qc)
                qc_mean = qc_mean * 1000  # convert from kg/m^3 to g/m^3
                qc_prime = qc_prime * 1000  # convert from kg/m^3 to g/m^3
                qc_var = np.nanmean(qc_prime * qc_prime)
                co2c_mean, co2c_prime = ecf.cal_meanprime(co2c)
                co2c_mean = co2c_mean * 10 ** 6  # convert from kg/m^3 to mg/m^3
                co2c_prime = co2c_prime * 10 ** 6  # convert from kg/m^3 to mg/m^3
                co2c_var = np.nanmean(co2c_prime * co2c_prime)
                Press_mean, Press_prime = ecf.cal_meanprime(Press)
                Press_var = np.nanmean(Press_prime * Press_prime)

                Tc_mean, Tc_prime = ecf.cal_meanprime(Tc)
                Tc_var = np.nanmean(Tc_prime * Tc_prime)
                Tc_vp_mean, Tc_vp_prime = ecf.cal_meanprime(Tc_vp)
                Tc_vp_var = np.nanmean(Tc_vp_prime * Tc_vp_prime)

                Winddirc, WS_ave = ecf.cal_winddir(np.nanmean(ux), np.nanmean(uy), angle)

                cor_wu = np.nanmean(uz_DR_prime * ux_DR_prime) / (uz_DR_var ** (1 / 2) * ux_DR_var ** (1 / 2))
                cor_wv = np.nanmean(uz_DR_prime * uy_DR_prime) / (uz_DR_var ** (1 / 2) * uy_DR_var ** (1 / 2))

                cor_wTc = np.nanmean(uz_DR_prime * Tc_prime) / (uz_DR_var ** (1 / 2) * Tc_var ** (1 / 2))
                cor_wTc_vp = np.nanmean(uz_DR_prime * Tc_vp_prime) / (uz_DR_var ** (1 / 2) * Tc_vp_var ** (1 / 2))
                cor_wqc = np.nanmean(uz_DR_prime * qc_prime) / (uz_DR_var ** (1 / 2) * qc_var ** (1 / 2))
                cor_wco2c = np.nanmean(uz_DR_prime * co2c_prime) / (uz_DR_var ** (1 / 2) * co2c_var ** (1 / 2))

                rho_a_mean = np.nanmean(rho_a)
                TKE = (ux_DR_var + uy_DR_var + uz_DR_var) / 2  # m^2/s^2
                wu = np.nanmean(uz_DR_prime * ux_DR_prime)
                wv = np.nanmean(uz_DR_prime * uy_DR_prime)
                wTc = np.nanmean(uz_DR_prime * Tc_prime)
                wTc_vp = np.nanmean(uz_DR_prime * Tc_vp_prime)
                wqc = np.nanmean(uz_DR_prime * qc_prime)
                wco2c = np.nanmean(uz_DR_prime * co2c_prime)

                tau = (wu ** 2 + wv ** 2) ** (1 / 2)
                u_star = tau ** (1 / 2)
                tau = rho_a_mean * tau  # Reynolds' stress
                L = -Tc_vp_mean * (u_star ** 3) / (0.4 * 9.8 * wTc_vp)  # k = 0.4
                T_star = -wTc_vp / u_star
                q_star = -wqc / u_star
                Bo = SHc_vp / LEc

                diag_irga_mean = np.nanmean(diag_irga)
                diag_irga_std = np.nanstd(diag_irga)
                diag_csat_mean = np.nanmean(diag_csat)
                diag_csat_std = np.nanstd(diag_csat)
                result = [SH, SHc, SHc_vp, LE, LEc, Fc, Fcc,
                          Bo, TKE, tau, u_star, T_star, q_star, L, WS_ave, Winddirc, rho_a_mean,
                          ux_DR_mean, uy_DR_mean, uz_DR_mean, qc_mean, co2c_mean, Press_mean, Tc_mean, Tc_vp_mean,
                          ux_DR_var, uy_DR_var, uz_DR_var, qc_var, co2c_var, Press_var, Tc_var, Tc_vp_var,
                          cor_wu, cor_wv, cor_wqc, cor_wco2c, cor_wTc, cor_wTc_vp,
                          wu, wv, wqc, wco2c, wTc, wTc_vp,
                          diag_csat_mean,diag_csat_std,diag_irga_mean, diag_irga_std, nan_num
                          ]
                all_rows.append(
                    {
                        "Timestamp": str(
                            (date + pd.to_timedelta(time_block_start_array[row_num] / 10, "s")) + pd.to_timedelta(15,
                                                                                                                  'min')),
                    }
                    | dict(zip(flux_cols, result))
                )
    if all_rows:
        df_all = pd.DataFrame(all_rows)
        df_all = df_all.fillna(-999)
        # Add unit row
        unit_row = {col: flux_units.get(col, "") for col in df_all.columns}
        df_with_unit = pd.concat([pd.DataFrame([unit_row]), df_all], ignore_index=True)
        df_with_unit.to_excel(outfile, index=False)


if __name__ == '__main__':
    main()
