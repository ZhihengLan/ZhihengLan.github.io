#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name :  plot_website.py
   Description :  [plot html interactive figures shown on lab423 website, 1 gantt figure and weekly flux figure]
   Author :       Micromet
   Date :         6/3/2026 4:53 PM
   Project :      ZhihengLan
   Email: zhiheng.lan@wsu.edu
-------------------------------------------------
   Change Activity:
                   6/3/2026 - Created
-------------------------------------------------
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import plotly.figure_factory as ff

def main():
    """Main execution block"""
    # flux result
    # different site names and path
    site_name = '300a'
    path = 'E:/Data/Hanford_300A/300A_nc/'

    site_name = 'a3'
    path = 'E:/Data/Ellensburg_A3/A3_nc/'

    site_name = 'a5'
    path = 'E:/Data/Ellensburg_A5/A5_nc/'

    # same codes
    dat_in = path + 'flux_results/'
    fig_out = path + 'figures/'
    os.makedirs(fig_out, exist_ok=True)
    files = glob.glob(os.path.join(dat_in, '*.xlsx'))
    df = pd.read_excel(files[0], skiprows=[1]) # skip the unit row
    # Ensure Timestamp is datetime and set as index
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    # Replace -999.000 (sentinel missing) with NaN
    df = df.replace(-999.0, pd.NA)

    # variable need to plot
    variables = ['SHc', 'LEc', 'Fcc','u_star','Tc_mean','qc_mean','co2c_mean','Press_mean',
                 'WS_ave','Winddirc','nan_num_max','diag_csat_mean','diag_irga_mean']
    units = {
        'SHc': '$W/m^2$',
        'LEc': '$W/m^2$',
        'Fcc': '$umol/m^2/s$',
        'u_star': '$m/s$',
        'Tc_mean': '$K$',
        'qc_mean': '$g/m^3$',
        'co2c_mean': '$mg/m^3$',
        'Press_mean': '$kPa$',
        'WS_ave': '$m/s$',
        'Winddirc': '$degree$',
        'nan_num_max': '#',
        'diag_csat_mean': '#',
        'diag_irga_mean': '#',
    }
    # gantt
    gantt_data = []
    for var in variables:
        # Check for valid (non-NaN) data
        valid  = df[var].notna()
        if not valid .any():
            continue
        # Find contiguous blocks of valid data
        # Shift and diff to detect changes
        groups = (valid != valid.shift()).cumsum()
        # Keep only groups where valid is True
        valid_groups = groups[valid].unique()
        for g in valid_groups:
            block = df[valid & (groups == g)]
            start = block.index[0]
            end = block.index[-1]
            gantt_data.append(dict(Task=var, Start=start, Finish=end, Resource="has data"))
    fig = ff.create_gantt(gantt_data, group_tasks=True, show_colorbar=False,
                          title="Data availability per parameter")
    fig.update_layout(
        autosize=True,
        height=500,   # dynamic height
        xaxis_title="Date"
    )
    fig.write_html(fig_out+f"{site_name}_gantt.html", include_plotlyjs='cdn',include_mathjax='cdn')

    # weekly plot
    for week_start, week_data in df.resample('W-MON',label='left', closed='left'):
        # Skip weeks with no data (all NaN)
        if week_data[variables].isna().all().all():
            continue
        fig = make_subplots(rows=len(variables), cols=1, shared_xaxes=True,
                            subplot_titles=variables,vertical_spacing=0.01,)
        for i, var in enumerate(variables, start=1):
            # Extract the week's data for this variable
            y = week_data[var]
            # Create a time series line plot
            fig.add_trace(go.Scatter(x=y.index, y=y, mode='lines', name=var),
                          row=i, col=1)
            # Add layout improvements
            fig.update_yaxes(title_text=f"{units[var]}" if var in units else var, row=i, col=1)
        week_label = week_start.strftime('%Y-%m-%d')
        fig.update_layout(
            title=f'{site_name} measurements for week starting {week_label}',
            height=100*len(variables),
            showlegend=False,
            autosize=True, width=None,
        )
        filename = f'{week_label}.html'
        filepath = os.path.join(fig_out, filename)
        fig.write_html(filepath,include_plotlyjs='cdn',include_mathjax='cdn')
        print(f'Saved: {filepath}')



if __name__ == '__main__':
    main()
