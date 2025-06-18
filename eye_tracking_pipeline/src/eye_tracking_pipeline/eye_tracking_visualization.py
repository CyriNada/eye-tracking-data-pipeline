import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go 
from PIL import Image

import numpy as np
import pandas as pd

def fixation_proportion_line(aggregated_df):
    aggregated_df['timeBin'] = pd.cut(aggregated_df["timeFromOnsetMs"], 
                                     bins = np.arange(-4, 3, 0.1).round(decimals=1), 
                                     labels = np.arange(-4, 3, 0.1).round(decimals=1)[1:], 
                                     right=True,ordered=False)
    binned_rois = pd.merge(aggregated_df.loc[:, 'timeBin'], 
             pd.get_dummies(aggregated_df['AOI'], dummy_na=True), 
             left_index=True, right_index=True).groupby('timeBin', observed=False).sum()
    binned_rois = binned_rois.div(binned_rois.sum(axis=1), axis=0)
    binned_rois.index = binned_rois.index.astype(float)*1000
    binned_rois = binned_rois.dropna()
    
    fig = px.line(binned_rois.loc[:, binned_rois.columns.notna()])
    
    fig.add_vrect(x0=0, x1=2000, line_width=0, fillcolor="blue", opacity=0.1)
    fig.update_layout(
        # title=f"{country}:{condition}:{institution} - Fixation proportion normalized around stimulus onset",  # Add title
        xaxis_title="Time relative to the onset of the stimulus (ms)",  # Add x-axis name
        yaxis_title="Fixation Proportion (%)",  # Add y-axis name
        xaxis=dict(tickmode='linear', tick0=0, dtick=100),  # Set x-axis ticks interval
        yaxis_tickformat=".0%",  # Format y-axis as percentage
    )
    
    return fig