import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go 
from PIL import Image

from ipywidgets import interact, Dropdown, HBox, VBox
from IPython.display import display, clear_output

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

def filterable_fixation_proportion_line(aggregated_df: pd.DataFrame, width: int = 1000):
    # --- 1. Prepare unique values for dropdowns ---
    # Get all unique values from the columns, and add an 'All' option
    unique_countries = ['All'] + sorted(aggregated_df['Country'].unique().tolist())
    unique_institutions = ['All'] + sorted(aggregated_df['Institution'].unique().tolist())
    unique_versions = ['All'] + sorted(aggregated_df['Version'].unique().tolist())
    unique_sessions = ['All'] + sorted(aggregated_df['Session'].unique().tolist())
    
    # --- 2. Create ipywidgets Dropdown instances ---
    country_dd = Dropdown(options=unique_countries, value='All', description='Country:')
    institution_dd = Dropdown(options=unique_institutions, value='All', description='Institution:')
    version_dd = Dropdown(options=unique_versions, value='All', description='Version:')
    session_dd = Dropdown(options=unique_sessions, value='All', description='Session:')
    
    # --- 3. Create a Plotly FigureWidget for dynamic updates ---
    plot_output = go.FigureWidget()
    # Set the figure to fill available width
    plot_output.layout.width = width  # Let the container control the width
    plot_output.layout.margin = dict(l=20, r=20, t=40, b=20)
    
    # --- 4. Define the update function that filters data and redraws the plot ---
    def update_plot_with_filters(country, institution, version, session):
        """
        Filters the aggregated DataFrame based on dropdown selections and updates the Plotly graph.
        """
        clear_output(wait=True) # Clear previous plot output in Jupyter
    
        # Create a copy to avoid modifying the original DataFrame
        current_filtered_df = aggregated_df.copy()
    
        # Apply filters based on dropdown selections
        if country != 'All':
            current_filtered_df = current_filtered_df[current_filtered_df['Country'] == country]
        if institution != 'All':
            current_filtered_df = current_filtered_df[current_filtered_df['Institution'] == institution]
        if version != 'All':
            current_filtered_df = current_filtered_df[current_filtered_df['Version'] == version]
        if session != 'All':
            current_filtered_df = current_filtered_df[current_filtered_df['Session'] == session]
    
        # Call your `et.fixation_proportion_line` function with the *filtered* DataFrame
        fig = fixation_proportion_line(current_filtered_df)
    
        # --- Crucial Fix for ValueError: ---
        plot_output.data = []
    
        for trace in fig.data:
            plot_output.add_trace(trace)
    
        # Update the layout of the FigureWidget (fill width)
        plot_output.layout = fig.layout
        plot_output.layout.autosize = True
        plot_output.layout.width = None
        plot_output.layout.margin = dict(l=20, r=20, t=40, b=20)
        plot_output.frames = fig.frames
    
    # --- 5. Initial display of the plot (before any dropdown changes) ---
    update_plot_with_filters(country_dd.value, institution_dd.value, version_dd.value, session_dd.value)
    
    # --- 6. Link dropdowns to the update function using .observe() ---
    country_dd.observe(lambda change: update_plot_with_filters(
        change.new, institution_dd.value, version_dd.value, session_dd.value), names='value')
    institution_dd.observe(lambda change: update_plot_with_filters(
        country_dd.value, change.new, version_dd.value, session_dd.value), names='value')
    version_dd.observe(lambda change: update_plot_with_filters(
        country_dd.value, institution_dd.value, change.new, session_dd.value), names='value')
    session_dd.observe(lambda change: update_plot_with_filters(
        country_dd.value, institution_dd.value, version_dd.value, change.new), names='value')
    
    # --- 7. Arrange and display the widgets and the plot ---
    controls = HBox([country_dd, institution_dd, version_dd, session_dd])
    dashboard = VBox([plot_output, controls], layout={'width': '100%'})
    return dashboard