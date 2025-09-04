import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go 
import plotly.io as pio
from PIL import Image

from ipywidgets import interact, Dropdown, Checkbox, HBox, VBox 
from IPython.display import display, clear_output

import numpy as np
import pandas as pd

def fixation_proportion_line(aggregated_df,
                             xaxis_title="Time relative to the onset of the stimulus (ms)",
                             yaxis_title="Fixation Proportion (%)"):
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
        xaxis_title=xaxis_title,  # Add x-axis name
        yaxis_title=yaxis_title,  # Add y-axis name
        xaxis=dict(tickmode='linear', tick0=0, dtick=100),  # Set x-axis ticks interval
        yaxis_tickformat=".0%",  # Format y-axis as percentage
        yaxis_range=[-0.02,0.5]
    )
    
    return fig

def filterable_fixation_proportion_line(aggregated_df: pd.DataFrame, width: int = 1000):
    # --- 1. Prepare unique values for dropdowns ---
    # Get all unique values from the columns, and add an 'All' option
    unique_countries = ['All', 'Not Control Group'] + sorted(aggregated_df['Country'].unique().tolist())
    unique_institutions = ['All'] + sorted(aggregated_df['Institution'].unique().tolist())
    unique_versions = ['All'] + sorted(aggregated_df['Version'].unique().tolist())
    unique_conditions = ['All'] + sorted(aggregated_df['condition'].unique().tolist())
    unique_sessions = ['All'] + sorted(aggregated_df['Session'].dropna().unique().tolist())
    unique_stimuli = ['All'] + sorted(aggregated_df['sound'].dropna().unique().tolist())
    
    # --- 2. Create ipywidgets Dropdown instances ---
    country_dd = Dropdown(options=unique_countries, value='All', description='Country:')
    institution_dd = Dropdown(options=unique_institutions, value='All', description='Institution:')
    version_dd = Dropdown(options=unique_versions, value='All', description='Version:')
    condition_dd = Dropdown(options=unique_conditions, value='All', description='Condition:')
    session_dd = Dropdown(options=unique_sessions, value=1.0, description='Session:')
    stimuli_dd = Dropdown(options=unique_stimuli, value='All', description='Stimuli:')
    
    # --- 3. Create a Plotly FigureWidget for dynamic updates ---
    plot_output = go.FigureWidget()
    # Set the figure to fill available width
    plot_output.layout.width = width  # Let the container control the width
    plot_output.layout.margin = dict(l=20, r=20, t=40, b=20)
    
    # --- 4. Define the update function that filters data and redraws the plot ---
    def update_plot_with_filters(country, institution, version, condition, session, stimuli):
        """
        Filters the aggregated DataFrame based on dropdown selections and updates the Plotly graph.
        """
        clear_output(wait=True) # Clear previous plot output in Jupyter
    
        # Create a copy to avoid modifying the original DataFrame
        current_filtered_df = aggregated_df.copy()
    
        # Apply filters based on dropdown selections
        if country != 'All':
            if country == 'Not Control Group':
                current_filtered_df = current_filtered_df[current_filtered_df['Country'] != 'Control Group']
            else:
                current_filtered_df = current_filtered_df[current_filtered_df['Country'] == country]
        if institution != 'All':
            current_filtered_df = current_filtered_df[current_filtered_df['Institution'] == institution]
        if version != 'All':
            current_filtered_df = current_filtered_df[current_filtered_df['Version'] == version]
        if condition != 'All':
            current_filtered_df = current_filtered_df[current_filtered_df['condition'] == condition]
        if session != 'All':
            if country != 'Control Group':
                current_filtered_df = current_filtered_df[current_filtered_df['Session'] == session]
        if stimuli != 'All':
            current_filtered_df = current_filtered_df[current_filtered_df['sound'] == stimuli]
    
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
    update_plot_with_filters(country_dd.value, institution_dd.value, version_dd.value, condition_dd.value, session_dd.value, stimuli_dd.value)
    
    # --- 6. Link dropdowns to the update function using .observe() ---
    country_dd.observe(lambda change: update_plot_with_filters(
        change.new, institution_dd.value, version_dd.value, condition_dd.value, session_dd.value, stimuli_dd.value), names='value')
    institution_dd.observe(lambda change: update_plot_with_filters(
        country_dd.value, change.new, version_dd.value, condition_dd.value, session_dd.value, stimuli_dd.value), names='value')
    version_dd.observe(lambda change: update_plot_with_filters(
        country_dd.value, institution_dd.value, change.new, condition_dd.value, session_dd.value, stimuli_dd.value), names='value')
    condition_dd.observe(lambda change: update_plot_with_filters(
        country_dd.value, institution_dd.value, version_dd.value, change.new, session_dd.value, stimuli_dd.value), names='value')
    session_dd.observe(lambda change: update_plot_with_filters(
        country_dd.value, institution_dd.value, version_dd.value, condition_dd.value, change.new, stimuli_dd.value), names='value')
    stimuli_dd.observe(lambda change: update_plot_with_filters(
        country_dd.value, institution_dd.value, version_dd.value, condition_dd.value, session_dd.value, change.new), names='value')
    
    # --- 7. Arrange and display the widgets and the plot ---
    controls = HBox([session_dd, country_dd, institution_dd, condition_dd, stimuli_dd])
    dashboard = VBox([plot_output, controls], layout={'width': '100%'})
    return dashboard


filters = [
      {'folder':'0.Session', 'prefix':'0.0.', 'session':'All'},
      {'folder':'0.Session', 'prefix':'0.1.', 'session':1},
      {'folder':'0.Session', 'prefix':'0.2.', 'session':2},
      {'folder':'1.Overall mean', 'prefix':'1.1.', 'condition':'standard'},
      {'folder':'1.Overall mean', 'prefix':'1.2.', 'condition':'alemannic_austria'},
      {'folder':'2.Mean kindergarden', 'prefix':'2.0.', 'institution':'KiGa'},
      {'folder':'2.Mean kindergarden', 'prefix':'2.1.', 'condition':'standard', 'institution':'KiGa'},
      {'folder':'2.Mean kindergarden', 'prefix':'2.2.', 'condition':'alemannic_austria', 'institution':'KiGa'},
      {'folder':'3.Mean preschool', 'prefix':'3.0.', 'institution':'Grundschule'},
      {'folder':'3.Mean preschool', 'prefix':'3.1.', 'condition':'standard', 'institution':'Grundschule'},
      {'folder':'3.Mean preschool', 'prefix':'3.2.', 'condition':'alemannic_austria', 'institution':'Grundschule'},
      {'folder':'4.Mean Germany/4.0.all_institutions', 'prefix':'4.0.0.', 'country':'Deutschland'},
      {'folder':'4.Mean Germany/4.0.all_institutions', 'prefix':'4.0.1.', 'country':'Deutschland', 'condition':'standard'},
      {'folder':'4.Mean Germany/4.0.all_institutions', 'prefix':'4.0.2.', 'country':'Deutschland', 'condition':'alemannic_austria'},
      {'folder':'4.Mean Germany/4.1.kindergarden', 'prefix':'4.1.0.', 'country':'Deutschland', 'institution':'KiGa'},
      {'folder':'4.Mean Germany/4.1.kindergarden', 'prefix':'4.1.1.', 'country':'Deutschland', 'condition':'standard', 'institution':'KiGa'},
      {'folder':'4.Mean Germany/4.1.kindergarden', 'prefix':'4.1.2.', 'country':'Deutschland', 'condition':'alemannic_austria', 'institution':'KiGa'},
      {'folder':'4.Mean Germany/4.2.grundschule', 'prefix':'4.2.0.', 'country':'Deutschland', 'institution':'Grundschule'},
      {'folder':'4.Mean Germany/4.2.grundschule', 'prefix':'4.2.1.', 'country':'Deutschland', 'condition':'standard', 'institution':'Grundschule'},
      {'folder':'4.Mean Germany/4.2.grundschule', 'prefix':'4.2.2.', 'country':'Deutschland', 'condition':'alemannic_austria', 'institution':'Grundschule'},
      {'folder':'5.Mean Austria/5.0.all_institutions', 'prefix':'5.0.0.', 'country':'Österreich'},
      {'folder':'5.Mean Austria/5.0.all_institutions', 'prefix':'5.0.1.', 'country':'Österreich', 'condition':'standard'},
      {'folder':'5.Mean Austria/5.0.all_institutions', 'prefix':'5.0.2.', 'country':'Österreich', 'condition':'alemannic_austria'},
      {'folder':'5.Mean Austria/5.1.kindergarden', 'prefix':'5.1.0.', 'country':'Österreich', 'institution':'KiGa'},
      {'folder':'5.Mean Austria/5.1.kindergarden', 'prefix':'5.1.1.', 'country':'Österreich', 'condition':'standard', 'institution':'KiGa'},
      {'folder':'5.Mean Austria/5.1.kindergarden', 'prefix':'5.1.2.', 'country':'Österreich', 'condition':'alemannic_austria', 'institution':'KiGa'},
      {'folder':'5.Mean Austria/5.2.grundschule', 'prefix':'5.2.0.', 'country':'Österreich', 'institution':'Grundschule'},
      {'folder':'5.Mean Austria/5.2.grundschule', 'prefix':'5.2.1.', 'country':'Österreich', 'condition':'standard', 'institution':'Grundschule'},
      {'folder':'5.Mean Austria/5.2.grundschule', 'prefix':'5.2.2.', 'country':'Österreich', 'condition':'alemannic_austria', 'institution':'Grundschule'},
      {'folder':'6.Mean Switzerland/6.0.all_institutions', 'prefix':'6.0.0.', 'country':'Schweiz'},
      {'folder':'6.Mean Switzerland/6.0.all_institutions', 'prefix':'6.0.1.', 'country':'Schweiz', 'condition':'standard'},
      {'folder':'6.Mean Switzerland/6.0.all_institutions', 'prefix':'6.0.2.', 'country':'Schweiz', 'condition':'alemannic_austria'},
      {'folder':'6.Mean Switzerland/6.1.kindergarden', 'prefix':'6.1.0.', 'country':'Schweiz', 'institution':'KiGa'},
      {'folder':'6.Mean Switzerland/6.1.kindergarden', 'prefix':'6.1.1.', 'country':'Schweiz', 'condition':'standard', 'institution':'KiGa'},
      {'folder':'6.Mean Switzerland/6.1.kindergarden', 'prefix':'6.1.2.', 'country':'Schweiz', 'condition':'alemannic_austria', 'institution':'KiGa'},
      {'folder':'6.Mean Switzerland/6.2.grundschule', 'prefix':'6.2.0.', 'country':'Schweiz', 'institution':'Grundschule'},
      {'folder':'6.Mean Switzerland/6.2.grundschule', 'prefix':'6.2.1.', 'country':'Schweiz', 'condition':'standard', 'institution':'Grundschule'},
      {'folder':'6.Mean Switzerland/6.2.grundschule', 'prefix':'6.2.2.', 'country':'Schweiz', 'condition':'alemannic_austria', 'institution':'Grundschule'},
]
def generate_line_plots_by_filter(aggregated_df, filters=filters):
      country_map = {'All countries': 'All-ctry',
            'Deutschland': 'DE',
            'Österreich':'AT',
            'Schweiz':'CH'}
      condition_map = {'All conditions': 'All-cond',
            'standard': 'de-std',
            'alemannic_austria':'de-at'}
      institution_map = {'All institutions': 'All-inst',
            'KiGa': 'KiGa',
            'Grundschule':'GS'}
      # no_session_table = pd.read_csv("../Participants/no_session_participants.csv", delimiter=';')
      for filter_dict in filters:
            filter = {'country':'All countries', 
                  'condition':'All conditions',
                  'institution':'All institutions',
                  'session':1}
            filter.update(filter_dict)
            
            title = f"{country_map[filter['country']]}.{condition_map[filter['condition']]}.{institution_map[filter['institution']]}.Ses{filter['session']}"  
            if 'title' in filter: plot_title = filter['title']
            else: 
                  plot_title = filter['prefix'] + title # condition_map[filter['condition']]
            
            # Filtration
            def filter_aggregated_df_copy(df,
                                    country = 'All countries', 
                                    condition = 'All conditions', 
                                    institution = 'All institutions' , 
                                    session = 'All'):
                assert country in df.Country.unique() or country == 'All countries', f'"{country}" not in df and not default'
                assert condition in df.condition.unique() or condition =='All conditions', f'"{condition}" not in df and not default'
                assert institution in df.Institution.unique() or institution == 'All institutions', f'"{institution}" not in df and not default'
                assert session in df.Session.unique() or session == 'All', f'"{session}" not in df and not default'
                
                return df[((df.Country == country) if country in df.Country.unique() else True) &
                                        ((df.condition == condition) if condition in df.condition.unique() else True) &
                                        ((df.Institution == institution) if institution in df.Institution.unique() else True) & 
                                        ((df.Session == session) if session in df.Session.unique() else True) & 
                                        (df.fixation.notna())].copy()
                
            aggr_fix_df = filter_aggregated_df_copy(
                aggregated_df,
                country = filter['country'],
                condition = filter['condition'],
                institution = filter['institution'],
                session = filter['session'])
            
            # print(filter, aggr_fix_df.shape)

            fig = et.fixation_proportion_line(aggr_fix_df)

            
            fig.add_vrect(x0=0, x1=2000, line_width=0, fillcolor="blue", opacity=0.1)
            fig.update_layout(
                  title=title,
                  title_font=dict(size=14,
                              color='grey',
                              family='Arial'),
                  xaxis_title="Zeit in Abhängigkeit zum Stimulus-Onset (ms)",  # Add x-axis name
                  yaxis_title="Fixationsproportion (%)",  # Add y-axis name
                  xaxis=dict(tickmode='linear', tick0=0, dtick=200),  # Set x-axis ticks interval
                  yaxis_tickformat=".0%",  # Format y-axis as percentage
                  yaxis_range=[-0.02,0.5],
                  legend=dict(
                        title=dict(
                              text="Variable"
                        )
                  )
            )

            # fig.show(config=fig_config)
            file_dir = "C:/Users/Cyril/Desktop/" + filter_dict['folder'] + '/'
            os.makedirs(os.path.dirname(file_dir), exist_ok=True)
            file_name = plot_title + '.png'
            print(os.path.join(file_dir, plot_title))
            fig.write_image(os.path.join(file_dir, file_name),
                        format='.png', width=1000, height=500, scale=3,engine='orca')