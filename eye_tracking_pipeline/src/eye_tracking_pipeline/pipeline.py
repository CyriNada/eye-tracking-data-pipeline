import os

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from .eye_tracking_metadata import generate_metatable
from .eye_tracking_data_manager import eye_tracking_hdf5_to_df, IDTFixationSaccadeClassifier

def run_pipeline(input_dir: str, 
                 output_dir: str, 
                 existing_meta_table: pd.DataFrame = pd.DataFrame(),
                 no_session_table: pd.DataFrame = pd.DataFrame()):
    meta_table = generate_metatable(input_dir)
    
    
    # TODO: Participants without marked session
    # if not no_session_table.empty:
    #     session_lookup = {
    #         (row['Subject'], row['Version']): row['Session']
    #         for _, row in no_session_table.iterrows()
    #     }
        
    #     # Apply to meta_table where Country is Deutschland
    #     for idx in meta_table[meta_table['Country'] == 'Deutschland'].index:
    #         key = (meta_table.at[idx, 'Subject'], meta_table.at[idx, 'Version'])
    #         if key in session_lookup:
    #             meta_table.at[idx, 'Session'] = session_lookup[key]
        
    # TODO: Skip processed files with existing_meta_table
    
    # Ensure output directories exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  
    for i, file_meta_row in meta_table.iterrows(): # meta_table[meta_table['in'] == '082_trackerTest2022_2_5_2024-06-14_08h11.24.487.hdf5']
        
        # Check if already processed
        # if os.path.exists(os.path.join(output_dir, file_meta_row['out'])):
        #     logger.info(f"Already exists: '{file_meta_row['out']}'")
        #     continue
        
        data_file_path = os.path.join(input_dir, file_meta_row['path'], file_meta_row['in'])
        result_dict = eye_tracking_hdf5_to_df(data_file_path)
        
        meta_table.at[i, 'conversionSuccess'] = result_dict['conversionSuccess']
        if result_dict['conversionSuccess'] == 0:
            # logger.warning(f"Conversion of '{file_meta_row['in']}' failed: with {result_dict['errorMessage']}")
            meta_table.at[i, 'conversionError'] = result_dict['errorMessage']
            continue
        
        # Don't overwrite if already got from no_session_table      
        if meta_table.at[i, 'Session'] is None:
            meta_table.at[i, 'Session'] = result_dict['session']
        
        experiment_df = result_dict['df']
        experiment_df.to_csv(os.path.join(output_dir, file_meta_row['out']), index=False)

        meta_table.at[i, 'numberOfTrials'] = experiment_df.trial.max()
        meta_table.at[i, 'missingCount'] = experiment_df.trackloss.sum()
        meta_table.at[i, 'missingPercent'] = experiment_df.trackloss.mean()
        
        # Find fixations in each trial
        for trial_number, trial_df in experiment_df.groupby('trial'):
            # Isolate and fill in up to 12 consecutively missing values ~ 100 ms
            gaze_df = trial_df.loc[:, ['gaze_x', 'gaze_y']].interpolate(method='linear', limit = 12, limit_area = 'inside', axis=0)

            # Identify the fixations
            classifier = IDTFixationSaccadeClassifier()
            fixations, saccades, fixation_indices, saccades_indices = \
                classifier.fit_predict(gaze_df.loc[:, 'gaze_x'].to_numpy(), 
                                       gaze_df.loc[:, 'gaze_y'].to_numpy())
            experiment_df.loc[fixations + trial_df.index[0], 'fixation'] = fixation_indices
            experiment_df.loc[saccades + trial_df.index[0], 'saccades'] = saccades_indices
    return meta_table
    
                
                

                    
                    
    
    