import os
from pathlib import Path
from typing import Tuple, List
from collections import defaultdict
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from .tobii_files_manager import tobii_hdf5_to_pandas
from .config import country_folders, country_codes, country_metatable_codes, institution_folders, institution_codes, institution_metatable_codes, exp_versions

def run_pipeline(input_dir: str, output_dir: str, suppress_warnings: bool = False):
    # Ensure output directories exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    meta_table = pd.DataFrame()
    for country_folder_name, country_short_name, country_metatable_code in zip(country_folders, country_codes, country_metatable_codes):
        for institution_folder_name, institution_short_name, institution_metatable_code in zip(institution_folders, institution_codes, institution_metatable_codes):
            for exp_version in exp_versions:
                # print(f'{country_code}_{institution_code}_{exp_version}')
                experiment_path = os.path.join(input_dir,
                                               country_folder_name,
                                               institution_folder_name,
                                               exp_version)
                if not os.path.isdir(experiment_path) and not suppress_warnings:
                    logger.warning(f"Missing: {experiment_path}")
                    continue
                
                hdf5_files = [f for f in os.listdir(experiment_path) if f.endswith('.hdf5')]
                if not hdf5_files and not suppress_warnings:
                    logger.warning(f"No .hdf5 files found in: {experiment_path}")
                    continue
                
                meta_dict = defaultdict(list)
                for hdf5_file_name in hdf5_files:
                    file_size = os.path.getsize(os.path.join(experiment_path, hdf5_file_name))
                    split_name = hdf5_file_name.split('_')
                    
                    if len(split_name) == 6: # Older subject code format
                        institution_code = "None"
                        subject_code = split_name[0]
                    elif len(split_name) == 8: # Updated subject code format
                        institution_code = split_name[1]
                        subject_code = split_name[2]
                    else:
                        raise ValueError(f"Non-standard file name: {hdf5_file_name}")
                    
                    date = split_name[-2]
                    time = split_name[-1][:-5]                    
                    
                    # TODO: subject_code not unique with session in hdf5?
                    output_file_name = f"{subject_code}_{{}}_{country_short_name}_{institution_short_name}_{exp_version}.csv" # Later inserted with Run
                    meta_dict[subject_code].append({
                        "out": output_file_name, 
                        "Subject": int(subject_code),
                        "Country": country_metatable_code,
                        "Institution": institution_metatable_code,
                        "Version": exp_version,
                        "Run": 1,               # Updated later
                        "conversionSuccess": 0, # Updated later
                        "hasMultiple": 0,       # Updated later
                        "FileSize": file_size,
                        "Date": date,
                        "Time": time,
                        "in": hdf5_file_name,
                        "missingCount": None,   # Updated later
                        "missingPercent": None, # Updated later
                        "numberOfTrials": None, # Updated later
                        "Session": None,        # Updated later     
                        "institutionCode": institution_code,
                        "path": os.path.join(country_folder_name,
                                               institution_folder_name,
                                               exp_version)
                    })
                
                # Subjects with multiple attempts
                for file_list in meta_dict.values():
                    if len(file_list) > 1:
                        def convert_time(time_str):
                            return datetime.strptime(time_str, '%Hh%M.%S.%f')
                        for run_label, file_dict in enumerate(sorted(file_list, key=lambda x: convert_time(x["Time"])), 1):
                            file_dict["hasMultiple"] = 1
                            file_dict["Run"] = run_label
                            file_dict["out"] = file_dict["out"].format(run_label) # Insert run_label into {} placeholder
                    else: 
                        file_list[0]["out"] = file_list[0]["out"].format(1) # Insert 1 into {} placeholder if there is only one run
                    meta_table = pd.concat([meta_table, pd.DataFrame(file_list)],ignore_index=True)
    return meta_table
                
                

                    
                    
    
    