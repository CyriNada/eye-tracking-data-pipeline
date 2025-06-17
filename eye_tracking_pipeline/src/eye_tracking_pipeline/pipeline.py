import os
from pathlib import Path
from typing import Tuple, List

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from .eye_tracking_metadata import generate_metatable
from .eye_tracking_data_manager import tobii_hdf5_to_pandas

def run_pipeline(input_dir: str, 
                 output_dir: str, 
                 existing_meta_table: pd.DataFrame = pd.DataFrame()):
    # Ensure output directories exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    meta_table = generate_metatable(input_dir)
    return meta_table
    
                
                

                    
                    
    
    