import sys
import os

# Add src to the path
sys.path.insert(0, os.path.abspath("../src"))
from eye_tracking_pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline(input_dir=r'C:\dev\grk-2700\eye_tracking_pipeline\tests\testing_data\Data_from_different_participants',
             output_dir=r'C:\dev\grk-2700\eye_tracking_pipeline\tests\results',
             suppress_warnings=True)