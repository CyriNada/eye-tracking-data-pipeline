import h5py
import pandas as pd
import numpy as np

tobii_cols_to_remove =['left_gaze_z','left_angle_x','left_angle_y', 'left_raw_x', 'left_raw_y',\
    'left_pupil_measure1_type','left_pupil_measure2','left_pupil_measure2_type',\
    'left_ppd_x', 'left_ppd_y','left_velocity_x', 'left_velocity_y', 'left_velocity_xy',\
    'right_gaze_z','right_angle_x','right_angle_y', 'right_raw_x', 'right_raw_y',\
    'right_pupil_measure1_type','right_pupil_measure2','right_pupil_measure2_type',\
    'right_ppd_x', 'right_ppd_y','right_velocity_x', 'right_velocity_y', 'right_velocity_xy',\
                'device_time', 'logged_time',\
    'confidence_interval', 'delay', 'filter_id',\
    'left_eye_cam_x', 'left_eye_cam_y', 'left_eye_cam_z',\
    'right_eye_cam_x', 'right_eye_cam_y', 'right_eye_cam_z']

def AOI_mapper(row):
    if row['region']=="":
        return ""
    else:
        tl,tr,br,bl=row.posPerm.strip('[]').split(' ')
        if row['region']=="tl":
            return int(tl)
        elif row['region']=="tr":
            return int(tr)
        elif row['region']=="br":
            return int(br)
        elif row['region']=="bl":
            return int(bl)
        else:
            print("Am here")
            return ""

def clean_pupil(row):
    left = row['left_pupil_measure1']
    right = row['right_pupil_measure1']
    
    if pd.notna(left) and pd.notna(right) and left > 0 and right > 0:
        return (left + right) / 2

    elif pd.notna(left) and left > 0:
        return left
    elif pd.notna(right) and right > 0:
        return right
    else:
        return np.nan  # both missing or invalid

def tobii_hdf5_to_pandas(data_file_path: str) -> dict:
    """
    Reads a Tobii eye-tracking HDF5 file and converts relevant data into a pandas DataFrame.

    Parameters:
    ----------
    data_file_path : str
        The full file path to the HDF5 data file exported from a Tobii eye-tracker.

    Returns:
    -------
    dict
        A dictionary containing the results of the conversion with the following keys:
            - "conversionSuccess": int
                1 if the conversion was successful, 0 otherwise.
            - "errorMessage": str or None
                A message describing the error if one occurred, or None on success.
            - "df": pandas.DataFrame or None
                The DataFrame containing the trial-level data if successful, else None.
    
    Notes:
    ------
    Any exceptions during the file reading or conversion process are caught,
    and a failure result is returned with an appropriate error message.
    """
    with h5py.File(data_file_path, 'r') as hdf: 
        try:
            # Handle meta data
            session_meta_data = hdf['data_collection']['session_meta_data']
            recorded_session = int(session_meta_data['code'][0])
            
            trial_file_csv = data_file_path.replace('.hdf5','.csv')
            trial_meta_df = pd.read_csv(trial_file_csv,skiprows=range(1,7),usecols=range(0,6))
            trial_meta_df = trial_meta_df.dropna()

            if trial_meta_df.empty:
                raise ValueError(f"Meta file is empty: {trial_file_csv}")
            
            # Handle MessageEvent
            events_data = hdf['/data_collection/events/experiment/MessageEvent']
            events_df=pd.DataFrame(np.array(events_data))
            if events_df.empty:
                raise ValueError(f"MessageEvent is empty: {data_file_path}")

            events_df.text=events_df.text.str.decode('utf-8')
            

            if events_df.loc[events_df.text=='trackerTest2022_2_5'].shape[0]!=1:
                #print(events_df.loc[events_df.text=='trackerTest2022_2_5'])
                raise ValueError(events_df.loc[events_df.text=='trackerTest2022_2_5'])
            
            #print(trial_file)
            
            events_df=events_df.drop(['device_time','logged_time','confidence_interval','delay','msg_offset','category'],axis=1)

            start_index=events_df.loc[events_df.text=="trackerTest2022_2_5"].index.tolist()[0]
            start_time=events_df.iloc[start_index]['time']
            
            end_index=events_df.iloc[start_index:].index.tolist()[-1]
            end_time=events_df.iloc[end_index]['time']
            
            # Handle Eye Tracking Data
            eye_tracking_data = hdf['/data_collection/events/eyetracker/BinocularEyeSampleEvent']
            eye_tracking_df=pd.DataFrame(np.array(eye_tracking_data))
            
            eye_tracking_df=eye_tracking_df.drop(tobii_cols_to_remove,axis=1)
            eye_tracking_df=eye_tracking_df.loc[eye_tracking_df['time'].between(start_time,end_time)]

            if eye_tracking_df.empty:
                raise ValueError(f"BinocularEyeSampleEvent is empty: {data_file_path}")
            
            eye_tracking_df=eye_tracking_df.loc[eye_tracking_df.status==0]
            eye_tracking_df['gaze_x']=eye_tracking_df[['left_gaze_x','right_gaze_x']].mean(axis=1)
            eye_tracking_df['gaze_y']=eye_tracking_df[['left_gaze_y','right_gaze_y']].mean(axis=1)

            eye_tracking_df['pupil_size'] = eye_tracking_df.apply(clean_pupil, axis=1)   
            
            eye_tracking_df.loc[eye_tracking_df.gaze_x.between(-0.75,-0.25) & eye_tracking_df.gaze_y.between(0.0,0.5,inclusive='neither'),"region"]="tl"
            eye_tracking_df.loc[eye_tracking_df.gaze_x.between(0.25,0.75) & eye_tracking_df.gaze_y.between(0.0,0.5,inclusive='neither'),"region"]="tr"
            eye_tracking_df.loc[eye_tracking_df.gaze_x.between(0.25,0.75) & eye_tracking_df.gaze_y.between(-0.5,0.0,inclusive='neither'),"region"]="br"
            eye_tracking_df.loc[eye_tracking_df.gaze_x.between(-0.75,-0.25) & eye_tracking_df.gaze_y.between(-0.5,0.0,inclusive='neither'),"region"]="bl"
            eye_tracking_df=eye_tracking_df.drop(['left_gaze_x', 'left_gaze_y', 'left_pupil_measure1', 'right_gaze_x','right_gaze_y', 'right_pupil_measure1', 'status'],axis=1)
            eye_tracking_df.region=eye_tracking_df.region.fillna("")
            
            # Process meta data
            trial_meta_df = pd.read_csv(trial_file_csv,skiprows=range(1,7),usecols=range(0,6))  # Trial metadata
            trial_meta_df=trial_meta_df.reset_index()
            trial_meta_df['trial']=trial_meta_df.index+1
            trial_meta_df=trial_meta_df.drop(trial_meta_df.tail(1).index)
            trial_meta_df=trial_meta_df.drop('index',axis=1)
            trial_meta_df=trial_meta_df.melt(id_vars=['trial','sound','condition'],value_vars=['target_image','phonological_image','semantic_image','unrelated_image'],value_name='image',var_name='AOI')
            trial_meta_df = trial_meta_df.sort_values('trial').reset_index(drop=True)
            
        
            tstart_events = events_df[events_df['text'].str.startswith('tStart', na=False)].copy()
            tstart_events['trial'] = tstart_events['text'].str.extract(r'tStart\s+(\d+)').astype(int)+1
            tstart_events['end_time'] = events_df.loc[events_df['text'].str.startswith('tEnd', na=False),'time'].tolist()
            
            if len(events_df.loc[events_df['text'].str.startswith('Target:', na=False)].text.str.split(': ',expand=True)[3].tolist())==20:
                tstart_events['posPerm'] = events_df.loc[events_df['text'].str.startswith('Target:', na=False)].text.str.split(': ',expand=True)[3].tolist()
            else:
                raise ValueError(f'Not 20: {data_file_path}')
            tstart_events=tstart_events.rename(columns={'time': 'start_time'})
            
            tstart_events = tstart_events.sort_values('start_time').reset_index(drop=True)    
            eye_tracking_df = eye_tracking_df.sort_values('time').reset_index(drop=True)

            
            trial_df=pd.merge_asof(eye_tracking_df, tstart_events[['start_time','trial','end_time','posPerm']], left_on='time',right_on='start_time')
            trial_df = trial_df[
                (trial_df['time'] >= trial_df['start_time']) &
                (trial_df['time'] < trial_df['end_time'])
            ]
            
            trial_df['AOI']=trial_df.apply(AOI_mapper,axis=1)
            trial_df['trial'] = trial_df['trial'].astype(int)
            trial_df=trial_df.replace({'AOI': {0: 'target_image', 1: 'phonological_image', 2:'semantic_image',3:'unrelated_image'}})
            trial_df=trial_df.merge(trial_meta_df[['trial','sound','condition']].drop_duplicates(),on=['trial'])
            trial_df=trial_df.merge(trial_meta_df,on=['trial','sound','condition','AOI'],how='left')
            
            trial_df=trial_df.drop(['experiment_id','session_id','device_id','event_id','type'],axis=1)
            trial_df['TimeFromTrialOnset']=trial_df.time-trial_df.start_time


            trial_df['image']=trial_df['image'].fillna(value='')

            trial_df['semantic_image']=np.where(trial_df['AOI'] == 'semantic_image', True, False)
            trial_df['target_image']=np.where(trial_df['AOI'] == 'target_image', True, False)
            trial_df['unrelated_image']=np.where(trial_df['AOI'] == 'unrelated_image', True, False)
            trial_df['phonological_image']=np.where(trial_df['AOI'] == 'phonological_image', True, False)

            trial_df['semantic_image'] = trial_df['semantic_image'].astype('object')
            trial_df['target_image'] = trial_df['target_image'].astype('object')
            trial_df['unrelated_image'] = trial_df['unrelated_image'].astype('object')
            trial_df['phonological_image'] = trial_df['phonological_image'].astype('object')
            
            trial_df.loc[trial_df.AOI=="","semantic_image"]=None
            trial_df.loc[trial_df.AOI=="","target_image"]=None
            trial_df.loc[trial_df.AOI=="","unrelated_image"]=None
            trial_df.loc[trial_df.AOI=="","phonological_image"]=None
            trial_df['trackloss']=False
            trial_df.loc[trial_df.gaze_y.isna()==True,'trackloss']=True

            if 'Adult' in data_file_path:
                school,speech,edit,group2,fname=data_file_path.split('\\')[-5:]
                trial_df['school']=school
                trial_df['speech']=speech
                trial_df['edit']=edit
                trial_df['group2']=group2
                trial_df['fname']=fname
            else:
                land,school,group2,fname=data_file_path.split('\\')[-4:]
                trial_df['land']=land
                trial_df['school']=school
                trial_df['group2']=group2
                trial_df['fname']=fname
            return {
                "conversionSuccess": 1,
                "errorMessage": None,
                "df": trial_df
            }
        
        except Exception as e:
            # raise e
            return {
                "conversionSuccess": 0,
                "errorMessage": f"Error: {str(e)}",
                "df": None
            }
            
if __name__ == "__main__":
    input_dir=r'C:\Users\Cyril\HESSENBOX\Eye-Tracking_LAVA (Jasmin Devi Nuscheler)\Data_from_different_participants'
    output_dir=r'C:\dev\grk-2700\eye_tracking_pipeline\tests\results'
    data_file_path = 'C:\\Users\\Cyril\\HESSENBOX\\Eye-Tracking_LAVA (Jasmin Devi Nuscheler)\\Data_from_different_participants\\2.Germany\\Primary_school\\A\\069_trackerTest2022_2_5_2024-06-13_08h05.39.112.hdf5'
    tobii_hdf5_to_pandas(data_file_path)