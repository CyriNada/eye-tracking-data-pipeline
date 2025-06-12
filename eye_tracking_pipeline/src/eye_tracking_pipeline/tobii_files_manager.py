import h5py

def tobii_hdf5_to_pandas(data_file_path: str):
    with h5py.File(data_file_path, 'r') as f: 
        # print("Keys: %s" % f.keys())
        session_meta_data = f['data_collection']['session_meta_data']
        data_array = session_meta_data[:]
        # meta_table.loc[data_table_file, 'Session'] = int(data_array['code'][0])