import pandas as pd
import numpy as np
from os import path
import json

if __name__ == '__main__':
    train_flnm="../WLASL100_train_25fps.pkl"
    val_flnm="../WLASL100_val_25fps.pkl"
    test_flnm="../WLASL100_test_25fps.pkl"
    data_path = "/Users/dra/Workspace/SMOG_GoogleGlass/ASL_Dataset/WLASL/data/WLASL2000/"
    json_file_path = '/Users/dra/Workspace/SMOG_GoogleGlass/ASL_Dataset/WLASL/data/splits/asl100.json'

    set_name = 'test'
    flnm_list = {'train': train_flnm, 'val': val_flnm, 'test': test_flnm}
    flnm = flnm_list[set_name]
    df = pd.read_pickle(flnm)
    if (df.duplicated('video_id')).any():
        print('dup')
    n_rows = len(df.index)
    with open(json_file_path) as ipf:
        content = json.load(ipf)
    n = 0
    for gloss_id, ent in enumerate(content):
        for inst in ent['instances']:
            split = inst['split']
            video_id = inst['video_id']
    
            if split != set_name:
                continue
            n += 1
            if not (video_id in df['video_id'].to_numpy()):
                print(video_id)
    print(n_rows)
    print(n)
