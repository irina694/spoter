import pandas as pd
import numpy as np
from os import path
import json
import cv2

if __name__ == '__main__':
    train_flnm="../WLASL100_train_25fps.pkl"
    val_flnm="../LASL100_val_25fps.pkl"
    test_flnm="../LASL100_test_25fps.pkl"
    data_path = "/Users/dra/Workspace/SMOG_GoogleGlass/ASL_Dataset/WLASL/data/WLASL2000/"
    json_file_path = '/Users/dra/Workspace/SMOG_GoogleGlass/ASL_Dataset/WLASL/data/splits/asl100.json'

    flnm_list = {'train': train_flnm, 'val': val_flnm, 'test': test_flnm}
    for set_name in flnm_list.keys():
        flnm = flnm_list[set_name]
        df = pd.read_pickle(flnm)
        with open(json_file_path) as ipf:
            content = json.load(ipf)
        for gloss_id, ent in enumerate(content):
            for inst in ent['instances']:
                split = inst['split']
                video_id = inst['video_id']
                nFrames = inst['frame_end'] - inst['frame_start'] + 1
        
                if split != set_name:
                    continue
                df = pd.read_pickle(flnm)
                se = df.loc[df['video_id'] == video_id].squeeze()
                nFrames_data = len(se[1])
                if (nFrames - nFrames_data ) == 0:
                    continue
                elif (nFrames - nFrames_data) == 1:
                    with open('Results/test_nFrames_{}.txt'.format(set_name), 'a+') as f:
                        f.write(f"OOO: video {video_id} has blank frame.\n")
                elif nFrames_data > nFrames:
                    read_flnm = path.join(data_path, video_id+".mp4")
                    cap = cv2.VideoCapture(read_flnm)
                    nFrames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if nFrames_video == nFrames_data:
                        with open('Results/test_nFrames_{}.txt'.format(set_name), 'a+') as f:
                            f.write(f"OOO: video {video_id} tagged starting/ending frame wrongly\n")
                    else:
                        with open('Results/test_nFrames_{}.txt'.format(set_name), 'a+') as f:
                            f.write(f"XXX: video {video_id} has problemetic number of frames\n")
                            f.write(f"XXX: nFrames={nFrames}, nFrames_data={nFrames_data}, nFrames_video={nFrames_video}\n")
                else:
                    with open('Results/test_nFrames_{}.txt'.format(set_name), 'a+') as f:
                        f.write(f"XXX: video {video_id} has problemetic number of frames\n")
                        f.write(f"XXX: nFrames={nFrames}, nFrames_data={nFrames_data}\n")
