import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import cv2
import time
import json

def test_plot(current_row, image, save_name):
    plt.imshow(image)
    current_row = current_row*(np.asarray(image.shape[:2][::-1])-1)
    plt.scatter(*current_row.T[:,:12], c='r', s=4.)
    plt.scatter(*current_row.T[:,12:33], c= 'g', s=3.)
    plt.scatter(*current_row.T[:,33:], c= 'b', s=3.)
    plt.savefig(save_name)
    plt.axis('off')
    plt.close()

if __name__ == '__main__':
    train_flnm="../WLASL100_train_25fps.pkl"
    val_flnm="../WLASL100_val_25fps.pkl"
    test_flnm="../WLASL100_test_25fps.pkl"
    data_path = "/Users/dra/Workspace/SMOG_GoogleGlass/ASL_Dataset/WLASL/data/WLASL2000/"
    json_file_path = '/Users/dra/Workspace/SMOG_GoogleGlass/ASL_Dataset/WLASL/data/splits/asl100.json'

    id_split = {'train':[], 'val':[], 'test':[]}
    with open(json_file_path) as ipf:
        content = json.load(ipf)
    for gloss_id, ent in enumerate(content):
        for inst in ent['instances']:
            split = inst['split']
            video_id = inst['video_id']
            id_split[split].append(video_id)

    flnm_list = {'train': train_flnm, 'val': val_flnm, 'test': test_flnm}

    for set_name, set_size in zip(id_split.keys(), [70,20,10]):
        df = pd.read_pickle(flnm_list[set_name])
        selected_id = np.random.choice(id_split[set_name], size = set_size, replace = False)
        for video_id in selected_id:
            se = df.loc[df['video_id'] == video_id].squeeze()
            num_frames = len(se[1])
            flnm = video_id+".mp4" 
            read_flnm = os.path.join(data_path, flnm)
 
            cap = cv2.VideoCapture(read_flnm)
            iframe = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if (not ret) or (iframe == num_frames):
                    break
                
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                compare = []
                for data in se[1:-1]:
                    compare.append(data[iframe])
                compare = np.array(compare)
                compare = compare.reshape((54,2))
                
                save_dir = f"Figs/{set_name}/{video_id}/"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_name = os.path.join(save_dir, "frame_{:03d}.png".format(iframe))
                test_plot(compare,image, save_name)
            
                iframe+=1
            
            cap.release()
            cv2.destroyAllWindows()
