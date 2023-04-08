import pandas as pd
import numpy as np
from os import path
import cv2
import mediapipe as mp
import time
import json
from pose_model_identifier import BODY_IDENTIFIERS, HAND_IDENTIFIERS, mp_holistic_data
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles

def compare(df, video_id, i, holistic):
    comp_flag = True
    def neck(pose_results):
        ls=pose_results.pose_landmarks.landmark[11]
        rs=pose_results.pose_landmarks.landmark[12]
        no=pose_results.pose_landmarks.landmark[0]
        if (ls.visibility > 0.5) & (rs.visibility > 0.5) & (no.visibility > 0.5):
            cx = (ls.x+rs.x)/2
            cy = (ls.y+rs.y)/2
            dx = no.x-cx
            dy = no.y-cy
            x = cx+0.3*dx
            y = cy+0.3*dy
        else:
            x = 0
            y = 0
        return x, y
    if not holistic_results.pose_landmarks:
        return False
    for id_name, lm_id in BODY_IDENTIFIERS.items():
        if id_name == "neck":
            x, y = neck(holistic)
            tmp = [x, y]
        else:
            lm = holistic.pose_landmarks.landmark[lm_id]
            tmp = [lm.x * float(lm.visibility >= 0.5), lm.y * float(lm.visibility >= 0.5)]
        for xy in ["_X", "_Y"]:
            names_df = id_name+xy
            test_data = df.loc[df['video_id'] == str(video_id)].squeeze()[names_df][i]
            if xy == "_X":
                data = tmp[0]
            elif xy == "_Y":
                data = tmp[1]
           
            if (data - test_data) != 0:
                with open('Results/test_out_{}.txt'.format(set_name), 'a+') as f:
                    f.write(f"{video_id} frame {i} got inconsistancy in pose\n")
                    f.write(f"The difference is {data-test_data}\n")
                comp_flag = False
                break
        if comp_flag == False:
           break
         

    for lr, lm in zip(["_Right", "_Left"], [holistic_results.right_hand_landmarks, holistic_results.left_hand_landmarks]):
        if lm:
            for id_name,lm_id in HAND_IDENTIFIERS.items():
                for xy in ["_X", "_Y"]:
                    names_df = id_name+lr+xy
                   # test_data = df.loc[df['video_id'] == str(video_id),names_df][i]
                    test_data = df.loc[df['video_id'] == str(video_id)].squeeze()[names_df][i]
                    if xy == "_X":
                        data = lm.landmark[lm_id].x
                    elif xy == "_Y":
                        data = lm.landmark[lm_id].y
                    if (data - test_data) != 0:
                        with open('Results/test_out_{}.txt'.format(set_name), 'a+') as f:
                            f.write(f"{video_id} frame {i} got inconsistancy in hand\n")
                            f.write(f"The difference is {data-test_data}\n")
                        comp_flag = False
                        break
                if comp_flag == False:
                   break
        if comp_flag == False:
           break
    return True
    
if __name__ == '__main__':
    train_flnm="../WLASL100_train_25fps.pkl"
    val_flnm="../WLASL100_val_25fps.pkl"
    test_flnm="../WLASL100_test_25fps.pkl"
    data_path = "/Users/dra/Workspace/SMOG_GoogleGlass/ASL_Dataset/WLASL/data/WLASL2000/"
    json_file_path = '/Users/dra/Workspace/SMOG_GoogleGlass/ASL_Dataset/WLASL/data/splits/asl100.json'

    df = pd.read_pickle(train_flnm)

    holistic = mp_holistic.Holistic()

    flnm_list = {'train': train_flnm, 'val': val_flnm, 'test': test_flnm}
    for set_name in ['train', 'val', 'test']:
    #for set_name in ['val']:
        flnm = flnm_list[set_name]
        df = pd.read_pickle(flnm)

        with open(json_file_path) as ipf:
            content = json.load(ipf)
        for gloss_id, ent in enumerate(content):
            gloss = ent['gloss']
            with open('Results/test_out_{}.txt'.format(set_name), 'a+') as f:
                f.write(f"gloss={gloss}\n")
                f.write(f"gloss_id={gloss_id}\n")
        
            for inst in ent['instances']:
                split = inst['split']
                if split != set_name:
                    continue

                video_id = inst['video_id']
                with open('Results/test_out_{}.txt'.format(set_name), 'a+') as f:
                    f.write(f"video_id={video_id}\n")

                read_flnm = path.join(data_path, video_id+".mp4")
                cap = cv2.VideoCapture(read_flnm)
                iframe = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Recolor image to RGB
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          
                    # Make detection
                    holistic_results = holistic.process(image)
                    # compare
                    suc = compare(df, video_id, iframe, holistic_results)
                    if suc:
                        iframe += 1
                cap.release()
