import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from os import path
import cv2
import mediapipe as mp
import json
from pose_model_identifier import BODY_IDENTIFIERS, HAND_IDENTIFIERS, mp_holistic_data
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles

def create_df(flnm, column_names):
    df = pd.DataFrame(columns = column_names)
#    df.to_pickle(flnm)
    return df

def save_data(df, data, flnm):
    df = df.append(data.get_series(), ignore_index=True)
    df.to_pickle(flnm)

if __name__ == '__main__':
    train_flnm="WLASL100_train_25fps.pkl"
    val_flnm="WLASL100_val_25fps.pkl"
    test_flnm="WLASL100_test_25fps.pkl"
    data_path = "/Users/dra/Workspace/SMOG_GoogleGlass/ASL_Dataset/WLASL/data/WLASL2000/"
    json_file_path = '/Users/dra/Workspace/SMOG_GoogleGlass/ASL_Dataset/WLASL/data/splits/asl100.json'

    holistic = mp_holistic.Holistic()

    column_names = []
    column_names.append('video_id')
    for id_name in BODY_IDENTIFIERS.keys():
        for xy in ["_X", "_Y"]:
            column_names.append(id_name+xy)
    for lr in ["_Right", "_Left"]:
        for id_name in HAND_IDENTIFIERS.keys():
            for xy in ["_X", "_Y"]:
                column_names.append(id_name+lr+xy)
    column_names.append('labels')

    with open(json_file_path) as ipf:
        content = json.load(ipf)
    for gloss_id, ent in enumerate(content):
        gloss = ent['gloss']
        print(f"gloss={gloss}")
        print(f"gloss_id={gloss_id}")
    
        for inst in ent['instances']:
            split = inst['split']
            video_id = inst['video_id']
            print(f"video_id={video_id}")
    
            if split == 'train':
                # open train csv
                flnm = train_flnm
            elif split == 'val':
                continue
            elif split == 'test':
                continue
            else:
                raise ValueError("Invalid split.")
            try:
                df = pd.read_pickle(flnm)
            except FileNotFoundError:
                df = create_df(flnm, column_names)

            read_flnm = path.join(data_path, video_id+".mp4")
            cap = cv2.VideoCapture(read_flnm)
            data = mp_holistic_data(video_id, gloss_id, column_names)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      
                # Make detection
                holistic_results = holistic.process(image)
                # Extract feature and save to mp_pose_data class
                data.extract_data(holistic_results)
            cap.release()
            # save data
            save_data(df, data, flnm)
