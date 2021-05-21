"""
script to compute the number
of jsons created for each
word. The output is a .csv
file where for each word the
number of jsons corresponding
to train, val, and test subsets
is mentioned
"""

import os
import json
import re
import numpy as np


def get_rect_and_landmarks(rect, landmarks):
    # converts and returns the face rectangle and landmarks
    # in formats appropriate for the display function

    x = rect["left"]
    y = rect["top"]
    w = rect["width"]
    h = rect["height"]

    if landmarks is not None:
        temp_agg = list()
        for i in range(len(landmarks)):
            temp = list()
            temp.append(landmarks["point-" + str(i+1)]["x"])
            temp.append(landmarks["point-" + str(i+1)]["y"])
            temp_agg.append(temp)
        return (x, y, w, h), np.asarray(temp_agg)
    else:
        return (x, y, w, h), np.empty((0, 0))


def choose_the_largest_face(faces_list):
    if len(faces_list) == 1: 
        return faces_list[0]
    
    area_max = 0
    area_max_id = 0
    for i,face in enumerate(faces_list):
        (face_rect,landmarks) = face
        area = face_rect[2] * face_rect[3] # area = width * height
        if area > area_max:
            area_max = area
            area_max_id = i
    return faces_list[area_max_id]


def load_one_json_file(filename, isDebug=False):
    # load the metadata and facial landmarks

    face_rect_list = []
    landmarks_list = []
    with open(filename) as f:
        video_data_dict = json.load(f)
        # extract duration
        if video_data_dict["metaData"] is not None:
            duration = float(re.findall(r"[-+]?\d*\.\d+|\d+", video_data_dict["metaData"]["Duration"])[0])
            if isDebug:
                print("duration metadata: %.3f" % duration)

        # extract frame information aggregated for all frames
        agg_frame_data = video_data_dict["aggFrameInfo"]  # list of frame-wise visual data

        for frame_data in agg_frame_data:
            n_faces = frame_data["numFaces"]
            if isDebug:
                print("frame index: %d number of faces: %d" % (frame_data["frameIndex"], n_faces))
            
            if frame_data["facialAttributes"] is not None:# if so, the n_faces should > 0 
                faces_list = []
                for attr in frame_data["facialAttributes"]:
                    face_idx = attr["faceIndex"]
                    face_rect, landmarks = get_rect_and_landmarks(attr["faceRectangle"],
                                                                  attr["faceLandmarks"])
                    faces_list.append((face_rect, landmarks))

                face_rect_chosen, landmarks_chosen = choose_the_largest_face(faces_list)    
                face_rect_list.append(face_rect_chosen)
                landmarks_list.append(landmarks_chosen)
    
    face_rect_array = np.array(face_rect_list)
    landmarks_array = np.array(landmarks_list)
    return face_rect_array, landmarks_array








i_data = "/cache/lrw/lipread_landmarks/dlib68_2d_sparse_json/lipread_mp4"
# or i_data = "/cache/lrw/lipread_landmarks/dlib68_2d_sparse_json_defects_not_one_face/lipread_mp4"
selected_n_classes = 10 # the max is 500

cnt = 0
data = dict()

for word in os.listdir(i_data):
    if not word.startswith('.'):
        cnt += 1
        if cnt > selected_n_classes:
            break
        print(cnt,word)
        splits = dict() # 'train' 'val' and 'test' sets
        # print("analysing data for the word: '%s'" % word)
        p = os.path.join(i_data, word)
        
        for sub_dir in os.listdir(p):
            if not sub_dir.startswith('.'):
                # print(sub_dir)
                p_sub = os.path.join(p, sub_dir)
                for _, _, files in os.walk(p_sub):
                    samples_list = []
                    for filename in files:
                        if filename.endswith('.json'):
                            face_rect_array, landmarks_array = load_one_json_file(os.path.join(p_sub, filename))
                            samples_list.append(landmarks_array)
                    splits[sub_dir] = samples_list
        data[word] = splits

print('-------------------------------')
print(data.keys()) # names of all the 'selected_n_classes' classes  
print(data['THOUGHT'].keys()) # print the names of the 3 splits for the first class 'THOUGHT'
print(len(data['THOUGHT']['train'])) # print the number of train samples of the first class
print(data['THOUGHT']['train'][0].shape) # print the shape (29 frames, 68 landmarks, 2 coordinates) of the first training sample of the first class    
print('-------------------------------')                     
        
