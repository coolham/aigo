
import cv2
import os
import dlib

from skimage import io
import csv
import numpy as np
import pandas as pd

def check_path_and_create(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

current_path = os.getcwd()  # 获取当前路径
parent_path = os.path.dirname(os.getcwd())    #取当前目录上一级目录
data_path = os.path.join(parent_path, 'data')


pathPic =  os.path.join(data_path, 'pic')
check_path_and_create(pathPic)

pathCsv =  os.path.join(data_path, 'csv')
pathModel = os.path.join(current_path, 'model')


# detector to find the faces
detector = dlib.get_frontal_face_detector()

# shape predictor to find the face landmarks
predictor = dlib.shape_predictor(os.path.join(pathModel, "shape_predictor_68_face_landmarks.dat"))

# face recognition model, the object maps human faces into 128D vectors
facerec = dlib.face_recognition_model_v1(os.path.join(pathModel, "dlib_face_recognition_resnet_model_v1.dat"))




# 返回单张图像的128D特征
def return_128d_features(path_img):
    img = io.imread(path_img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(img_gray, 1)
    if (len(dets) != 0):
        shape = predictor(img_gray, dets[0])
        face_descriptor = facerec.compute_face_descriptor(img_gray, shape)
    else:
        face_descriptor = 0
        print("no face")
    # print(face_descriptor)
    return face_descriptor

# 将文件夹中照片特征提取出来，写入csv
# path_pics:  图像文件夹的路径
# path_csv:   要生成的csv路径
def write_into_csv(pics_path, csv_file):
    dir_pics = os.listdir(pics_path)
    with open(csv_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(dir_pics)):
            # 调用return_128d_features()得到128d特征
            pic_full_name = os.path.join(pics_path, dir_pics[i])
            print('pic_full_name=%s' % pic_full_name)
            features_128d = return_128d_features(pic_full_name)
            #  print(features_128d)
            # 遇到没有检测出人脸的图片跳过
            if features_128d == 0:
                i += 1
            else:
                writer.writerow(features_128d)



# 从csv中读取数据，计算128d特征的均值
def compute_the_mean(path_csv_rd):
    column_names = []
    for i in range(128):
        column_names.append("features_" + str(i + 1))
    rd = pd.read_csv(path_csv_rd, names=column_names)
    # 存放128维特征的均值
    feature_mean = []
    for i in range(128):
        tmp_arr = rd["features_" + str(i + 1)]
        tmp_arr = np.array(tmp_arr)

        # 计算某一个特征的均值
        tmp_mean = np.mean(tmp_arr)
        feature_mean.append(tmp_mean)
    print(feature_mean)

    return feature_mean

csv_file_name = os.path.join(pathCsv, 'default_person.csv')


write_into_csv(pathPic, csv_file_name)

feature_mean = compute_the_mean(csv_file_name)
print(feature_mean)