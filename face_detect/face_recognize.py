
import os
import dlib
import cv2
import numpy as np
import pygame
import time

'''
调用摄像头，捕获摄像头中的人脸，然后如果检测到人脸，将摄像头中的人脸提取出128D的特征，
然后和预设的default_person的128D特征进行计算欧式距离，如果比较小，可以判定为一个人，否则不是一个人；

欧氏距离对比的阈值设定，是在 return_euclidean_distance 函数的 dist 变量；


'''

current_path = os.getcwd()  # 获取当前路径
parent_path = os.path.dirname(os.getcwd())    #取当前目录上一级目录
data_path = os.path.join(parent_path, 'data')


pathPic =  os.path.join(data_path, 'pic')
pathCsv =  os.path.join(data_path, 'csv')
pathModel = os.path.join(current_path, 'model')

# detector to find the faces
detector = dlib.get_frontal_face_detector()

# shape predictor to find the face landmarks
predictor = dlib.shape_predictor(os.path.join(pathModel, "shape_predictor_68_face_landmarks.dat"))

# face recognition model, the object maps human faces into 128D vectors
facerec = dlib.face_recognition_model_v1(os.path.join(pathModel, "dlib_face_recognition_resnet_model_v1.dat"))


def play_voice(filename):
    pygame.mixer.init()  # 初始化音频
    track = pygame.mixer.music.load(filename)#载入音乐文件
    pygame.mixer.music.play()#开始播放
    time.sleep(3)#播放10秒
    pygame.mixer.music.stop()#停止播放



# 计算两个向量间的欧式距离
def calc_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    print(dist)
    return dist

def get_128d_features(img_gray):
    dets = detector(img_gray, 1)
    if (len(dets) != 0):
        shape = predictor(img_gray, dets[0])
        face_descriptor = facerec.compute_face_descriptor(img_gray, shape)
    else:
        face_descriptor = 0
    return face_descriptor


features_mean_default_person = [-0.07705531540242108, 0.07676272128116, 0.13210357996550473, 0.03064705204980617, -0.09546984935348685, 0.027531069576401602, -0.1160457655787468, -0.11258635940876874, 0.050513754514130677, -0.08522002331235191, 0.24521919814023105, -0.04173731448298151, -0.18470219184051861, -0.09172973104498604, 0.020464210347695785, 0.1628022478385405, -0.1367759663950313, -0.12343623556874016, -0.04555372013287111, -0.009162182242355564, 0.032958223960701034, 0.023703536092811686, 0.03996129079975865, -0.06970496103167534, -0.0991098007017916, -0.38584142381494696, -0.059166101908141915, -0.10057362507690083, 0.04651932884007692, -0.06769290363246744, -0.08806091513146054, 0.023967684161933987, -0.1630119586532766, -0.022285405715758152, 0.028537997891279785, 0.09174829450520602, -0.03312370871109041, -0.020625371020287275, 0.1174350076101043, -0.022544379261406986, -0.2242660481821407, 0.05070193636823784, -0.005131255581297658, 0.21818098425865173, 0.23056791451844302, 0.07762527973814444, 0.04744326594201001, -0.08439763770862059, 0.05290746587243947, -0.21378954296762293, 0.06785540553656491, 0.13244773989373987, 0.06559509885582057, 0.06439795819195834, 0.023577657909217207, -0.10808700729500163, -0.01314428440210494, 0.06816738335923715, -0.12759541584686798, 0.04721045697277242, 0.11471805992451581, -0.06154500185088678, -0.008174627172676, -0.07296180386434901, 0.23204267431389203, 0.06811616508374838, -0.12267485396428542, -0.1389559338038618, 0.10173859244043176, -0.11614503101869063, -0.08529971506107938, 0.023583633740517227, -0.15699567848985846, -0.17194177887656473, -0.2983568852598017, 0.04088061827827583, 0.407772958278656, 0.06950331445444714, -0.21185370602390982, 0.026000293479724365, -0.04075561540031975, 0.007488849826834418, 0.09474406391382217, 0.1268184401772239, 0.02714495767246593, -0.010728444904088976, -0.04813129861246456, -0.02012388433583758, 0.21128164638172497, -0.08881686898795041, -0.008431811351329088, 0.1889972754500129, -0.045770422809503296, 0.03872521458701654, 0.02900741520253095, 0.07455238649113612, -0.002987166697328741, 0.022201961647211152, -0.09027900580655444, -0.019217771097001703, -0.0015827392397279089, -0.09790844131599773, 5.628621544350411e-05, 0.10225225375457243, -0.16975689069791275, 0.17277309569445523, -0.027787732997570525, 0.06858387555588376, 0.0817337002266537, 0.003289770186794075, -0.05043841424313458, -0.05677817626432939, 0.15104964578693564, -0.2713212492791089, 0.21611447903242978, 0.24847084012898532, 0.050187159668315544, 0.12007317692041397, 0.0713130088353699, 0.11935501274737445, -0.0023335663771087475, 0.034602618352933365, -0.21019391580061478, -0.07785245911641554, 0.011482518377967856, -0.01126012278043411, -0.020467010580680588, 0.06728753650730307]



cap = cv2.VideoCapture(1)
# 设置视频参数，propId设置的视频参数，value设置的参数值
cap.set(3, 480)

while (cap.isOpened()):
    flag, im_rd = cap.read()
    # 每帧数据延时1ms，延时为0读取的是静态帧
    kk = cv2.waitKey(1)

    # 取灰度
    img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)
    # 人脸数rects
    rects = detector(img_gray, 0)
    #print(len(rects))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im_rd, "q: quit", (20, 400), font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    if len(rects) != 0:
        # 检测到人脸
        # 将捕获到的人脸提取特征和内置特征进行比对
        features_rd = get_128d_features(im_rd)
        dist = calc_euclidean_distance(features_rd, features_mean_default_person)
        matched = False
        if dist <0.4:
            matched = True
        if matched:
            im_rd = cv2.putText(im_rd, "default_person", (20, 350), font, 0.8, (0, 255, 255), 1,
                                cv2.LINE_AA)
        else:
            im_rd = cv2.putText(im_rd, 'not match', (20, 350), font, 0.8, (0, 255, 255),
                                1, cv2.LINE_AA)
        # 矩形框
        for k, d in enumerate(rects):
            # 绘制矩形框
            im_rd = cv2.rectangle(im_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255),
                                  2)
            cv2.putText(im_rd, "faces: " + str(len(rects)), (20, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

        if matched:
            sound_file = os.path.join(data_path, 'a.mp3')
            play_voice(sound_file)

    else:
        # 没有检测到人脸
        cv2.putText(im_rd, "no face", (20, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

    if (kk == ord('q')):
        break

    cv2.imshow("camera", im_rd)

cap.release()
cv2.destroyAllWindows()

