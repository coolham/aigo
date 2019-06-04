import dlib

import numpy as np
import cv2

'''
实验中使用了两个模型：

shape_predictor_68_face_landmarks.dat：
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

dlib_face_recognition_resnet_model_v1.dat：
http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

'''
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./model/shape_predictor_68_face_landmarks.dat')

camera_id = 1
capture = cv2.VideoCapture(camera_id)

#width, height = capture.get(3), capture.get(4)
width, height = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width, height)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)

# 定义编码方式并创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
outfile = cv2.VideoWriter('output.avi', fourcc, 25., (640, 480))


cnt_ss = 0
cnt_p =0
path_save = '../data/pic/'

save_video = False

while capture.isOpened():
    # 获取一帧
    flag, frame = capture.read()
    if not flag:
        print('Error: video_cap.read')
        continue

    kk = cv2.waitKey(1)
    img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    rects = detector(img_gray, 0)
    #print(len(rects))
    font = cv2.FONT_HERSHEY_SIMPLEX
    if len(rects) != 0:
        for k, d in enumerate(rects):
            pos_start = tuple([d.left(), d.top()])
            pos_end = tuple([d.right(), d.bottom()])
            height = d.bottom() - d.top()
            width = d.right() - d.left()
            frame = cv2.rectangle(frame, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255), 2)
            im_blank = np.zeros((height, width, 3), np.uint8)
            if kk == ord('s'):
                cnt_p += 1
                for h in range(height):
                    for w in range(width):
                        im_blank[h][w] = frame[d.top() + h][d.left() + w]
                print(path_save + "img_face_" + str(cnt_p) + ".jpg")
                cv2.imwrite(path_save + "img_face_" + str(cnt_p) + ".jpg", im_blank)
                cv2.putText(frame, "faces: " + str(len(rects)), (20, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        # 没有检测到人脸
        cv2.putText(frame, "no face", (20, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

    frame = cv2.putText(frame, "s: save face", (20, 400), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    frame = cv2.putText(frame, "q: quit", (20, 450), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("camera", frame)

    if save_video:
        outfile.write(frame)  # 写入文件

    if kk == ord('q'):
        break


capture.release()
cv2.destroyAllWindows()
