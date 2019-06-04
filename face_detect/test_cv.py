
import time
import cv2
import sys
import os
import numpy as np

'''
如果期望达到更好的识别效果，可能需要去自己训练模型（https://coding-robin.de/2013/07/22/train-your-own-opencv-haar-classifier.html）
或者做一些图片的预处理使图片更容易识别、调节detectMultiScal函数中的各个参数来达到期望的效果。
'''

if len(sys.argv) >=2:
    cascPath = sys.argv[1]
else:
    cascPath = './'

current_path = os.getcwd()
parent_path = os.path.dirname(current_path)

model_path = os.path.join(current_path, 'model')

# 加载级联分类器模型：
# 在opencv的‘\sources\data\haarcascades’目录下可以找到这个官方训练好的普适性模型
faceCascade = cv2.CascadeClassifier(os.path.join(model_path, 'haarcascade_frontalface_default.xml'))


capture = cv2.VideoCapture(1)
width, height = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width, height)

font = cv2.FONT_HERSHEY_SIMPLEX
cnt_p =0
path_save = os.path.join(current_path, 'pic')

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        print('Error: video_cap.read')
        time.sleep(1)
        continue

    kk = cv2.waitKey(1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(30, 30),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        # 在原先的彩图上画出包围框（绿色框，边框宽度为2）
        #print('rectangle: x=%d, y=%d, w=%d, h=%d' % (x, y, w, h))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if (kk & 0xFF) == ord('s'):
            cnt_p += 1
            im_blank = np.zeros((h, w, 3), np.uint8)
            for i in range(h):
                for j in range(w):
                    im_blank[j][i] = frame[y + j][x + i]

            pic_file = os.path.join(path_save,  "cv_face_%d.jpg" % cnt_p)
            print(pic_file)
            cv2.imwrite(pic_file, im_blank)
            cv2.putText(frame, "faces: ", (20, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

    frame = cv2.putText(frame, "s: save face", (20, 400), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    frame = cv2.putText(frame, "q: quit", (20, 450), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    #显示图片
    cv2.imshow('Video', frame)

    if (kk & 0xFF) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

