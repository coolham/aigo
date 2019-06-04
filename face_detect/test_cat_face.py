# -*- coding: utf-8 -*-
import cv2
import logging

# 设置日志
logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 待检测的图片路径
ImagePath = './pic/cat/c1.jpg'

# 读取图片
logger.info('Reading image...')
image = cv2.imread(ImagePath)
# 把图片转换为灰度模式
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 探测图片中的人脸
logger.info('Detect faces...')
# 获取训练好的人脸的参数数据,进行人脸检测
face_cascade = cv2.CascadeClassifier(r'./model/haarcascade_frontalcatface.xml')
faces = face_cascade.detectMultiScale(gray,scaleFactor=1.15,minNeighbors=5,minSize=(3, 3))

search_info = "Find %d face."%len(faces) if len(faces) <= 1 else "Find %d faces."%len(faces)
logger.info(search_info)

# 绘制人脸的矩形区域(红色边框)
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)

# 显示图片
cv2.imshow('Find faces!', image)
cv2.waitKey(0)
