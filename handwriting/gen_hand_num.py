# Blog:      http://www.cnblogs.com/AdaminXie/
# Github:       https://github.com/coneypo/Generate_handwritten_number
# 生成手写体数字

import random
import os
from PIL import Image, ImageDraw, ImageFont

random.seed(3)
current_path = os.getcwd()  # 获取当前路径
img_data_path = os.path.join(current_path, 'data')


# 在目录下生成用来存放数字 1-9 的 9个文件夹，分别用 1-9 命名
def mkdir_for_imgs():
    if not os.path.exists(img_data_path):
        os.makedirs(img_data_path)

    for i in range(0, 10):
        num_path = os.path.join(img_data_path, 'Num_%d' % i)
        if not os.path.exists(num_path):
            os.makedirs(num_path)

# 删除路径下的图片
def del_imgs():
    for i in range(1, 10):
        img_file = img_data_path + "Num_" + str(i)
        if not os.path.exists(img_file):
            continue
        dir_nums = os.listdir(img_data_path + "Num_" + str(i))
        for tmp_img in dir_nums:
            if tmp_img in dir_nums:
                # print("delete: ", tmp_img)
                os.remove(img_data_path + "Num_" + str(i) + "/" + tmp_img)
    print("Delete finish", "\n")


# 生成单张扭曲的数字图像
def generate_single():
    # 先绘制一个50*50的空图像
    im_50_blank = Image.new('RGB', (50, 50), (255, 255, 255))

    # 创建画笔
    draw = ImageDraw.Draw(im_50_blank)

    # 生成随机数1-9
    num = str(random.randint(1, 9))

    # 设置字体，这里选取字体大小25
    font = ImageFont.truetype('simsun.ttc', 20)

    # xy是左上角开始的位置坐标
    draw.text(xy=(18, 11), font=font, text=num, fill=(0, 0, 0))

    # 随机旋转-10-10角度
    random_angle = random.randint(-10, 10)
    im_50_rotated = im_50_blank.rotate(random_angle)

    # 图形扭曲参数
    params = [1 - float(random.randint(1, 2)) / 100,
              0,
              0,
              0,
              1 - float(random.randint(1, 10)) / 100,
              float(random.randint(1, 2)) / 500,
              0.001,
              float(random.randint(1, 2)) / 500]

    # 创建扭曲
    im_50_transformed = im_50_rotated.transform((50, 50), Image.PERSPECTIVE, params)

    # 生成新的30*30空白图像
    im_30 = im_50_transformed.crop([10, 10, 40, 40])

    return im_30, num


# 生成手写体数字1-9存入指定文件夹1-9
def generate_1to9(n):
    # 用cnt_num[1]-cnt_num[9]来计数数字1-9生成的个数，方便之后进行命名
    cnt_num = []
    for i in range(10):
        cnt_num.append(0)

    for m in range(1, n + 1):
        # 调用生成图像文件函数
        im, generate_num = generate_single()

        # 取灰度
        im_gray = im.convert('1')

        # 计数生成的数字1-9的个数,用来命名图像文件
        for j in range(1, 10):
            if generate_num == str(j):
                cnt_num[j] = cnt_num[j] + 1

                # 路径如 "F:/code/***/P_generate_handwritten_number/data_pngs/1/1_231.png"
                # 输出显示路径
                output_path = os.path.join(img_data_path, 'Num_%d' % j)
                # if not os.path.exists(output_path):
                #     os.makedirs(output_path)
                output_filename = os.path.join(output_path, '%d_%d.png' % (j, cnt_num[j]))
                print('Generate: {}'.format(output_filename))
                # 将图像保存在指定文件夹中
                im_gray.save(output_filename)

    print("\n")
    # 输出显示1-9的分布
    print("生成的1-9的分布：")
    for k in range(9):
        print("Num", k + 1, ":", cnt_num[k + 1], "in all")


if __name__ == '__main__':

    del_imgs()
    mkdir_for_imgs()

    # generate n times
    generate_1to9(100)
