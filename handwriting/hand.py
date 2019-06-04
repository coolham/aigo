
import os

current_path = os.getcwd()
data_path = os.path.join(current_path, 'data')

csv_path = os.path.join(data_path, 'csv')



def get_feature(img):
    '''
     # 提取特征
     # 30*30的图像，
    :param img:
    :return:
    '''
    width, height = img.size
    pixel_cnt_list = []

    height = 30
    for y in range(height):
        pixel_cnt_x = 0
        for x in range(width):
            if img.getpixel((x, y)) == 0:  # 黑点
                pixel_cnt_x += 1
        pixel_cnt_list.append(pixel_cnt_x)

    for x in range(width):
        pixel_cnt_y = 0
        for y in range(height):
            if img.getpixel((x, y)) == 0:  # 黑点
                pixel_cnt_y += 1
        pixel_cnt_list.append(pixel_cnt_y)

    return pixel_cnt_list


# 遍历访问文件夹 num_1-9 中的所有图像文件，进行特征提取，然后写入 CSV 文件中
with open(path_csv + "tmp.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # 访问文件夹 1-9
    for i in range(1, 10):
        num_list = os.listdir(path_images + "num_" + str(i))
        print(path_images + "num_" + str(i))
        print("num_list:", num_list)
        # 读到图像文件
        if os.path.isdir(path_images + "num_" + str(i)):
            print("样本个数：", len(num_list))
            sum_images = sum_images + len(num_list)

            # Travsel every single image to generate the features
            for j in range(0, (len(num_list))):
                # 处理读取单个图像文件提取特征
                img = Image.open(path_images + "num_" + str(i) + "/" + num_list[j])
                get_features_single(img)
                pixel_cnt_list.append(num_list[j][0])

                # 写入CSV
                writer.writerow(pixel_cnt_list)

