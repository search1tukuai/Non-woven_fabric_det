import os
import cv2

from interface.correct import correct
from interface.fuse import img_fuse

from utils.util import get_files, get_dirs, get_cur_files

files_dir = "C:/Download/test/"
save_path = "C:/Download/res/"
# """
# 1、读取文件
files_name, files_path = get_files(files_dir)

# 2.1、对文件进行分类
dir_list = []
for file_name in files_name:
    file_prex = file_name.split("-")
    dir_list.append(file_prex)

# 2.2、建立文件夹
for dir in dir_list:
    path = ""
    if len(dir) == 3:
        path = save_path + "/" + dir[0] + "_fix"
    elif len(dir) == 2:
        path = save_path + "/" + dir[0]
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)

# 3、复制文件
for file_name in files_name:
    file_prex = file_name.split("-")
    dir_name = ""
    if len(file_prex) == 2:
        dir_name = file_prex[0]
    elif len(file_prex) == 3:
        dir_name = file_prex[0] + "_fix"
    img_ori = cv2.imread(files_dir + "/" + file_name)
    file_save_path = save_path + "/" + dir_name + "/" + file_name
    is_exist = os.path.exists(file_save_path)
    if not is_exist:
        cv2.imwrite(file_save_path, img_ori)
# """
# 4、图像拼接
dirs_list = get_dirs(save_path)
for dir in dirs_list:
    img_fuse_name = dir + "_fuse" + ".jpg"
    img_cor_name = dir + "_cor" + ".jpg"
    files_list = get_cur_files(os.path.join(save_path, dir))
    img1_path = ""
    img2_path = ""
    if files_list[0].split("-")[1].split(".")[0] == '1':
        img1_path = files_list[0]
        img2_path = files_list[1]
    elif files_list[0].split("-")[1].split(".")[0] == '2':
        img1_path = files_list[1]
        img2_path = files_list[0]
    else:
        print("图片顺序拜访不对")

    if len(img1_path) == 0 or len(img2_path) == 0:
        print("初始图片路径不对！")
    else:
        img1_path = os.path.join(save_path, dir, img1_path).replace("\\", "/")
        img2_path = os.path.join(save_path, dir, img2_path).replace("\\", "/")
        # 拼接
        img_fuse_res = img_fuse(img1_path, img2_path, False)
        cv2.imwrite(os.path.join(save_path, dir, img_fuse_name).replace("\\", "/"), img_fuse_res[:, 0:1200])
        # 矫正
        # img_correct_res = correct(img_fuse_res)
        # cv2.imwrite(os.path.join(save_path, dir, img_cor_name).replace("\\", "/"), img_correct_res)
        print("当前处理完：" + dir + "文件夹")

# 5、区域分割


