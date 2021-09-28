import cv2


def get_file_name(path):
    """
    从路径中分割出文件名
    :param path: 图片的路径
    :return: [文件名, 后缀]
    """
    path_split = path.split("\\")
    name = path_split[len(path_split) - 1]
    return name