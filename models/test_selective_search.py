import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import numpy as np
import cv2
from PIL import Image
# def norm_image(image):
#     """
#     标准化图像
#     :param image: [H,W,C]
#     :return:
#     """
#     image = image.copy()
#     image = image - np.max(np.min(image), 0)
#     image = image/np.max(image)
#     image = image
#     return np.uint8(image)
# # 加载图片数据
# # img = skimage.data.checkerboard()
# img_path = './selective_search.jpg'
# img = Image.open(img_path)
# original_image = np.asarray(img)
# H, W = original_image.shape[:2]
# import time
# start = time.time()
# img_lbl, regions = selectivesearch.selective_search(original_image, scale=500, sigma=0.9, min_size=200)
# # 返回regions 为左上角坐标和宽和长
# end = time.time()
# print(end-start)
# # 计算一共分割了多少个原始候选区域
# temp = set()
# for i in range(img_lbl.shape[0]):
#     for j in range(img_lbl.shape[1]):
#         temp.add(img_lbl[i, j, 3])
# print(len(temp))
#
# print(len(regions))  # 计算利用Selective Search算法得到了多少个候选区域

# 创建一个集合 元素list(左上角x，左上角y,宽,高)
# candidates = set()
# for r in regions:
#     if r['rect'] in candidates:  # 排除重复的候选区
#         continue
#     if r['size'] < 2000:  # 排除小于 2000 pixels的候选区域(并不是bounding box中的区域大小)
#         continue
#     x, y, w, h = r['rect']
#     if w / h > 2 or h / w > 2:  # 排除扭曲的候选区域边框  即只保留近似正方形的
#         continue
#     candidates.add(r['rect'])
# search_region = {}
# fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
# ax.imshow(original_image)
# i = 0
# for x, y, w, h in candidates:
#     # rect = mpatches.Rectangle(
#     #     (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
#     # ax.add_patch(rect)
#     norm_x, norm_y, norm_w, norm_h = x/W, y/H, w/W, h/H
#     search_region['selective_region_' + str(i)] = (norm_x, norm_y, norm_w, norm_h)
#     i = i + 1
# # plt.show()
# print(search_region) # x, y, w, h
print(0.81**0.2)