import cv2
import numpy as np


def gamma_trans(img, gamma):  # gamma大于1时图片变暗，小于1图片变亮
    # 具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    # 实现映射用的是Opencv的查表函数
    return cv2.LUT(img, gamma_table)


P = cv2.imread("image/lady.jpg", 0)  # 打开文件，并以灰度图显示
reverse_img = 255 - P

cv2.imshow('srcimg', reverse_img)  # 图像反转

# 执行Gamma矫正，小于1的值让暗部细节大量提升，同时亮部细节少量提升
img_corrected = gamma_trans(P, 0.5)
cv2.imshow('a1', img_corrected)  # 图像变亮
cv2.imshow('as1', P)  # 原图
img_corrected = gamma_trans(P, 2)
cv2.imshow('a1g', img_corrected)  # 图像变暗

cv2.waitKey(0)
cv2.destroyAllWindows()
