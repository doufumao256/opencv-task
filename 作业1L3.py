import cv2
import numpy as np
import matplotlib.image

# 以灰度图的方式读取图片
img_gray = cv2.imread("image/lady.jpg", 0)
cv2.imshow("Scr", img_gray)
cv2.waitKey(0)

img_matrix = matplotlib.image.imread("image/ladyB.jpg")
average = np.mean(img_matrix)

# 阈值
T = average / 255
print(T)

# 归一化(0,1)
img_gray = img_gray / 255
cv2.imshow("Normalization", img_gray)
cv2.waitKey(0)

# 二值化
img_gray_h, img_gray_w = img_gray.shape
for i in range(img_gray_h):
    if i <= img_gray_h / 2:
        for j in range(img_gray_w):
            if img_gray[i][j] <= T:
                img_gray[i][j] = 0
            else:
                img_gray[i][j] = 1
    else:
        pass
cv2.imshow("Binarization", img_gray)
cv2.waitKey(0)
