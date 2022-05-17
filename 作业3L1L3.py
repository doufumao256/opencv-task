import cv2
import numpy as np
from matplotlib import pyplot as plt

# 获取灰度图像
img = cv2.imread("image/lady.jpg", 0)


def gray_change(grayImg, T):
    # 获取图像的高和宽
    grayImg_height = grayImg.shape[0]
    grayImg_width = grayImg.shape[1]

    # 创建新图像
    newImg_move = np.zeros((grayImg_height, grayImg_width), np.uint8)

    for i in range(grayImg_height):
        for j in range(grayImg_width):
            if int(grayImg[i, j] * T) > 255:  # 溢出判断
                gray = 255
            else:
                gray = int(grayImg[i, j] * T)

            newImg_move[i, j] = np.uint8(gray)
    return newImg_move


img1 = gray_change(img, 2)  # P2
img2 = gray_change(img, 0.5)  # P3
cv2.imshow('P, P2 ,P3', np.hstack([img, img1, img2]))


# 灰度图像的直方图
origin_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
img1_hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
img2_hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(img)
dst1 = cv2.equalizeHist(img1)
dst2 = cv2.equalizeHist(img2)
cv2.imshow("P11, P21 ,P30", np.hstack([dst, dst1, dst2]))

# 直方图
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
hist1 = cv2.calcHist([dst1], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([dst2], [0], None, [256], [0, 256])

plt.figure()
plt.subplot(2, 3, 1), plt.plot(origin_hist), plt.title('H1')
plt.subplot(2, 3, 2), plt.plot(img1_hist1), plt.title('H2')
plt.subplot(2, 3, 3), plt.plot(img2_hist2), plt.title('H3')
plt.subplot(2, 3, 4), plt.plot(hist), plt.title('H11')
plt.subplot(2, 3, 5), plt.plot(hist1), plt.title('H21')
plt.subplot(2, 3, 6), plt.plot(hist2), plt.title('H3')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()



