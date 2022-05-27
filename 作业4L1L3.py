import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 转换为灰度图像
imgG1 = cv.imread("image/lena.jpg", 0)
cv.imwrite("image/lenaG.jpg", imgG1)
imgG2 = cv.imread("image/images.jpg", 0)
cv.imwrite("image/imagesG.jpg", imgG2)


# 定义得到前9行与前9列交叉的81个像素的灰度矩阵（9X9）
def get_img_array(grayImg):
    new_img_array = np.zeros((9, 9), np.uint8)
    for i in range(9):
        for j in range(9):
            gray = int(grayImg[i, j])

            new_img_array[i, j] = np.uint8(gray)
    print(f'新数组：\n{new_img_array}\n')
    return new_img_array


# 输出G1,G2的前9行与前9列交叉的81个像素的灰度矩阵H1,H2（9X9）
H1 = get_img_array(imgG1)  # H1
H2 = get_img_array(imgG2)  # H2

# 均值滤波
G11 = cv.blur(imgG1, (5, 5))  # 可以更改核的大小
G21 = cv.blur(imgG2, (5, 5))
# 输出G11，G21的前9行与前9列交叉的81个像素的灰度矩阵H11,H21(9X9)
H11 = get_img_array(G11)  # H1
H21 = get_img_array(G21)  # H2

# L3部分
# 高斯滤波
G12 = cv.GaussianBlur(imgG1, (5, 5), 0)
G22 = cv.GaussianBlur(imgG2, (5, 5), 0)
# 中值滤波
G23 = cv.medianBlur(imgG2, 5)
# 显示图像
cv.imshow("LENA GRAY G1,NoiseGray G2", np.hstack([imgG1, imgG2]))
cv.imshow("medianBlurG11,medianBlurG21", np.hstack([G11, G21]))
cv.imshow("GaussianBlurG12,GaussianBlurG22,medianBlurG23", np.hstack([G12, G22, G23]))

# 等待显示
cv.waitKey(0)
cv.destroyAllWindows()

# # 显示图形
# titles = ["LENA GRAY G1", "NoiseGray G2", "medianBlurG11", "medianBlurG21", "GaussianBlurG12", "GaussianBlurG22",
#           "medianBlurG23"]
# images = [imgG1, imgG2, G11, G21, G12, G22, G23]
#
# for i in range(7):
#     plt.subplot(3, 3, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()
"""
1）随着核大小逐渐变大，会让图像变得更加模糊；

2）核必须是大于1的奇数，如3、5、7等；

3）在代码 dst = cv2.medianBlur(src, ksize) 中 填写核大小时，只需填写一个数即可，如3、5、7等，对比均值滤波函数用法。

1）随着核大小逐渐变大，会让图像变得更加模糊；

2）如果设置为核大小为（1，1），则结果就是原始图像。
"""
