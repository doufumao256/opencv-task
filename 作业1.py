import cv2 as cv
import numpy as np
import matplotlib.image

src = cv.imread("image/lady.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)

cv.imshow("input", src)

T = 127

# 转换为灰度图像
img1 = cv.imread("image/lady.jpg", 0)
cv.imshow("lady", img1)

cv.imwrite("image/ladyB.jpg", img1)

gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
h, w = gray.shape
T = cv.mean(gray)[0]
print("current threshold value : ", T)

# 二值图像
binary = np.zeros((h, w), dtype=np.uint8)
for row in range(h):
    for col in range(w):
        pv = gray[row, col]
        if pv > T:
            binary[row, col] = 255
        else:
            binary[row, col] = 0
cv.imshow("binary", binary)

cv.waitKey(0)
cv.destroyAllWindows()

# 将图片转换为矩阵

img_matrix = matplotlib.image.imread("image/ladyB.jpg")
print(img_matrix)  # 输出图像转化的矩阵

# 计算每行的均值
line_average = np.mean(img_matrix, axis=1)
T1 = line_average[0]
print(f'得到第一行的平均值为{T1}')
T2 = line_average[1]
print(f'得到第二行的平均值为{T2}')

# 计算每行的中位数
line_median = np.median(img_matrix, axis=1)
T3 = line_median[0]
print(f'得到第一行的中位数为{T3}')
T4 = line_median[1]
print(f'得到第二行的中位数为{T4}')


def change(line, t):
    for i in range(len(line)):
        if line[i] >= t:
            line[i] = 255
        else:
            line[i] = 0


try:
    change(line_average, T1)
    change(line_average, T2)
    change(line_median, T3)
    change(line_median, T4)
except Exception as result:
    print(result)
