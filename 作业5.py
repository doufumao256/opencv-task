import cv2 as cv
import numpy as np

imgG1 = cv.imread("image/lena.jpg", 0)

x = cv.Sobel(imgG1, cv.CV_16S, 1, 0)
y = cv.Sobel(imgG1, cv.CV_16S, 0, 1)
imgG11 = cv.convertScaleAbs(x)  # 转回uint8
imgG12 = cv.convertScaleAbs(y)

A = np.array([
    [5, 5, 5, 10, 10, 10],
    [5, 5, 5, 10, 10, 10],
    [5, 5, 5, 10, 10, 10],
    [5, 5, 5, 10, 10, 10],
    [5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5],
])
Adx = A
Ady = A
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # sobel算子X方向的模板
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # sobel算子Y方向的模板
# Soble算子的两种模板分别对A处理
for i in range(0, 5):
    for j in range(0, 4):
        Adx[i + 1, j + 1] = abs(np.sum(A[i:i + 3, j:j + 3] * sobel_x))
        Ady[i + 1, j + 1] = abs(np.sum(A[i:i + 3, j:j + 3] * sobel_y))
# Soble算子的两种模板分别对A处理后的结果
print(Adx)
print(Ady)

# laplacian处理G1
laplacian_img = cv.Laplacian(imgG1, cv.CV_16S, ksize=3)
imgG13 = cv.convertScaleAbs(laplacian_img)

# X方向Sobel  Y方向Sobel  laplacian处理后的图像
cv.imshow("G1(by X)   ,G2(by Y),   G3 (by laplacian)", np.hstack([imgG11, imgG12, imgG13]))
cv.waitKey(0)
cv.destroyAllWindows()
