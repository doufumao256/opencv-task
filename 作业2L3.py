import numpy as np
import cv2
import math

P = cv2.imread("image/lady.jpg", cv2.IMREAD_UNCHANGED)
imgo = cv2.imread("image/lady.jpg", 1)
Blue, Green, Red = cv2.split(P)  # 原来通道顺序BGR
cv2.imshow("o", imgo)
# 显示单个通道图像
cv2.imshow('channel_blue', Blue)
cv2.imshow('channel_green', Green)
cv2.imshow('channel_red', Red)

# 通道合并
merge_img = cv2.merge([Red, Green, Blue])  # 以RGB显示
cv2.imshow('merge_imgfile', merge_img)

# 提取通道参数
rows, cols, chns = P.shape
blue = np.zeros((rows, cols), P.dtype)
green = np.zeros((rows, cols), P.dtype)

zeromerge_img = cv2.merge([blue, green, Red])  # 以什么颜色结尾，图像总体就偏向其颜色
cv2.imshow('zeromerge_imgfile', zeromerge_img)
onemerge_img = cv2.merge([green, blue, Red])
cv2.imshow('onemerge_imgfile', onemerge_img)  # zero 和one对比显示前两个对换通道改变图像总体颜色较小


def gray_change(grayImg, T):
    # 获取图像的高和宽
    grayImg_height = grayImg.shape[0]
    grayImg_width = grayImg.shape[1]

    # 创建新图像
    newImg_move = np.zeros((grayImg_height, grayImg_width), np.uint8)  # 上移

    for i in range(grayImg_height):
        for j in range(grayImg_width):
            if int(grayImg[i, j] + T) > 255:  # 溢出判断
                gray = 255
            else:
                gray = int(grayImg[i, j] + T)

            newImg_move[i, j] = np.uint8(gray)
    return newImg_move


newImg_move1 = gray_change(Red, 20)
cv2.imshow("RED ++", newImg_move1)
newImg_move2 = gray_change(Green, -20)
cv2.imshow("GREEN --", newImg_move2)

# 通道合并
merge_img2 = cv2.merge([Blue, newImg_move2, newImg_move1])  # 以RGB显示
cv2.imshow('merge_imgfile2', merge_img2)

cv2.waitKey(0)


# 对数变换
def logTransform(c, img):
    h, w = img.shape[0], img.shape[1]
    new_img = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if new_img[i, j] < 127:
                new_img[i, j] = c * (math.log(1.0 + img[i, j]))
            else:
                pass
    new_img = cv2.normalize(new_img, new_img, 0, 255, cv2.NORM_MINMAX)

    return new_img


# 伽马对换
def gammaTranform(c, gamma, image):
    h, w, d = image.shape[0], image.shape[1], image.shape[2]
    new_img = np.zeros((h, w, d), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            new_img[i, j, 0] = c * math.pow(image[i, j, 0], gamma)
            new_img[i, j, 1] = c * math.pow(image[i, j, 1], gamma)
            new_img[i, j, 2] = c * math.pow(image[i, j, 2], gamma)
    cv2.normalize(new_img, new_img, 0, 255, cv2.NORM_MINMAX)
    new_img = cv2.convertScaleAbs(new_img)

    return new_img


p = cv2.imread("image/ladyB.jpg", 0)  # 读取已保存的灰度图
log_img = logTransform(1.0, p)
cv2.imshow('log_img', log_img)

img1 = cv2.imread("image/ladyB.jpg", 1)
new_img = gammaTranform(1, 2.5, img1)

cv2.imshow('x', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# titles = ['Original Image', 'channel_blue', 'channel_green', 'channel_red', 'merge_imgfile', 'zeromerge_imgfile',
#           'onemerge_imgfile', 'log_img', 'x']
# images = [P, Blue, Green, Red, merge_img, zeromerge_img, onemerge_img, log_img, new_img]
# # try:
# #     for i in range(9):
# #         images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)  # CV BGR转变RGB
# #
# #     for i in range(9):
# #         plt.subplot(3, 3, i+1), plt.imshow(images[i])
# #         plt.title(titles[i])
# #         plt.xticks([]), plt.yticks([])
# #     plt.show()


# except Exception as result:
#     print(result)
