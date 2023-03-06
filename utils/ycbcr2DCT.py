import cv2
import numpy as np
import matplotlib.pyplot as plt
from .rgb2ycbcrF import *
import imageio.v2 as imageio


# 整张图 DCT 变换
def whole_img_dct(img_f32):
    img_dct = cv2.dct(img_f32)            # 进行离散余弦变换
    img_dct_log = np.log(abs(img_dct))    # 进行log处理
    # img_idct = cv2.idct(img_dct)          # 进行离散余弦反变换
    return img_dct_log

# 分块图 DCT 变换
def block_img_dct(img_f32):
    height,width = img_f32.shape[:2]
    block_y = height // 8
    block_x = width // 8
    height_ = block_y * 8
    width_ = block_x * 8
    img_f32_cut = img_f32[:height_, :width_]
    img_dct = np.zeros((height_, width_), dtype=np.float32)
    # new_img = img_dct.copy()
    for h in range(block_y):
        for w in range(block_x):
            # 对图像块进行dct变换
            img_block = img_f32_cut[8*h: 8*(h+1), 8*w: 8*(w+1)]
            img_dct[8*h: 8*(h+1), 8*w: 8*(w+1)] = cv2.dct(img_block)

            # 进行 idct 反变换
            # dct_block = img_dct[8*h: 8*(h+1), 8*w: 8*(w+1)]
            # img_block = cv2.idct(dct_block)
            # new_img[8*h: 8*(h+1), 8*w: 8*(w+1)] = img_block

    img_dct_log2 = np.log(abs(img_dct))
    # return img_dct_log2, new_img
    return img_dct_log2


if __name__ == '__main__':
    # img_u8 = cv2.imread("plane.jpg")
    rgb_image = imageio.imread('testimg.png')
    #转化为cbcry
    ycbcr_image = rgb2ycbcr(rgb_image)
    print(ycbcr_image.shape) #(513, 513, 3)

    img_f32 = ycbcr_image[:,:,0].astype(np.float32)  # 数据类型转换 转换为浮点型
    img_dct_log  = whole_img_dct(img_f32)
    
    print(img_dct_log.shape) #(513, 513)

    img_dct_block= block_img_dct(img_f32)

    # plt.figure(1, figsize=(12, 8))
    # plt.subplot(231)
    # plt.imshow(rgb_image)
    # plt.title('or'), plt.xticks([]), plt.yticks([])
    # plt.subplot(232)
    # plt.imshow(ycbcr_image[:,:,0])
    # plt.title('rgb2-k'), plt.xticks([]), plt.yticks([])

    # plt.subplot(233)
    # plt.imshow(img_dct_log)
    # plt.title('DCT-all'), plt.xticks([]), plt.yticks([])

    # plt.subplot(234)
    # plt.imshow(img_dct_block)
    # plt.title('DCT-block'), plt.xticks([]), plt.yticks([])

    # plt.subplot(235)
    # plt.imshow(new_img1)
    # plt.title('IDCT-all'), plt.xticks([]), plt.yticks([])

    # plt.subplot(236)
    # plt.imshow(new_img2)
    # plt.title('IDCT-block'), plt.xticks([]), plt.yticks([])
    # plt.show()
    plt.imsave('img_dct_log.png',img_dct_log)
    plt.imsave('img_dct_block.png',img_dct_block)
