import numpy as np
# import imageio #有个警告
import imageio.v2 as imageio
import matplotlib.pyplot as plt

def rgb2ycbcr(rgb_image):
    """convert rgb into ycbcr"""
    if len(rgb_image.shape)!=3 or rgb_image.shape[2]!=3:
        raise ValueError("input image is not a rgb image")
    rgb_image = rgb_image.astype(np.float32)
    # 1：创建变换矩阵，和偏移量
    transform_matrix = np.array([[0.257, 0.564, 0.098],
                                 [-0.148, -0.291, 0.439],
                                 [0.439, -0.368, -0.071]])
    shift_matrix = np.array([16, 128, 128])
    ycbcr_image = np.zeros(shape=rgb_image.shape)
    w, h, _ = rgb_image.shape
    # 2：遍历每个像素点的三个通道进行变换
    for i in range(w):
        for j in range(h):
            ycbcr_image[i, j, :] = np.dot(transform_matrix, rgb_image[i, j, :]) + shift_matrix       
    return ycbcr_image

if __name__ == "__main__":
    
    rgb_image = imageio.imread("testimg.png")
    ycbcr_image = rgb2ycbcr(rgb_image)

    images = [rgb_image, ycbcr_image]
    titles = ["orignal", "ycbcr"]
    for i in range(1, len(images)+1):
        plt.subplot(1, 2, i)
        plt.title(titles[i-1])
        plt.imshow(images[i-1]/255)
        plt.imsave(titles[i-1]+'.png',images[i-1]/255)