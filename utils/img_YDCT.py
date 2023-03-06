import torch
# from rgb2ycbcrF import *
# from ycbcr2DCT import * 
from .rgb2ycbcrF import *
from .ycbcr2DCT import * 

def img2ycbcrDct(image):
    # image = torch.from_numpy(image.astype(np.float32) * 255)
    image =image.permute(0,2,3,1)
    # print(image.shape)
    [n,w,h,c] = image.shape
    imageDCT = torch.zeros(n,w,h)

    # 进行一个循环修改
    for i in range(n):
        ycbcrImg = rgb2ycbcr(image[i].cpu().numpy())
        # print(ycbcrImg.shape)
        yImg = ycbcrImg[:,:,0].astype(np.float32)
        # print(yImg.shape)
        dctImg =  whole_img_dct(yImg)
        # print(dctImg.shape)

        dctTensor = torch.from_numpy(dctImg)
        imageDCT[i] = dctTensor

        # if(i == 2): #测试需要
        #     plt.imsave("ycbcrImg.png", ycbcrImg/255)
        #     plt.imsave("yImg.png",yImg)
        #     plt.imsave('dctImg.png',dctImg) 

    device = torch.device('cuda:1')
    imageDCT= imageDCT.to(device)

    return imageDCT

if __name__ == '__main__':
    image =  torch.randn(10,3,256,256)
    # print(image.permute(0,2,3,1).shape)
    imageDct = img2ycbcrDct(image.permute(0,2,3,1))