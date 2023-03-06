import numpy as np
import torch.utils.data as data
from os.path import join
from PIL import Image
import random
from random import randrange
import torch
import imageio


def generate_4masks(mask):
    # 调整mask的大小,以适应四个训练
    mask_pil = Image.fromarray(mask)

    (width2, height2) = (mask_pil.width // 2, mask_pil.height // 2)
    (width3, height3) = (mask_pil.width // 4, mask_pil.height // 4)
    (width4, height4) = (mask_pil.width // 8, mask_pil.height // 8)

    mask2 = mask_pil.resize((width2, height2))
    mask3 = mask_pil.resize((width3, height3))
    mask4 = mask_pil.resize((width4, height4))

    mask = mask.astype(np.float32) / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    mask2 = np.asarray(mask2).astype(np.float32) / 255
    mask2[mask2 > 0.5] = 1
    mask2[mask2 <= 0.5] = 0

    mask3 = np.asarray(mask3).astype(np.float32) / 255
    mask3[mask3 > 0.5] = 1
    mask3[mask3 <= 0.5] = 0

    mask4 = np.asarray(mask4).astype(np.float32) / 255
    mask4[mask4 > 0.5] = 1
    mask4[mask4 <= 0.5] = 0

    mask = torch.from_numpy(mask)
    mask2 = torch.from_numpy(mask2)
    mask3 = torch.from_numpy(mask3)
    mask4 = torch.from_numpy(mask4)

    return mask, mask2, mask3, mask4


def data_aug(img, data_aug_ind):
    #图片的一个旋转
    img = Image.fromarray(img)
    if data_aug_ind == 0:
        return np.asarray(img)
    elif data_aug_ind == 1:
        return np.asarray(img.rotate(90, expand=True))
    elif data_aug_ind == 2:
        return np.asarray(img.rotate(180, expand=True))
    elif data_aug_ind == 3:
        return np.asarray(img.rotate(270, expand=True))
    elif data_aug_ind == 4:
        return np.asarray(img.transpose(Image.FLIP_TOP_BOTTOM))
    elif data_aug_ind == 5:
        return np.asarray(img.rotate(90, expand=True).transpose(Image.FLIP_TOP_BOTTOM))
    elif data_aug_ind == 6:
        return np.asarray(img.rotate(180, expand=True).transpose(Image.FLIP_TOP_BOTTOM))
    elif data_aug_ind == 7:
        return np.asarray(img.rotate(270, expand=True).transpose(Image.FLIP_TOP_BOTTOM))
    else:
        raise Exception('Data augmentation index is not applicable.')


class TrainData(data.Dataset):
    def __init__(self, args):
        super(TrainData, self).__init__()
        path, crop_size, train_num, train_ratio, val_num = args['path'], args['crop_size'], args['train_num'], args['train_ratio'], args['val_num']

        # authentic
        authentic_names = []
        authentic_path = join(path, 'authenticF')

        with open(join(authentic_path, 'authentic.txt')) as f:
            contents = f.readlines()
            for content in contents[val_num:]:
                authentic_names.append(join(authentic_path, content.strip())) #Python strip() 方法用于移除字符串头尾指定的字符（默认为空格）或字符序列

        # splice  0111
        splice_names = []  #存储所有图片的路径+ 图片名
        splice_path = join(path, 'spliceF')

        with open(join(splice_path, 'Splic1_from512cut.txt')) as f:
            contents = f.readlines()
            for content in contents[val_num:]: #留了前val_num 200张作为验证集
                splice_names.append(join(splice_path, content.strip()))

        # copymove 0111
        copymove_names = []
        copymove_path = join(path, 'copymoveF')

        with open(join(copymove_path, 'Copymove1_from512.txt')) as f:
            contents = f.readlines()
            for content in contents[val_num:]:
                copymove_names.append(join(copymove_path, content.strip()))


        self.image_names = [authentic_names, splice_names, copymove_names]  

        self.train_num = train_num
        self.train_ratio = train_ratio
        self.crop_size = crop_size

    def rgba2rgb(self, rgba, background=(255, 255, 255)):
        row, col, ch = rgba.shape

        rgb = np.zeros((row, col, 3), dtype='float32')
        r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

        a = np.asarray(a, dtype='float32') / 255.0

        R, G, B = background

        rgb[:, :, 0] = r * a + (1.0 - a) * R
        rgb[:, :, 1] = g * a + (1.0 - a) * G
        rgb[:, :, 2] = b * a + (1.0 - a) * B

        return np.asarray(rgb, dtype='uint8')

    def get_item(self, index):

        crop_width, crop_height = self.crop_size
        train_num = self.train_num
        train_ratio = self.train_ratio

        # get 3 class
        if index < train_num * train_ratio[0]:
            cls = 0
        elif train_num * train_ratio[0] <= index < train_num * (train_ratio[0] + train_ratio[1]):
            cls = 1
        else:
            cls = 2

        one_cls_names = self.image_names[cls]

        index = randrange(0, len(one_cls_names))

        # read the chosen image
        image_name = one_cls_names[index]
        image = imageio.imread(image_name)

        im_height, im_width, im_channel = image.shape

        if im_channel != 3:
            #确保输入的图像都是三通道的
            print(image_name)
            raise Exception('Image channel is not 3.')

        # authentic
        if cls == 0:
            if image.shape[-1] == 4:
                image = self.rgba2rgb(image)

            if im_height != crop_height or im_width != crop_width:
                # resize image
                image = Image.fromarray(image.astype(np.uint8))
                image = image.resize((crop_height, crop_width), resample=Image.BICUBIC)
                image = np.asarray(image)

            mask = np.zeros((crop_height, crop_width)).astype(np.uint8) #生成一个全为0的图片，表示该图片没有被修改

        # splice
        elif cls == 1:
            mask_name = image_name.replace('2cut', '2cut_Mask')

            mask = imageio.imread(mask_name)
            ma_height, ma_width = mask.shape[:2]

            if im_width != ma_width or im_height != ma_height:
                raise Exception('the sizes of image and mask are different: {}'.format(image_name))

            if im_height != crop_height or im_width != crop_width:
                # resize image
                image = Image.fromarray(image)
                image = image.resize((crop_height, crop_width), resample=Image.BICUBIC)
                image = np.asarray(image)
                # resize mask
                mask = Image.fromarray(mask)
                mask = mask.resize((crop_height, crop_width), resample=Image.BICUBIC)
                mask = np.asarray(mask)

        # copymove
        elif cls == 2:
            mask = imageio.imread(image_name.replace('2cut', '2cut_Mask'))
            ma_height, ma_width = mask.shape[:2]

            if im_width != ma_width or im_height != ma_height:
                raise Exception('the sizes of image and mask are different: {}'.format(image_name))

            if im_height != crop_height or im_width != crop_width:
                # resize image
                image = Image.fromarray(image)
                image = image.resize((crop_height, crop_width), resample=Image.BICUBIC)
                image = np.asarray(image)
                # resize mask
                mask = Image.fromarray(mask)
                mask = mask.resize((crop_height, crop_width), resample=Image.BICUBIC)
                mask = np.asarray(mask)

        else:
            raise Exception('class is not defined!')


        # 任意选择一个方式进行图片/mask的旋转 -增强图片
        # image   
        aug_index = randrange(0, 8)
        image = data_aug(image, aug_index)
        image = torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1)

        # mask
        mask = data_aug(mask, aug_index)

        mask, mask2, mask3, mask4 = generate_4masks(mask)

        return image, [mask, mask2, mask3, mask4], cls

    def __getitem__(self, index):
        res = self.get_item(index)
        return res

    def __len__(self):
        return self.train_num


class ValData(data.Dataset):
    def __init__(self, args):
        super(ValData, self).__init__()

        path, val_num = args['path'], args['val_num']

        # authentic
        authentic_names = []
        authentic_path = join(path, 'authenticF')

        with open(join(authentic_path, 'authentic.txt')) as f:
            contents = f.readlines()
            for content in contents[:val_num]:
                authentic_names.append(join(authentic_path, content.strip()))

        authentic_cls = [0] * val_num

        # 0111
        # splice
        splice_names = []
        splice_path = join(path, 'spliceF')

        with open(join(splice_path, 'Splic1_from512cut.txt')) as f:
            contents = f.readlines()
            for content in contents[:val_num]:
                splice_names.append(join(splice_path, content.strip()))

        splice_cls = [1] * val_num

        #这里也要注释掉
        # copymove
        copymove_names = []
        copymove_path = join(path, 'copymoveF')

        with open(join(copymove_path, 'Copymove1_from512.txt')) as f:
            contents = f.readlines()
            for content in contents[:val_num]:
                copymove_names.append(join(copymove_path, content.strip()))

        copymove_cls = [2] * val_num

        self.image_names = authentic_names + splice_names + copymove_names  
        self.image_class = authentic_cls + splice_cls + copymove_cls

    def get_item(self, index):
        image_name = self.image_names[index]
        cls = self.image_class[index]

        image = imageio.imread(image_name)

        im_height, im_width, im_channel = image.shape

        if im_channel != 3:
            print(image_name)
            raise Exception('Image channel is not 3.')

        # image
        # 这边进行了一个归一化/255
        image = torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1)

        # authentic
        if cls == 0:
            # mask
            mask = np.zeros((im_height, im_width))
            mask = torch.from_numpy(mask.astype(np.float32))

        # splice
        elif cls == 1:
            # mask
            #因为后面有个cut 所以只能找 2cut
            mask = imageio.imread(image_name.replace('2cut', '2cut_Mask'))
            mask = torch.from_numpy(mask.astype(np.float32) / 255)

        # copymove
        elif cls == 2:
            # mask
            mask = imageio.imread(image_name.replace('2cut', '2cut_Mask'))
            mask = torch.from_numpy(mask.astype(np.float32) / 255)

        else:
            raise Exception('class is not defined!')

        return image, mask, cls, image_name

    def __getitem__(self, index):
        res = self.get_item(index)

        return res

    def __len__(self):
        return len(self.image_names)
