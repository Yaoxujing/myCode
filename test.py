import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.utils import save_image

from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_hrnet_cfg
from utils.config import get_pscc_args
from models.NLCDetection import NLCDetection
from models.detection_head import DetectionHead
from utils.load_vdata import TestData

# 把这两个关了
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device('cuda:0') if torch.cuda.is_available() else "cpu"
# device = torch.device('cpu')


# 显存不够的时候做的尝试 失败了
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# CUDA_LAUNCH_BLOCKING=1

def load_network_weight(net, checkpoint_dir, name):
    weight_path = '{}/{}.pth'.format(checkpoint_dir, name)
    net_state_dict = torch.load(weight_path, map_location='cuda:0')
    net_state_dict = torch.load(weight_path)

    net.load_state_dict(net_state_dict)
    print('{} weight-loading succeeds'.format(name))


def test(args):
    # define backbone
    FENet_name = 'HRNet'
    FENet_cfg = get_hrnet_cfg()
    FENet = get_seg_model(FENet_cfg)

    # define localization head
    SegNet_name = 'NLCDetection'
    SegNet = NLCDetection(args)

    # define detection head
    ClsNet_name = 'DetectionHead'
    ClsNet = DetectionHead(args)

    # FENet_checkpoint_dir = './checkpoint/{}_checkpoint'.format(FENet_name)
    FENet_checkpoint_dir = 'checkpoint0113files/{}_checkpoint'.format(FENet_name)
    SegNet_checkpoint_dir = 'checkpoint0113files/{}_checkpoint'.format(SegNet_name)
    ClsNet_checkpoint_dir = 'checkpoint0113files/{}_checkpoint'.format(ClsNet_name)

    # load FENet weight
    FENet = FENet.to(device)
    # 并行运行程序
    FENet = nn.DataParallel(FENet, device_ids=device_ids)
    load_network_weight(FENet, FENet_checkpoint_dir, FENet_name)

    # load SegNet weight
    SegNet = SegNet.to(device)
    # 并行运行程序
    SegNet = nn.DataParallel(SegNet, device_ids=device_ids)
    load_network_weight(SegNet, SegNet_checkpoint_dir, SegNet_name)

    # load ClsNet weight
    ClsNet = ClsNet.to(device)
    ClsNet = nn.DataParallel(ClsNet, device_ids=device_ids)
    load_network_weight(ClsNet, ClsNet_checkpoint_dir, ClsNet_name)

    test_data_loader = DataLoader(TestData(args), batch_size=1, shuffle=False,
                                  num_workers=8)
    
   

    for batch_id, test_data in enumerate(test_data_loader):

        image, cls, name = test_data
        image = image.to(device)

        with torch.no_grad():

            # backbone network
            FENet.eval()
            feat = FENet(image)

            # localization head
            SegNet.eval()
            pred_mask = SegNet(feat)[0]

            # 调整输出的大小
            pred_mask = F.interpolate(pred_mask, size=(image.size(2), image.size(3)), mode='bilinear',
                                      align_corners=True)

            # classification head
            ClsNet.eval()
            pred_logit = ClsNet(feat)

        # ce
        sm = nn.Softmax(dim=1)
        pred_logit = sm(pred_logit)
        _, binary_cls = torch.max(pred_logit, 1)

        # if binary_cls.item() == 0 :
        #     pred_tag = 'authentic'
        # elif binary_cls.item() == 1 :
        #     pred_tag = 'splice'
        # elif binary_cls.item() == 2 :
        #     pred_tag = 'copymove'
        # else:
        #     pred_tag = 'removal'

        pred_tag = 'forged' if binary_cls.item() == 1 else 'authentic'

        if args.save_tag:
            save_image(pred_mask, name, 'mask')

        print_name = name[0].split('/')[-1].split('.')[0]

        print(f'The image {print_name} is {pred_tag}')
    # print(f'save in ')


if __name__ == '__main__':
    args = get_pscc_args()
    test(args)
