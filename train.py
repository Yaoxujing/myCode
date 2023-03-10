import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.utils import findLastCheckpoint, save_image, adjust_learning_rate

from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_hrnet_cfg
from models.NLCDetection import NLCDetection
# from models.DiceLoss import DiceLoss  #引入这个损失
from models.detection_head import DetectionHead
from utils.config import get_pscc_args
from utils.load_tdata import TrainData
from utils.load_tdata import ValData
import logging

#在cuda 的 1 核上
device = torch.device('cuda:1')
device_ids = [1]


def train(args):
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

    # load train data
    train_data_loader = DataLoader(TrainData(args), batch_size=args['train_bs'], shuffle=True, num_workers=8)

    FENet = FENet.to(device)
    SegNet = SegNet.to(device)
    ClsNet = ClsNet.to(device)

    #parameters Adam
    params = list(FENet.parameters()) + list(SegNet.parameters()) + list(ClsNet.parameters())
    optimizer = torch.optim.Adam(params, lr=args['learning_rate'])

    #GPU并行操作，但是device_ids 只有一个，就是默认没有并行的操作
    FENet = nn.DataParallel(FENet, device_ids=device_ids)
    SegNet = nn.DataParallel(SegNet, device_ids=device_ids)
    ClsNet = nn.DataParallel(ClsNet, device_ids=device_ids)

    #这个定义一个路径保存训练参数 不知道要不要加./
    # 重新定义一个训练参数，不用预训练 原来是checkpoint
    # FENet_dir = './checkpoint0223files/{}_checkpoint'.format(FENet_name)
    FENet_dir = 'myCode/checkpoint0307files/{}_checkpoint'.format(FENet_name)

    if not os.path.exists(FENet_dir):
        os.mkdir(FENet_dir)

    SegNet_dir = 'myCode/checkpoint0307files/{}_checkpoint'.format(SegNet_name)
    if not os.path.exists(SegNet_dir):
        os.mkdir(SegNet_dir)

    ClsNet_dir = 'myCode/checkpoint0307files/{}_checkpoint'.format(ClsNet_name)
    if not os.path.exists(ClsNet_dir):
        os.mkdir(ClsNet_dir)

    
    # 配置日志记录器
    logging.basicConfig(filename='myCode/只修改了分类的网络.log', level=logging.INFO)
    # 记录日志
    log = logging.getLogger(__name__)
    log.info('开始训练模型')

    # load FENet weight
    try:
        FENet_weight_path = '{}/{}.pth'.format(FENet_dir, FENet_name)
        FENet_state_dict = torch.load(FENet_weight_path, map_location='cuda:0')
        FENet.load_state_dict(FENet_state_dict)
        print('{} weight-loading succeeds: {}'.format(FENet_name, FENet_weight_path))
    except:
        print('{} weight-loading fails'.format(FENet_name))

    # load SegNet weight
    try:
        SegNet_weight_path = '{}/{}.pth'.format(SegNet_dir, SegNet_name)
        SegNet_state_dict = torch.load(SegNet_weight_path, map_location='cuda:0')
        SegNet.load_state_dict(SegNet_state_dict)
        print('{} weight-loading succeeds: {}'.format(SegNet_name, SegNet_weight_path))
    except:
        print('{} weight-loading fails'.format(SegNet_name))

    # load ClsNet weight
    try:
        ClsNet_weight_path = '{}/{}.pth'.format(ClsNet_dir, ClsNet_name)
        ClsNet_state_dict = torch.load(ClsNet_weight_path, map_location='cuda:0')
        ClsNet.load_state_dict(ClsNet_state_dict)
        print('{} weight-loading succeeds: {}'.format(ClsNet_name, ClsNet_weight_path))
    except:
        print('{} weight-loading fails'.format(ClsNet_name))

    # validation
    print('length of traindata: {}'.format(len(train_data_loader)))
    previous_score = validation(FENet, SegNet, ClsNet, args)
    print('previous_score {0:.4f}'.format(previous_score))

    # cross entropy loss
    authentic_ratio = args['train_ratio'][0]
    # fake_ratio = 1 - authentic_ratio
    copymove_ratio = args['train_ratio'][1]
    splice_ratio = args['train_ratio'][2]
    text_ratio = args['train_ratio'][3]
    erase_ratio = args['train_ratio'][4]
    scrawl_ratio = args['train_ratio'][5]

    # print('authentic_ratio: {}'.format(authentic_ratio), 'fake_ratio: {}'.format(fake_ratio))
    print('authentic_ratio: {}'.format(authentic_ratio),
           'copymove_ratio: {}'.format(copymove_ratio),
           'splice_ratio: {}'.format(splice_ratio),
           'text_ratio: {}'.format(text_ratio),
           'erase_ratio: {}'.format(erase_ratio),
           'scrawl_ratio: {}'.format(scrawl_ratio))

    # weights = [1. / authentic_ratio, 1. /fake_ratio]
    weights = [1. / authentic_ratio, 1. / copymove_ratio , 1./ splice_ratio , 1./text_ratio,1./erase_ratio,1./scrawl_ratio]
    weights = torch.tensor(weights)
    
    # 交叉熵损失
    CE_loss = nn.CrossEntropyLoss(weight=weights).to(device)

    # 原本是BCE损失
    BCE_loss_full = nn.BCELoss(reduction='none').to(device)
    # 现在改成 DICEloss
    # DICE_loss_full = DiceLoss()

    # 从上一次训练的banch开始 
    initial_epoch = findLastCheckpoint(save_dir=SegNet_dir)
    if initial_epoch > 0:
        try:
            FENet_checkpoint = torch.load(
                '{0}/{1}_{2}.pth'.format(FENet_dir, FENet_name, initial_epoch))
            FENet.load_state_dict(FENet_checkpoint['model'])
            print("resuming FENet by loading epoch {}".format(initial_epoch))

            SegNet_checkpoint = torch.load(
                '{0}/{1}_{2}.pth'.format(SegNet_dir, SegNet_name, initial_epoch))
            SegNet.load_state_dict(SegNet_checkpoint['model'])
            print("resuming SegNet by loading epoch {}".format(initial_epoch))

            ClsNet_checkpoint = torch.load(
                '{0}/{1}_{2}.pth'.format(ClsNet_dir, ClsNet_name, initial_epoch))
            ClsNet.load_state_dict(ClsNet_checkpoint['model'])
            optimizer.load_state_dict(ClsNet_checkpoint['optimizer'])
            print("resuming ClsNet by loading epoch {}".format(initial_epoch))
        except:
            print('cannot load checkpoint on epoch {}'.format(initial_epoch))
            initial_epoch = 0
            print("resuming by loading epoch {}".format(initial_epoch))

    # 这个循环开始训练了
    for epoch in range(initial_epoch, args['num_epochs']):

        #自适应修改学习率 
        adjust_learning_rate(optimizer, epoch, args['lr_strategy'], args['lr_decay_step'])
        seg_total, seg_correct, seg_loss_sum = [0] * 3
        cls_total, cls_correct, cls_loss_sum = [0] * 3

        for batch_id, train_data in enumerate(train_data_loader):

            image, masks, cls = train_data
            
            #简单规划成两类 要改这里
            # cls[cls != 0] = 1

            mask1, mask2, mask3, mask4 = masks

            # median-frequency class weighting
            mask1_balance = torch.ones_like(mask1)
            if (mask1 == 1).sum():
                mask1_balance[mask1 == 1] = 0.5 / ((mask1 == 1).sum().to(torch.float) / mask1.numel())
                mask1_balance[mask1 == 0] = 0.5 / ((mask1 == 0).sum().to(torch.float) / mask1.numel())
            else:
                print('Mask1 balance is not working!')

            mask2_balance = torch.ones_like(mask2)
            if (mask2 == 1).sum():
                mask2_balance[mask2 == 1] = 0.5 / ((mask2 == 1).sum().to(torch.float) / mask2.numel())
                mask2_balance[mask2 == 0] = 0.5 / ((mask2 == 0).sum().to(torch.float) / mask2.numel())
            else:
                print('Mask2 balance is not working!')

            mask3_balance = torch.ones_like(mask3)
            if (mask3 == 1).sum():
                mask3_balance[mask3 == 1] = 0.5 / ((mask3 == 1).sum().to(torch.float) / mask3.numel())
                mask3_balance[mask3 == 0] = 0.5 / ((mask3 == 0).sum().to(torch.float) / mask3.numel())
            else:
                print('Mask3 balance is not working!')

            mask4_balance = torch.ones_like(mask4)
            if (mask4 == 1).sum():
                mask4_balance[mask4 == 1] = 0.5 / ((mask4 == 1).sum().to(torch.float) / mask4.numel())
                mask4_balance[mask4 == 0] = 0.5 / ((mask4 == 0).sum().to(torch.float) / mask4.numel())
            else:
                print('Mask4 balance is not working!')


            #全部转化为cuda型
            image = image.to(device)
            mask1, mask2, mask3, mask4 = mask1.to(device), mask2.to(device), mask3.to(device), mask4.to(device)
            mask1_balance, mask2_balance, mask3_balance, mask4_balance, = mask1_balance.to(device), mask2_balance.to(
                device), mask3_balance.to(device), mask4_balance.to(device)

            cls = cls.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # feature extraction network
            FENet.train()
            feat = FENet(image)
            SegNet.train()
            [pred_mask1, pred_mask2, pred_mask3, pred_mask4] = SegNet(feat)

            # 这个分类就分了两类
            
            # 现在把它改成分四类

            ClsNet.train()
            pred_logit = ClsNet(feat)


            pred_mask1, pred_mask2, pred_mask3, pred_mask4 = pred_mask1.squeeze(dim=1), pred_mask2.squeeze(
                dim=1), pred_mask3.squeeze(dim=1), pred_mask4.squeeze(dim=1)

            # print("pred_mask1",pred_mask1.shape)
            # pred_mask1 torch.Size([10, 256, 256])

            # print("mask1",mask1.shape)
            # mask1 torch.Size([10, 256, 256])

            # 这里的改动
            mask1_loss = torch.mean(BCE_loss_full(pred_mask1, mask1) * mask1_balance)
            mask2_loss = torch.mean(BCE_loss_full(pred_mask2, mask2) * mask2_balance)
            mask3_loss = torch.mean(BCE_loss_full(pred_mask3, mask3) * mask3_balance)
            mask4_loss = torch.mean(BCE_loss_full(pred_mask4, mask4) * mask4_balance)
            seg_loss = mask1_loss + mask2_loss + mask3_loss + mask4_loss

            # 就是这里的问题吧 现在这个loss改了 也不用改
            # crossEntropy
            cls_loss = CE_loss(pred_logit, cls)

            # print("pred_logit",pred_logit.shape)
            # pred_logit torch.Size([10, 2])
            # print("cls",cls.shape)
            # cls torch.Size([10])
            loss = seg_loss + cls_loss

            loss.backward()
            optimizer.step()

            # localization
            binary_mask1 = torch.zeros_like(pred_mask1)
            binary_mask1[pred_mask1 > 0.5] = 1
            binary_mask1[pred_mask1 <= 0.5] = 0

            seg_correct += (binary_mask1 == mask1).sum().item()
            seg_total += int(torch.ones_like(mask1).sum().item())

            # detection
            _, binary_cls = torch.max(pred_logit, 1)

            # print(binary_cls)

            # 正确率
            cls_correct += (binary_cls == cls).sum().item()
            
            # 总数
            cls_total += int(torch.ones_like(cls).sum().item())

            # print statistics
            seg_loss_sum += seg_loss.item()
            cls_loss_sum += cls_loss.item()

            if batch_id % 100 == 99:
                print(
                    '[{0}, {1}] batch_loc_acc: [{2}/{3}] {4:.2f}, seg_loss: {5:.4f}; batch_cls_acc: [{6}/{7}] {8:.2f}, '
                    'cls_loss: {9:.4f}'.format(epoch + 1, batch_id + 1, seg_correct, seg_total,
                                               seg_correct / seg_total * 100,
                                               seg_loss_sum / 100, cls_correct, cls_total,
                                               cls_correct / cls_total * 100,
                                               cls_loss_sum / 100))
                log.info('[{0}, {1}] batch_loc_acc: [{2}/{3}] {4:.2f}, seg_loss: {5:.4f}; batch_cls_acc: [{6}/{7}] {8:.2f}, '
                    'cls_loss: {9:.4f}'.format(epoch + 1, batch_id + 1, seg_correct, seg_total,
                                               seg_correct / seg_total * 100,
                                               seg_loss_sum / 100, cls_correct, cls_total,
                                               cls_correct / cls_total * 100,
                                               cls_loss_sum / 100))

                seg_total, seg_correct, seg_loss_sum, cls_total, cls_correct, cls_loss_sum = [0] * 6

        # 每一个epoch都保存一个checkpoint
        if epoch % 1 == 0:
            FENet_checkpoint = {'model': FENet.state_dict(),
                                'optimizer': optimizer.state_dict()}
            torch.save(FENet_checkpoint,
                       '{0}/{1}_{2}.pth'.format(FENet_dir, FENet_name, epoch + 1))

            SegNet_checkpoint = {'model': SegNet.state_dict(),
                                 'optimizer': optimizer.state_dict()}
            torch.save(SegNet_checkpoint,
                       '{0}/{1}_{2}.pth'.format(SegNet_dir, SegNet_name, epoch + 1))

            ClsNet_checkpoint = {'model': ClsNet.state_dict(),
                                 'optimizer': optimizer.state_dict()}
            torch.save(ClsNet_checkpoint,
                       '{0}/{1}_{2}.pth'.format(ClsNet_dir, ClsNet_name, epoch + 1))

            current_score = validation(FENet, SegNet, ClsNet, args)
            print('current_score: {0:.4f}'.format(current_score))
            log.info('current_score: {0:.4f}'.format(current_score))

            if current_score >= previous_score:
                torch.save(FENet.state_dict(), '{0}/{1}.pth'.format(FENet_dir, FENet_name))
                torch.save(SegNet.state_dict(), '{0}/{1}.pth'.format(SegNet_dir, SegNet_name))
                torch.save(ClsNet.state_dict(), '{0}/{1}.pth'.format(ClsNet_dir, ClsNet_name))
                previous_score = current_score


def validation(FENet, SegNet, ClsNet, args):
    val_data_loader = DataLoader(ValData(args), batch_size=args['val_bs'], shuffle=False, num_workers=8)

    pred_soft_ncls = []
    ncls = []
    seg_correct, seg_total, cls_correct, cls_total = [0] * 4

    for batch_id, val_data in enumerate(val_data_loader):

        image, mask, cls, name = val_data

        image = image.to(device)
        mask = mask.to(device)

        # cls[cls != 0] = 1  # 这边cls的分类
        cls = cls.to(device)

        with torch.no_grad():

            # feature extraction network
            FENet.eval()
            feat = FENet(image)

            # localization network
            SegNet.eval()
            pred_mask = SegNet(feat)[0]

            # classification network
            ClsNet.eval()
            pred_logit = ClsNet(feat)

        # 解决一个多尺度问题
        if pred_mask.shape != mask.shape:
            pred_mask = F.interpolate(pred_mask, size=(mask.size(1), mask.size(2)), mode='bilinear', align_corners=True)

        binary_mask1 = torch.zeros_like(pred_mask)
        binary_mask1[pred_mask > 0.5] = 1
        binary_mask1[pred_mask <= 0.5] = 0

        # 这个精确度可以改一下 输出多一点
        seg_correct += (binary_mask1 == mask).sum().item()
        seg_total += int(torch.ones_like(mask).sum().item())

        # ce 二分类改成多分类这边需要改正
        sm = nn.Softmax(dim=1)
        pred_logit = sm(pred_logit)
        pred_soft_ncls.extend(pred_logit[:, 1])
        ncls.extend(cls)

        _, binary_cls = torch.max(pred_logit, 1)

        # 一个一维的cls的数吧 0~1  -  能不能改成 0~3
        # 这里我感觉好像不用修改了吧,取分类中最大的那个值
        

        # print(binary_cls)  # tensor([0], device='cuda:0') 
        # print(binary_cls.shape) # torch.Size([1])
        cls_correct += (binary_cls == cls).sum().item()
        cls_total += int(torch.ones_like(cls).sum().item())

        if args['save_tag']:
            save_image(pred_mask, name, 'mask')
    return (seg_correct / seg_total + cls_correct / cls_total) / 2


if __name__ == '__main__':
    args = get_pscc_args()
    train(args)
