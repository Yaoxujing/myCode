from yacs.config import CfgNode as CN

pscc_args = CN()
# pscc_args.path = '../dataset/'
pscc_args.path = 'datasetF/'
# debug的时候需要改一下路径
# pscc_args.path = 'tamper/dataset/'

pscc_args.num_epochs = 25
pscc_args.lr_strategy = [2e-4, 1e-4, 5e-5, 2.5e-5, 1.25e-5]
pscc_args.learning_rate = pscc_args.lr_strategy[0]
pscc_args.learning_step = 5

pscc_args.lr_decay_step = pscc_args.num_epochs // pscc_args.learning_step

# 这边也是截切过的
pscc_args.crop_size = [256, 256]
pscc_args.val_num = 200

# 训练的时候不用保存
# pscc_args.save_tag = False

# 测试的时候才保存 
pscc_args.save_tag = True

pscc_args.train_bs = 10
pscc_args.val_bs = 1
pscc_args.train_num = 10000  # 100000
# authentic, splice, copymove, removal
# pscc_args.train_ratio = [0.25, 0.25, 0.25, 0.25]
# pscc_args.train_ratio = [0.5,0.5]
pscc_args.train_ratio = [0.2, 0.4, 0.4] #更改


def get_pscc_args():
  return pscc_args.clone()

