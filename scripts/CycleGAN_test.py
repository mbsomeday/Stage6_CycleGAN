import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import torch
from torch.utils.data import DataLoader

from options.test_options import TestOptions
from models.cycle_gan_model import CycleGANModel
from util.my_dataset import cycleGAN_Dataset


opt = TestOptions().parse()  # get test options
opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(opt)

# 加载模型
cycleGAN = CycleGANModel(opt)
cycleGAN.setup(opt)
cycleGAN.eval()

# 加载数据
test_dataset = cycleGAN_Dataset(dataset_name_list=opt.dataset_name_list, path_key=opt.data_key, txt_name='test.txt')
test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False)


for i, data in enumerate(test_loader):
    cycleGAN.set_input(data)
    ret = cycleGAN.test()
    print(ret.shape)


    break























