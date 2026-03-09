import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import torch
from torch.utils.data import DataLoader

from options.test_options import TestOptions
from models.cycle_gan_model import CycleGANModel
from util.my_dataset import cycleGAN_Dataset
from util.my_utils import save_image_tensor


opt = TestOptions().parse()  # get test options
opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(opt)

# 加载模型
cycleGAN = CycleGANModel(opt)
cycleGAN.setup(opt)
cycleGAN.eval()

# 加载数据
test_dataset = cycleGAN_Dataset(dataset_name_list=opt.dataset_name_list, path_key=opt.data_key, txt_name=opt.test_txt_name)
test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False)


for i, data in enumerate(test_loader):
    cycleGAN.set_input(data)
    ret = cycleGAN.test()

    gen_images = cycleGAN.fake_B.shape

    for idx, img_path in enumerate(data['A_paths']):
        path_content = img_path.split(os.sep)
        org_img_name = path_content[-1].split('.')
        new_img_name = org_img_name[0] + '_D2toD1.' + org_img_name[-1]
        img_save_path = os.path.join(opt.gen_img_save_dir, path_content[-2], new_img_name)

        cur_img = gen_images[idx].unsqueeze(0)

        save_image_tensor(cur_img, img_save_path)
        break




    break























