import torch

from options.test_options import TestOptions
from models.cycle_gan_model import CycleGANModel
# from util.my_dataset import cycleGAN_Dataset


opt = TestOptions().parse()  # get test options
opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(opt)

cycleGAN = CycleGANModel(opt)
cycleGAN.setup(opt)
























