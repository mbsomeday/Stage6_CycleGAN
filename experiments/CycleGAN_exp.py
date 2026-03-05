from torch.utils.data import DataLoader

from models.cycle_gan_model import CycleGANModel
from util.my_dataset import cycleGAN_Dataset


class CycleGAN_experiment():
    def __init__(self, opts):
        self.opts = opts

        # # ------------------------ 创建模型 ------------------------
        # self.cycleGAN = CycleGANModel()

        # ------------------------ 加载数据 ------------------------
        train_dataset = cycleGAN_Dataset(dataset_name_list=opts.dataset_name_list, path_key=opts.data_key, txt_name=opts.train_txt)
        train_loader = DataLoader(train_dataset, batch_size=opts.train_batch_size, shuffle=True)

        val_dataset = cycleGAN_Dataset(dataset_name_list=opts.dataset_name_list, path_key=opts.data_key, txt_name=opts.val_txt)
        val_loader = DataLoader(val_dataset, batch_size=opts.val_batch_size, shuffle=True)



    def training_setup(self):
        pass


    def print_args(self):
        pass

    def train(self):
        pass

    def test(self):
        pass























