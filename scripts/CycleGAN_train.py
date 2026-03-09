# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import argparse, random, time
import datetime

from options.train_options import TrainOptions
from experiments.CycleGAN_exp import CycleGAN_experiment
from util.util import init_ddp


# def get_args():
#     parser = argparse.ArgumentParser()
#
#     # train
#     parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
#
#     # dataset
#     parser.add_argument('--data_key', type=str, default='Stage6_org')
#     parser.add_argument('--train_txt', type=str, default='train.txt')
#     parser.add_argument('--val_txt', type=str, default='val.txt')
#     parser.add_argument('--train_batch_size', type=int, default=4)
#     parser.add_argument('--val_batch_size', type=int, default=4)
#     parser.add_argument('--dataset_name_list', nargs='+', default=['D1', 'D2'], help='style transfer from A to B')
#     parser.add_argument("--dataset_mode", type=str, default="unaligned", help="chooses how datasets are loaded. [unaligned | aligned | single | colorization]")
#     parser.add_argument("--direction", type=str, default="AtoB", help="AtoB or BtoA")
#     parser.add_argument('--cur_seed', type=int, default=6)
#     parser.add_argument('--isTrain', action='store_true', default=True)
#
#
#     # save
#     parser.add_argument('--save_base_dir', type=str, default=r'/kaggle/working/model')
#     parser.add_argument('--save_counter', type=str, default=None, help='to save timestamp in txt')
#
#     # reload
#     parser.add_argument('--reload', action='store_true')
#     parser.add_argument('--reloadweights', nargs='+', default=None)
#
#     args = parser.parse_args()
#
#     return args


# args = get_args()

args = TrainOptions().parse()
args.device = init_ddp()

# 开始时间
start_time = datetime.datetime.now()
print(f'Started at {str(start_time.strftime("%Y-%m-%d %H:%M:%S"))}')

print('Current Mode: 【Training】')
cyclegan_exp = CycleGAN_experiment(args)
cyclegan_exp.train()




































