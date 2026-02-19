import argparse, itertools
import numpy as np


from models.networks import *
from torch.utils.data import DataLoader
from util.my_dataset import cycleGAN_Dataset
from util.image_pool import ImagePool
from util.strategy import EarlyStopping_CycleGAN
from util.cyclegan_training import train_one_epoch, val_cycleGAN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_args():
    parser = argparse.ArgumentParser()

    # train
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')

    # dataset
    parser.add_argument('--data_key', type=str, default='Stage6_org')
    parser.add_argument('--train_txt', type=str, default='train.txt')
    parser.add_argument('--val_txt', type=str, default='val.txt')
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--dataset_name_list', nargs='+', default=['D1', 'D2'], help='style transfer from A to B')
    parser.add_argument("--dataset_mode", type=str, default="unaligned", help="chooses how datasets are loaded. [unaligned | aligned | single | colorization]")
    parser.add_argument("--direction", type=str, default="AtoB", help="AtoB or BtoA")

    # save
    parser.add_argument('--save_base_dir', type=str, default=r'/kaggle/working/model')
    parser.add_argument('--save_counter', type=str, default=r'/kaggle/working', help='to save timestamp in txt')

    # reload
    parser.add_argument('--reload', action='store_true')
    parser.add_argument('--reloadweights', nargs='+', default=None)

    args = parser.parse_args()

    return args


def get_init_CycleGAN():
    norm_layer = get_norm_layer(norm_type='batch')
    netG_A2B = ResnetGenerator(3, 3, ngf=64, norm_layer=norm_layer, use_dropout=True, n_blocks=9)
    netG_B2A = ResnetGenerator(3, 3, ngf=64, norm_layer=norm_layer, use_dropout=True, n_blocks=9)
    netD_A = Discriminator(3)
    netD_B = Discriminator(3)

    return netG_A2B, netG_B2A, netD_A, netD_B



def init_Hyperparameter(netG_A2B, netG_B2A, netD_A, netD_B):
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

    start_epoch = 0
    loss_G = np.inf
    loss_D_A = np.inf
    loss_D_B = np.inf

    return [optimizer_G, optimizer_D_A, optimizer_D_B], [loss_G, loss_D_A, loss_D_B], start_epoch

# ------------------------ 【函数】加载训练了一半的模型 ------------------------
def reload_weights(model, weights_path):
    checkpoints = torch.load(weights_path, map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoints['model_state_dict'])
    model.to(DEVICE)
    return model

def load_optimizer_loss(loaded_optimizer, weights_path, key_opt, key_loss=None, get_start_epoch=False):
    checkpoints = torch.load(weights_path, map_location=torch.device(DEVICE))
    loaded_optimizer.load_state_dict(checkpoints[key_opt])
    for state in loaded_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()


    if get_start_epoch:
        start_epoch = checkpoints['epoch']
        loaded_loss = checkpoints[key_loss]
        loss_D_A = checkpoints['loss_D_A']
        loss_D_B = checkpoints['loss_D_B']
        return loaded_optimizer, loaded_loss, loss_D_A, loss_D_B, start_epoch
    else:
        return loaded_optimizer

def reload(preTrainedWeights):
    # 先确定各个模型的权重
    netG_A2B_weights = preTrainedWeights[0]
    netG_B2A_weights = preTrainedWeights[1]
    netD_A_weights = preTrainedWeights[2]
    netD_B_weights = preTrainedWeights[3]

    # load model
    norm_layer = get_norm_layer(norm_type='batch')
    netG_A2B = ResnetGenerator(3, 3, ngf=64, norm_layer=norm_layer, use_dropout=True, n_blocks=9)
    netG_B2A = ResnetGenerator(3, 3, ngf=64, norm_layer=norm_layer, use_dropout=True, n_blocks=9)

    netG_A2B = reload_weights(netG_A2B, netG_A2B_weights)
    netG_B2A = reload_weights(netG_B2A, netG_B2A_weights)

    netD_A = Discriminator(3)
    netD_B = Discriminator(3)
    netD_A = reload_weights(netD_A, netD_A_weights)
    netD_B = reload_weights(netD_B, netD_B_weights)

    # load optimizer
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

    optimizer_G, loss_G, loss_D_A, loss_D_B, start_epoch = load_optimizer_loss(optimizer_G, netG_A2B_weights,
                                                                              key_opt='optimizer_G', key_loss='loss_G',
                                                                              get_start_epoch=True)
    optimizer_D_A = load_optimizer_loss(optimizer_D_A, netD_A_weights, key_opt='optimizer_D_A', key_loss=None, get_start_epoch=False)
    optimizer_D_B = load_optimizer_loss(optimizer_D_B, netD_B_weights, key_opt='optimizer_D_B', key_loss=None, get_start_epoch=False)

    optimizers = [optimizer_G, optimizer_D_A, optimizer_D_B]
    losses = [loss_G, loss_D_A, loss_D_B]

    model = [netG_A2B, netG_B2A, netD_A, netD_B]

    return start_epoch, model, optimizers, losses



def train(opts):

    # 中断后重新训练的情况
    if opts.reload:
        start_epoch, models, optimizers, losses = reload(opts.reloadweights)
        netG_A2B, netG_B2A, netD_A, netD_B = models
    else:       # 从头训练
        models = get_init_CycleGAN()
        netG_A2B, netG_B2A, netD_A, netD_B = models
        optimizers, losses, start_epoch = init_Hyperparameter(netG_A2B, netG_B2A, netD_A, netD_B)

    netG_A2B.to(DEVICE)
    netG_B2A.to(DEVICE)
    netD_A.to(DEVICE)
    netD_B.to(DEVICE)

    optimizer_G, optimizer_D_A, optimizer_D_B = optimizers
    loss_G, loss_D_A, loss_D_B = losses

    # ------------------------ 加载数据 ------------------------
    train_dataset = cycleGAN_Dataset(dataset_name_list=opts.dataset_name_list, path_key=opts.data_key, txt_name=opts.train_txt)
    train_loader = DataLoader(train_dataset, batch_size=opts.train_batch_size, shuffle=True)

    val_dataset = cycleGAN_Dataset(dataset_name_list=opts.dataset_name_list, path_key=opts.data_key, txt_name=opts.val_txt)
    val_loader = DataLoader(val_dataset, batch_size=opts.val_batch_size, shuffle=True)

    # ------------------------ 加载其他训练相关 ------------------------
    fake_A_pool = ImagePool(pool_size=50)
    fake_B_pool = ImagePool(pool_size=50)

    criterionGAN = torch.nn.MSELoss()
    criterionCycle = torch.nn.L1Loss()
    criterionIdt = torch.nn.L1Loss()

    early_stopping = EarlyStopping_CycleGAN(from_ds_name=opts.dataset_name_list[0],
                                            to_ds_name=opts.dataset_name_list[1], save_base_dir=opts.save_base_dir,
                                            loss_G=loss_G, loss_D_A=loss_D_A, loss_D_B=loss_D_B, save_counter=opts.save_counter
                                            )

    for epoch in range(start_epoch, opts.epochs):
        netG_A2B.train()
        netG_B2A.train()
        netD_A.train()
        netD_B.train()

        netG_A2B, netG_B2A, netD_A, netD_B, optimizer_G, optimizer_D_A, optimizer_D_B = train_one_epoch(netG_A2B, netG_B2A,
                                                                                                        netD_A, netD_B,
                                                                                                        fake_A_pool,
                                                                                                        fake_B_pool,
                                                                                                        criterionGAN,
                                                                                                        criterionCycle,
                                                                                                        criterionIdt,
                                                                                                        optimizer_G,
                                                                                                        optimizer_D_A,
                                                                                                        optimizer_D_B,
                                                                                                        epoch,
                                                                                                        train_dataset,
                                                                                                        train_loader)
        torch.cuda.empty_cache()
        loss_G, loss_D_A, loss_D_B = val_cycleGAN(netG_A2B, netG_B2A, netD_A, netD_B,
                                                  val_dataset, val_loader,
                                                  criterionGAN, criterionCycle, criterionIdt,
                                                  )
        torch.cuda.empty_cache()

        print(f'Validation results: loss_G:{loss_G:.6f}, loss_D_A:{loss_D_A:.6f}, loss_D_B:{loss_D_B:.6f}.')
        # Early Stopping 策略
        early_stopping(loss_G, loss_D_A, loss_D_B, netG_A2B, netG_B2A, netD_A, netD_B,
                       optimizer_G, optimizer_D_A, optimizer_D_B, epoch=epoch + 1)

        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出迭代，结束训练


        # 每次重新加载dataloader
        train_dataset = cycleGAN_Dataset(dataset_name_list=opts.dataset_name_list, path_key=opts.data_key, txt_name=opts.train_txt)
        train_loader = DataLoader(train_dataset, batch_size=opts.train_batch_size, shuffle=True)





if __name__ == '__main__':
    args = get_args()
    print(args)

    train(args)













































