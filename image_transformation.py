import argparse, os

from models.networks import *
from configs.ds_paths import PATHS
from util.my_dataset import my_dataset
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--generator_weights', type=str, default=r'E:\Bias_Reduction_Summary\Model_Weights\CycleGAN\15\netG_A2B-D2toD1-15-1.0993.pth')
    parser.add_argument('--org_ds_name', nargs='+', default=['D2'])
    parser.add_argument('--path_key', type=str, default='Stage6_org')
    parser.add_argument('--txt_name', type=str, default='test.txt')

    # save
    parser.add_argument('--save_dir', type=str, default=r'')

    args = parser.parse_args()

    return args

def get_initGenerator():
    norm_layer = get_norm_layer(norm_type='batch')
    G_model = ResnetGenerator(3, 3, ngf=64, norm_layer=norm_layer, use_dropout=True, n_blocks=9)
    return G_model

def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    vutils.save_image(input_tensor, filename)


# def gen_biased_image(runOn=PATHS, gen_model, org_ds_name, txt_name, gen_image_save_dir):

def gen_image(opts):
    '''
        将 D1,D2,D3 转换为 D4
    :param gen_weights_path: 训练好的 image transfer 的 generator
    :param org_ds_name: 从哪一个数据集转换为D4
    :param txt_name:
    :param gen_image_save_dir: 生成图片保存的文件夹
    :return:
    '''

    # 加载 generator 模型
    generator = get_initGenerator()
    gen_weights_path = opts.generator_weights
    checkpoints = torch.load(gen_weights_path, map_location=torch.device(DEVICE), weights_only=False)
    generator.load_state_dict(checkpoints['model_state_dict'])

    generator.to(DEVICE)
    generator.eval()

    print(f'Using {gen_weights_path} to generate images from {opts.org_ds_name} - {opts.txt_name}')
    print(f'Save dir: {gen_weights_path}')

    # 加载 origin数据集的dataset
    org_dataset = my_dataset(ds_name_list=opts.org_ds_name, path_key=opts.path_key, txt_name=opts.txt_name)
    org_loader = DataLoader(org_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for data_dict in tqdm(org_loader):
            images = data_dict['image']
            cur_name = data_dict['img_name'][0]

            image_save_name = cur_name.split('.')[0] + '_D2TD1' + cur_name.split('.')[1]

            # 生成图片
            transformed_image = generator(images)

            save_path = os.path.join(opts.save_dir, image_save_name)
            save_image_tensor(transformed_image, save_path)


            break



        # for images, names, ped_labels in tqdm(org_loader):
        #
        #     images = images.to(DEVICE)
        #     cur_name = names[0]
        #
        #     cur_name = cur_name.replace('\\', os.sep)
        #     name_contents = cur_name.split(os.sep)
        #
        #     print(cur_name, name_contents)
        #
        #     # # 判断在该位置是否为ped和 nonped
        #     # temp = name_contents[-2]
        #     # img_time = ''
        #     # if temp == 'pedestrian' or temp == 'nonPedestrian':
        #     #     obj_cls = temp
        #     # else:
        #     #     obj_cls = name_contents[-3]
        #     #     img_time = temp
        #     #
        #     # individual_name = name_contents[-1]
        #
        #     out = generator(images)
        #     print(out.shape)
        #     save_name = 'a.jpg'
        #     # save_image_tensor(out, save_name)
        #     break
    #
    #         if txt_name == 'augmentation_train.txt':
    #             augTrain_dir = os.path.join(gen_image_save_dir, 'augmentation_train', obj_cls, img_time)
    #             if not os.path.exists(augTrain_dir):
    #                 os.makedirs(augTrain_dir)
    #             save_name = os.path.join(augTrain_dir, individual_name)
    #         else:
    #             # 如果存储路径文件夹没有创建，则创建
    #             save_dir_path = os.path.join(gen_image_save_dir, obj_cls, img_time)
    #             if not os.path.exists(save_dir_path):
    #                 os.makedirs(save_dir_path)
    #             save_name = os.path.join(save_dir_path, individual_name)
    #
    #         save_image_tensor(out, save_name)


if __name__ == '__main__':
    opts = get_args()
    gen_image(opts)

























