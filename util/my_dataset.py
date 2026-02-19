import os, random
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

from configs.ds_paths import PATHS


class cycleGAN_Dataset(Dataset):
    '''
        返回：imgA, imgB, pathA, pathB
        把图片从A转换到B
        get_num: 若没有指定，则取两个数据集数据量的最小值
    '''
    def __init__(self, dataset_name_list, path_key, txt_name):

        self.path_key = path_key
        self.dataset_dir_list = [PATHS[self.path_key][ds_name] for ds_name in dataset_name_list]

        self.dsA_name = dataset_name_list[0]
        self.dsB_name = dataset_name_list[1]

        self.dsA_dir = self.dataset_dir_list[0]
        self.dsB_dir = self.dataset_dir_list[1]
        self.txt_name = txt_name

        self.image_transformer = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.imgA, self.imgB = self.init_ImageandLabel()

        # 输出一些必要信息
        print(f'[{txt_name[:-4]}] CycleGAN Dataset from {self.dsA_name} to {self.dsB_name}, get {len(self.imgA)} images from {self.dsA_name}, {len(self.imgB)} images from {self.dsB_name}.')

    def init_ImageandLabel(self):
        dsA_txt_path = os.path.join(self.dsA_dir, 'dataset_txt', self.txt_name)
        with open(dsA_txt_path, 'r') as f:
            dsA_data = f.readlines()

        dsB_txt_path = os.path.join(self.dsB_dir, 'dataset_txt', self.txt_name)
        with open(dsB_txt_path, 'r') as f:
            dsB_data = f.readlines()

        dsA_num = len(dsA_data)
        dsB_num = len(dsB_data)

        assert dsA_num == dsB_num, "两数据集样本量不一致！"

        self.get_num = dsA_num

        imgA = self.get_image(dsA_data, self.dsA_dir)
        imgB = self.get_image(dsB_data, self.dsB_dir)

        return imgA, imgB


    def get_image(self, data, base_dir):
        '''
        :param data: 读取的txt文件
        :param base_dir:
        :return: 根据读取的txt返回get_num数量的image_path
        '''

        random.shuffle(data)
        images = []

        for i in range(self.get_num):
            line = data[i]
            line = line.replace('\\', os.sep)
            line = line.strip()
            words = line.split()

            image_path = os.path.join(base_dir, words[0])
            images.append(image_path)

        return images

    def __len__(self):
        return self.get_num

    def __getitem__(self, item):
        imgA_path = self.imgA[item]
        imgB_path = self.imgB[item]

        imgA = Image.open(imgA_path)
        imgA = self.image_transformer(imgA)
        imgB = Image.open(imgB_path)
        imgB = self.image_transformer(imgB)

        return imgA, imgB


class my_dataset(Dataset):
    def __init__(self, ds_name_list, path_key, txt_name):
        '''
        :param ds_name_list:
        :param path_key: org_dataset
        :param ds_labels: 用于打乱数据集标签
        '''
        self.ds_name_list = ds_name_list
        self.ds_label_list = []
        self.path_key = path_key

        # 用于测试打乱dataset name和label的对应实验
        for ds_name in ds_name_list:
            self.ds_label_list.append(int(ds_name[1]) - 1)

        if len(self.ds_name_list) != len(self.ds_label_list):
            raise ValueError(f'Dataset loader name and label match error. Names:{self.ds_name_list}, labels:{self.ds_label_list}')

        print(f'Mapping dataset names and labels: {ds_name_list} - {self.ds_label_list}')

        self.txt_name = txt_name
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.images, self.ped_labels, self.ds_labels = self.init_ImagesLabels()
        print(f'Get dataset: {ds_name_list}, txt_name: {txt_name}, total {len(self.images)} images')

    def init_ImagesLabels(self):
        images, ped_labels, ds_labels = [], [], []

        for ds_idx, ds_name in enumerate(self.ds_name_list):
            ds_label = self.ds_label_list[ds_idx]
            ds_dir = PATHS[self.path_key][ds_name]
            txt_path = os.path.join(ds_dir, 'dataset_txt', self.txt_name)

            print(f'Lodaing {txt_path}')

            with open(txt_path, 'r') as f:
                data = f.readlines()

            for data_idx, line in enumerate(data):
                line = line.replace('\\', os.sep)
                line = line.strip()
                contents = line.split()

                image_path = os.path.join(ds_dir, contents[0])
                images.append(image_path)
                ped_labels.append(contents[-1])
                ds_labels.append(ds_label)

        return images, ped_labels, ds_labels

    def get_ped_cls_num(self):
        '''
            获取行人和非行人类别的数量
        '''
        nonPed_num, ped_num = 0, 0
        for item in self.ped_labels:
            if item == '0':
                nonPed_num += 1
            elif item == '1':
                ped_num += 1
        return nonPed_num, ped_num

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        ped_label = self.ped_labels[idx]
        ds_label = self.ds_labels[idx]

        image = Image.open(image_path).convert('RGB')
        image = self.img_transforms(image)
        ds_label = np.array(ds_label).astype(np.int64)

        image_name = image_path.split(os.sep)[-1]

        return image, image_name, ped_label


# if __name__ == '__main__':
#     ds = cycleGAN_Dataset(dataset_name_list=['D1', 'D2'], path_key='Stage6_org', txt_name='test.txt')






















