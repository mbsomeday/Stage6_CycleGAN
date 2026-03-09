import time
from torch.utils.data import DataLoader

from models.cycle_gan_model import CycleGANModel
from util.my_dataset import cycleGAN_Dataset
from util.visualizer import Visualizer



class CycleGAN_experiment():
    def __init__(self, opts):
        self.opts = opts


    def training_setup(self):
        pass


    def print_args(self):
        pass

    def train(self):
        # ------------------------ 创建模型 ------------------------
        self.cycleGAN = CycleGANModel(self.opts)
        self.cycleGAN.setup(self.opts)

        # ------------------------ 加载数据 ------------------------
        self.train_dataset = cycleGAN_Dataset(dataset_name_list=self.opts.dataset_name_list, path_key=self.opts.data_key, txt_name=self.opts.train_txt)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.opts.train_batch_size, shuffle=True)

        self.val_dataset = cycleGAN_Dataset(dataset_name_list=self.opts.dataset_name_list, path_key=self.opts.data_key, txt_name=self.opts.val_txt)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.opts.val_batch_size, shuffle=True)

        # ------------------------ 一些配置 ------------------------
        visualizer = Visualizer(self.opts)  # create a visualizer that display/save images and plots
        total_iters = 0  # the total number of training iterations

        # epoch_idx = 0
        for epoch in range(self.opts.epoch_count, self.opts.n_epochs + self.opts.n_epochs_decay + 1):
            # epoch_idx += 1
            # if epoch_idx == 2:
            #     break
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()  # timer for data loading per iteration
            epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

            for i, data in enumerate(self.train_loader):  # inner loop within one epoch

                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % self.opts.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                # total_iters += self.opts.batch_size
                # epoch_iter += self.opts.batch_size

                total_iters += self.opts.train_batch_size
                epoch_iter += self.opts.train_batch_size

                self.cycleGAN.set_input(data)  # unpack data from dataset and apply preprocessing
                self.cycleGAN.optimize_parameters()  # calculate loss functions, get gradients, update network weights

                if total_iters % self.opts.display_freq == 0:  # display images on visdom and save images to a HTML file
                    save_result = total_iters % self.opts.update_html_freq == 0
                    self.cycleGAN.compute_visuals()
                    visualizer.display_current_results(self.cycleGAN.get_current_visuals(), epoch, total_iters, save_result)

                if total_iters % self.opts.print_freq == 0:  # print training losses and save logging information to the disk
                    losses = self.cycleGAN.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / self.opts.train_batch_size     # 这里原来是self.opts.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    visualizer.plot_current_losses(total_iters, losses)

                if total_iters % self.opts.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                    print(f"saving the latest model (epoch {epoch}, total_iters {total_iters})")
                    save_suffix = f"iter_{total_iters}" if self.opts.save_by_iter else "latest"
                    self.cycleGAN.save_networks(save_suffix)

                iter_data_time = time.time()


            self.cycleGAN.update_learning_rate()  # update learning rates at the end of every epoch

            if epoch % self.opts.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
                print(f"saving the model at the end of epoch {epoch}, iters {total_iters}")
                self.cycleGAN.save_networks("latest")
                self.cycleGAN.save_networks(epoch)

            print(f"End of epoch {epoch} / {self.opts.n_epochs + self.opts.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time:.0f} sec")

            # break


    def test(self):
        pass























