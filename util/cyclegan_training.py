import torch
from time import time
from tqdm import tqdm

from util.image_pool import ImagePool


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'




def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def train_one_epoch(netG_A2B, netG_B2A, netD_A, netD_B,
                    fake_A_pool, fake_B_pool,
                    criterionGAN, criterionCycle, criterionIdt,
                    optimizer_G, optimizer_D_A, optimizer_D_B,
                    epoch, train_dataset, train_loader):
    print('-' * 30, f'Epoch {epoch + 1}', '-' * 30)

    netG_A2B.train()
    netG_B2A.train()
    netD_A.train()
    netD_B.train()

    # Tensor = torch.cuda.FloatTensor if DEVICE == 'cuda' else torch.Tensor

    epoch_start_time = time()  # timer for entire epoch

    for batch_id, data in enumerate(tqdm(train_loader)):
        real_A = data[0].to(DEVICE)
        real_B = data[1].to(DEVICE)

        num_sample = real_A.shape[0]


        # target_real = Variable(Tensor(num_sample, 1).fill_(1.0), requires_grad=False)
        # target_fake = Variable(Tensor(num_sample, 1).fill_(0.0), requires_grad=False)

        # target_real = torch.tensor(torch.zeros(num_sample, 1), device='cuda')
        # target_fake = torch.tensor(torch.ones(num_sample, 1), device='cuda')

        target_real = torch.zeros(num_sample, 1).clone().detach().requires_grad_(True).to(DEVICE)
        target_fake = torch.ones(num_sample, 1).clone().detach().requires_grad_(True).to(DEVICE)

        ###### Generators A2B and B2A ######

        # 先 train G_A and G_B, 此时将 discriminator 的 gradient 设置为 0
        set_requires_grad([netD_A, netD_B], False)
        optimizer_G.zero_grad()

        # Identity loss
        same_B = netG_A2B(real_B)
        loss_identity_B = criterionIdt(same_B, real_B) * 5.0
        same_A = netG_B2A(real_A)
        loss_identity_A = criterionIdt(same_A, real_A) * 5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterionGAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterionGAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterionCycle(recovered_A, real_A) * 10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterionCycle(recovered_B, real_B) * 10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()

        optimizer_G.step()
        ###################################

        ###### Discriminator A ######

        set_requires_grad([netD_A, netD_B], True)
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterionGAN(pred_real, target_real)

        # Fake loss
        #         fake_A = fake_A_buffer.push_and_pop(fake_A)
        fake_A = fake_A_pool.query(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterionGAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterionGAN(pred_real, target_real)

        # Fake loss
        #         fake_B = fake_B_buffer.push_and_pop(fake_B)
        fake_B = fake_B_pool.query(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterionGAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################
        # break

    print(f'Time for training epoch {epoch + 1}: {int(time() - epoch_start_time)} s. loss_G:{loss_G:.6f}, loss_D_A:{loss_D_A:.6f}, loss_D_B:{loss_D_B:.6f}.')

    return netG_A2B, netG_B2A, netD_A, netD_B, optimizer_G, optimizer_D_A, optimizer_D_B


def val_cycleGAN(netG_A2B, netG_B2A, netD_A, netD_B, val_dataset, val_loader, criterionGAN, criterionCycle,
                 criterionIdt):
    netG_A2B.eval()
    netG_B2A.eval()
    netD_A.eval()
    netD_B.eval()

    # Buffer
    fake_A_pool = ImagePool(pool_size=50)
    fake_B_pool = ImagePool(pool_size=50)

    Tensor = torch.cuda.FloatTensor if DEVICE == 'cuda' else torch.Tensor

    with torch.no_grad():
        for batch_id, data in enumerate(tqdm(val_loader)):
            real_A = data[0].to(DEVICE)
            real_B = data[1].to(DEVICE)

            num_sample = real_A.shape[0]

            # target_real = Variable(Tensor(num_sample, 1).fill_(1.0), requires_grad=False)
            # target_fake = Variable(Tensor(num_sample, 1).fill_(0.0), requires_grad=False)

            # target_real = torch.tensor(torch.zeros(num_sample, 1), device='cuda')
            # target_real = torch.tensor(torch.ones(num_sample, 1), device='cuda')

            target_real = torch.zeros(num_sample, 1).clone().detach().requires_grad_(True).to(DEVICE)
            target_fake = torch.ones(num_sample, 1).clone().detach().requires_grad_(True).to(DEVICE)

            ###### Generators A2B and B2A ######

            # Identity loss
            same_B = netG_A2B(real_B)
            loss_identity_B = criterionIdt(same_B, real_B) * 5.0
            same_A = netG_B2A(real_A)
            loss_identity_A = criterionIdt(same_A, real_A) * 5.0

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterionGAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterionGAN(pred_fake, target_real)

            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterionCycle(recovered_A, real_A) * 10.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterionCycle(recovered_B, real_B) * 10.0

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            ###################################

            ###### Discriminator A ######
            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterionGAN(pred_real, target_real)

            # Fake loss
            # fake_A = fake_A_buffer.push_and_pop(fake_A)
            fake_A = fake_A_pool.query(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterionGAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            ###################################

            ###### Discriminator B ######
            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterionGAN(pred_real, target_real)

            # Fake loss
            # fake_B = fake_B_buffer.push_and_pop(fake_B)
            fake_B = fake_B_pool.query(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterionGAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            ###################################
            # break


    return loss_G, loss_D_A, loss_D_B






