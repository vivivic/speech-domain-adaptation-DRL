import os

from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
from dataloader import npyDataset2d
from model import Generator, Discriminator, ContentEncoder, StyleEncoder
from utils import Tanhize 
from hparams import get_hparams

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

lambda_cy = 10 
lambda_f = 1 
lambda_s = 1 
lambda_c = 1 


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_z_random(batch, dim1):

    z = torch.randn(batch, dim1).cuda()

    return z 


def recon_criterion(x,y):
    return torch.mean(torch.abs(x-y))

    
def train():

    hparams = get_hparams()
    model_path = os.path.join( hparams.model_path, hparams.task_name, hparams.spec_opt )
    if not os.path.exists(model_path):
        os.makedirs(model_path)


    # Load Dataset Loader


    normalizer_clean = Tanhize('clean')
    normalizer_noisy = Tanhize('noisy')


   
    print('Load dataset2d loader')
    dataset_A_2d = npyDataset2d(hparams.dataset_root,hparams.list_dir_train_A_2d, hparams.frame_len, normalizer = normalizer_noisy)
    dataset_B_2d = npyDataset2d(hparams.dataset_root,hparams.list_dir_train_B_2d, hparams.frame_len, normalizer = normalizer_clean)
    
    dataloader_A = DataLoader(dataset_A_2d, batch_size = hparams.batch_size,
                            shuffle = True,
                            drop_last = True,
                            )
    dataloader_B = DataLoader(dataset_B_2d, batch_size = hparams.batch_size,
                            shuffle = True,
                            drop_last = True,
                            )
    
    
    # Load Generator / Disciminator model
    generator_A = Generator()
    generator_B = Generator()

    discriminator_A = Discriminator()
    discriminator_B = Discriminator()

    ContEncoder_A = ContentEncoder()
    ContEncoder_B = ContentEncoder()

    StEncoder_A = StyleEncoder()
    StEncoder_B = StyleEncoder()


    generator_A.apply(weights_init)
    generator_B.apply(weights_init) 

    discriminator_A.apply(weights_init) 
    discriminator_B.apply(weights_init) 

    ContEncoder_A.apply(weights_init)
    ContEncoder_B.apply(weights_init)

    StEncoder_A.apply(weights_init)
    StEncoder_B.apply(weights_init)

    
    real_label = 1
    fake_label = 0
    real_tensor = Variable(torch.FloatTensor(hparams.batch_size))
    _ = real_tensor.data.fill_(real_label)

    fake_tensor = Variable(torch.FloatTensor(hparams.batch_size))
    _ = fake_tensor.data.fill_(fake_label)
   
    # Define Loss function
    d = nn.MSELoss()
    bce = nn.BCELoss()

    # Cuda Process
    if hparams.cuda == True:
        print('-- Activate with CUDA --')

        generator_A = nn.DataParallel(generator_A).cuda()
        generator_B = nn.DataParallel(generator_B).cuda()
        discriminator_A = nn.DataParallel(discriminator_A).cuda()
        discriminator_B = nn.DataParallel(discriminator_B).cuda()
        ContEncoder_A = nn.DataParallel(ContEncoder_A).cuda()
        ContEncoder_B = nn.DataParallel(ContEncoder_B).cuda()
        StEncoder_A = nn.DataParallel(StEncoder_A).cuda()
        StEncoder_B = nn.DataParallel(StEncoder_B).cuda()

        d.cuda()
        bce.cuda()
        real_tensor = real_tensor.cuda()
        fake_tensor = fake_tensor.cuda()

    else:
        print('-- Activate without CUDA --')

    


    gen_params = chain(
            generator_A.parameters(),
            generator_B.parameters(),
            ContEncoder_A.parameters(),
            ContEncoder_B.parameters(),
            StEncoder_A.parameters(),
            StEncoder_B.parameters(),
            )

    dis_params = chain(
            discriminator_A.parameters(), 
            discriminator_B.parameters(), 
            )

    optimizer_g = optim.Adam( gen_params, lr=hparams.learning_rate)
    optimizer_d = optim.Adam( dis_params, lr=hparams.learning_rate)

    iters = 0
    for e in range(hparams.epoch_size):

  
        # input Tensor 

        A_loader, B_loader = iter(dataloader_A), iter(dataloader_B)
    
        for i in range(len(A_loader)-1):
            
            batch_A = A_loader.next()
            batch_B = B_loader.next()

            A_indx = torch.LongTensor(list( range(hparams.batch_size)))
            B_indx = torch.LongTensor(list( range(hparams.batch_size)))


            A_ = torch.FloatTensor(batch_A)            
            B_ = torch.FloatTensor(batch_B)

            if hparams.cuda == True:
               

                x_A = Variable(A_.cuda())
                x_B = Variable(B_.cuda())


            else:
                x_A = Variable(A_)
                x_B = Variable(B_)

            real_tensor.data.resize_(hparams.batch_size).fill_(real_label)
            fake_tensor.data.resize_(hparams.batch_size).fill_(fake_label)

        
         

            ## Discrominator Update Steps

            discriminator_A.zero_grad()
            discriminator_B.zero_grad()

            # x_A, x_B, x_AB, x_BA
            # [#_batch, max_time_len, dim]

            A_c = ContEncoder_A(x_A).detach()
            B_c = ContEncoder_B(x_B).detach()

            # A,B :  N ~ (0,1)
            A_s = Variable(get_z_random(hparams.batch_size, 8))
            B_s = Variable(get_z_random(hparams.batch_size, 8))
    

            x_AB = generator_B(A_c, B_s).detach()
            x_BA = generator_A(B_c, A_s).detach()



            # We recommend LSGAN-loss for adversarial loss
 
            l_d_A_real = 0.5 * torch.mean( (discriminator_A(x_A) -  real_tensor) **2 ) 
            l_d_A_fake = 0.5 * torch.mean( (discriminator_A(x_BA) -  fake_tensor) **2 )

            l_d_B_real = 0.5 * torch.mean( (discriminator_B(x_B) - real_tensor)** 2) 
            l_d_B_fake = 0.5 * torch.mean( (discriminator_B(x_AB) -  fake_tensor) ** 2)


            l_d_A = l_d_A_real + l_d_A_fake
            l_d_B = l_d_B_real + l_d_B_fake


            l_d = l_d_A + l_d_B  
            
            l_d.backward()
            optimizer_d.step()
    

            ## Generator Update Steps

            generator_A.zero_grad()
            generator_B.zero_grad()
            ContEncoder_A.zero_grad()
            ContEncoder_B.zero_grad()
            StEncoder_A.zero_grad()
            StEncoder_B.zero_grad()

            A_c = ContEncoder_A(x_A)
            B_c = ContEncoder_B(x_B)

            A_s_prime = StEncoder_A(x_A)
            B_s_prime = StEncoder_B(x_B)
            

            # A,B : N ~ (0,1)
            A_s = Variable(get_z_random(hparams.batch_size, 8))
            B_s = Variable(get_z_random(hparams.batch_size, 8))
    

            x_BA = generator_A(B_c, A_s)
            x_AB = generator_B(A_c, B_s)

            x_A_recon = generator_A(A_c, A_s_prime)
            x_B_recon = generator_B(B_c, B_s_prime)


            B_c_recon = ContEncoder_A(x_BA)
            A_s_recon = StEncoder_A(x_BA)

            A_c_recon = ContEncoder_B(x_AB)
            B_s_recon = StEncoder_B(x_AB)


    
            x_ABA = generator_A(A_c_recon, A_s_prime)
            x_BAB = generator_B(B_c_recon, B_s_prime)

            l_cy_A = recon_criterion(x_ABA, x_A)
            l_cy_B = recon_criterion(x_BAB, x_B)

            l_f_A = recon_criterion(x_A_recon, x_A)
            l_f_B = recon_criterion(x_B_recon, x_B)

            l_c_A = recon_criterion(A_c_recon, A_c)
            l_c_B = recon_criterion(B_c_recon, B_c)

            l_s_A = recon_criterion(A_s_recon, A_s)
            l_s_B = recon_criterion(B_s_recon, B_s)


            # We recommend LSGAN-loss for adversarial loss
            
            l_gan_A = 0.5 * torch.mean( (discriminator_A(x_BA) -  real_tensor) **2)
            l_gan_B = 0.5 * torch.mean( (discriminator_B(x_AB) -  real_tensor ) **2)

            l_g = l_gan_A + l_gan_B + lambda_f *( l_f_A + l_f_B)  + lambda_s * (l_s_A + l_s_B) + lambda_c * (l_c_A + l_c_B) + lambda_cy * ( l_cy_A + l_cy_B)

            l_g.backward()
            optimizer_g.step()

       

            if iters % hparams.log_interval == 0:
                print ("---------------------")

                print ("Gen Loss :{} disc loss :{}".format(l_g/hparams.batch_size , l_d/hparams.batch_size))
                print ("epoch :" , e , " " , "total ", hparams.epoch_size)
                print ("iteration :", iters )

            if iters % hparams.model_save_interval == 0:
                torch.save( generator_A.state_dict(), os.path.join(model_path, 'model_gen_A_{}.pth'.format(iters)))
                torch.save( generator_B.state_dict(), os.path.join(model_path, 'model_gen_B_{}.pth'.format(iters)))
                torch.save( discriminator_A.state_dict(), os.path.join(model_path, 'model_dis_A_{}.pth'.format(iters)))
                torch.save( discriminator_B.state_dict(), os.path.join(model_path, 'model_dis_B_{}.pth'.format(iters)))

                torch.save( ContEncoder_A.state_dict(), os.path.join(model_path, 'model_ContEnc_A_{}.pth'.format(iters)))
                torch.save( ContEncoder_B.state_dict(), os.path.join(model_path, 'model_ContEnc_B_{}.pth'.format(iters)))
                torch.save( StEncoder_A.state_dict(), os.path.join(model_path, 'model_StEnc_A_{}.pth'.format(iters)))
                torch.save( StEncoder_B.state_dict(), os.path.join(model_path, 'model_StEnc_B_{}.pth'.format(iters)))

            iters += 1
            
if __name__ == '__main__':
    train()

