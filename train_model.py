import torch
import numpy as np
import os
import pickle
import time
from PAN_model import PAN
from torch.utils.data import DataLoader
from data_loader import *

def main():
    opt = {
        'train_data': './datasets/facades/train/',
        'display_data': './datasets/facades/val/',
        'checkpoints_dir': './checkpoints',  # model
        'name':'facades',
        'gan_mode': 'vanilla',    # the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.
        'start': 0,
        'epoch': 400,
        'use_gpu': True,
        'gpu_id': 0,    # 1 gpu 0 cpu
        'direction': 'AtoB',
        'isTrain': True,
        'shuffle_': True,
        'use_h5py': 0,
        'batchSize': 4,
        'loadSize': 286,
        'fineSize': 256,
        'flip': True,
        'ngf': 64,
        'ndf': 64,
        'input_nc': 3,
        'output_nc': 3,
        'training_method': 'adam',
        'lr_G': 0.0002,
        'lr_D': 0.0002,
        'beta1': 0.5,
        'preprocess': 'regular',
        'save_freq': 100,
        'continue_train': False,
        'use_Pix': 'L1',
        'which_netG': 'unet_nodrop',
        'which_netD': 'basic',
        'lam_pix': 25.,
        'lam_p1': 5.,
        'lam_p2': 1.5,
        'lam_p3': 1.5,
        'lam_p4': 1.,
        'lam_gan_d': 1.,
        'lam_gan_g': 1.,
        'm': 3.0,
    }
    PANModel = PAN(opt)
    PANModel.print_networks(True)
    dataset = FacadesDataset(root=opt['train_data'])
    loader = DataLoader(dataset=dataset, batch_size=opt['batchSize'], shuffle=opt['shuffle_'])
    for epoch in range(opt['start'], opt['start'] + opt['epoch']):
        cnt = 0
        batch_cnt = 0

        for imag_A, imag_B in loader:
            # Image.fromarray(np.uint8(imag_A[0])).save('test.jpg')
            st = time.time()
            batch_cnt += 1
            if opt['direction'] == 'AtoB':
                gt_img = imag_B
            else:
                gt_img = imag_A
            input = {"A": imag_A, "B": imag_B}
            PANModel.set_input(input)
            output, Dloss, Gloss = PANModel.optimize_parameters()
            if epoch % opt['save_freq'] == 0:
                output = np.transpose(np.array(output.cpu().data), (0, 2, 3, 1))*255
                gt_img = np.array(gt_img.cpu().data)
                if not os.path.isdir('./result/' + '%04d' % epoch):
                    os.makedirs('./result/' + '%04d' % epoch)
                temp = np.concatenate((gt_img[:, :, :, :], output[:, :, :, :]), axis=1)
                for i in range(temp.shape[0]):
                    Image.fromarray(np.uint8(temp[i])).save('./result/' + '%04d/%05d.jpg' % (epoch, cnt))
                    cnt += 1
            print("%d %d GLoss = %.3f DLoss = %.3f Time=%.3f" % (epoch + 1, batch_cnt, Gloss.item(),Dloss.item(), time.time() - st))


        if epoch % opt['save_freq'] == 0:
            PANModel.save_networks(str(epoch))
if __name__ == '__main__':
    main()