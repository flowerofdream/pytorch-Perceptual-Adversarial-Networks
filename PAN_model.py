import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
from base_model import BaseModel
from PIL import Image

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights. 网络权重初始化

    Parameters:
        net (network)   -- network to be initialized  网络
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal 网络初始化类型
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal. 缩放因子

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies. 只适用于正态分布
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02):
    """Initialize a network: 1. register CPU/GPU device; 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.

    Return an initialized network.
    """
    init_weights(net, init_type, init_gain=init_gain)
    return net

class G_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=2, padding=1):
        super(G_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(ch_out),
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class G_de_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=4, stride=2, padding=1):
        super(G_de_block, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(ch_out),
        )

    def forward(self, x):
        out = self.deconv(x)
        return out

class Genertor_Unet(nn.Module):
    def __init__(self):
        super(Genertor_Unet, self).__init__()
        self.LReLU = nn.LeakyReLU(0.2, inplace=True)
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = G_block(64, 128)
        self.conv3 = G_block(128, 256)
        self.conv4 = G_block(256, 512)
        self.conv5 = G_block(512, 512)
        self.conv6 = G_block(512, 512)

        self.deconv7 = G_de_block(512, 512)
        self.deconv8 = G_de_block(1024, 256)
        self.deconv9 = G_de_block(768, 128)
        self.deconv10 = G_de_block(384, 64)
        self.deconv11 = G_de_block(192, 64)

        self.deconv12 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # encode
        out1_ = self.conv1(x)
        out1 = self.LReLU(out1_)    # 64 128 128

        out2_ = self.conv2(out1)

        out2 = self.LReLU(out2_)    # 128 64 64
        out3_ = self.conv3(out2)

        out3 = self.LReLU(out3_)    # 256 32 32
        out4_ = self.conv4(out3)

        out4 = self.LReLU(out4_)    # 512 16 16
        out5_ = self.conv5(out4)

        out5 = self.LReLU(out5_)    # 512 8 8
        out6_ = self.conv6(out5)
        out6 = self.LReLU(out6_)    # 512 4 4

        # decode
        out7_ = self.deconv7(out6)
        out7 = torch.cat((out7_,out5_), dim=1)
        out7 = self.ReLU(out7)     # 1024 8 8

        out8_ = self.deconv8(out7)
        out8 = torch.cat((out8_, out4_), dim=1)
        out8 = self.ReLU(out8)     # 768 16 16

        out9_ = self.deconv9(out8)
        out9 = torch.cat((out9_, out3_), dim=1)
        out9 = self.ReLU(out9)     # 384 32 32

        out10_ = self.deconv10(out9)
        out10 = torch.cat((out10_, out2_), dim=1)
        out10 = self.ReLU(out10)       # 192 64 64

        out11_ = self.deconv11(out10)
        out11 = self.ReLU(out11_)      # 64 128 128
        out12 = self.deconv12(out11)

        return out12


class D_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, BN=True):
        super(D_block, self).__init__()
        if BN:
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, x):
        out = self.conv(x)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = D_block(3, 64, 3, 1, 1, False)
        self.conv2 = D_block(64, 128, 3, 2, 1, True)
        self.conv3 = D_block(128, 128, 3, 1, 1, True)
        self.conv4 = D_block(128, 256, 3, 2, 1, True)
        self.conv5 = D_block(256, 256, 3, 1, 1, True)
        self.conv6 = D_block(256, 512, 3, 2, 1, True)
        self.conv7 = D_block(512, 512, 3, 1, 1, True)
        self.conv8 = D_block(512, 512, 3, 2, 1, True)
        self.conv9 = D_block(512, 8, 3, 2, 1, False)
        self.fc = nn.Sequential(
            nn.Linear(8*8*8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out1 = self.conv1(x)        # 64 256 256
        out2 = self.conv2(out1)     # 128 128 128
        out3 = self.conv3(out2)     # 128 128 128
        out4 = self.conv4(out3)     # 256 64 64
        out5 = self.conv5(out4)     # 256 64 64
        out6 = self.conv6(out5)     # 512 32 32
        out7 = self.conv7(out6)     # 512 32 32
        out8 = self.conv8(out7)     # 512 16 16
        out9 = self.conv9(out8)     # 8 8 8
        out9 = out9.view(out9.size(0), -1)  # 512
        out10 = self.fc(out9)
        return out1, out4, out6, out8, out10

    def get_perception(self, x):
        out1 = self.conv1(x)        # 64 256 256
        out2 = self.conv2(out1)     # 128 128 128
        out3 = self.conv3(out2)     # 128 128 128
        out4 = self.conv4(out3)     # 256 64 64
        out5 = self.conv5(out4)     # 256 64 64
        out6 = self.conv6(out5)     # 512 32 32
        out7 = self.conv7(out6)     # 512 32 32
        out8 = self.conv8(out7)     # 512 16 16
        out9 = self.conv9(out8)     # 8 8 8
        return out1, out4, out6, out8

##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class PAN(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.netG = Genertor_Unet().to(self.device)
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        if self.isTrain:
            self.netG = init_net(self.netG)
            self.m = torch.tensor(self.opt['m']).to(self.device)
            self.netD = Discriminator().to(self.device)
            self.netD = init_net(self.netD)

            # define loss functions
            self.CrossEntropyLoss = nn.CrossEntropyLoss()
            self.criterionGAN = GANLoss(opt['gan_mode']).to(self.device)
            # self.criterionGAN = networks.GANLoss(opt['gan_mode']).to(self.device)
            # self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt['lr_G'], betas=(opt['beta1'], 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt['lr_D'], betas=(opt['beta1'], 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt['direction'] == 'AtoB'
        self.real_A = np.transpose(input['A' if AtoB else 'B'], (0, 3, 1, 2)).clone().detach().float().to(self.device)/255

        self.real_B = np.transpose(input['B' if AtoB else 'A'], (0, 3, 1, 2)).clone().detach().float().to(self.device)/255
        # self.real_A = torch.tensor(np.transpose(input['A' if AtoB else 'B'], (0, 3, 1, 2))).float().to(self.device)
        # self.real_B = torch.tensor(np.transpose(input['B' if AtoB else 'A'], (0, 3, 1, 2))).float().to(self.device)

    def get_Perception(self, real_feature, fake_feature, lam_p):
        return lam_p * torch.mean(real_feature-fake_feature)

    def forward(self):
        self.fake_B = self.netG(self.real_A)
        return self.fake_B

    def backward_D(self):
        dis1_f, dis2_f, dis3_f, dis4_f, disout_f= self.netD(self.real_B)
        dis1_ff, dis2_ff, dis3_ff, dis4_ff, disout_ff= self.netD(self.fake_B)
        p1 = self.get_Perception(dis1_f, dis1_ff, self.opt['lam_p1'])
        p2 = self.get_Perception(dis2_f, dis2_ff, self.opt['lam_p2'])
        p3 = self.get_Perception(dis3_f, dis3_ff, self.opt['lam_p3'])
        p4 = self.get_Perception(dis4_f, dis4_ff, self.opt['lam_p4'])
        self.l2_norm = p1 + p2 + p3 + p4

        self.loss_D_real = self.criterionGAN(disout_f, True)

        self.loss_D_fake = self.criterionGAN(disout_ff, False)

        self.D_Loss = self.opt['lam_gan_d'] * torch.mean(self.loss_D_fake + self.loss_D_real) + torch.max((self.m - self.l2_norm), torch.tensor(0).float().to(self.device))

        self.D_Loss.backward(retain_graph=True)
        return self.D_Loss

    def backward_G(self):
        dis1_f, dis2_f, dis3_f, dis4_f, disout_ff = self.netD(self.fake_B)
        self.G_loss= self.criterionGAN(disout_ff, True)

        # self.G_loss = self.opt['lam_gan_g'] * torch.mean(self.CrossEntropyLoss(self.disout_ff, 0)) + self.l2_norm
        if self.opt['use_Pix'] == 'L1':
            pixel_loss = self.opt['lam_pix'] * torch.mean(torch.abs(self.fake_B - self.real_B))
        elif self.opt['use_Pix'] == 'L2':
            pixel_loss = self.opt['lam_pix'] * torch.mean(torch.sqrt(self.fake_B - self.real_B))
        else:
            pixel_loss = torch.tensor(0).to(self.device)
        self.G_loss = self.G_loss + pixel_loss

        self.G_loss.backward()
        return self.G_loss

    def optimize_parameters(self):
        output = self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        D_loss = self.backward_D()
        self.optimizer_D.step()
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        G_loss = self.backward_G()
        self.optimizer_G.step()
        return (output, D_loss, G_loss)