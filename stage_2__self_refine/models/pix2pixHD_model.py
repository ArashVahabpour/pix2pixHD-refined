import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'

    @staticmethod
    def stack_images(image_and_edge):
        return torch.cat([x.squeeze(dim=1) for x in image_and_edge.split(split_size=1, dim=1)], dim=1).unsqueeze(dim=1)

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True)
        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake):
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,d_real,d_fake),flags) if f]
        return loss_filter
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        # self.use_features = opt.instance_feat or opt.label_feat
        # self.gen_features = self.use_features and not self.opt.load_features
        input_nc = opt.input_nc #opt.label_nc if opt.label_nc != 0 else opt.input_nc

        ##### define networks        
        # Generator network
        netG1_input_nc = input_nc
        netG1_output_nc = opt.output_nc1
        netG2_input_nc = netG1_output_nc
        netG2_output_nc = opt.output_nc2

        self.netG1 = networks.define_G(netG1_input_nc, netG1_output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)        

        self.netG2 = networks.define_G(netG2_input_nc, netG2_output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)
        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD1_input_nc = input_nc + 1
            netD2_input_nc = input_nc + 1

            self.netD1 = networks.define_D(netD1_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            self.netD2 = networks.define_D(netD2_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG1, 'G1', opt.which_epoch, pretrained_path)
            self.load_network(self.netG2, 'G2', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD1, 'D1', opt.which_epoch, pretrained_path)
                self.load_network(self.netD2, 'D2', opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
                
        
            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','D_real', 'D_fake')

            params = list(self.netG1.parameters()) + list(self.netG2.parameters())

            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            

            # optimizer D                        
            params = list(self.netD1.parameters()) + list(self.netD2.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, label_map, real_image=None, real_edge=None, infer=False):

        input_label = Variable(label_map.data.cuda(), volatile=infer)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())
        if real_edge is not None:
            real_edge = Variable(real_edge.data.cuda())

        return input_label, real_image, real_edge

    def discriminate(self, input_label, test_image, netD_idx, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return (self.netD1 if netD_idx==1 else self.netD2).forward(fake_query)
        else:
            return (self.netD1 if netD_idx==1 else self.netD2).forward(input_concat)

    def forward(self, label, image, edge, infer=False):
        # Encode Inputs
        input_label, real_image, real_edge = self.encode_input(label, image, edge)

        input_concat = input_label
        fake_image_and_edge1 = self.netG1.forward(input_concat)
        fake_image1, fake_edge1 = torch.split(fake_image_and_edge1, 1, dim=1)

        fake_image_and_edge2 = self.netG2.forward(fake_image_and_edge1)
        fake_image2, fake_edge2 = torch.split(fake_image_and_edge2, 1, dim=1)

        fake_images = torch.cat((fake_image1, fake_image2), 0)
        fake_edges = torch.cat((fake_edge1, fake_edge2), 0)
        # Fake Detection and Loss
        pred_fake_pool1 = self.discriminate(input_label.repeat(2, 1, 1, 1), fake_images, netD_idx=1, use_pool=True)
        pred_fake_pool2 = self.discriminate(input_label.repeat(2, 1, 1, 1), fake_edges, netD_idx=2, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool1, False) + self.criterionGAN(pred_fake_pool2, False)

        # Real Detection and Loss        
        pred_real1 = self.discriminate(input_label.repeat(2, 1, 1, 1), real_image.repeat(2, 1, 1, 1), netD_idx=1)  # real images, netD1
        pred_real2 = self.discriminate(input_label.repeat(2, 1, 1, 1), real_edge.repeat(2, 1, 1, 1), netD_idx=2)  # real edges, netD2
        loss_D_real = (self.criterionGAN(pred_real1, True) + self.criterionGAN(pred_real2, True))/2

        # GAN loss (Fake Passability Loss)        
        pred_fake1 = self.netD1.forward(torch.cat((input_label.repeat(2, 1, 1, 1), fake_images), dim=1))
        pred_fake2 = self.netD2.forward(torch.cat((input_label.repeat(2, 1, 1, 1), fake_edges), dim=1))
        loss_G_GAN = (self.criterionGAN(pred_fake1, True) + self.criterionGAN(pred_fake2, True))/2
        
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):

                for j in range(len(pred_fake1[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake1[i][j], pred_real1[i][j].detach()) * self.opt.lambda_feat / 2

                for j in range(len(pred_fake2[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake2[i][j], pred_real2[i][j].detach()) * self.opt.lambda_feat / 2
                   
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            fake_image1 = fake_image_and_edge1[:, 0:1]
            fake_image2 = fake_image_and_edge2[:, 0:1]
            loss_G_VGG = (self.criterionVGG(fake_image1.repeat(1,3,1,1), real_image.repeat(1,3,1,1)) +
                          self.criterionVGG(fake_image2.repeat(1,3,1,1), real_image.repeat(1,3,1,1))) * self.opt.lambda_feat / 2
        
        # Only return the fake_B image if necessary to save BW
        loss_filter = self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake)
        generated1, generated2 = Pix2PixHDModel.stack_images(fake_image_and_edge1), Pix2PixHDModel.stack_images(fake_image_and_edge2)
        return [loss_filter, (None, None) if not infer else (generated1, generated2)]

    def inference(self, label):
        # Encode Inputs        
        input_label, real_image, real_edge = self.encode_input(Variable(label), infer=True)

        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image_and_edge1 = self.netG1.forward(input_label)
                fake_image_and_edge2 = self.netG2.forward(fake_image_and_edge1)
        else:
            fake_image_and_edge1 = self.netG1.forward(input_label)
            fake_image_and_edge2 = self.netG2.forward(fake_image_and_edge1)

        generated1, generated2 = Pix2PixHDModel.stack_images(fake_image_and_edge1), Pix2PixHDModel.stack_images(fake_image_and_edge2)
        return generated1, generated2

    def save(self, which_epoch):
        for net, net_name in zip((self.netG1, self.netG2, self.netD1, self.netD2), ('G1', 'G2', 'D1', 'D2')):
            self.save_network(net, net_name, which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG1.parameters()) + list(self.netG2.parameters())  #<<<TODO check if they don't shadow each other
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label = inp
        return self.inference(label)


