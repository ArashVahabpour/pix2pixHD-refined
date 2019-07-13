### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True)

        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake):
            return [l for (l, f) in zip((g_gan, g_gan_feat, g_vgg, d_real, d_fake), flags) if f]

        return loss_filter

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain

        ##### define networks
        # Generator network
        netG1_input_nc = opt.input_nc
        netG2_input_nc = opt.output_nc1

        self.netG1 = networks.define_G(netG1_input_nc, opt.output_nc1, opt.ngf, opt.netG,
                                       opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                       opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)
        self.netG2 = networks.define_G(netG2_input_nc, opt.output_nc2, opt.ngf, opt.netG,
                                       opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                       opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD1_input_nc = netG1_input_nc + opt.output_nc1
            netD2_input_nc = netG1_input_nc + opt.output_nc2
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

            self.criterionL1 = torch.nn.L1Loss()

            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake')

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:
                import sys
                if sys.version_info >= (3, 0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [value]
                        finetune_list.add(key.split('.')[0])
                print(
                    '------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))
            else:
                params = list(self.netG1.parameters()) + list(self.netG2.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.netD1.parameters()) + list(self.netD2.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, label_map, real_edge=None, real_image=None,
                     context_all=None, context_single=None, infer=False):
        input_label, input_context_all, input_context_single = \
            [Variable(x.data.cuda(), volatile=infer) for x in (label_map, context_all, context_single)]

        input_label = torch.cat([input_label, input_context_all, input_context_single], dim=1)

        # real and edge images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())
        if real_edge is not None:
            real_edge = Variable(real_edge.data.cuda())

        return input_label, real_edge, real_image

    def discriminate(self, input_label, test_image, netD_idx, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return (self.netD1 if netD_idx == 1 else self.netD2).forward(fake_query)
        else:
            return (self.netD1 if netD_idx == 1 else self.netD2).forward(input_concat)

    def forward(self, label, edge, image, context_all, context_single, infer=False):
        # Encode Inputs
        input_label, real_edge, real_image = \
            self.encode_input(label_map=label, real_edge=edge, real_image=image, context_all=context_all, context_single=context_single)

        fake_image_and_edge_1 = self.netG1.forward(input_label)
        fake_image_and_edge_2 = self.netG2.forward(fake_image_and_edge_1)

        # Fake Detection and Loss
        pred_fake_pool1 = self.discriminate(input_label, fake_image_and_edge_1, netD_idx=1, use_pool=True)
        pred_fake_pool2 = self.discriminate(input_label, fake_image_and_edge_2, netD_idx=2, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool1, False) + self.criterionGAN(pred_fake_pool2, False)

        # Real Detection and Loss
        pred_real1 = self.discriminate(input_label, torch.cat((real_image, real_edge), dim=1), netD_idx=1)
        pred_real2 = self.discriminate(input_label, torch.cat((real_image, real_edge), dim=1), netD_idx=2)
        loss_D_real = self.criterionGAN(pred_real1, True) + self.criterionGAN(pred_real2, True)

        # GAN loss (Fake Passability Loss)
        pred_fake1 = self.netD1.forward(torch.cat((input_label, fake_image_and_edge_1), dim=1))
        pred_fake2 = self.netD2.forward(torch.cat((input_label, fake_image_and_edge_2), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake1, True) + self.criterionGAN(pred_fake2, True)

        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake1[i]) - 1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                                       self.criterionFeat(pred_fake1[i][j],
                                                          pred_real1[i][j].detach()) * self.opt.lambda_feat

                for j in range(len(pred_fake2[i]) - 1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                                       self.criterionFeat(pred_fake2[i][j],
                                                          pred_real2[i][j].detach()) * self.opt.lambda_feat

        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = (self.criterionVGG(fake_image_and_edge_1[:, 0:1].repeat(1, 3, 1, 1),
                                            real_image.repeat(1, 3, 1, 1)) +
                          self.criterionVGG(fake_image_and_edge_2[:, 0:1].repeat(1, 3, 1, 1),
                                            real_image.repeat(1, 3, 1, 1)) #+
                          # self.criterionL1(fake_image_and_edge_1[:, 1:2], real_edge) +
                          # self.criterionL1(fake_image_and_edge_2[:, 1:2], real_edge)
                          ) * self.opt.lambda_feat

        # Only return the fake_B image if necessary to save BW
        stack_images = lambda a: torch.cat([x.squeeze() for x in a.split(split_size=1, dim=1)], dim=1).unsqueeze(dim=1)
        return [self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake),
                None if not infer else (stack_images(fake_image_and_edge_1), stack_images(fake_image_and_edge_2))]

    def inference(self, label, context_all=None, context_single=None):
        # Encode Inputs
        input_label, real_edge, _ = \
            self.encode_input(label_map=Variable(label), context_all=context_all, context_single=context_single, infer=True)

        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image_and_edge_1 = self.netG1.forward(input_label)
                fake_image_and_edge_2 = self.netG2.forward(fake_image_and_edge_1)
        else:
            fake_image_and_edge_1 = self.netG1.forward(input_label)
            fake_image_and_edge_2 = self.netG2.forward(fake_image_and_edge_1)
        stack_images = lambda a: torch.cat([x.squeeze(dim=1) for x in a.split(split_size=1, dim=1)], dim=1).unsqueeze(
            dim=1)
        return stack_images(fake_image_and_edge_1), stack_images(fake_image_and_edge_2)

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        if self.opt.data_type == 16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG1, 'G1', which_epoch, self.gpu_ids)
        self.save_network(self.netG2, 'G2', which_epoch, self.gpu_ids)
        self.save_network(self.netD1, 'D1', which_epoch, self.gpu_ids)
        self.save_network(self.netD2, 'D2', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
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
        label, context_all, context_single = inp
        return self.inference(label=label, context_all=context_all, context_single=context_single)

