import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys
import torch.nn.functional as F
import scipy.stats as st
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import clip
import torchvision.models as models
# ~~~~~~
from models.model import Hed

def no_sigmoid_cross_entropy(sig_logits, label):
    # print(sig_logits)
    count_neg = torch.sum(1.-label)
    count_pos = torch.sum(label)

    beta = count_neg / (count_pos+count_neg)
    pos_weight = beta / (1-beta)

    cost = pos_weight * label * (-1) * torch.log(sig_logits) + (1-label)* (-1) * torch.log(1-sig_logits)
    cost = torch.mean(cost * (1-beta))

    return cost
# ~~~~~~

class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def gauss_kernel(self, kernlen=21, nsig=3, channels=1):
        interval = (2 * nsig + 1.) / (kernlen)
        x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        out_filter = np.array(kernel, dtype=np.float32)
        out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
        out_filter = np.repeat(out_filter, channels, axis=2)
        return out_filter


    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.device = torch.device("cuda:" + str(self.gpu_ids[0]) if torch.cuda.is_available() else "cpu")

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

            self.netD_ink = networks.define_D(opt.output_nc, opt.ndf,
                                              opt.which_model_netD,
                                              opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

            g_kernel = self.gauss_kernel(21, 3, 1).transpose((3, 2, 1, 0))
            self.gauss_conv = nn.Conv2d(1, 1, kernel_size=21, stride=1, padding=1, bias=False)
            self.gauss_conv.weight.data.copy_(torch.from_numpy(g_kernel))
            self.gauss_conv.weight.requires_grad = False
            self.gauss_conv.cuda()

            # ~~~~~~
            self.HED = Hed()
            self.HED = self.HED.cuda() 
            save_path = './35.pth'
            self.HED.load_state_dict(torch.load(save_path))
            for param in self.HED.parameters():
                param.requires_grad = False
            # ~~~~~~
            # Geom
            self.netGeom = networks.GlobalGenerator2(input_nc=768, output_nc=3, n_downsampling=1, n_UPsampling=3)
            self.netGeom.load_state_dict(torch.load(opt.feats2Geom_path))
            self.netGeom = self.netGeom.to(self.device)
            print("记载预训练好的geom网络自 %s" % opt.feats2Geom_path)
            
        
         # CLIP
        ### load pretrained inception
        self.net_recog = networks.InceptionV3(num_classes=7, isTrain = True, use_aux=True, pretrain=True, freeze=True)
        self.net_recog.cuda() 
        self.net_recog.eval()
        self.clip_model, self.preprocess = clip.load("ViT-B/32", jit=False)
        clip.model.convert_weights(self.clip_model)


        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)
                self.load_network(self.netD_ink, 'D_ink', which_epoch)
                self.load_network(self.netGeom, 'Geom', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.ink_fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_ink = torch.optim.Adam(self.netD_ink.parameters(),lr=opt.lr,betas=(opt.beta1, 0.999))
            self.optimizer_Geom = torch.optim.Adam(self.netGeom.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            self.optimizers.append(self.optimizer_D_ink)
            self.optimizers.append(self.optimizer_Geom)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
        networks.print_network(self.net_recog)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
            networks.print_network(self.netD_ink)
            networks.print_network(self.netGeom)
        print('-----------------------------------------------')

        self.set_weights()

        # 创建文件
        if not os.path.exists(self.save_dir):
            print("创建文件夹 [%s] " % (self.save_dir))
            os.mkdir(self.save_dir)

    def set_input(self, data):
        self.input_A = data['A']
        self.input_B = data['B']
        self.A_depth = data['C']

        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_depth = Variable(self.A_depth)

        self.real_A = self.real_A.cuda()
        self.real_B = self.real_B.cuda()
        self.real_depth = self.real_depth.cuda()

        # print("Input shape before padding:", self.real_A.shape)
        # print("Input shape before padding:", self.real_B.shape)
        # print("Input shape before padding:", self.real_depth.shape)


        # 腐蚀操作
        kernel_size = 5
        pad_size = kernel_size // 2
        p1d = (pad_size, pad_size, pad_size, pad_size) 
        p_real_B = F.pad(self.real_B, p1d, "constant", 1)
        erode_real_B = -1 * (F.max_pool2d(-1 * p_real_B, kernel_size, 1))
        erode_real_B = erode_real_B.cuda() 

        res1 = self.gauss_conv(erode_real_B[:, 0, :, :].unsqueeze(1))
        res2 = self.gauss_conv(erode_real_B[:, 1, :, :].unsqueeze(1))
        res3 = self.gauss_conv(erode_real_B[:, 2, :, :].unsqueeze(1))
        self.ink_real_B = torch.cat((res1, res2, res3), dim=1)  # 在通道上融合 重新拼接回RGB


    def test(self):
        with torch.no_grad():
            real_A = Variable(self.input_A)
            real_B = Variable(self.input_B)
        fake_B = self.netG_A(real_A)
        self.rec_A = self.netG_B(fake_B).data
        self.fake_B = fake_B.data
        self.edge_fake_B = self.fake_B
        self.edge_real_A = self.fake_B
        fake_A = self.netG_B(real_B)
        self.rec_B = self.netG_A(fake_A).data
        self.fake_A = fake_A.data
        self.ink_real_B = fake_A
        self.ink_fake_B = self.fake_A

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def set_weights(self):
        self.lambda_idt = self.opt.identity
        self.lambda_A = self.opt.lambda_A
        self.lambda_B = self.opt.lambda_B
        self.lambda_sup = self.opt.lambda_sup  # 会更改
        self.lambda_ink = self.opt.lambda_ink
        self.lambda_Geom = self.opt.lambda_Geom
        self.lambda_recog = self.opt.lambda_recog


    # 重构损失
    def get_identityLoss(self, real_A, real_B):
        idt_A = self.netG_A(real_B)
        loss_idt_A = self.criterionIdt(idt_A, real_B) * self.lambda_B * self.lambda_idt
        idt_B = self.netG_B(real_A)
        loss_idt_B = self.criterionIdt(idt_B, real_A) * self.lambda_A * self.lambda_idt

        return idt_A, idt_B, loss_idt_A.item(), loss_idt_B.item()

    # 边缘损失
    def get_edgeLoss(self, real_A, fake_B):
        edge_real_A = torch.sigmoid(self.HED(real_A).detach())
        edge_fake_B = torch.sigmoid(self.HED(fake_B))
        loss_edge_1 = no_sigmoid_cross_entropy(edge_fake_B, edge_real_A) * self.lambda_sup
        return edge_real_A, edge_fake_B, loss_edge_1

    def get_G_Loss(self, fake_B, fake_A):
        pred_fake = self.netD_A(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, True)

        pred_fake = self.netD_B(fake_A)
        loss_G_B = self.criterionGAN(pred_fake, True)
        return loss_G_A, loss_G_B

    def get_ink_fake_B(self, fake_B):
        kernel_size = 5
        pad_size = kernel_size // 2
        p1d = (pad_size, pad_size, pad_size, pad_size)
        p_fake_B = F.pad(fake_B, p1d, "constant", 1)
        erode_fake_B = -1 * (F.max_pool2d(-1 * p_fake_B, kernel_size, 1))
        res1 = self.gauss_conv(erode_fake_B[:, 0, :, :].unsqueeze(1))
        res2 = self.gauss_conv(erode_fake_B[:, 1, :, :].unsqueeze(1))
        res3 = self.gauss_conv(erode_fake_B[:, 2, :, :].unsqueeze(1))
        ink_fake_B = torch.cat((res1, res2, res3), dim=1)
        return ink_fake_B

    def get_G_ink_loss(self, ink_fake_B):
        pred_fake_ink = self.netD_ink(ink_fake_B)
        loss_G_ink = self.criterionGAN(pred_fake_ink, True) * self.lambda_ink
        return loss_G_ink


    def get_cycle_loss(self, rec_A, rec_B):
        loss_cycle_A = self.criterionCycle(rec_A, self.real_A) * self.lambda_A
        loss_cycle_B = self.criterionCycle(rec_B, self.real_B) * self.lambda_B
        return loss_cycle_A, loss_cycle_B

    def get_geom_loss(self,geom_input, recover_geom):  
        geom_input = F.interpolate(geom_input, (299, 299))  # 网络要求299x299
        recover_geom = F.interpolate(recover_geom, (304, 304))  # 预测的大小304x304
        # if geom_input.size()[1] == 1:
        #     geom_input = geom_input.repeat(1, 3, 1, 1)
        _, geom_input = self.net_recog(geom_input)
        # print("Input shape before padding:", geom_input.shape)
        geom_input = geom_input.repeat(1, 768, 7, 1)
        pred_geom = self.netGeom(geom_input)
        # print("Input shape before padding:", pred_geom.shape)
        pred_geom = F.interpolate(pred_geom, (304, 304))
        pred_geom = (pred_geom + 1) / 2.0
        criterionGeom = torch.nn.BCELoss(reduce=True)
        loss_cycle_Geom = criterionGeom(pred_geom, recover_geom) * self.lambda_Geom
        fake_depth = F.interpolate(pred_geom, (256, 256))
        return fake_depth, loss_cycle_Geom
        
    def get_recog_loss(self, real_A, fake_B):
        recog_real = real_A
        recog_real0 = (recog_real[:, 0, :, :].unsqueeze(1) - 0.48145466) / 0.26862954
        recog_real1 = (recog_real[:, 1, :, :].unsqueeze(1) - 0.4578275) / 0.26130258
        recog_real2 = (recog_real[:, 2, :, :].unsqueeze(1) - 0.40821073) / 0.27577711
        recog_real = torch.cat([recog_real0, recog_real1, recog_real2], dim=1)

        line_input = fake_B
        # if out_channels == 1:
        #     line_input_channel0 = (line_input - 0.48145466) / 0.26862954
        #     line_input_channel1 = (line_input - 0.4578275) / 0.26130258
        #     line_input_channel2 = (line_input - 0.40821073) / 0.27577711
        #     line_input = torch.cat([line_input_channel0, line_input_channel1, line_input_channel2], dim=1)

        patches_r = [F.interpolate(recog_real, size=224)]  # The resize operation on tensor.
        patches_l = [F.interpolate(line_input, size=224)]

        # if N_patches > 1:
        #     patches_r2, patches_l2 = createNRandompatches(recog_real, line_input, opt.N_patches,opt.patch_size)
        #     patches_r += patches_r2
        #     patches_l += patches_l2

        loss_recog = 0

        for patchnum in range(len(patches_r)):

            real_patch = patches_r[patchnum]
            line_patch = patches_l[patchnum]
            
            clip_model, preprocess = clip.load("ViT-B/32", jit=False)
            clip_model = clip_model.to()
            clip.model.convert_weights(clip_model)

            feats_r = clip_model.encode_image(real_patch).detach()
            feats_line = clip_model.encode_image(line_patch)

            criterionCLIP = torch.nn.MSELoss(reduce=True)

            myloss_recog = criterionCLIP(feats_line, feats_r.detach())

            # if opt.cos_clip == 1:
            #     myloss_recog = 1.0 - loss_recog
            #     myloss_recog = torch.mean(loss_recog)

            patch_factor = (1.0 / float(1))
            if patchnum == 0:
                patch_factor = 1.0
            loss_recog += patch_factor * myloss_recog

        return loss_recog * self.lambda_recog

    # def total_variation_loss(self, image):
    #     # Calculate the total variation loss
    #     h_tv = torch.sum(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]))
    #     v_tv = torch.sum(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    #     tv_loss = h_tv + v_tv
    #     return tv_loss * 0.1

    
    def backward_G(self,lambda_sup):

        # 重构
        idt_A, idt_B, loss_idt_A, loss_idt_B = self.get_identityLoss(self.real_A, self.real_B)
        # 生成
        fake_B = self.netG_A(self.real_A)
        fake_A = self.netG_B(self.real_B)
        # 边缘
        edge_real_A, edge_fake_B, loss_edge_1 = self.get_edgeLoss(self.real_A, fake_B)
        loss_G_A, loss_G_B = self.get_G_Loss(fake_B, fake_A)
        # 水墨
        ink_fake_B = self.get_ink_fake_B(fake_B)
        loss_G_ink = self.get_G_ink_loss(ink_fake_B)
        # 循环
        rec_A = self.netG_B(fake_B)
        rec_B = self.netG_A(fake_A)
        loss_cycle_A, loss_cycle_B = self.get_cycle_loss(rec_A, rec_B)
        # # geom
        fake_depth, loss_geom = self.get_geom_loss(fake_B, self.real_depth)
        # recog
        loss_recog = self.get_recog_loss(self.real_A, fake_B)

        # tv_loss_fake_A = self.total_variation_loss(fake_A)
        # tv_loss_fake_B = self.total_variation_loss(fake_B) 
        # tv_loss = tv_loss_fake_A + tv_loss_fake_B

        # combined loss
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B + loss_edge_1 + loss_recog + loss_geom

        loss_G.backward()

        self.fake_B = fake_B
        self.fake_A = fake_A
        self.rec_A = rec_A
        self.rec_B = rec_B
        self.edge_real_A = edge_real_A
        self.edge_fake_B = edge_fake_B
        self.ink_fake_B = ink_fake_B
        self.fake_depth = fake_depth

        self.loss_G_A = loss_G_A.item()
        self.loss_G_B = loss_G_B.item()
        self.loss_G_ink = loss_G_ink.item()
        self.loss_cycle_A = loss_cycle_A.item()
        self.loss_cycle_B = loss_cycle_B.item()
        self.loss_edge_1 = loss_edge_1.item()
        self.loss_idt_A = loss_idt_A
        self.loss_idt_B = loss_idt_B
        self.loss_geom = loss_geom.item()
        self.loss_recog = loss_recog.item()

        pass

    # 辨别器对误差回传
    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B
        loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        self.loss_D_A = loss_D_A.item()
        pass

    def backward_D_B(self):
        fake_A = self.fake_A
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        self.loss_D_B = loss_D_B.item()
        pass

    def backward_D_ink(self):
        ink_fake_B = self.ink_fake_B
        loss_D_ink = self.backward_D_basic(self.netD_ink, self.ink_real_B, ink_fake_B)
        self.loss_D_ink = loss_D_ink.item()
        pass

    def optimize_parameters(self,lambda_sup):
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G(lambda_sup)
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()
        # D_ink
        self.optimizer_D_ink.zero_grad()
        self.backward_D_ink()
        self.optimizer_D_ink.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A), ('Cyc_A', self.loss_cycle_A),
                                  ('D_B', self.loss_D_B), ('G_B', self.loss_G_B), ('Cyc_B', self.loss_cycle_B),
                                  ('edge1', self.loss_edge_1), ('D_ink', self.loss_D_ink),
                                  ('G_ink', self.loss_G_ink), ('idt_A', self.loss_idt_A), ('idt_B', self.loss_idt_B),
                                  ('Geom', self.loss_geom), ('recog', self.loss_recog)])
        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.input_A)
        fake_B = util.tensor2im(self.fake_B)
        rec_A = util.tensor2im(self.rec_A)
        real_B = util.tensor2im(self.input_B)
        fake_A = util.tensor2im(self.fake_A)
        rec_B = util.tensor2im(self.rec_B)
        edge_fake_B = util.tensor2im(self.edge_fake_B)
        edge_real_A = util.tensor2im(self.edge_real_A)
        ink_real_B = util.tensor2im(self.ink_real_B.data)
        ink_fake_B = util.tensor2im(self.ink_fake_B)
        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                   ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B),
                                   ('edge_fake_B', edge_fake_B), ('edge_real_A', edge_real_A),
                                   ('ink_real_B', ink_real_B), ('ink_fake_B', ink_fake_B)])
        # if self.opt.isTrain and self.opt.identity > 0.0:
        #     ret_visuals['idt_A'] = util.tensor2im(self.idt_A)
        #     ret_visuals['idt_B'] = util.tensor2im(self.idt_B)
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
        self.save_network(self.netD_ink, 'D_ink', label, self.gpu_ids)
        # self.save_network(self.netGeom, 'Geom', label, self.gpu_ids)
