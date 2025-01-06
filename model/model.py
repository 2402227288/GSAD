import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger('base')


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)

        if opt['dist']: ## 如果dist为true启用分布式训练
            self.local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(self.local_rank)
            device = torch.device("cuda", self.local_rank)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt)) # netG可能是主模型也可能是不确定训练模型
        if opt['dist']:
            self.netG.to(device)
       
        # self.netG.to(device) 一般都要调用两次，第二次专门为分布式训练设计
        if not opt['uncertainty_train']: #如果不是不确定性训练，还会定义一个不确定性模型。netGU用来指导
            self.netGU = self.set_device(networks.define_G(opt)) # uncertainty model 不确定性训练模型
            if opt['dist']:
                self.netGU.to(device)
       

        self.schedule_phase = None
        self.opt = opt

        # set loss and load resume state
        self.set_loss()

        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train') #设置噪声调度
        if self.opt['phase'] == 'train': ##配置训练过程的优化器
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()

        if not opt['uncertainty_train'] and self.opt['phase'] == 'train': #非不确定训练，要借助不确定性训练来辅助主模型训练
            self.netGU.load_state_dict(torch.load(self.opt['path']['resume_state']+'_gen.pth'), strict=True) # 载入不确定性模型权重 
            if opt['dist']:
                self.netGU = DDP(self.netGU, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

        if self.opt['phase'] == 'test': ## 测试阶段加载主模型权重
            try:
                self.netG.load_state_dict(torch.load(self.opt['path']['resume_state']), strict=True)
            except Exception:
                self.netG = nn.DataParallel(self.netGU)
                self.netG.load_state_dict(torch.load(self.opt['path']['resume_state']), strict=True)
        else:
            self.load_network() ## 如果是训练阶段 调用 load_network() 方法加载预训练模型。注意是给netG载入权重，载入权重的路径在函数中有详细说明
            if opt['dist']:
                self.netG = DDP(self.netG, device_ids=[self.local_rank], output_device=self.local_rank,find_unused_parameters=True)
        self.print_network()

    def feed_data(self, data): ## 将数据中的LQ以及GT移动到指定设备上

        dic = {}

        if self.opt['dist']:
            dic = {}
            dic['LQ'] = data['LQ'].to(self.local_rank)
            dic['GT'] = data['GT'].to(self.local_rank)
            self.data = dic
        else:
            dic['LQ'] = data['LQ']
            dic['GT'] = data['GT']

            self.data = self.set_device(dic)

    def optimize_parameters(self): ## 包含两个模型的训练过程

        self.optG.zero_grad()

        if not self.opt['uncertainty_train']:
            if self.opt['dist']:
                l_pix, l_gsad = self.netG(self.data, self.netGU.module.denoise_fn) ##此时需要借助netGU进行辅助训练
            else:
                l_pix, l_gsad = self.netG(self.data, self.netGU.denoise_fn) # diffusion.py会返回两个loss

            # need to average in multi-gpu
            b, c, h, w = self.data['LQ'].shape

            num_clusters = 6
            l_pix = l_pix.sum()/int(b*c*h*w)
            l_gsad = l_gsad.sum()/int(b*num_clusters)
            # l_svd = l_svd.sum()/int(b*c*h*w)
            loss = l_pix + l_gsad
            loss.backward()
            self.optG.step()

            # set log
            self.log_dict['total_loss'] = loss.item()
            self.log_dict['l_1'] = l_pix.item()
            self.log_dict['l_gsad'] = l_gsad.item()
        else:
            l_pix = self.netG(self.data) ## 此时仅仅进行不确定性训练，注意Pt只返回一个loss

            b, c, h, w = self.data['LQ'].shape

            l_pix = l_pix.sum()/int(b*c*h*w)
            l_pix.backward()
            self.optG.step()

            # set log
            self.log_dict['l_u'] = l_pix.item() ## 日志记录总损失


    def test(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data['LQ'], continous) ## 从低光图像逐步恢复到正常图像
                
            else:
                if self.opt['dist']:
                    self.SR = self.netG.module.super_resolution(self.data['LQ'], continous) # self.SR代表恢复的图像
                else:
                    self.SR = self.netG.super_resolution(self.data['LQ'], continous) ## 当然默认执行这一步

        self.netG.train() #切换回训练模式

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

        if not self.opt['uncertainty_train']:
            if isinstance(self.netGU, nn.DataParallel):
                self.netGU.module.set_loss(self.device)
            else:
                self.netGU.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        # 该函数用于设置扩散模型（netG 和 netGU）中的噪声调度计划，
        # 根据指定的 schedule_opt 参数调整模型的噪声分布，以适应不同的训练或测试阶段。

        if self.opt['dist']:
            # local_rank = torch.distributed.get_rank()
            device = torch.device("cuda", self.local_rank)
            if self.schedule_phase is None or self.schedule_phase != schedule_phase:
                self.schedule_phase = schedule_phase
                if isinstance(self.netG, nn.DataParallel):
                    self.netG.module.set_new_noise_schedule(
                        schedule_opt, self.device)
                else:
                    self.netG.set_new_noise_schedule(schedule_opt, device)

                if not self.opt['uncertainty_train']:
                    if isinstance(self.netGU, nn.DataParallel):
                        self.netGU.module.set_new_noise_schedule(
                            schedule_opt, self.device)
                    else:
                        self.netGU.set_new_noise_schedule(schedule_opt, device)
        else:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

            if not self.opt['uncertainty_train']:
                if isinstance(self.netGU, nn.DataParallel):
                    self.netGU.module.set_new_noise_schedule(
                        schedule_opt, self.device)
                else:
                    self.netGU.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False): ## 从计算图和GPU上分离出来进行指标计算
        out_dict = OrderedDict() # 创建一个有序字典，应该是输出字典
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else: ## 执行以下内容
            out_dict['HQ'] = self.SR.detach().float().cpu() ## 恢复图像
            out_dict['INF'] = self.data['LQ'].detach().float().cpu() #输入图像
            out_dict['GT'] = self.data['GT'].detach()[0].float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LQ'] = self.data['LQ'].detach().float().cpu() ## 低光图像
            else: 
                out_dict['LQ'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(s)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch)) #保存路径
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()# 将所有张量参数值保存到cpu上
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path) 

        if self.opt['uncertainty_train']:
            uncertainty_save_dir = './checkpoints/uncertainty/'
            os.makedirs(uncertainty_save_dir, exist_ok=True)
            ut_gen_path = os.path.join(
                './checkpoints/uncertainty/', 'latest_gen.pth'.format(iter_step, epoch))
            ut_opt_path = os.path.join(
                './checkpoints/uncertainty/', 'latest_opt.pth'.format(iter_step, epoch))
            torch.save(state_dict, ut_gen_path)
            torch.save(opt_state, ut_opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module

            # network = nn.DataParallel(network).cuda()

            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
            # network.load_state_dict(torch.load(
            #     gen_path), strict=False)
            
            if self.opt['phase'] == 'train':
                # optimizer
                # opt = torch.load(opt_path)
                # self.optG.load_state_dict(opt['optimizer'])
                # self.begin_step = opt['iter']
                # self.begin_epoch = opt['epoch']
                self.begin_step = 0
                self.begin_epoch = 0
      
      
      
    # ###########恢复训练            
    # def load_network(self):
    #     # 新增的恢复路径
    #     gen_resume = "./experiments/lolv1_train_241220_155652/checkpoint/I2000000_E33334_gen.pth"
    #     opt_resume = "./experiments/lolv1_train_241220_155652/checkpoint/I2000000_E33334_opt.pth"

    #     # 加载生成器模型
    #     logger.info(f"Loading pretrained model for G from {gen_resume} ...")
    #     network = self.netG
    #     if isinstance(self.netG, nn.DataParallel):  # 如果是多GPU训练，获取实际模型
    #         network = network.module

    #     # 加载生成器权重
    #     network.load_state_dict(
    #         torch.load(gen_resume), strict=(not self.opt['model']['finetune_norm'])
    #     )
        
    #     # 如果是训练阶段，加载优化器状态和训练进度
    #     if self.opt['phase'] == 'train':
    #         if os.path.exists(opt_resume):  # 检查优化器状态文件是否存在
    #             logger.info(f"Loading optimizer state from {opt_resume} ...")
    #             opt_state = torch.load(opt_resume)
    #             self.optG.load_state_dict(opt_state['optimizer'])  # 恢复优化器状态
    #             self.begin_step = opt_state['iter']
    #             self.begin_epoch = opt_state['epoch']
    #             logger.info(f"Resumed training from step {self.begin_step}, epoch {self.begin_epoch}.")


