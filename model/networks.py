import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import modules
logger = logging.getLogger('base')
####################
# initialize
####################


def weights_init_normal(m, std=0.02):## 正态分布初始化
    classname = m.__class__.__name__ ##返回层的名称 'Conv2d'等
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std) ##使用正态分布对权重 m.weight.data 进行初始化。
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None: 
            m.bias.data.zero_() #偏置归零
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0) # 使用常量值 0.0 初始化偏置 m.bias.data。


def weights_init_kaiming(m, scale=1): ## kaiming初始化
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m): # Orthogonal 初始化
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)

# 选择初始化方法
def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################


# Generator
def define_G(opt):
    model_opt = opt['model'] #提取json文件中model配置
    print(model_opt['which_model_G']) #打印model中的模型名称
    if model_opt['which_model_G'] == 'ddpm': ##模型名如果是ddpm就导入这些模块
        from .ddpm_modules import diffusion, unet, diffusion_Pt #如果配置中 which_model_G 为 'ddpm'，导入与 DDPM（扩散概率模型）相关的模块
    if ('norm_groups' not in model_opt['unet']) or model_opt['unet']['norm_groups'] is None:
        model_opt['unet']['norm_groups']=32 #添加默认参数
    model = unet.UNet( # 配置Unet网络
        in_channel=model_opt['unet']['in_channel'],
        out_channel=model_opt['unet']['out_channel'],
        norm_groups=model_opt['unet']['norm_groups'],
        inner_channel=model_opt['unet']['inner_channel'],
        channel_mults=model_opt['unet']['channel_multiplier'],
        attn_res=model_opt['unet']['attn_res'],
        res_blocks=model_opt['unet']['res_blocks'],
        dropout=model_opt['unet']['dropout'],
        image_size=model_opt['diffusion']['image_size']
    )
    # True: 训练不确定性模型阶段
    # False: 使用不确定性模型指导训练阶段
    if opt['uncertainty_train']: ## 训练不确定模型
        netG = diffusion_Pt.GaussianDiffusion(
            model,
            image_size=model_opt['diffusion']['image_size'],
            channels=model_opt['diffusion']['channels'],
            loss_type='l1',   
            conditional=model_opt['diffusion']['conditional'],
            schedule_opt=model_opt['beta_schedule']['train'] #噪声调度
        )
    else:
        netG = diffusion.GaussianDiffusion( ## 使用不确定性模型指导训练阶段
            model,
            image_size=model_opt['diffusion']['image_size'],
            channels=model_opt['diffusion']['channels'],
            loss_type='l1',   
            conditional=model_opt['diffusion']['conditional'],
            schedule_opt=model_opt['beta_schedule']['train']
        )
    if opt['phase'] == 'train':
        # init_weights(netG, init_type='kaiming', scale=0.1)
        init_weights(netG, init_type='orthogonal') #默认使用 Orthogonal 初始化，也可切换到 Kaiming 初始化
    if opt['gpu_ids'] and opt['distributed']: # 分布式支持训练
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG

    
# Generator，额这个压根就没有用到，甚至可以删掉
def define_GGG(opt): ## 这个应该是作者进行消融实验的，没有包含不确定性训练
    model_opt = opt['model']
    print(model_opt['which_model_G'])
    if model_opt['which_model_G'] == 'ddpm':
        from .ddpm_modules import diffusion, unet
    if ('norm_groups' not in model_opt['unet']) or model_opt['unet']['norm_groups'] is None:
        model_opt['unet']['norm_groups']=32
    model = unet.UNet(
        in_channel=model_opt['unet']['in_channel'],
        out_channel=model_opt['unet']['out_channel'],
        norm_groups=model_opt['unet']['norm_groups'],
        inner_channel=model_opt['unet']['inner_channel'],
        channel_mults=model_opt['unet']['channel_multiplier'],
        attn_res=model_opt['unet']['attn_res'],
        res_blocks=model_opt['unet']['res_blocks'],
        dropout=model_opt['unet']['dropout'],
        image_size=model_opt['diffusion']['image_size']
    )
    netGVar = diffusion.GaussianDiffusion(
        model,
        image_size=model_opt['diffusion']['image_size'],
        channels=model_opt['diffusion']['channels'],
        loss_type='l1',  
        conditional=model_opt['diffusion']['conditional'],
        schedule_opt=model_opt['beta_schedule']['train']
    )
    if opt['phase'] == 'train':
        # init_weights(netG, init_type='kaiming', scale=0.1)
        init_weights(netGVar, init_type='orthogonal')
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        netGVar = nn.DataParallel(netGVar)
    return netGVar