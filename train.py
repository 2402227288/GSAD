import os
from os.path import basename
import math
import argparse
import logging
import cv2
import sys
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torchvision import transforms
import options.options as option
from utils import util
from data import create_dataloader
import data as Data
from data.LoL_dataset import LOLv1_Dataset, LOLv2_Dataset
import torchvision.transforms as T
import model as Model
import core.logger as Logger
import core.metrics as Metrics
import random

import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass

def init_dist(backend='nccl', **kwargs):             ##分布式训练初始化
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Path to option YMAL file.',
                            default='./config/lolv1.yml') # 
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',  #--launcher：用于分布式训练，pytorch 为多进程启动方式，none 表示单机单卡训练。
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('-c', '--config', type=str, default='config/lolv1_train.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default="1")
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('-uncertainty', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args) #解析json文件
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    opt_dataset = option.parse(args.dataset, is_train=True) ##解析train的yml文件 args.dataset就是--dataset，这个opt是专门给数据的


    # os.environ['CUDA_VISIBLE_DEVICES'] = '0' #指定可见GPU为0
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids  # 从命令行参数设置可见的GPU


    if args.uncertainty: ##是否进行不确定训练
        opt['uncertainty_train'] = True
    else:
        opt['uncertainty_train'] = False

    #### distributed training settings
    #### distributed training settings
    # 如果 launcher 为 none，关闭分布式训练。
    # 如果是分布式训练，使用 torch.distributed 初始化通信环境。
    # 获取当前进程的 rank 和总进程数 world_size。
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        # init_dist()
        torch.distributed.init_process_group(
            'nccl',
            init_method='env://'
        ) 
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        device = torch.device("cuda", rank)


    #### mkdir and loggers  配置日志输出格式
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO, # base记录train
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO, # val记录val
                          screen=True, tofile=True)
        logger = logging.getLogger('base') # 获取base日志
        logger.info(option.dict2str(opt)) # 将配置文件的字典转化为字符串在日志中记录

        # tensorboard logger ，tensorboard没用到
        if opt.get('use_tb_logger', False) and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                # from torch.utils.tensorboard import SummaryWriter
                if sys.platform != 'win32':
                    from tensorboardX import SummaryWriter
                else:
                    from tensorboardX import SummaryWriter
                    # from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboard import SummaryWriter
            conf_name = basename(args.opt).replace(".yml", "")
            exp_dir = opt['path']['experiments_root']
            log_dir_train = os.path.join(exp_dir, 'tb', conf_name, 'train')
            log_dir_valid = os.path.join(exp_dir, 'tb', conf_name, 'valid')
            tb_logger_train = SummaryWriter(log_dir=log_dir_train)
            tb_logger_valid = SummaryWriter(log_dir=log_dir_valid)
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed  设置随机种子
    seed =  1999  # 优先使用配置中的seed，否则使用1999
    if rank <= 0:
        logger.info('Seed: {}'.format(seed))
    util.set_random_seed(seed)
    # seed = opt['seed']
    # if seed is None:
    #     seed = random.randint(1, 10000)
    # if rank <= 0:
    #     logger.info('Random seed: {}'.format(seed))
    # util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    if opt_dataset['dataset'] == 'LOLv1':
        dataset_cls = LOLv1_Dataset
    elif opt_dataset['dataset'] == 'LOLv2':
        dataset_cls = LOLv2_Dataset
    else:
        raise NotImplementedError()

    for phase, dataset_opt in opt_dataset['datasets'].items():
        if phase == 'train':
            train_set = dataset_cls(opt=dataset_opt, train=True, all_opt=opt_dataset)
            train_loader = create_dataloader(train_set, dataset_opt, opt_dataset, None)
        elif phase == 'val':
            val_set = dataset_cls(opt=dataset_opt, train=False, all_opt=opt_dataset)
            val_loader = create_dataloader(val_set, dataset_opt, opt_dataset, None)

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase']) # 噪声调度


    # training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(current_epoch, current_step))
    avg_psnr = 0
    while current_step < n_iter: # 以迭代次数计算epoch
        # if opt['dist']:
        #     train_sampler.set_epoch(current_epoch)

        current_epoch += 1
        for _, train_data in enumerate(train_loader):

            current_step += 1
            if current_step > n_iter: # 终止训练
                break

            diffusion.feed_data(train_data) # 数据传入设备
            diffusion.optimize_parameters() # 模型训练集成
            # log
            if current_step % opt['train']['print_freq'] == 0 and rank <= 0: # 每隔 print_freq 步输出一次训练日志，只有主进程 (rank <= 0) 执行日志记录。
                logs = diffusion.get_current_log() # 主要是损失函数值的log
                message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                    current_epoch, current_step)
                for k, v in logs.items():  # logs.items()是一个字典，kv代表对应名称和对应值（损失）
                    message += '{:s}: {:.4e} '.format(k, v)
                logger.info(message) # 输出包含迭代次数以及损失值的日志信息，主模型3个loss，不确定性训练1个loss

            if current_step % opt['train']['save_checkpoint_freq'] == 0 and rank <= 0: # 保证在主进程上保存模型
                logger.info('Saving models and training states.')
                diffusion.save_network(current_epoch, current_step)

            # validation
            if current_step % opt['train']['val_freq'] == 0 and rank <= 0:

                avg_psnr = 0.0
                avg_ssim = 0.0
                idx = 0
  
                result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                result_path_gt = result_path+'/gt/'
                result_path_out = result_path+'/output/'
                result_path_input = result_path+'/input/'
                os.makedirs(result_path_gt, exist_ok=True)
                os.makedirs(result_path_out, exist_ok=True)
                os.makedirs(result_path_input, exist_ok=True)

                if opt['dist']:
                    diffusion.netG.module.set_new_noise_schedule(
                            opt['model']['beta_schedule']['val'], device) # 设置噪声调度，从封装中剥离出来
                else:
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                
                for val_data in val_loader:

                    idx += 1
                    diffusion.feed_data(val_data) #将数据移到device上
                    diffusion.test(continous=False)

                    visuals = diffusion.get_current_visuals()
                    # HQ恢复图像 GT LQ
                    normal_img = Metrics.tensor2img(visuals['HQ'])
                    if normal_img.shape[0] != normal_img.shape[1]: # lolv1 and lolv2-real 检查图像是否是正方形
                        normal_img = normal_img[8:408, 4:604,:]
                    gt_img = Metrics.tensor2img(visuals['GT'])
                    ll_img = Metrics.tensor2img(visuals['LQ'])

                    img_mode = 'single'
                    if img_mode == 'single':
                        util.save_img(
                            gt_img, '{}/{}_gt.png'.format(result_path_gt, idx))
                        util.save_img(
                            ll_img, '{}/{}_in.png'.format(result_path_input, idx))
                        # util.save_img(
                        #     normal_img, '{}/{}_normal.png'.format(result_path_out, idx))
                    else:
                        util.save_img(
                            gt_img, '{}/{}_{}_gt.png'.format(result_path, current_step, idx))
                        util.save_img(
                            normal_img, '{}/{}_{}_normal_process.png'.format(result_path, current_step, idx))
                        util.save_img(
                            Metrics.tensor2img(visuals['HQ'][-1]), '{}/{}_{}_normal.png'.format(result_path, current_step, idx))
                        normal_img = Metrics.tensor2img(visuals['HQ'][-1])
  

                    # Similar to LLFlow, 
                    # we follow a similar way of 'Kind' to finetune the overall brightness as illustrated 
                    # in Line 73 (https://github.com/zhangyhuaee/KinD/blob/master/evaluate_LOLdataset.py).
                   
                    ####GT mean矫正操作！！！！！！
                    gt_img = gt_img / 255. ##gt
                    normal_img = normal_img / 255. ##模型输出
                    #计算平均灰度值 astype代表转化为这个数据类型
                    mean_gray_out = cv2.cvtColor(normal_img.astype(np.float32), cv2.COLOR_BGR2GRAY).mean() # 计算灰度图所有像素点的平均值
                    mean_gray_gt = cv2.cvtColor(gt_img.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                    normal_img_adjust = np.clip(normal_img * (mean_gray_gt / mean_gray_out), 0, 1) ##得到矫正值

                    normal_img = (normal_img_adjust * 255).astype(np.uint8)
                    gt_img = (gt_img * 255).astype(np.uint8)

                    psnr = util.calculate_psnr(normal_img, gt_img)
                    ssim = util.calculate_ssim(normal_img, gt_img)
                    
                    util.save_img(normal_img, '{}/{}_normal.png'.format(result_path_out, idx))

                    logger.info('cPSNR: {:.4e} cSSIM: {:.4e}'.format(psnr, ssim))
                    avg_ssim += ssim
                    avg_psnr += psnr

                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx

                logger.info('# Validation # PSNR: {:.4e} SSIM: {:.4e}'.format(avg_psnr, avg_ssim))
                logger_val = logging.getLogger('val')  
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e} SSIM: {:.4e}'.format(
                    current_epoch, current_step, avg_psnr, avg_ssim))

 
if __name__ == '__main__':
    main()
