import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import cv2
from .kmean_torch import kmeans_core
from kmeans_pytorch import kmeans

# 在扩散模型中，这段代码的目的是对 Beta 序列的前 warmup_time 步进行热启动调整：

# 让 Beta 值从 linear_start 开始逐步增加到 linear_end。
# 热启动的前 warmup_time 步通常用于帮助模型逐渐适应噪声的加入。
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):#beta起始值和终止值，总的时间步数（扩散过程的步长），热启动比例
    betas = linear_end * np.ones(n_timestep, dtype=np.float64) # beta长度为n_timestep
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':# 平方调度
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':# 线性调度
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10': # 热启动调度
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const': # 常量调度
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1 递减调度
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine": # 余弦调度
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val # val非空就返回val
    return d() if isfunction(d) else d # d是函数返回函数，否则就返回d的值


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn, #去噪网络，networks中为unet网络
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt) #这里应该要调度一下噪声

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device) # 固定三个参数

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance( #如果betas为tensor则返回其numpy格式
            betas, torch.Tensor) else betas # 转化为numpy数组
        alphas = 1. - betas #计算 Alpha 序列 [500,]
        alphas_cumprod = np.cumprod(alphas, axis=0) #Alpha的累积乘积[500,]
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])#前一个Alpha的累积乘积 [500,]
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod)) #这么写确实没错，但维度会到T+1 [501,]

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas)) #将变量作为一个常量缓存，在模型中不会更新
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) ##反向传播的方差beta~
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', # 反向传播方差
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20)))) ##防止log出现异常
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise ##由x_t得到x_0

    def q_posterior(self, x_start, x_t, t):  ## 计算后验均值和后验方差 这一部分属于q
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):## 这一部分是p
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)#把单个噪声等级值扩展成一个批次大小的张量，使得批次中的每个样本都能获得相同的噪声等级信息。
        if condition_x is not None: ## condition_x为低光照图像
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)[0]) ## 将低光图像+噪声+噪声等级输入unet网络就可以得到noise，从而预测x_recon也即恢复图像也即x_0
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level)) # denoise_fn为噪声预测网络，文中也即Unet，从噪声中恢复x_0

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance ##得到后验均值u~和beta~

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):   #单步逆扩散过程，由x_t得到x_t-1
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        # print(model_log_variance)
        # print(model_log_variance.shape)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False): # 从纯噪声​x_T逐步采样到清晰图像x_0的完整过程
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))# # 设置采样间隔，通常是每10次迭代保存一次图像
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):## 可以在终端看到逐步的去噪过程
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in   ## 执行这个，在test模块，我们可以看到此时x_in是属于LQ图像也即低光图像
            shape = x.shape
            img = torch.randn(shape, device=device) #随机生成一个噪声
            ret_img = x #用于储存恢复的图像
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1] ##默认返回最后一个图像

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )


    def draw_features(self, x, savename): #将特征图（Feature Map）进行可视化
        img = x[0, 0, :, :]#只选取第一个样本的第一个通道
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255   #归一化
        img = img.astype(np.uint8)   #编码类型
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET) #生成彩色图片
        cv2.imwrite(savename,img) #保存为savename
 
    def predict_start(self, x_t, continuous_sqrt_alpha_cumprod, noise):  ##没用到
        return (1. / continuous_sqrt_alpha_cumprod) * x_t - \
            (1. / continuous_sqrt_alpha_cumprod**2 - 1).sqrt() * noise

    def predict_t_minus1(self, x, t, continuous_sqrt_alpha_cumprod, noise, clip_denoised=True): ##没用到

        x_recon = self.predict_start(x, 
                    continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), 
                    noise=noise)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, model_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        noise_z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

        return model_mean + noise_z * (0.5 * model_log_variance).exp()       


    def to_patches(sefl, data, kernel_size): #将输入的 4D 图像张量 分割成固定大小的非重叠小块（patches）

        patches = nn.Unfold(kernel_size=kernel_size, stride=kernel_size)(torch.mean(data, dim=1, keepdim=True)) # [8, 64, 400]
        patches = patches.transpose(2,1) # [8, 400, 64]
        # patches = patches.view(b, 400, 8, 8).contiguous()
        return patches


    def calcu_kmeans(self, data, num_clusters): #将小块data聚类成num_clusters类别

        [b, h, w] = data.shape
        cluster_ids_all = np.empty([b, h])
        cluster_ids_all = torch.from_numpy(cluster_ids_all)
        for i in range(b):
            # cluster_ids, cluster_centers = kmeans(
            #     X=data[i,:,:], num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
            # )
            km = kmeans_core(k=num_clusters,data_array=data[i,:,:].cpu().numpy(),batch_size=400,epochs=1000)
            km.run() # 执行聚类
            cluster_ids = km.idx #  获取聚类结果（每个图像块的类别索引）
            cluster_ids_all[i, :] = cluster_ids  # 存储当前批次的聚类结果

        return cluster_ids_all # [batch_size, num_patches] 返回每个批次中每个图像块的聚类类别。

    def calcu_svd(self, data): #矩阵奇异值分解，返回奇异值

        u, sv, v = torch.svd(data)
        #sv_F2 = torch.norm(sv, dim=1)
        #sv_F2 = sv_F2.unsqueeze(1)

        #normalized_sv = sv / sv_F2
        return sv

    def calcu_svd_distance(self, data1, data2, cluster_ids, num_clusters): ## 计算奇异值之间的距离

        [b, h, w] = data1.shape # [8, 400, 64] 
        sv_ab_dis = np.empty([b, num_clusters])
        sv_ab_dis = torch.from_numpy(sv_ab_dis)
        for i in range(num_clusters):

            indices = (cluster_ids[0] ==i).nonzero(as_tuple=True)[0]
            
            if len(indices)==0:
                sv_ab_dis[:, i] = 1e-5
            else:
                data1_select = torch.index_select(data1, 1, indices.cuda())
                data2_select = torch.index_select(data2, 1, indices.cuda())
                sv1 = self.calcu_svd(data1_select.cpu())
                sv2 = self.calcu_svd(data2_select.cpu())
   
                sv_ab_dis_i = torch.abs(sv1 - sv2)
                sv_ab_dis[:, i] = torch.sum(sv_ab_dis_i, dim=1)
        return sv_ab_dis

    def uncertainty_train(self, x_in, noise=None):  ## 从这个x_in开始改
 
        x_start = x_in['GT'] ##输入的GT图像，定义为x_0
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1) # t 是从 1 到 num_timesteps 中的一个随机整数
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],  # 随机采样根号alfa的连乘法
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1) # [batch_size, 1]


        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise) #模拟从x_0添加噪声到x_t

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(
                torch.cat([x_in['LQ'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod) ## 对去噪网络输入x_t与低光图像LQ和alfa连乘积

        # for uncertainty training 
        Pt = x_recon[1] ##得到不确定性Pt，x_recon包含预测的噪声+预测的Pt
        epsilon_pred = torch.mul(x_recon[0], torch.exp(-Pt)) # 加权的噪声预测
        epsilon = torch.mul(noise, torch.exp(-Pt)) # 加权的真实噪声预测
        loss = self.loss_func(epsilon_pred, epsilon) + 2 * torch.mean(Pt) ## 文中Lu，公式5

        return loss

    def forward(self, x, *args, **kwargs):
        return self.uncertainty_train(x, *args, **kwargs)
