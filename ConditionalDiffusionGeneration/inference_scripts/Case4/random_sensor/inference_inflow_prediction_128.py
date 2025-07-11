#!/usr/bin/env python3
"""
河流流量预测的条件扩散推理脚本
（版本：专注于测试和可视化前128个时间步）
"""

import os
import torch
import numpy as np
from functools import partial
import sys
import matplotlib.pyplot as plt
import yaml

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 添加路径
sys.path.append("../../../..")
from ConditionalDiffusionGeneration.src.guided_diffusion.unet import create_model
from ConditionalDiffusionGeneration.src.guided_diffusion.condition_methods import get_conditioning_method
from ConditionalDiffusionGeneration.src.guided_diffusion.measurements import get_operator, get_noise
from ConditionalDiffusionGeneration.src.guided_diffusion.gaussian_diffusion import create_sampler
from ConditionalNeuralField.cnf.inference_function import decoder
from einops import rearrange

def process_single_window(measurement, no_of_samples, time_length, latent_size, 
                        device, sample_fn, operator, cnf_config, feature_info):
    """
    处理单个窗口的推理
    （使用循环逐个生成样本，以避免批处理错误）
    """
    
    measurement_device = measurement.to(device)
    
    samples = []
    for i in range(no_of_samples):
        print(f"  生成样本 {i+1}/{no_of_samples}...")
        
        # 每次只生成一个样本 (batch size = 1)
        x_start = torch.randn(1, 1, time_length, latent_size, device=device)
        
        sample = sample_fn(x_start=x_start, measurement=measurement_device, record=False, save_root=None)
        samples.append(sample)
    
    print("  样本生成完成，正在解码...")
    
    # 将所有样本的潜在向量拼接在一起
    gen_latents = torch.cat(samples, dim=0)
    gen_latents = operator._unnorm(gen_latents)
    gen_latents = gen_latents[:, 0]
    
    # 使用 decoder 函数将潜在向量解码回真实测量空间
    coords = torch.tensor(np.load('/home/yif47/river/river-dl/temporal/datasets/colorado/test_coords.npy'), device=device, dtype=torch.float32)
    
    xnorm = operator.x_normalizer
    ynorm = operator.y_normalizer
    model = operator.model
    
    gen_latents_cnf_input = rearrange(gen_latents, "s t l -> (s t) l")
    

    
    gen_fields = decoder(coords, gen_latents_cnf_input, model, xnorm, ynorm, 
                         batch_size=cnf_config['cnf_batch_size'], device=device)
    gen_fields = rearrange(gen_fields, "(s t) co c -> s t co c", t=time_length)
    
    # 提取预测的inflow
    predicted_inflow = gen_fields[..., feature_info['inflow_index']:feature_info['inflow_index']+1]
    
    return predicted_inflow

def main():
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 加载训练配置文件
    print("加载训练配置文件...")
    training_config_path = '/home/yif47/river/river-dl/temporal/Confield/UnconditionalDiffusionTraining_and_Generation/training_recipes/combined_case4.yml'
    with open(training_config_path, 'r') as f:
        training_config = yaml.safe_load(f)
    
    # 加载特征信息
    print("加载特征信息...")
    feature_info = np.load('/home/yif47/river/river-dl/temporal/datasets/colorado/test_feature_info.npy', allow_pickle=True).item()
    
    # --- 数据加载和切片 ---
    time_length = training_config['time_length']

    # 加载完整的已知特征数据（原始尺度）
    true_measurement_full = torch.from_numpy(np.load('/home/yif47/river/river-dl/temporal/datasets/colorado/test_known_features.npy')).to(device)
    # 加载完整的真实inflow值（原始尺度）
    true_inflow_full = torch.from_numpy(np.load('/home/yif47/river/river-dl/temporal/datasets/colorado/test_true_inflow.npy')).to(device)

    print(f"原始数据范围:")
    print(f"known_features: [{true_measurement_full.min():.3f}, {true_measurement_full.max():.3f}]")
    print(f"true_inflow: [{true_inflow_full.min():.3f}, {true_inflow_full.max():.3f}]")

    # 仅截取前128个时间步进行测试
    true_measurement = true_measurement_full[:time_length]
    true_inflow = true_inflow_full[:time_length]
    
    print(f"\n--- 仅测试前 {time_length} 个时间步 ---")
    print(f"截取后的已知特征数据形状: {true_measurement.shape}")
    print(f"截取后的真实inflow数据形状: {true_inflow.shape}\n")
    
    # 加载训练好的无条件模型
    print("加载无条件扩散模型...")
    u_net_model = create_model(image_size=training_config['image_size'],
                               num_channels=training_config['num_channels'],
                               num_res_blocks=training_config['num_res_blocks'],
                               channel_mult=training_config['channel_mult'],
                               num_heads=training_config['num_heads'],
                               num_head_channels=training_config['num_head_channels'],
                               attention_resolutions=training_config['attention_resolutions'],
                               dims=training_config['dims'],
                               model_path='/home/yif47/river/river-dl/temporal/Confield/Input/diff_model/checkpoint/ema_0.9999_045000.pt'
                            )
    u_net_model.to(device)
    u_net_model.eval()

    # 加载CNF配置文件 - 使用combined_case4.yml而不是stream.yml
    cnf_config_path = '/home/yif47/river/river-dl/temporal/Confield/UnconditionalDiffusionTraining_and_Generation/training_recipes/combined_case4.yml'
    with open(cnf_config_path, 'r') as f:
        cnf_config = yaml.safe_load(f)
    
    # 设置 operator 和 noise
    print('set operator and noise')
    # 从合并配置文件中获取正确的batch_size参数



    
    operator = get_operator(device=device, name='inflow_prediction',
                           ckpt_path="/home/yif47/river/river-dl/temporal/Confield/Input/cnf_model/checkpoint/checkpoint_99.pt",
                           max_val_path="/home/yif47/river/river-dl/temporal/Confield/Input/data_scale/data_max.npy",
                           min_val_path="/home/yif47/river/river-dl/temporal/Confield/Input/data_scale/data_min.npy",
                           normalizer_params_path="/home/yif47/river/river-dl/temporal/Confield/Input/cnf_model/checkpoint/normalizer_params.pt",  
                           batch_size=cnf_config['cnf_batch_size'],
                           inflow_col_index=feature_info['inflow_index'],
                           cnf_config_path=cnf_config_path)
    
    noiser = get_noise(sigma=0.0, name='gaussian')
    
    # 条件化方法
    cond_method = get_conditioning_method(operator=operator, noiser=noiser, name='ps', scale=0.05)
    measurement_cond_fn = partial(cond_method.conditioning)
    
    # 采样器
    sampler = create_sampler(sampler='ddpm',
                             steps=training_config['steps'],
                             noise_schedule=training_config['noise_schedule'],
                             model_mean_type="epsilon",
                             model_var_type="fixed_large",
                             dynamic_threshold=False,
                             clip_denoised=True,
                             rescale_timesteps=False,
                             timestep_respacing="")
    
    sample_fn = partial(sampler.p_sample_loop, model=u_net_model, measurement_cond_fn=measurement_cond_fn)
    
    # 生成参数
    no_of_samples = 10
    latent_size = training_config['latent_length']
    
    print(f"生成参数:")
    print(f"样本数量: {no_of_samples}")
    
    # --- 直接进行单窗口推理 ---
    print("\n开始单窗口推理...")
    predicted_inflow = process_single_window(
        true_measurement, no_of_samples, 
        time_length, latent_size, device, sample_fn, operator, 
        cnf_config, feature_info
    )
    print("推理完成！")

    # --- 结果可视化 ---
    print("\n正在生成可视化结果...")

    # 将预测结果转换为numpy数组用于可视化
    pred_mean = torch.mean(predicted_inflow, dim=0).cpu().detach().numpy()
    pred_std = torch.std(predicted_inflow, dim=0).cpu().detach().numpy()
    true_inflow_np = true_inflow.cpu().detach().numpy()
    
    print(f"预测结果数据范围:")
    print(f"预测均值: [{pred_mean.min():.3f}, {pred_mean.max():.3f}]")
    print(f"真实值: [{true_inflow_np.min():.3f}, {true_inflow_np.max():.3f}]")

    # 获取流域信息
    all_basins = feature_info['all_basins']
    num_basins_to_plot = 5  # 选择要绘制的流域数量
    selected_basin_indices = np.linspace(0, len(all_basins) - 1, num_basins_to_plot, dtype=int)
    time_steps = np.arange(time_length)

    # 创建子图
    fig, axes = plt.subplots(num_basins_to_plot, 1, figsize=(15, 4 * num_basins_to_plot), sharex=True)
    if num_basins_to_plot == 1:
        axes = [axes] # 确保在只有一个子图时也能迭代

    for i, basin_idx in enumerate(selected_basin_indices):
        basin_name = all_basins[basin_idx]
        
        # 绘制真实inflow（原始空间）
        axes[i].plot(time_steps, true_inflow_np[:, basin_idx, 0], 'b-', linewidth=2, label='True Inflow')
        
        # 绘制预测的inflow均值（原始空间）
        axes[i].plot(time_steps, pred_mean[:, basin_idx, 0], 'r-', linewidth=2, label='Predicted Mean Inflow')
        
        # 绘制不确定性区间 (±2 sigma)（原始空间）
        upper_bound = pred_mean[:, basin_idx, 0] + 2 * pred_std[:, basin_idx, 0]
        lower_bound = pred_mean[:, basin_idx, 0] - 2 * pred_std[:, basin_idx, 0]
        axes[i].fill_between(time_steps, lower_bound, upper_bound, color='red', alpha=0.2, label='±2σ Uncertainty')
        
        axes[i].set_title(f"Inflow Prediction for Basin: {basin_name}")
        axes[i].set_ylabel("Inflow")
        axes[i].legend()
        axes[i].grid(True)

    axes[-1].set_xlabel("Time Step")
    plt.suptitle(f"Inflow Prediction for First {time_length} Timesteps", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # 调整布局以适应总标题
    plt.savefig('inflow_prediction_first_128_steps.png')
    plt.show()
    
    print("\n可视化图像已保存到 inflow_prediction_first_128_steps.png")

if __name__ == "__main__":
    main()