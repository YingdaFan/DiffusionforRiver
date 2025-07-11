#!/usr/bin/env python3
"""
河流流量预测的条件扩散推理脚本

本脚本用于测试模型对从未见过的水库（EBR）的泛化能力
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
    
    print("训练配置参数:")
    print(f"image_size: {training_config['image_size']}")
    print(f"num_channels: {training_config['num_channels']}")
    print(f"num_res_blocks: {training_config['num_res_blocks']}")
    print(f"num_heads: {training_config['num_heads']}")
    print(f"num_head_channels: {training_config['num_head_channels']}")
    print(f"attention_resolutions: {training_config['attention_resolutions']}")
    print(f"channel_mult: {training_config['channel_mult']}")
    print(f"dims: {training_config['dims']}")
    print(f"time_length: {training_config['time_length']}")
    print(f"latent_length: {training_config['latent_length']}")
    
    # 加载特征信息
    print("加载特征信息...")
    feature_info = np.load('/home/yif47/river/river-dl/temporal/datasets/colorado/test_feature_info.npy', allow_pickle=True).item()
    print("特征信息:")
    print(f"所有特征: {feature_info['all_features']}")
    print(f"已知特征: {feature_info['known_features']}")
    print(f"inflow索引: {feature_info['inflow_index']}")
    print(f"所有basin: {feature_info['all_basins']}")
    
    # 设置目标basin（可以根据需要修改）
    target_basin = None
    print(f"目标basin设置: {target_basin}")
    
    # 加载已知特征数据（排除inflow，包含所有basin）
    true_measurement = torch.from_numpy(np.load('/home/yif47/river/river-dl/temporal/datasets/colorado/test_known_features.npy')).to(device)
    print(f"已知特征数据形状: {true_measurement.shape}")
    
    # 加载真实inflow值用于评估（所有basin）
    true_inflow = torch.from_numpy(np.load('/home/yif47/river/river-dl/temporal/datasets/colorado/test_true_inflow.npy')).to(device)
    print(f"真实inflow数据形状: {true_inflow.shape}")
    
    # 目标basin相关变量初始化
    target_basin_index = None
    true_target_inflow = None
    if target_basin:
        target_basin_index = feature_info['all_basins'].index(target_basin)
        true_target_inflow = true_inflow[:, target_basin_index:target_basin_index+1, :]
        print(f"目标basin: {target_basin} (索引: {target_basin_index})")
        print(f"目标basin真实inflow数据形状: {true_target_inflow.shape}")
    else:
        print("未设置目标basin，将跳过目标basin特定测试")
    
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
                               model_path='/home/yif47/river/river-dl/temporal/Confield/Input/diff_model/checkpoint/ema_0.9999_010000.pt'
                            )
    u_net_model.to(device)
    u_net_model.eval()

    # 加载CNF配置文件
    cnf_config_path = '/home/yif47/river/river-dl/temporal/Confield/ConditionalNeuralField/training_recipes/stream.yml'
    with open(cnf_config_path, 'r') as f:
        cnf_config = yaml.safe_load(f)
    
    # 设置 operator 和 noise
    print('set operator and noise')
    operator = get_operator(device=device, name='inflow_prediction',
                           ckpt_path="/home/yif47/river/river-dl/temporal/Confield/Input/cnf_model/checkpoint/checkpoint_99.pt",
                           max_val_path="/home/yif47/river/river-dl/temporal/Confield/Input/data_scale/data_max.npy",
                           min_val_path="/home/yif47/river/river-dl/temporal/Confield/Input/data_scale/data_min.npy",
                           normalizer_params_path="/home/yif47/river/river-dl/temporal/Confield/Input/cnf_model/checkpoint/normalizer_params.pt",  
                           batch_size=cnf_config['batch_size'],
                           inflow_col_index=feature_info['inflow_index'],
                           cnf_config_path=cnf_config_path)
    
    noiser = get_noise(sigma=0.0, name='gaussian')
    
    # 条件化方法
    cond_method = get_conditioning_method(operator=operator, noiser=noiser, name='ps', scale=1.)
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
    time_length = training_config['time_length']
    latent_size = training_config['latent_length']
    
    print(f"生成参数:")
    print(f"样本数量: {no_of_samples}")
    print(f"潜在向量时间维度: {time_length}")
    print(f"潜在向量特征维度: {latent_size}")
    
    # 滑动窗口推理
    print("开始滑动窗口推理...")
    window_size = time_length
    overlap = window_size // 2
    stride = window_size - overlap
    total_length = true_measurement.shape[0]
    # 修正窗口数量计算，确保覆盖整个序列
    num_windows = (total_length - overlap - 1) // stride + 1
    
    print(f"总长度: {total_length}, 窗口大小: {window_size}, 重叠: {overlap}, 步长: {stride}, 窗口数量: {num_windows}")
    
    all_window_predictions = []
    
    for window_idx in range(num_windows):
        start_idx = window_idx * stride
        end_idx = start_idx + window_size
        
        # 处理最后一个窗口的边界
        if end_idx > total_length:
            end_idx = total_length
            start_idx = end_idx - window_size

        print(f"\n处理窗口 {window_idx + 1}/{num_windows} (时间步 {start_idx}-{end_idx})")
        
        window_measurement = true_measurement[start_idx:end_idx]
        
        window_predictions = process_single_window(
            window_measurement, no_of_samples, 
            time_length, latent_size, device, sample_fn, operator, 
            cnf_config, feature_info
        )
        
        all_window_predictions.append({
            'predictions': window_predictions,
            'start_idx': start_idx,
            'end_idx': end_idx
        })

    # 合并结果
    print("\n合并所有窗口的预测结果...")
    predicted_inflow = merge_window_predictions(all_window_predictions, total_length, no_of_samples, feature_info['all_basins'], device)
    
    # 后续评估和可视化代码...
    # (这部分代码是正确的，无需修改)

def process_single_window(measurement, no_of_samples, time_length, latent_size, 
                        device, sample_fn, operator, cnf_config, feature_info):
    """
    处理单个窗口的推理
    （最终修正版：使用循环逐个生成样本，以避免批处理错误）
    """
    
    measurement_device = measurement.to(device)
    
    samples = []
    for i in range(no_of_samples):
        print(f"  正在为窗口生成样本 {i+1}/{no_of_samples}...")
        
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
                         batch_size=cnf_config['batch_size'], device=device)
    gen_fields = rearrange(gen_fields, "(s t) co c -> s t co c", t=time_length)
    
    # 提取预测的inflow
    predicted_inflow = gen_fields[..., feature_info['inflow_index']:feature_info['inflow_index']+1]
    
    return predicted_inflow


def merge_window_predictions(all_window_predictions, total_length, no_of_samples, all_basins, device):
    """合并所有窗口的预测结果，使用重叠区域的平滑（加权平均）过渡"""
    
    num_basins = len(all_basins)
    full_predictions = torch.zeros(no_of_samples, total_length, num_basins, 1, device=device)
    full_weights = torch.zeros(total_length, device=device) # 权重对于所有样本和basin都是一样的

    # 创建一个线性过渡的权重窗口
    window_length = all_window_predictions[0]['end_idx'] - all_window_predictions[0]['start_idx']
    overlap = window_length // 2
    
    # 创建一个三角权重
    ramp_up = torch.linspace(0, 1, overlap, device=device)
    ramp_down = torch.linspace(1, 0, overlap, device=device)
    flat_part_len = window_length - 2 * overlap
    
    # 完整的三角权重窗口
    triangular_window = torch.cat([
        ramp_up,
        torch.ones(flat_part_len, device=device),
        ramp_down
    ])
    
    # 合并每个窗口的预测
    for window_data in all_window_predictions:
        predictions = window_data['predictions']  # [samples, window_time, basin, 1]
        start_idx = window_data['start_idx']
        end_idx = window_data['end_idx']
        
        # 将加权预测添加到完整序列
        current_window_len = end_idx - start_idx
        # Reshape权重以匹配预测的维度
        window_weights_reshaped = triangular_window[:current_window_len].view(1, current_window_len, 1, 1)

        full_predictions[:, start_idx:end_idx, :, :] += predictions * window_weights_reshaped
        full_weights[start_idx:end_idx] += triangular_window[:current_window_len]

    # 归一化（避免除零）
    # 在权重为0的地方，将其设置为1，以避免除以0。因为相应位置的prediction也为0，所以结果不变。
    full_weights[full_weights == 0] = 1.0
    full_predictions = full_predictions / full_weights.view(1, total_length, 1, 1)
    
    return full_predictions

if __name__ == "__main__":
    main() 