## Imports
import sys
import os

# 添加正确的路径到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
unconditional_dir = os.path.dirname(current_dir)  # UnconditionalDiffusionTraining_and_Generation
confield_root = os.path.dirname(unconditional_dir)  # Confield根目录

sys.path.append(unconditional_dir)  # 为了导入src模块
sys.path.append(confield_root)  # 为了导入ConditionalNeuralField模块

import torch
import numpy as np
from src.script_util import create_model, create_gaussian_diffusion
from ConditionalNeuralField.scripts.train import trainer
from basicutility import ReadInput as ri

## Setup
if torch.cuda.is_available():  
  dev = "cuda" 
else:  
  dev = "cpu"
  
device = torch.device(dev)  
torch.manual_seed(42)
np.random.seed(42)
inp = ri.basic_input(sys.argv[1])

## Hyperparams
test_batch_size = inp.test_batch_size # num of samples to generate
time_length = inp.time_length
latent_length = inp.latent_length
image_size= inp.image_size
num_channels= inp.num_channels
num_res_blocks= inp.num_res_blocks
num_heads= inp.num_heads
num_head_channels = inp.num_head_channels
attention_resolutions = inp.attention_resolutions
channel_mult = getattr(inp, 'channel_mult', None)  # 添加channel_mult支持
steps= inp.steps
noise_schedule= inp.noise_schedule

## Create model and diffusion
unet_model = create_model(image_size=image_size,
                          num_channels= num_channels,
                          num_res_blocks= num_res_blocks,
                          num_heads=num_heads,
                          num_head_channels=num_head_channels,
                          attention_resolutions=attention_resolutions,
                          channel_mult=channel_mult,
                          dims=1  # 添加dims=1，与训练时保持一致
                        )


unet_model.load_state_dict(torch.load(inp.ema_path, map_location=device))
unet_model.to(device)
unet_model.eval()

diff_model = create_gaussian_diffusion(steps=steps,
                                       noise_schedule=noise_schedule
                                    )

## Unconditional sample
sample_fn = diff_model.p_sample_loop

# 根据你的数据格式调整采样形状
# 你的数据是1D的，所以应该是 (batch_size, 1, latent_length)
gen_latents = sample_fn(unet_model, (test_batch_size, 1, latent_length))
print(f"Generated latents shape: {gen_latents.shape}")

## Denormalizing the latents (load the max and min of your training latent data)
print("Denormalizing latents...")
max_val, min_val = np.load(inp.max_val), np.load(inp.min_val)
max_val, min_val = torch.tensor(max_val, device=device), torch.tensor(min_val, device=device)
gen_latents = (gen_latents + 1)*(max_val - min_val)/2. + min_val

## Decode the latents
print("Loading CNF model for decoding...")
yaml = ri.basic_input(inp.cnf_case_file_path)
fptrainer = trainer(yaml, infer_mode=True)  # 使用infer_mode=True
fptrainer.load(-1, siren_only=True)
fptrainer.nf.to(device)
################
# ===== 在这里添加检查 =====
print("Checking train/eval mode difference...")
# 创建一个测试用的latent vector
test_latents = torch.randn(1, 256).to(device)  # 假设latent_length=256

# 测试train模式
fptrainer.nf.train()
with torch.no_grad():  # 即使在train模式也不需要梯度
    output1 = fptrainer.infer(None, test_latents)

# 测试eval模式  
fptrainer.nf.eval()
with torch.no_grad():
    output2 = fptrainer.infer(None, test_latents)

# 比较差异
diff = torch.abs(output1 - output2).max()
print(f"Train vs Eval difference: {diff.item():.8f}")

if diff > 1e-6:
    print("⚠️  模型在train/eval模式下有差异")
else:
    print("✅ 模型在train/eval模式下无差异")
# ===== 检查结束 =====












##################3
fptrainer.nf.eval()




coord = None # Define your query points here, by default None will result into using training query points




batch_size = 1 # if you are limited by your GPU Memory, please change the batch_size variable accordingly

n_samples = gen_latents.shape[0]
gen_fields = []

for sample_index in range(n_samples):
    print(f"Decoding sample {sample_index + 1}/{n_samples}")
    
    # 调整这部分，因为你的latents是(batch_size, 1, latent_length)
    current_latents = gen_latents[sample_index].squeeze(0)  # 移除channel维度
    
    # 如果latents需要分批处理
    if len(current_latents.shape) > 1:
        for i in range(0, current_latents.shape[0], batch_size):
            batch_latents = current_latents[i:i+batch_size]
            decoded = fptrainer.infer(coord, batch_latents)
            gen_fields.append(decoded.detach().cpu().numpy())
    else:
        # 单个latent vector
        decoded = fptrainer.infer(coord, current_latents.unsqueeze(0))
        gen_fields.append(decoded.detach().cpu().numpy())

gen_fields = np.concatenate(gen_fields, axis=0)
print(f"Final generated fields shape: {gen_fields.shape}")

# 保存结果
output_path = inp.save_path if hasattr(inp, 'save_path') else './generated_fields.npy'
np.save(output_path, gen_fields)
print(f"Results saved to: {output_path}")