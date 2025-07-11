#Imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from src.script_util import create_model, create_gaussian_diffusion
from src.train_util import TrainLoop
from torch.utils.data import DataLoader, TensorDataset
from src.dist_util import setup_dist, dev
from src.logger import configure, log
from basicutility import ReadInput as ri

## Setup
torch.manual_seed(42)
np.random.seed(42)

inp = ri.basic_input(sys.argv[1])

setup_dist()
configure(dir=inp.diffusion_checkpoint_path, format_strs=["stdout","log","tensorboard_new"])

## HyperParams (Change according to the case)
batch_size = inp.diffusion_batch_size
test_batch_size = inp.diffusion_test_batch_size

image_size= inp.image_size
num_channels= inp.num_channels
num_res_blocks= inp.num_res_blocks
num_heads=inp.num_heads
num_head_channels= inp.num_head_channels
attention_resolutions= inp.attention_resolutions
channel_mult = inp.channel_mult
dims = getattr(inp, 'dims', 2)  # 读取dims参数，默认为2

steps= inp.steps
noise_schedule= inp.noise_schedule

microbatch= inp.microbatch
lr = inp.diffusion_lr
ema_rate= inp.ema_rate
log_interval= inp.log_interval
save_interval= inp.save_interval
lr_anneal_steps= inp.lr_anneal_steps



## Data Preprocessing
train_data = np.load(inp.diffusion_train_data_path)
valid_data = np.load(inp.diffusion_valid_data_path)

print(f"Original data shapes - train: {train_data.shape}, valid: {valid_data.shape}")

max_val, min_val = np.max(train_data, keepdims=True), np.min(train_data, keepdims=True)
os.makedirs(inp.scale_path, exist_ok=True)
np.save(os.path.join(inp.scale_path, "data_max.npy"), max_val)
np.save(os.path.join(inp.scale_path, "data_min.npy"), min_val)

norm_train_data = -1 + (train_data - min_val)*2. / (max_val - min_val)
norm_valid_data = -1 + (valid_data - min_val)*2. / (max_val - min_val)

#sliding window
time_length = getattr(inp, 'time_length', norm_train_data.shape[0])
sliding_window = getattr(inp, 'sliding_window', False)
def create_windows(data, time_length):
    N = data.shape[0]
    if N < time_length:
        raise ValueError("数据长度小于窗口长度")
    return np.stack([data[i:i+time_length] for i in range(N - time_length + 1)], axis=0)

def split_sequences(data, time_length):
    N = data.shape[0]
    num_seq = N // time_length
    if num_seq == 0:
        raise ValueError("数据长度小于窗口长度")
    return data[:num_seq * time_length].reshape(num_seq, time_length, -1)

split_method = getattr(inp, 'split_method', 'sliding')  # 默认滑窗
window_size = getattr(inp, 'time_length', norm_train_data.shape[0])

if split_method == 'sliding':
    norm_train_data = create_windows(norm_train_data, time_length)
    norm_valid_data = create_windows(norm_valid_data, time_length)
elif split_method == 'split':
    norm_train_data = split_sequences(norm_train_data, time_length)
    norm_valid_data = split_sequences(norm_valid_data, time_length)
else:
    raise ValueError(f"未知的split_method: {split_method}")

# norm_train_data shape: (N, L)  N=256, L=256

norm_train_data = torch.tensor(norm_train_data[:, None, :], dtype=torch.float32)
norm_valid_data = torch.tensor(norm_valid_data[:, None, :], dtype=torch.float32)


dl_train = DataLoader(TensorDataset(norm_train_data), batch_size=batch_size, shuffle=False)
dl_valid = DataLoader(TensorDataset(norm_valid_data), batch_size=test_batch_size, shuffle=False)

def dl_iter(dl):
    while True:
        yield from dl 

## Unet Model
log("creating model and diffusion...")

unet_model = create_model(image_size=image_size,
                          num_channels= num_channels,
                          num_res_blocks= num_res_blocks,
                          num_heads=num_heads,
                          num_head_channels=num_head_channels,
                          attention_resolutions=attention_resolutions,
                          channel_mult=channel_mult,
                          dims=dims  # 使用配置中的dims参数
                        )

unet_model.to(dev())

## Gaussian Diffusion
diff_model = create_gaussian_diffusion(steps=steps,
                                       noise_schedule=noise_schedule
                                    )

## Training Loop
log("training...")

train_uncond_model = TrainLoop(
                                model=unet_model,
                                diffusion=diff_model,
                                train_data = dl_iter(dl_train),
                                valid_data=dl_iter(dl_valid),
                                batch_size= batch_size,
                                microbatch= microbatch,
                                lr = lr,
                                ema_rate=ema_rate,
                                log_interval=log_interval,
                                save_interval=save_interval,
                                lr_anneal_steps=lr_anneal_steps,
                                resume_checkpoint="")

train_uncond_model.run_loop()