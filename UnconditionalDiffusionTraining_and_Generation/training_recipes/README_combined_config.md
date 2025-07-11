# 合并配置文件使用说明

## 概述

`combined_case4.yml` 是一个合并的配置文件，包含了 CNF (Conditional Neural Field) 和 Diffusion 两个训练阶段的所有参数。

## 关键参数一致性

### 必须保持一致的参数：

1. **`hidden_size` (CNF) = `latent_length` (Diffusion)**
   - CNF 输出的潜在向量维度必须与 Diffusion 模型的输入维度匹配
   - 当前设置：128

2. **数据维度匹配**
   - CNF 输出的潜在向量 shape: `(time_steps, hidden_size)`
   - Diffusion 输入数据 shape: `(num_sequences, time_length, latent_length)`


## 使用方法

### 第一阶段：CNF 训练
```bash
cd Confield
python -m ConditionalNeuralField.scripts.train UnconditionalDiffusionTraining_and_Generation/training_recipes/combined_case4.yml
```

### 第二阶段：Diffusion 训练
```bash
cd Confield
CUDA_VISIBLE_DEVICES=1 python -m UnconditionalDiffusionTraining_and_Generation.scripts.train UnconditionalDiffusionTraining_and_Generation/training_recipes/combined_case4.yml
```

## 参数说明

### CNF 阶段参数
- `hidden_size: 128` - 潜在向量维度
- `cnf_batch_size: 64` - CNF 训练批次大小
- `cnf_epochs: 100` - CNF 训练轮数

### Diffusion 阶段参数
- `latent_length: 128` - 潜在向量维度（必须等于 CNF 的 hidden_size）
- `time_length: 128` - 时间序列切割长度
- `diffusion_batch_size: 16` - Diffusion 训练批次大小
- `image_size: 128` - 图像尺寸（通常等于 time_length）

## 数据流程

1. **CNF 训练**：原始时空数据 → 潜在向量 `(time_steps, hidden_size)`
2. **数据保存**：CNF 输出的潜在向量保存为 `train_data.npy`, `valid_data.npy`
3. **Diffusion 训练**：潜在向量 → 切割/重塑 → `(num_sequences, 1, time_length, latent_length)`

## 注意事项

1. **显存问题**：如果遇到显存不足，可以：
   - 降低 `diffusion_batch_size`
   - 降低 `num_channels`
   - 减少 `attention_resolutions` 的分辨率
   - 使用更小的 `channel_mult`

2. **参数调整**：修改任何参数时，确保相关的一致性参数也相应调整

3. **数据维度检查**：脚本会自动检查潜在向量维度是否匹配，如果不匹配会给出警告

## 当前配置优化建议

如果遇到显存问题，建议按以下顺序调整：

1. `diffusion_batch_size: 16` → `1` 或 `2`
2. `num_channels: 128` → `32` 或 `64`
3. `attention_resolutions: "64,32"` → `"32"` 或 `"16"`
4. `channel_mult: "1,1,2,2,4,4"` → `"1,2,4"`
5. `image_size: 128` → `64` 