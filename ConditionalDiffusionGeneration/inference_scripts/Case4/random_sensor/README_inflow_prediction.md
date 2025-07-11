# 河流流量预测条件扩散推理框架

本框架用于测试无条件扩散模型对特定水库的inflow预测能力，通过条件扩散生成来预测缺失的inflow特征。测试策略是使用训练时间段最后10%的所有basin已知特征来预测目标basin的inflow。评估分为两个部分：目标basin的预测效果和所有basin的整体预测效果。

## 文件结构

```
├── All_Reservoirs_Combined_preprocessing_Confield_test.py  # 测试集预处理脚本
├── inference_inflow_prediction_simple.py                   # 简化的推理脚本
├── run_inflow_prediction.py                               # 便捷运行脚本
├── README_inflow_prediction.md                            # 本说明文档
└── input/                                                 # 输入文件目录
    ├── diff_model/                                        # 训练好的无条件扩散模型
    │   └── ema_model.pt
    ├── cnf_model/                                         # 训练好的CNF模型
    │   ├── checkpoint.pt
    │   └── normalizer.pt
    └── data_scale/                                        # 数据缩放参数
        ├── data_max.npy
        └── data_min.npy
```

## 使用步骤

### 1. 准备测试集数据

首先运行测试集预处理脚本：

```bash
cd datasets/colorado/
python All_Reservoirs_Combined_preprocessing_Confield_test.py
```

这将生成以下文件：
- `test_coords.npy` - 测试集坐标信息（包含所有basin）
- `test_known_features.npy` - 已知特征数据（排除inflow，包含所有basin）
- `test_true_inflow.npy` - 真实inflow值（用于评估，包含所有basin）
- `test_feature_info.npy` - 特征信息字典

### 2. 运行推理

#### 方法1: 直接运行推理脚本
```bash
cd Confield/ConditionalDiffusionGeneration/inference_scripts/Case4/random_sensor/
python inference_inflow_prediction_simple.py
```

#### 方法2: 使用便捷运行脚本（推荐）
```bash
# 默认预测EBR
python run_inflow_prediction.py

# 预测指定basin
python run_inflow_prediction.py BSR
python run_inflow_prediction.py CAU
python run_inflow_prediction.py DCR
# 等等...
```

## 核心概念

### InflowPredictionOperator

这是新定义的操作员类，继承自`NonLinearOperator`，用于实现条件扩散推理：

```python
@register_operator(name='inflow_prediction')
class InflowPredictionOperator(NonLinearOperator):
    def __init__(self, device, ckpt_path, max_val_path, min_val_path, 
                 normalizer_params_path, batch_size, inflow_col_index):
        # 初始化CNF模型和相关参数
        # ...
    
    def forward(self, data, **kwargs):
        # 1. 将latent向量解码为完整的物理特征场
        # 2. 从完整特征中排除inflow，只返回已知特征
        # 这样实现了条件约束：y = F(x)，其中F是排除inflow的操作
```

### 条件扩散过程

1. **输入**: 所有basin的已知特征数据（排除inflow）
2. **目标**: 生成完整的特征场，包括缺失的inflow
3. **约束**: 生成的已知特征必须与输入一致
4. **输出**: 预测的inflow序列（重点关注目标basin）

### 评估策略

评估分为两个部分：

1. **目标basin评估**: 专门评估指定basin的预测效果
   - RMSE、MAE、相关系数
   - 预测vs真实值对比图
   - 误差时间序列图

2. **所有basin评估**: 评估整体预测效果
   - 所有basin的平均RMSE、MAE、相关系数
   - 代表性basin的预测对比图
   - 平均误差时间序列图

### 数据流程

```
所有basin的已知特征 (8个特征，排除inflow)
    ↓
条件扩散模型
    ↓
完整特征场 (9个特征，包括inflow)
    ↓
提取目标basin的inflow预测结果
    ↓
与真实值比较评估
    ├── 目标basin评估
    └── 所有basin评估
```

## 特征说明

根据您的预处理脚本，特征列定义为：

```python
FEATURE_COLUMNS = [
    'precipitation', 'temperature', 'daylight_duration_s', 
    'solar_radiation_W_m2', 'snow_water_equivalent_kg_m2', 
    'temp_max_C', 'temp_min_C', 'vapor_pressure_Pa', 'inflow'
]
```

- **已知特征**: 前8个特征（索引0-7）
- **待预测特征**: inflow（索引8）

## 输出结果

推理完成后会生成：

1. **inflow_prediction_results.npy** - 包含所有预测结果的字典
2. **inflow_prediction_results.png** - 可视化图表
3. **控制台输出** - 包含RMSE、MAE等评估指标

### 结果字典结构

```python
results = {
    'predicted_inflow': predicted_inflow,           # [samples, time, basin, 1]
    'predicted_target_inflow': predicted_target_inflow,  # [samples, time, 1, 1]
    'true_inflow': true_inflow,                    # [time, basin, 1]
    'true_target_inflow': true_target_inflow,      # [time, 1, 1]
    'pred_mean': pred_mean,                        # [time, basin, 1]
    'pred_std': pred_std,                          # [time, basin, 1]
    'pred_target_mean': pred_target_mean,          # [time, 1, 1]
    'pred_target_std': pred_target_std,            # [time, 1, 1]
    'rmse_all': rmse_all,                          # 标量
    'mae_all': mae_all,                            # 标量
    'corr_all': corr_all,                          # 标量
    'rmse_target': rmse_target,                    # 标量
    'mae_target': mae_target,                      # 标量
    'corr_target': corr_target,                    # 标量
    'target_basin': target_basin,                  # 字符串
    'target_basin_index': target_basin_index,      # 整数
    'feature_info': feature_info                   # 特征信息字典
}
```

## 关键修改点

### 1. 数据加载
- 原脚本：`true_measurement = torch.from_numpy(np.load(f'input/random_sensor/{no_of_sensors}/measures.npy'))`
- 新脚本：`true_measurement = torch.from_numpy(np.load('test_known_features.npy'))` (包含所有basin的已知特征)

### 2. 操作员选择
- 原脚本：`name='case4'`
- 新脚本：`name='inflow_prediction'`

### 3. 结果提取
- 原脚本：解码为3D物理场
- 新脚本：从完整特征场中提取inflow列，重点关注EBR的预测结果

### 4. 测试策略
- 使用训练时间段最后10%的所有basin数据作为测试集
- 利用所有basin的已知特征来预测目标basin的inflow
- 评估分为两个部分：目标basin和所有basin的整体效果
- 支持灵活切换目标basin进行预测

## 注意事项

1. **模型兼容性**: 确保CNF模型的输出维度与特征数量匹配
2. **数据格式**: 测试集数据格式必须与训练集一致
3. **内存需求**: 推理过程需要足够的GPU内存
4. **路径设置**: 确保所有模型文件路径正确

## 故障排除

### 常见问题

1. **ImportError**: 检查Python路径设置
2. **CUDA内存不足**: 减少batch_size或样本数量
3. **文件不存在**: 确认所有输入文件路径正确
4. **维度不匹配**: 检查CNF模型输出维度与特征数量

### 调试建议

1. 先运行预处理脚本确认数据格式
2. 检查模型文件是否存在且可加载
3. 使用较小的样本数量进行测试
4. 查看控制台输出的详细错误信息 