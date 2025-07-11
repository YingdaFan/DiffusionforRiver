import numpy as np
import glob
import re
def load_elbow_flow(path):
    return np.load(f"{path}")[1:]

def load_channel_flow(
    path,
    t_start=0,
    t_end=1200,
    t_every=1,
):
    return np.load(f"{path}")[t_start:t_end:t_every]

def load_periodic_hill_flow(path):
    data = np.load(f"{path}")
    return data

def load_3d_flow(path):
    data = np.load(f"{path}")
    return data



def load_stream_flow(data_path, coor_path=None, **kwargs):
    """
    加载河流流量数据
    
    Args:
        data_path: 路径到 train_data.npy，形状为 (t, N, c)
        coor_path: 路径到 train_coords.npy，形状为 (N, 3) 
        **kwargs: 其他参数（保持与其他 load 函数一致）
    
    Returns:
        data: numpy array，形状为 (t, N, c)
    """
    
    print(f"Loading stream flow data from: {data_path}")
    
    # 加载主数据
    data = np.load(data_path)
    print(f"Loaded data shape: {data.shape}")
    
    # 如果提供了坐标路径，验证数据一致性
    if coor_path is not None:
        print(f"Loading coordinates from: {coor_path}")
        coords = np.load(coor_path)
        print(f"Loaded coords shape: {coords.shape}")
        
        # 验证数据一致性
        assert data.shape[1] == coords.shape[0], \
            f"Data points {data.shape[1]} != coord points {coords.shape[0]}"
        
        return data




