'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
import torch
from ConditionalNeuralField.cnf.inference_function import pass_through_model_batch
from ConditionalNeuralField.cnf.utils.normalize import Normalizer_ts
from ConditionalNeuralField.cnf.nf_networks import SIRENAutodecoder_film
import numpy as np
from einops import rearrange
# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper

class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)

class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

@register_operator(name='inpainting')
class InpaintingOperator(LinearOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, device):
        self.device = device
    
    def forward(self, data, **kwargs):
        try:
            return data.to(self.device) * kwargs.get('mask', None).to(self.device)
        except:
            raise ValueError("Require mask")
    
    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)
    
@register_operator(name='case2')
class Case2Operator(NonLinearOperator):
    def __init__(self, device,
                 ckpt_path,
                 max_val,
                 min_val,
                 coords,
                 batch_size):
        
        self.device = device
        self.coords = torch.tensor(coords, dtype = torch.float32, device=device)

        self.x_normalizer = Normalizer_ts(method = '-11',dim=0,
                                    params = [torch.tensor([1.,1.], device = device),
                                            torch.tensor([0.,0.], device = device)])
        self.y_normalizer = Normalizer_ts(method = '-11',dim=0, 
                                    params = [torch.tensor([[0.9617, 0.2666, 0.2869, 0.0290]], device = device), 
                                            torch.tensor([[-0.0051, -0.2073, -0.2619, -0.0419]], device = device)])
        cin_size, cout_size = 2,4
        self.model = SIRENAutodecoder_film(cin_size,256,cout_size,10,256)
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        self.max_val = torch.from_numpy(max_val).to(device) 
        self.min_val = torch.from_numpy(min_val).to(device)
        
        self.batch_size = batch_size

    def _unnorm(self, norm_data):
        return ((norm_data[:, 0, ...] + 1)*(self.max_val- self.min_val)/2 + self.min_val)[:, None, ...]

    def forward(self, data, **kwargs):
        mask = kwargs.get('mask', None)
        data_reshaped = rearrange(self._unnorm(data), "s c t l -> (s c t) l")
        phy_fields = pass_through_model_batch(self.coords, data_reshaped, self.model, 
                                              self.x_normalizer, self.y_normalizer,
                                              self.batch_size, self.device)
        return mask*phy_fields

@register_operator(name='case3')
class Case3Operator(NonLinearOperator):
    def __init__(self, device,
                 coords,
                 batch_size,
                 max_val,
                 min_val,
                 normalizer_params_path,
                 ckpt_path) -> None:
        
        self.device = device
        self.coords = torch.tensor(coords, dtype = torch.float32, device=device)
        
        params = torch.load(normalizer_params_path)
        x_ub,x_lb = params['x_normalizer_params']
        y_ub,y_lb = params['y_normalizer_params']
        cin_size, cout_size = 2,2
        self.x_normalizer = Normalizer_ts(method = '-11',dim=0, params = (x_ub,x_lb))
        self.y_normalizer = Normalizer_ts(method = '-11',dim=0, params = (y_ub[:cout_size],y_lb[:cout_size]))
        
        ckpt = torch.load(ckpt_path)
        self.model = SIRENAutodecoder_film(cin_size,256,cout_size,17,256)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        self.max_val = torch.from_numpy(max_val).to(device) 
        self.min_val = torch.from_numpy(min_val).to(device)
        
        self.batch_size = batch_size
        
    def _unnorm(self, norm_data):
        return ((norm_data[:, 0, ...] + 1)*(self.max_val- self.min_val)/2 + self.min_val)[:, None, ...] 
    
    def forward(self, data, **kwargs):
        data_reshaped = rearrange(self._unnorm(data), "s c t l -> (s c t) l")
        return pass_through_model_batch(self.coords, data_reshaped, self.model, 
                                        self.x_normalizer, self.y_normalizer,
                                        self.batch_size, self.device)
        
@register_operator(name='case3_gappy')
class Case3Operator_gappy(NonLinearOperator):
    def __init__(self, device,
                 coords,
                 batch_size,
                 max_val,
                 min_val,
                 normalizer_params_path,
                 ckpt_path
                 ) -> None:
        
        self.device = device
        self.coords = torch.tensor(coords, dtype = torch.float32, device=device)
        
        params = torch.load(normalizer_params_path)
        x_ub,x_lb = params['x_normalizer_params']
        y_ub,y_lb = params['y_normalizer_params']
        cin_size, cout_size = 2,2
        self.x_normalizer = Normalizer_ts(method = '-11',dim=0, params = (x_ub,x_lb))
        self.y_normalizer = Normalizer_ts(method = '-11',dim=0, params = (y_ub[:cout_size],y_lb[:cout_size]))
        
        ckpt = torch.load(ckpt_path)
        self.model = SIRENAutodecoder_film(cin_size,256,cout_size,17,256)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        self.max_val = torch.from_numpy(max_val).to(device) 
        self.min_val = torch.from_numpy(min_val).to(device)
        
        self.batch_size = batch_size
        
    def _unnorm(self, norm_data):
        return ((norm_data[:, 0, ...] + 1)*(self.max_val- self.min_val)/2 + self.min_val)[:, None, ...] 
    
    def forward(self, data, **kwargs):
        data_reshaped = rearrange(self._unnorm(data), "s c t l -> (s c t) l")
        out =  pass_through_model_batch(self.coords, data_reshaped, self.model, 
                                              self.batch_size, self.x_normalizer, self.y_normalizer,
                                              self.device)
        out[:, :10, 1] = 0.
        out[:,10:, 0] = 0.
        return out

@register_operator(name='case4')
class Case4Operator(NonLinearOperator):
    def __init__(self, device,
                 coords_path,
                 batch_size,
                 max_val_path,
                 min_val_path,
                 normalizer_params_path,
                 ckpt_path
                 ) -> None:
        
        self.device = device
        coords = np.load(coords_path)
        self.coords = torch.tensor(coords, dtype = torch.float32, device=device)
        
        params = torch.load(normalizer_params_path)
        x_uub, x_llb = params['x_normalizer_params']
        y_uub,_ = params['y_normalizer0u_params']
        _,y_llb = params['y_normalizer0l_params']
        cin_size, cout_size = 3,3
        self.x_normalizer = Normalizer_ts(method = '-11',dim=0, params = (x_uub,x_llb))  # only take out xyz 
        self.y_normalizer = Normalizer_ts(method = '-11',dim=0, params = (y_uub[:cout_size],y_llb[:cout_size]))
        
        ckpt = torch.load(ckpt_path)
        self.model = SIRENAutodecoder_film(cin_size,384,cout_size,15,384) 
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        max_val = np.load(max_val_path)
        min_val = np.load(min_val_path)
        self.max_val = torch.from_numpy(max_val).to(device) 
        self.min_val = torch.from_numpy(min_val).to(device)
        
        self.batch_size = batch_size
        
    def _unnorm(self, norm_data):
        return ((norm_data[:, 0, ...] + 1)*(self.max_val- self.min_val)/2 + self.min_val)[:, None, ...] 
    
    def forward(self, data, **kwargs):
        data_reshaped = rearrange(self._unnorm(data), "s c t l -> (s c t) l")
        return pass_through_model_batch(self.coords, data_reshaped, self.model, 
                                            self.x_normalizer, self.y_normalizer, self.batch_size,
                                              self.device)

@register_operator(name='inflow_prediction')
class InflowPredictionOperator(NonLinearOperator):
    def __init__(self, device, ckpt_path, max_val_path, min_val_path, normalizer_params_path, batch_size, inflow_col_index, 
                 coords_path=None, cnf_config_path=None, **cnf_model_params):
        self.device = device
        self.inflow_col_index = inflow_col_index
        
        # 加载CNF模型参数
        params = torch.load(normalizer_params_path)
        x_ub, x_lb = params['x_normalizer_params']
        y_ub, y_lb = params['y_normalizer_params']
        
        # 从配置文件或参数中获取CNF模型参数
        if cnf_config_path:
            import yaml
            with open(cnf_config_path, 'r') as f:
                cnf_config = yaml.safe_load(f)
            cin_size = cnf_config['NF']['in_coord_features']
            cout_size = cnf_config['NF']['out_features']
            hidden_size = cnf_config['hidden_size']
            num_hidden_layers = cnf_config['NF']['num_hidden_layers']
            hidden_features = cnf_config['NF']['hidden_features']
            
        else:
            # 使用传入的参数
            cin_size = cnf_model_params.get('in_coord_features', 3)
            cout_size = cnf_model_params.get('out_features', 9)
            hidden_size = cnf_model_params.get('hidden_size', 128)
            num_hidden_layers = cnf_model_params.get('num_hidden_layers', 10)
            hidden_features = cnf_model_params.get('hidden_features', 128)
        
        # 初始化正规化器 - 使用与训练时相同的参数结构
        self.x_normalizer = Normalizer_ts(method='-11', dim=0, params=(x_ub, x_lb))
        self.y_normalizer = Normalizer_ts(method='-11', dim=0, params=(y_ub, y_lb))
        
        # 加载CNF模型
        self.model = SIRENAutodecoder_film(cin_size, hidden_size, cout_size, num_hidden_layers, hidden_features)
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval().to(device)
        
        # 加载数据缩放参数
        self.max_val = torch.from_numpy(np.load(max_val_path)).to(device)
        self.min_val = torch.from_numpy(np.load(min_val_path)).to(device)
        
        # 加载坐标数据 - 在初始化时加载一次
        if coords_path:
            coords = np.load(coords_path)
            self.coords = torch.tensor(coords, dtype=torch.float32, device=device)
        else:
            # 默认路径
            coords = np.load('/home/yif47/river/river-dl/temporal/Confield/Input/test/test_coords.npy')
            self.coords = torch.tensor(coords, dtype=torch.float32, device=device)
        
        self.batch_size = batch_size

    def _unnorm(self, norm_data):
        # 将 [-1, 1] 范围的latent还原
        return ((norm_data[:, 0, ...] + 1) * (self.max_val - self.min_val) / 2 + self.min_val)[:, None, ...]

    # def forward(self, data, **kwargs):

    #     data_reshaped = rearrange(self._unnorm(data), "s c t l -> (s c t) l")

    #     # full_features 的形状应为 [时间步数, basin数量, 所有特征数量]
    #     full_features = pass_through_model_batch(self.coords, data_reshaped, self.model, 
    #                                              self.x_normalizer, self.y_normalizer,
    #                                              self.batch_size, self.device)

    #     # 创建索引来选择除了 'inflow' 之外的所有列
    #     all_indices = list(range(full_features.shape[-1]))
    #     known_indices = [i for i in all_indices if i != self.inflow_col_index]

    #     # 尝试重塑以匹配期望的输出
    #     # 这里需要根据实际情况调整
    #     return full_features[..., known_indices]
    
# In file: ConditionalDiffusionGeneration/src/guided_diffusion/measurements.py
# Inside class InflowPredictionOperator:

    def forward(self, data, **kwargs):
        # data shape is [s, c, t, l], for example [10, 1, 128, 128]
        s, c, t, l = data.shape
        # 我们可以加一个断言来确保通道维度总是1，因为后续逻辑依赖于此
        assert c == 1, f"Channel dimension 'c' is expected to be 1, but got {c}."

        # 1. UNNORM: 这一步没问题，它会保持 [s, 1, t, l] 的形状
        data_unnormed = self._unnorm(data)

        # 2. REARRANGE for DECODER: 【核心修正】
        #    使用 "s 1 t l -> (s t) l" 模式。
        #    这明确告诉 einops：输入的第二个维度必须是1，并且在输出中将其“压平”，不再存在。
        #    这样就解决了 'c' 未被处理的问题。
        data_reshaped = rearrange(data_unnormed, "s 1 t l -> (s t) l")

        # 3. DECODER CALL: 这一步是正确的。
        #    输入是 [(s*t), l], 输出是 [(s*t), num_basins, num_features]
        full_features = pass_through_model_batch(self.coords, data_reshaped, self.model,
                                                self.x_normalizer, self.y_normalizer,
                                                self.batch_size, self.device)

        # 4. REARRANGE to FINAL SHAPE: 这一步也是正确的。
        #    将解码后的场恢复为 4D 批次结构 [s, t, num_basins, num_features]
        full_features_rearranged = rearrange(full_features, "(s t) co c_out -> s t co c_out", s=s)

        # 5. SLICING: 这一步是正确的。
        known_indices = [i for i in range(full_features_rearranged.shape[-1]) if i != self.inflow_col_index]

        # 6. RETURN: 返回一个干净的4D张量，例如 [10, 128, 30, 8]
        return full_features_rearranged[..., known_indices]
        


# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma

@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson
       
        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

    
        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0
       
        # return data.clamp(low_clip, 1.0)