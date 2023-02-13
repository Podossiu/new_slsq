import torch as t
import math
from .quantizer import Quantizer
import numpy as np


class BinaryStep(t.autograd.Function):
    @staticmethod 
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input>0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero_index = t.abs(input) > 1
        middle_index = (t.abs(input) <= 1) * (t.abs(input) > 0.4)
        additional = 2-4*t.abs(input)
        additional[zero_index] = 0.
        additional[middle_index] = 0.4
        return grad_input*additional

def soft_pruner(x, block_size, p):
    return x

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

def ste_w_quant(input, c, p, thd):
    eps = t.tensor([t.finfo(t.float32).eps], device = input.device)
    sign = input.sign()
    distance = c - p + eps
    s = distance / thd
    
    p_mask = (input.abs() < p).float()
    
    v_t = (input.abs()-p) * (1 - p_mask) * sign
    v_ste_t = v_t + p_mask * (input.abs() ** 2) / (p ** 2) * sign * (input.abs() - p)

    v_t = (v_t - v_ste_t).detach() + v_ste_t
    
    v_q = t.clamp(v_t / s, -thd, thd)
    #v_q = t.clamp(v_t / c, -thd, thd)
    v_q = (v_q.round() - v_q).detach() + v_q

    v_dq = v_q * s
    #v_dq = v_q * c
    
    return v_dq
 
def w_quant(input, c, p, thd):
    eps = t.tensor([t.finfo(t.float32).eps], device = input.device)
    sign = input.sign()
    distance = c - p + eps
    s = distance / thd
    
    p_mask = (input.abs() < p).float()

    v_t = (input.abs() - p) * (1. - p_mask) * sign
    v_q = t.clamp(v_t / (s+eps), -thd, thd)
    #v_q = t.clamp(v_t / c, -thd, thd)
    v_q = (v_q.round() - v_q).detach() + v_q
    v_dq = v_q * s
    #v_dq = v_q * c
    return v_dq

def grad_p_scale(c, p):
    x = c
    y = p
    scale = (1. - (p.detach() / c.detach() + 1e-12))
    x_grad = x * scale
    y_grad = p * scale
    return (x - x_grad).detach() + x_grad, (y - y_grad).detach() + y_grad

class ste_w_quan(t.autograd.Function):
    @staticmethod
    def forward(ctx, input, c, p, thd):
        eps = t.tensor([t.finfo(t.float32).eps], device = input.device)
        sign = input.sign()
        distance = c - p + eps
        s = distance / thd
        
        v_t = t.clamp(input.abs() - p, min = 0) * sign
        v_q = t.round(t.clamp(v_t / s, -thd, thd))
        v_dq = v_q * s

        ctx.save_for_backward(input,v_t, v_q, c, p, distance)
        ctx.thd = thd
        ctx.s = s
        return v_dq

    @staticmethod
    def backward(ctx, grad_output):
        input, v_t, v_q, c, p, distance = ctx.saved_tensors
        thd = ctx.thd
        s = ctx.s
        sign = input.sign()
        i_mask = (input.abs() <= c).float() * (input.abs() >= p).float()
        c_mask = (input.abs() > c).float()
        p_mask = (input.abs() < p).float()
        
        grad_c = (v_q / thd - v_t / distance) * i_mask + c_mask * sign
        grad_p = -grad_c - sign * i_mask

        grad_p = grad_p - p_mask * (input.abs() - p) * (input.abs() ** 2) * sign / (p ** 3) * 2

        grad_c = (grad_c * grad_output.clone()).sum().reshape(c.shape)
        grad_p = (grad_p * grad_output.clone()).sum().reshape(p.shape)
        
        grad_input = (i_mask + p_mask * (input.abs() -p) * 2 * input / (p ** 2) * sign) * grad_output.clone()

        return grad_input, grad_c, grad_p, None

class w_quan(t.autograd.Function):
    @staticmethod
    def forward(ctx, input, c, p, thd):
        eps = t.tensor([t.finfo(t.float32).eps], device = input.device)
        sign = input.sign()
        s =  (c - p) / thd
        
        quant_x = (input.abs() - p) / (s + eps)

        quant_x = t.clamp(quant_x, 0, thd) * sign

        quant_x = (t.round(quant_x) - quant_x).detach() + quant_x
        
        quant_x = quant_x * s
        return quant_x

class duq_quan(t.autograd.Function):
    @staticmethod
    def forward(ctx, input, c, p, thd):
        eps = t.tensor([t.finfo(t.float32).eps], device = input.device)
        sign = input.sign()
        distance = c - p + eps
        s = thd / distance

        v_g = (input.abs() - p) / distance
        v_c = t.clamp(v_g, 0, 1) * sign
        v_bar = t.round(v_c * thd).div(thd)
        v_hat = (v_bar.abs() * distance + p) * v_bar.sign()
        ctx.save_for_backward(input,v_g, v_c, v_bar, c, p)
        ctx.s = s
        ctx.n = thd
        return v_hat

    @staticmethod
    def backward(ctx, grad_output):
        input, v_g, v_c, v_bar, c, p = ctx.saved_tensors
        s = ctx.s
        n = ctx.n
        ste_constant = 2 / ( 1 + 2 * s * p)

        c_mask = (input.abs() > c).float()
        i_mask = (input.abs() <= c).float() * (v_bar != 0).float()
        step_mask = (v_bar == 0).float()
        sign = input.sign()

        grad_c = (v_bar - v_c) * i_mask
        grad_p = -grad_c

        grad_c = grad_c + sign * c_mask
        grad_c = ((grad_c - input * v_g * s * step_mask * ste_constant) * grad_output).sum().reshape(c.shape)

        grad_p = ((grad_p + input * ( -1 + v_g) * s * step_mask * ste_constant) * grad_output).sum().reshape(p.shape)

        grad_input = 1. - step_mask
        grad_input = (grad_input + step_mask * s * input.abs() * ste_constant) * grad_output.clone()

        return grad_input, grad_c, grad_p, None

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad
class weight_quant(t.autograd.Function):
    @staticmethod
    def forward(ctx, input, c, p, thd):
        sign = input.sign()
        distance = c - p + 1e-12
        s = distance / thd
        
        v_g = (input.abs() - p) / s
  
        v_c = t.clamp(v_g, 0, thd) * sign

        v_q = t.round(v_c)

        v_dq = v_q * s
        
        ctx.save_for_backward(input, v_q, c, p, distance)
        ctx.s = s
        ctx.thd = thd
        return v_dq
    
    @staticmethod
    def backward(ctx, grad_output):
        input, v_q, c, p, distance = ctx.saved_tensors
        s = ctx.s
        thd = ctx.thd
        c_mask =(input.abs() > c).float()
        i_mask =(input.abs() <= c).float() * (input.abs() >= p).float()
        s_mask = (v_q == 0.).float()
        sign = input.sign()
        
        v_t = (input.abs() - p) / distance
        
        grad_c = (v_q / thd - v_t * sign) * i_mask
        grad_p = -grad_c - sign * i_mask

        grad_c = grad_c + sign * c_mask
        grad_p = grad_p - sign * c_mask
        
        grad_c = (grad_c * grad_output.clone()).sum()
        grad_p = (grad_p * grad_output.clone()).sum()
        grad_input = grad_output.clone()
        #ste_constant = 2 * s / (2 * p + s)
        
        #grad_p = ((grad_p + s_mask * ste_constant * v_t * (-thd + 0.5) * sign) * grad_output.clone()).sum().reshape(p.shape)
        #grad_c = ((grad_c - s_mask * ste_constant * v_t * sign) * grad_output.clone()).sum().reshape(c.shape)

        #grad_input = ((1. - s_mask) + s_mask * thd * v_t * ste_constant) * grad_output.clone()
        return grad_input, grad_c, grad_p, None

class SLsqQuan(Quantizer):
    def __init__(self, bit, per_channel=False, symmetric=False, all_positive=False, hard_pruning=False, block_size=4, 
                temperature=1e-3, ste=False, z_param=False):
        super().__init__(bit)

        self.thd_neg = -2 ** (bit - 1) + 1
        self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.p = t.nn.Parameter(t.zeros(1))
        self.c = t.nn.Parameter(t.ones(1))
        self.block_size = block_size
        self.hard_pruning = hard_pruning
        self.temperature = t.tensor([temperature])
        self.register_buffer('eps', t.tensor([t.finfo(t.float32).eps]))
        
        self.gamma = t.tensor(t.ones(1))
        self.ste = ste
        self.init_mode = False
        self.z = t.nn.Parameter(t.zeros(1))
        self.z_param = z_param
        
        if ste:
            self.weight_quant = ste_w_quant
        else:
            self.weight_quant = w_quant
        self.weight_pruner = BinaryStep.apply
    def calculate_block_sparsity(self, x):
        with t.no_grad():
            if len(x.shape) == 4:
                co, ci, kh, kw = x.shape
                x_reshape = x.reshape(co // self.block_size, self.block_size, ci, kh, kw).detach()
                if self.z_param:
                    score = (x_reshape.abs().mean(dim = 1, keepdim = True) - (self.p + self.z)).detach()
                else:
                    score = (x_reshape.abs().mean(dim = 1, keepdim = True)) - (self.p).detach()
                hard_mask = (score > 0).float().detach()
            
            elif len(x.shape) == 2:
                co, ci = x.shape

                x_reshape = x.reshape(co // self.block_size, self.block_size, ci)
                if self.z_param:
                    score = x_reshape.abs().mean(dim = 1, keepdim = True) - (self.p + self.z)
                else:
                    score = x_reshape.abs().mean(dim = 1, keepdim = True) - (self.p)
                hard_mask = (score > 0).float().detach()
        return hard_mask.sum(), hard_mask.numel()

    def soft_pruner(self, x, p, z):
        if len(x.shape) == 4:
            co, ci, kh, kw = x.shape
            
            x_reshape = x.reshape(co // self.block_size, self.block_size, ci, kh, kw)
            if self.z_param:
                score = x_reshape.abs().mean(dim = 1,keepdim = True) - (p + z)
            else:
                score = x_reshape.abs().mean(dim = 1,keepdim = True) - p
            #score = score / (score.std().detach()) * 1e-3
            if not self.hard_pruning:
                temperature = t.tensor([1e-3], device = score.device)
                #_soft_mask = t.sigmoid(score/temperature)
                _soft_mask = self.weight_pruner(score)
                _soft_mask = _soft_mask.repeat(1, self.block_size, 1, 1, 1).reshape(co, ci, kh, kw)
                return _soft_mask, temperature
            hard_mask = (score > 0).float()
            hard_mask = hard_mask.repeat(1, self.block_size, 1, 1, 1).reshape(co,ci,kh,kw)

        elif len(x.shape) == 2:
            co, ci = x.shape
            x_reshape = x.reshape(co // self.block_size, self.block_size, ci)
            if self.z_param:
                score = x_reshape.abs().mean(dim = 1,keepdim = True) - (p + z)
            else:
                score = x_reshape.abs().mean(dim = 1,keepdim = True) - p
            #score = score / (score.std().detach()) * 1e-3
            if not self.hard_pruning:
                temperature = t.tensor([1e-3], device = score.device)
                #_soft_mask = t.sigmoid(score/temperature)
                _soft_mask = self.weight_pruner(score)
                _soft_mask = _soft_mask.repeat(1, self.block_size, 1).reshape(co, ci)
                return _soft_mask, temperature
            hard_mask = (score > 0).float()
            hard_mask = hard_mask.repeat(1, self.block_size, 1).reshape(co,ci)
        return hard_mask, None

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            s = x.detach().abs().mean(dim = list(range(1, x.dim())), keepdim = True) * 2 / (self.thd_pos ** 0.5)
            self.c = t.nn.Parameter(s * self.thd_pos)
        else:
            s = x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5)
            self.c.data = s.clone().detach() * self.thd_pos
            
    def forward(self, x):
        mask = None
        temperature = None
        if self.init_mode:
            self.init_mode = False
            return x, mask, temperature
        self.p.data.clamp_(min = 0.)
        self.c.data.clamp_(min = self.p.item() + self.eps.item())
        self.z.data.clamp_(min = -self.p.item())

        if (len(x.shape) == 4 and x.shape[1] != 1) or (len(x.shape) == 2):
            mask, temperature = self.soft_pruner(x, self.p, self.z)
            x = x * mask
        quant_x = self.weight_quant(x, self.c, self.p, self.thd_pos)
        return quant_x, mask, temperature

class pqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=True, per_channel=False, quant_mode = False, pruning_mode = False, block_size = 4, temperature = 1e-3, hard_pruning = False):
        super().__init__(bit)

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s = t.nn.Parameter(t.ones([]))
        self.init_mode = False
        self.quant_mode = quant_mode
        self.pruning_mode = pruning_mode

        self.p= t.nn.Parameter(t.zeros([]))
        self.soft_mask = None
        self.block_size = block_size
        self.hard_pruning = hard_pruning
        self.temperature = t.tensor([temperature])

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = t.nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
            self.s = t.nn.Parameter(t.zeros_like(self.s))
        else:
            self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))
            self.p = t.nn.Parameter(t.zeros_like(self.s))

    def soft_pruner(self, x, p):

        co, ci, kh, kw = x.shape
        x_reshape = x.reshape(co // self.block_size, self.block_size, ci, kh, kw)

        score = x_reshape.abs().mean(dim = 1,keepdim = True).detach() - p
        score = score / score.abs().max().detach()
        if not self.hard_pruning:
            _soft_mask = t.nn.functional.sigmoid(score/ self.temperature)
            self.soft_mask = _soft_mask
            self.soft_mask = self.soft_mask.repeat(1, self.block_size, 1, 1, 1).reshape(co,ci,kh,kw)
            return self.soft_mask
        
        hard_mask = (score > 0).float()
        hard_mask = hard_mask.repeat(1, self.block_size, 1, 1, 1).reshape(co,ci,kh,kw)
        return hard_mask

    def forward(self, x):
        self.p.data.clamp_(min = 0.)
        x_r = x
        if self.pruning_mode:
            if (len(x.shape) == 4 and x.shape[1] != 1):
                mask = self.soft_pruner(x, self.p)
                x_r = x_r * mask
        if self.quant_mode:
            if self.init_mode:
                self.init_from(x)
                self.init_mode = False
                return x
            if self.per_channel:
                #s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
                s_grad_scale = 1.0 / (x.numel() ** 0.5)
            else:
                #s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
                s_grad_scale = 1.0 / (x.numel() ** 0.5)

            s_scale = grad_scale(self.s, s_grad_scale)

            x_r = x_r / s_scale
            x_r = t.clamp(x_r, self.thd_neg, self.thd_pos)
            x_r = round_pass(x_r)
            x_r = x_r * s_scale
        return x_r


class LsqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=True, per_channel=False):
        super().__init__(bit)

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** (bit+1) -1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1
        self.register_buffer('eps', t.tensor([t.finfo(t.float32).eps]))
        self.per_channel = per_channel
        self.s = t.nn.Parameter(t.ones([]))
        self.init_mode = False
        self.beta = t.nn.Parameter(t.zeros([]))

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = t.nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
        else:
            x = (x - x.min()).detach()
            self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))
            #self.s = t.nn.Parameter(x.detach().abs().max() / self.thd_pos)
    def kl_divergence_activation(self, x):
        scaled_weight = x.mul(1.0).reshape([-1, 1, 1, 1])
        scaled_weight_npy = scaled_weight.detach().cpu().numpy()
        n = self.thd_pos
        scaled_weight_npy = scaled_weight_npy - np.min(scaled_weight_npy)
        mn = 0
        mx = np.abs(scaled_weight_npy).max()
        if np.isclose(mx, 0.0):
            return 0.0
        hist, bin_edges = np.histogram(np.abs(scaled_weight_npy), bins='sqrt', range = (mn, mx), density = True)
        hist = hist / np.sum(hist)
        cumsum = np.cumsum(hist)
        threshold = []
        scaling_factor = []
        d = []
        if n + 1 > len(bin_edges) - 1:
            th_layer_out = bin_edges[-1]
            return float(th_layer_out) / n

        for i in range(n + 1, len(bin_edges), 1):
            threshold_tmp = (i + 0.5) * (bin_edges[1] - bin_edges[0])
            threshold = np.concatenate((threshold, [threshold_tmp]))
            
            scaling_factor_tmp = threshold_tmp / n
            scaling_factor = np.concatenate((scaling_factor, [scaling_factor_tmp]))
            p = np.copy(cumsum)
            p[(i-1):] = 1
            x = np.linspace(0.0, 1.0, n)
            xp = np.linspace(0.0, 1.0, i)
            fp = p[:i]
            p_interp = np.interp(x, xp, fp)

            x = np.linspace(0.0, 1.0, i)
            xp = np.linspace(0.0, 1.0, n)
            fp = p_interp

            q_interp = np.interp(x, xp, fp)
            q = np.copy(p)
            q[:i] = q_interp
            d_tmp = np.sum((cumsum - q) * np.log2(cumsum / q)) # Kullback-Leibler-J
            d = np.concatenate((d, [d_tmp]))
        
        #th_layer_out = threshold[np.argmin(d)]
        #threshold = float(th_layer_out)
        scaling_factor_out = scaling_factor[np.argmin(d)]
        scaling_factor = float(scaling_factor_out)
        return scaling_factor

    def forward(self, x):
        if self.init_mode:
            thd = self.kl_divergence_activation(x)
            self.s.data = t.tensor([thd], device = self.s.device)
            #self.init_from(x)
            self.beta.data = x.min().detach() - self.s * self.thd_neg
            #self.init_from(x)
            self.init_mode = False
            return x
        
        self.s.data.clamp_(min = self.eps.item())
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)
        beta_scale = grad_scale(self.beta, s_grad_scale)
        x = (x - beta_scale) / s_scale
        x = t.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scale + beta_scale
        return x
'''
class SLsqQuan(Quantizer):
    def __init__(self, bit, per_channel=False, symmetric = False, all_positive = False, hard_pruning = False, block_size = 4, temperature = 1e-3):
        super().__init__(bit)
        
        self.thd_neg = -2 ** (bit - 1) + 1
        self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.p= t.nn.Parameter(t.zeros(1))
        self.c = t.nn.Parameter(t.ones(1))
        self.weight_quantizer = weight_quant.apply
        self.soft_mask = None
        self.block_size = block_size
        self.mask_mean = 0.
        self.hard_pruning = hard_pruning
        self.temperature = temperature

    def soft_pruner(self, x, p):

        co, ci, kh, kw = x.shape
        x_reshape = x.reshape(co // self.block_size, self.block_size, ci, kh, kw)

        score = x_reshape.abs().mean(dim = 1,keepdim = True).detach() - p
        if not self.hard_pruning:
            _soft_mask = t.nn.functional.sigmoid(score/ self.temperature)
            self.soft_mask = _soft_mask
            _soft_mask = _soft_mask.repeat(1, self.block_size, 1, 1, 1).reshape(co,ci,kh,kw)
            return _soft_mask
        
        hard_mask = (score > 0).float()
        hard_mask = hard_mask.repeat(1, self.block_size, 1, 1, 1).reshape(co,ci,kh,kw)
        return hard_mask

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            s = x.detach().abs().mean(dim = list(range(1, x.dim())), keepdim = True) * 2 / (self.thd_pos ** 0.5)
            self.c = t.nn.Parameter(s * self.thd_pos)
            self.p = t.nn.Parameter(t.zeros_like(self.s))
        else:
            s = x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5)
            self.c = t.nn.Parameter(s.clone().detach() * self.thd_pos)
            self.p = t.nn.Parameter(t.zeros([]))
    
    def forward(self, x):
        self.p.data.clamp_(0.,self.c.data)
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        c_scale = grad_scale(self.c, s_grad_scale)
        p_scale = grad_scale(self.p, s_grad_scale)
        #c_scale = self.c
        #p_scale = self.p
        
        sign = x.sign()
        s = (c_scale - p_scale + 1e-12) / self.thd_pos
        
        quant_x = (x.abs() - p_scale) / s
  
        quant_x = t.clamp(quant_x, 0, self.thd_pos) * sign

        quant_x = (t.round(quant_x) - quant_x).detach() + quant_x

        quant_x = quant_x * s

        if (len(x.shape) == 4):
            mask = self.soft_pruner(x, p_scale)
            quant_x = quant_x * mask
        return quant_x
'''
if __name__ == "__main__":
    module = SLsqQuan(bit = 8)
    x = t.randn((100,4,1,1))
    module.init_from(x = x)
    for i in range(0, 1000, 1):
        module.p.data = t.tensor(i * 0.001, dtype=t.float32)
        print(i * 0.001)
        y, mask, temperature= module(x)
        print(mask.mean())
        (y.sum() + mask.mean() * 10).backward()
        #y.sum().backward()
        print(module.p.grad)
        

