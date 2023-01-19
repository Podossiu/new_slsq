import torch as t
from .quantizer import *
class QuanConv2d(t.nn.Conv2d):
    def __init__(self, m: t.nn.Conv2d, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == t.nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn
        self.weight = t.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())
    def forward(self, x):
        out = ()
        if isinstance(self.quan_w_fn, SLsqQuan):
            quantized_weight, weight_mask, temperature = self.quan_w_fn(self.weight)
            #quantized_weight = self.quan_w_fn(self.weight)
        else:
            quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        out += (self._conv_forward(quantized_act, quantized_weight, self.bias),)
        if isinstance(self.quan_w_fn, SLsqQuan):
            out += (weight_mask, temperature)
        return out


class QuanLinear(t.nn.Linear):
    def __init__(self, m: t.nn.Linear, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == t.nn.Linear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.weight = t.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())

    def forward(self, x):
        out = ()
        if isinstance(self.quan_w_fn, lsq.SLsqQuan):
            quantized_weight, weight_mask, temperature = self.quan_w_fn(self.weight)
        else:
            quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        out += (t.nn.functional.linear(quantized_act, quantized_weight, self.bias),)
        if isinstance(self.quan_w_fn, SLsqQuan):
            out += (weight_mask, temperature)
        return out


QuanModuleMapping = {
    t.nn.Conv2d: QuanConv2d,
    t.nn.Linear: QuanLinear
}

