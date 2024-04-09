
from typing import *

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel

# 通常情况下，在创建 nn.Linear 层时不需要手动初始化权重矩阵和偏置项
# PyTorch会自动对它们进行初始化，默认正态分布。
class NonLinear(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 activation: Optional[Callable] = None, 
                 init_func: Optional[Callable] = None) -> None: 
        super(NonLinear, self).__init__()
        # y = wx + b
        # w为(in_features, out_features)矩阵
        # b为长度为out_features的列向量
        self._linear = nn.Linear(in_features, out_features)
        self._activation = activation

        print(f'NonLinear linear {self._linear}, {self._linear.weight.size()}, {self._linear.bias.size()} ; activation {self._activation}')
        # NonLinear linear Linear(in_features=768, out_features=800, bias=True), torch.Size([800, 768]), torch.Size([800]); activation LeakyReLU(negative_slope=0.1)
            # weight: tensor([[-0.0215,  0.0078, -0.0016,  ...,  0.0137, -0.0073, -0.0180],
            #         [ 0.0096,  0.0311, -0.0321,  ...,  0.0238,  0.0336,  0.0272],
            #         [ 0.0007, -0.0100, -0.0316,  ...,  0.0211,  0.0218,  0.0065],
            #         ...,
            #         [-0.0245,  0.0085,  0.0047,  ..., -0.0154, -0.0241, -0.0096],
            #         [ 0.0098, -0.0203, -0.0004,  ..., -0.0290,  0.0359,  0.0326],
            #         [-0.0228,  0.0151, -0.0317,  ...,  0.0040,  0.0359, -0.0360]],
            # bias:  tensor([ 7.5877e-03,  1.0182e-03,  9.2303e-03, -2.8242e-02,  2.5903e-02,...)
        self.reset_parameters(init_func=init_func)
    
    def reset_parameters(self, init_func: Optional[Callable] = None) -> None:
        if init_func:
            init_func(self._linear.weight)

    # 前向传播: σ(f(x))
        # 即 activitation(wx + b)
        # x应该是个维度=out_features的列向量或者矩阵
    def forward(self, x):
        if self._activation:
            return self._activation(self._linear(x))
        return self._linear(x)


class Biaffine(nn.Module):
    def __init__(self, 
                 in1_features: int, 
                 in2_features: int, 
                 out_features: int,
                 init_func: Optional[Callable] = None) -> None:
        super(Biaffine, self).__init__()
        # 初始化第一个输入特征的维度
        self.in1_features = in1_features
        # 初始化第二个输入特征的维度
        self.in2_features = in2_features
        # 初始化输出特征维度
        self.out_features = out_features

        # 线性层的输入和输出维度
        self.linear_in_features = in1_features 
        self.linear_out_features = out_features * in2_features

        # 创建线性层, 将输入特征映射到输出特征
        # with bias default
        # 这个设置的是输入矩阵和输出矩阵的最后一个维度的长度, 而不是第一个维度大小
        self._linear = nn.Linear(in_features=self.linear_in_features,
                                out_features=self.linear_out_features)

        print(f'Biaffine linear {self._linear}, {self._linear.weight.size()}, {self._linear.bias.size()}')
        # weight: tensor([[ 0.0171,  0.0297, -0.0273,  ...,  0.0006, -0.0331,  0.0108],
                        # [ 0.0249, -0.0295, -0.0191,  ...,  0.0161, -0.0014, -0.0054],
                        # [-0.0140, -0.0195,  0.0044,  ...,  0.0114,  0.0144, -0.0202],
                        # ...,
                        # [ 0.0470, -0.0103,  0.0086,  ...,  0.0488,  0.0495,  0.0461]],
                        # requires_grad=True)
        # bias: tensor([ 0.0153,  0.0190,  0.0025, -0.0174,  0.0013,  0.0293, -0.0493, -0.0272...])
        self.reset_parameters(init_func=init_func)
        sys.exit()
        
    def reset_parameters(self, init_func: Optional[Callable] = None) -> None:
        if init_func:
            init_func(self._linear.weight)
            
    # 所以这个biaffine的前向传播里面，就是
        # 需要训练w和b: 1个线性层 self._linear = nn.Linear(in_features=self.linear_in_features,out_features=self.linear_out_features)
        # 和一个矩阵乘法
        # 所以这个biaffine里面，需要训练的参数就是linear中的w权重矩阵?
        # σ: 形状重塑
        # biaffine(input1, input2) = σ(σ(linear(input1)) * σ(input2))
    def forward(self, input1: torch.Tensor, input2: torch.Tensor):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        # input1 torch.Size([32, 160, 400])
        # input2 torch.Size([32, 160, 400])

        # 将第一个输入特征经过线性层映射，然后重塑为三维张量
        affine = self._linear(input1)
        # print(f'affine {affine.size()}')
        # affine torch.Size([32, 160, 400])

        affine = affine.view(batch_size, len1*self.out_features, dim2)

        # 对第二个输入特征进行转置操作
        input2 = torch.transpose(input2, 1, 2)
        
        # 对映射后的第一个输入特征与转置后的第二个输入特征进行批量矩阵乘法
        # 最后得到的biaffine: (batch_size, len2, len1*self.out_features)
        # 第一个维度表示批量中的样本数量，第二个维度表示输入序列中的长度，第三个维度表示输出序列中的长度?
        biaffine = torch.transpose(torch.bmm(affine, input2), 1, 2)

        # print("adasd", batch_size, len2, len1, self.out_features)
        
        # biaffine 重新排列结果为四维张量，并返回
        #  batch_size 表示批量大小，len2 表示输入序列的长度，len1 表示输出序列的长度，而 self.out_features 表示每个输出位置的特征维度?
        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)
        return biaffine