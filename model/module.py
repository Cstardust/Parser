
from typing import *

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel
import numpy as np

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
        # sys.exit()
        
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


from torch._C import _ImperativeEngine as ImperativeEngine


__all__ = ["VariableMeta", "Variable"]


class VariableMeta(type):
    def __instancecheck__(cls, other):
        return isinstance(other, torch.Tensor)


class Variable(torch._C._LegacyVariableBase, metaclass=VariableMeta):  # type: ignore[misc]
    _execution_engine = ImperativeEngine()


def orthonormal_initializer(output_size, input_size):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py
    """
    print(output_size, input_size)
    I = np.eye(output_size)
    lr = .1
    eps = .05 / (output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in range(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI ** 2 / 2)
            Q2 = Q ** 2
            Q -= lr * Q.dot(QTQmI) / (
                    np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        print('Orthogonal pretrainer loss: %.2e' % loss)
    else:
        print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return np.transpose(Q.astype(np.float32))


# 自定义的LSTM类，继承自PyTorch的nn.Module
class MyLSTM(nn.Module):

    """A module that runs multiple steps of LSTM."""

    # 初始化函数，定义了模型的结构
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, \
                 bidirectional=False, dropout_in=0, dropout_out=0):
        super(MyLSTM, self).__init__()
        
        # 初始化模型参数
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.num_directions = 2 if bidirectional else 1

        # 前向LSTM单元的列表
        self.fcells = []
        # 后向LSTM单元的列表（如果是双向LSTM的话）
        self.bcells = []
        
        # 初始化前向和后向LSTM单元
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            self.fcells.append(nn.LSTMCell(input_size=layer_input_size, hidden_size=hidden_size))
            if self.bidirectional:
                self.bcells.append(nn.LSTMCell(input_size=layer_input_size, hidden_size=hidden_size))

        # 参数列表，用于存储权重和偏置参数
        self._all_weights = []
        for layer in range(num_layers):
            # 前向LSTM单元参数
            layer_params = (self.fcells[layer].weight_ih, self.fcells[layer].weight_hh, \
                            self.fcells[layer].bias_ih, self.fcells[layer].bias_hh)
            suffix = ''
            param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
            param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
            param_names = [x.format(layer, suffix) for x in param_names]
            # 将参数添加到模型中
            for name, param in zip(param_names, layer_params):
                setattr(self, name, param)
            self._all_weights.append(param_names)

            # 如果是双向LSTM，则初始化后向LSTM单元参数
            if self.bidirectional:
                layer_params = (self.bcells[layer].weight_ih, self.bcells[layer].weight_hh, \
                                self.bcells[layer].bias_ih, self.bcells[layer].bias_hh)
                suffix = '_reverse'
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]
                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)

        # 初始化参数
        self.reset_parameters()

    # 重置参数的函数
    def reset_parameters(self):
        for layer in range(self.num_layers):
            # 如果是双向LSTM，初始化后向LSTM单元的参数
            if self.bidirectional:
                param_ih_name = 'weight_ih_l{}{}'.format(layer, '_reverse')
                param_hh_name = 'weight_hh_l{}{}'.format(layer, '_reverse')
                param_ih = self.__getattr__(param_ih_name)
                param_hh = self.__getattr__(param_hh_name)
                if layer == 0:
                    W = orthonormal_initializer(self.hidden_size, self.hidden_size + self.input_size)
                else:
                    W = orthonormal_initializer(self.hidden_size, self.hidden_size + 2 * self.hidden_size)
                W_h, W_x = W[:, :self.hidden_size], W[:, self.hidden_size:]
                param_ih.data.copy_(torch.from_numpy(np.concatenate([W_x] * 4, 0)))
                param_hh.data.copy_(torch.from_numpy(np.concatenate([W_h] * 4, 0)))
            else:
                # 如果是单向LSTM，初始化前向LSTM单元的参数
                param_ih_name = 'weight_ih_l{}{}'.format(layer, '')
                param_hh_name = 'weight_hh_l{}{}'.format(layer, '')
                param_ih = self.__getattr__(param_ih_name)
                param_hh = self.__getattr__(param_hh_name)
                if layer == 0:
                    W = orthonormal_initializer(self.hidden_size, self.hidden_size + self.input_size)
                else:
                    W = orthonormal_initializer(self.hidden_size, self.hidden_size + self.hidden_size)
                W_h, W_x = W[:, :self.hidden_size], W[:, self.hidden_size:]
                param_ih.data.copy_(torch.from_numpy(np.concatenate([W_x] * 4, 0)))
                param_hh.data.copy_(torch.from_numpy(np.concatenate([W_h] * 4, 0)))

        # 初始化偏置参数
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant(self.__getattr__(name), 0)

    # 静态方法，用于执行前向传播的RNN
    @staticmethod
    def _forward_rnn(cell, input, masks, initial, drop_masks):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in range(max_time):
            h_next, c_next = cell(input=input[time], hx=hx)
            h_next = h_next*masks[time] + initial[0]*(1-masks[time])
            c_next = c_next*masks[time] + initial[1]*(1-masks[time])
            output.append(h_next)
            if drop_masks is not None: h_next = h_next * drop_masks
            hx = (h_next, c_next)
        output = torch.stack(output, 0)
        return output, hx

    # 静态方法，用于执行前向传播的双向RNN
    @staticmethod
    def _forward_brnn(cell, input, masks, initial, drop_masks):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in reversed(range(max_time)):
            h_next, c_next = cell(input=input[time], hx=hx)
            h_next = h_next*masks[time] + initial[0]*(1-masks[time])
            c_next = c_next*masks[time] + initial[1]*(1-masks[time])
            output.append(h_next)
            if drop_masks is not None: h_next = h_next * drop_masks
            hx = (h_next, c_next)
        output.reverse()
        output = torch.stack(output, 0)
        return output, hx

    # 前向传播函数
    def forward(self, input, masks, initial=None):
        if self.batch_first:
            input = input.transpose(0, 1)
            masks = torch.unsqueeze(masks.transpose(0, 1), dim=2)
        max_time, batch_size, _ = input.size()
        masks = masks.expand(-1, -1, self.hidden_size)

        if initial is None:
            initial = Variable(input.data.new(batch_size, self.hidden_size).zero_())
            initial = (initial, initial)
        h_n = []
        c_n = []

        # 循环处理每一层的LSTM单元
        for layer in range(self.num_layers):
            max_time, batch_size, input_size = input.size()
            input_mask, hidden_mask = None, None
            if self.training:
                input_mask = input.data.new(batch_size, input_size).fill_(1 - self.dropout_in)
                input_mask = Variable(torch.bernoulli(input_mask), requires_grad=False)
                input_mask = input_mask / (1 - self.dropout_in)
                input_mask = torch.unsqueeze(input_mask, dim=2).expand(-1, -1, max_time).permute(2, 0, 1)
                input = input * input_mask

                hidden_mask = input.data.new(batch_size, self.hidden_size).fill_(1 - self.dropout_out)
                hidden_mask = Variable(torch.bernoulli(hidden_mask), requires_grad=False)
                hidden_mask = hidden_mask / (1 - self.dropout_out)

            layer_output, (layer_h_n, layer_c_n) = MyLSTM._forward_rnn(cell=self.fcells[layer], \
                input=input, masks=masks, initial=initial, drop_masks=hidden_mask)
            if self.bidirectional:
                blayer_output, (blayer_h_n, blayer_c_n) = MyLSTM._forward_brnn(cell=self.bcells[layer], \
                    input=input, masks=masks, initial=initial, drop_masks=hidden_mask)

            # 将前向和后向的隐藏状态拼接在一起
            h_n.append(torch.cat([layer_h_n, blayer_h_n], 1) if self.bidirectional else layer_h_n)
            c_n.append(torch.cat([layer_c_n, blayer_c_n], 1) if self.bidirectional else layer_c_n)
            input = torch.cat([layer_output, blayer_output], 2) if self.bidirectional else layer_output

        # 将每一层的隐藏状态和细胞状态整理成一个张量
        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        return input, (h_n, c_n)

    @property
    def out_size(self) -> int:
        return self.hidden_size * 2

"""
class LSTM(nn.LSTM):

    __constants__ = nn.LSTM.__constants__ + ["recurrent_dropout"]

    def __init__(self, *args, **kwargs):
        self.recurrent_dropout = float(kwargs.pop("recurrent_dropout", 0.0))
        super().__init__(*args, **kwargs)

    def forward(self, input, hx=None):
        if not self.training or self.recurrent_dropout == 0.0:
            if id(self._flat_weights) != self._flat_weights_id:
                self.flatten_parameters()
            return super().forward(input, hx)
        __flat_weights = self._flat_weights
        p = self.recurrent_dropout
        self._flat_weights = [
            F.dropout(w, p) if name.startswith("weight_hh_") else w
            for w, name in zip(__flat_weights, self._flat_weights_names)
        ]
        self.flatten_parameters()
        ret = super().forward(input, hx)
        self._flat_weights = __flat_weights
        return ret

    def flatten_parameters(self) -> None:
        super().flatten_parameters()
        self._flat_weights_id = id(self._flat_weights)

class EmbeddingDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p

    def forward(self, xs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if not self.training or self.p == 0.0:
            return list(xs)
        with torch.no_grad():
            masks = torch.rand((len(xs),) + xs[0].size()[:-1], device=xs[0].device) >= self.p
            scale = masks.size(0) / torch.clamp(masks.sum(dim=0, keepdims=True), min=1.0)
            masks = (masks * scale)[..., None]
        return [x * mask for x, mask in zip(xs, masks)]

    def extra_repr(self) -> str:
        return f"p={self.p}"

class BiLSTMEncoder(nn.Module):
    def __init__(
        self,
        # embeddings: Iterable[Union[torch.Tensor, Tuple[int, int]]],
        reduce_embeddings: Optional[Sequence[int]] = None,
        n_lstm_layers: int = 3,
        lstm_hidden_size: Optional[int] = None,
        embedding_dropout: float = 0.0,
        lstm_dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
    ):
        super().__init__()
        self.embeds = nn.ModuleList()
        # for item in embeddings:
        #     if isinstance(item, tuple):
        #         size, dim = item
        #         emb = nn.Embedding(size, dim)
        #     else:
        #         emb = nn.Embedding.from_pretrained(item, freeze=False)
        #     self.embeds.append(emb)
        self._reduce_embs = sorted(reduce_embeddings or [])

        embed_dims = [emb.weight.size(1) for emb in self.embeds]
        lstm_in_size = sum(embed_dims)
        if len(self._reduce_embs) > 1:
            lstm_in_size -= embed_dims[self._reduce_embs[0]] * (len(self._reduce_embs) - 1)
        if lstm_hidden_size is None:
            lstm_hidden_size = lstm_in_size
        self.bilstm = LSTM(
            lstm_in_size,
            lstm_hidden_size,
            n_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
            recurrent_dropout=recurrent_dropout,
        )
        self.embedding_dropout = EmbeddingDropout(embedding_dropout)
        self.lstm_dropout = nn.Dropout(lstm_dropout)
        self._hidden_size = lstm_hidden_size

    def freeze_embedding(self, index: Optional[Union[int, Iterable[int]]] = None) -> None:
        if index is None:
            index = range(len(self.embeds))
        elif isinstance(index, int):
            index = [index]
        for i in index:
            self.embeds[i].weight.requires_grad = False

    def forward(self, *input_ids: Sequence[torch.Tensor], return_dict=False) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(input_ids) != len(self.embeds):
            raise ValueError(f"exact {len(self.embeds)} types of sequences must be given")
        lengths = torch.tensor([x.size(0) for x in input_ids[0]])
        xs = [emb(torch.cat(ids_each, dim=0)) for emb, ids_each in zip(self.embeds, input_ids)]
        if len(self._reduce_embs) > 1:
            xs += [torch.sum(torch.stack([xs.pop(i) for i in reversed(self._reduce_embs)]), dim=0)]
        seq = self.lstm_dropout(torch.cat(self.embedding_dropout(xs), dim=-1))  # (B * n, d)

        if torch.all(lengths == lengths[0]):
            hs, _ = self.bilstm(seq.view(len(lengths), lengths[0], -1))
        else:
            seq = torch.split(seq, tuple(lengths), dim=0)
            seq = nn.utils.rnn.pack_sequence(seq, enforce_sorted=False)
            hs, _ = self.bilstm(seq)
            hs, _ = nn.utils.rnn.pad_packed_sequence(hs, batch_first=True)
        return self.lstm_dropout(hs), lengths

    @property
    def out_size(self) -> int:
        return self._hidden_size * 2
"""



# 初始化编码器
input_size = 100  # 输入大小
hidden_size = 128  # 隐藏层大小
num_layers = 2  # 层数
batch_first = True  # 输入数据的第一个维度是否为batch_size
encoder = MyLSTM(input_size, hidden_size, num_layers, batch_first, bidirectional=True)

# 准备输入数据
batch_size = 32
sequence_length = 10
input_data = torch.randn(batch_size, sequence_length, input_size)
masks = torch.ones(batch_size, sequence_length)  # 假设没有填充，所有时间步均有效

# 执行编码
output, (final_hidden_state, final_cell_state) = encoder(input_data, masks)

# 将输出转置为(batch_size, sequence_length, hidden_size * num_directions)
output = output.transpose(0, 1)

# 输出的形状为(batch_size, sequence_length, hidden_size * num_directions)
print("Encoder output shape:", output.shape)

# 最终隐藏状态和细胞状态的形状为(num_layers * num_directions, batch_size, hidden_size)
print("Final hidden state shape:", final_hidden_state.shape)
print("Final cell state shape:", final_cell_state.shape)