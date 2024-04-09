
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




# class BLSTM(nn.Module):
#     """
#         Implementation of BLSTM Concatenation for sentiment classification task
#     """

#     def __init__(self, embeddings, input_dim, hidden_dim, num_layers, output_dim, max_len=40, dropout=0.5):
#         super(BLSTM, self).__init__()

#         self.emb = nn.Embedding(num_embeddings=embeddings.size(0),
#                                 embedding_dim=embeddings.size(1),
#                                 padding_idx=0)
#         self.emb.weight = nn.Parameter(embeddings)

#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim

#         # sen encoder
#         self.sen_len = max_len
#         self.sen_rnn = nn.LSTM(input_size=input_dim,
#                                hidden_size=hidden_dim,
#                                num_layers=num_layers,
#                                dropout=dropout,
#                                batch_first=True,
#                                bidirectional=True)

#         self.output = nn.Linear(2 * self.hidden_dim, output_dim)

#     def bi_fetch(self, rnn_outs, seq_lengths, batch_size, max_len):
#         rnn_outs = rnn_outs.view(batch_size, max_len, 2, -1)

#         # (batch_size, max_len, 1, -1)
#         fw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([0])).cuda())
#         fw_out = fw_out.view(batch_size * max_len, -1)
#         bw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([1])).cuda())
#         bw_out = bw_out.view(batch_size * max_len, -1)

#         batch_range = Variable(torch.LongTensor(range(batch_size))).cuda() * max_len
#         batch_zeros = Variable(torch.zeros(batch_size).long()).cuda()

#         fw_index = batch_range + seq_lengths.view(batch_size) - 1
#         fw_out = torch.index_select(fw_out, 0, fw_index)  # (batch_size, hid)

#         bw_index = batch_range + batch_zeros
#         bw_out = torch.index_select(bw_out, 0, bw_index)

#         outs = torch.cat([fw_out, bw_out], dim=1)
#         return outs

#     def forward(self, sen_batch, sen_lengths, sen_mask_matrix):
#         """
#         :param sen_batch: (batch, sen_length), tensor for sentence sequence
#         :param sen_lengths:
#         :param sen_mask_matrix:
#         :return:
#         """

#         ''' Embedding Layer | Padding | Sequence_length 40'''
#         sen_batch = self.emb(sen_batch)

#         batch_size = len(sen_batch)

#         ''' Bi-LSTM Computation '''
#         sen_outs, _ = self.sen_rnn(sen_batch.view(batch_size, -1, self.input_dim))
#         sen_rnn = sen_outs.contiguous().view(batch_size, -1, 2 * self.hidden_dim)  # (batch, sen_len, 2*hid)

#         ''' Fetch the truly last hidden layer of both sides
#         '''
#         sentence_batch = self.bi_fetch(sen_rnn, sen_lengths, batch_size, self.sen_len)  # (batch_size, 2*hid)

#         representation = sentence_batch
#         out = self.output(representation)
#         out_prob = F.softmax(out.view(batch_size, -1))

#         return out_prob

class LSTM(nn.LSTM):
    """LSTM with DropConnect."""

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
        """Drop embeddings with scaling.
        https://github.com/tdozat/Parser-v2/blob/304c638aa780a5591648ef27060cfa7e4bee2bd0/parser/neural/models/nn.py#L50  # noqa
        """
        if not self.training or self.p == 0.0:
            return list(xs)
        with torch.no_grad():
            masks = torch.rand((len(xs),) + xs[0].size()[:-1], device=xs[0].device) >= self.p
            scale = masks.size(0) / torch.clamp(masks.sum(dim=0, keepdims=True), min=1.0)
            masks = (masks * scale)[..., None]
        return [x * mask for x, mask in zip(xs, masks)]

    def extra_repr(self) -> str:
        return f"p={self.p}"

class BiLSTMEncoder():
    def __init__(
        self,
        embeddings: Iterable[Union[torch.Tensor, Tuple[int, int]]],
        reduce_embeddings: Optional[Sequence[int]] = None,
        n_lstm_layers: int = 3,
        lstm_hidden_size: Optional[int] = None,
        embedding_dropout: float = 0.0,
        lstm_dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
    ):
        super().__init__()
        self.embeds = nn.ModuleList()
        for item in embeddings:
            if isinstance(item, tuple):
                size, dim = item
                emb = nn.Embedding(size, dim)
            else:
                emb = nn.Embedding.from_pretrained(item, freeze=False)
            self.embeds.append(emb)
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
