
from typing import *
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel
from model.module import NonLinear, Biaffine, BiLSTMEncoder
from utils import arc_rel_loss

class DepParser(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg

        mlp_arc_size:int = cfg.hidden_size
            # arc: 表示用于建模弧的隐藏层的大小。在依存句法分析中，弧通常指的是句子中单词之间的依存关系，即单词之间的连接或连接线。MLP 的一个部分用于对这些弧进行建模，以学习它们之间的复杂关系和特征。
        mlp_rel_size:int = cfg.hidden_size  
            # relation: 表示用于建模关系的隐藏层的大小。在依存句法分析中，关系通常指的是依存弧的标签或类型，即描述连接单词之间关系的类别或属性。MLP 的另一个部分用于对这些关系进行建模，以学习它们之间的特征表示和分类。
        dropout = cfg.dropout
##
        # # 加载预训练的ELECTRA模型作为编码器
        # self.encoder = AutoModel.from_pretrained(cfg.plm)
        # self.encoder.resize_token_embeddings(len(cfg.tokenizer))
        #         # 两个多层感知器（MLP）模型，用于学习依存句法分析中的弧（arc）和头（head）之间的关系
        # # weight是形状为(out_features, in_features)的矩阵
        # # bias是长度为out_features的列向量
        # # nn.LeakyReLU(0.1): 激活函数
        # # 一般MLP是个固定类别的分类器: si = W * ri+b
        # # 无法处理不定类别分类，所以作者提出先将ri重新encode为R(k×1)，也就是过一遍MLP：
        # # h_arc-dep = self.mlp_arc_dep(ri)
        # self.mlp_arc_dep = NonLinear(in_features=self.encoder.config.hidden_size, 
        #                              out_features=mlp_arc_size+mlp_rel_size, 
        #                              activation=nn.LeakyReLU(0.1))
        # # h_arc-head = self.mlp_arc_head(rj)
        # self.mlp_arc_head = NonLinear(in_features=self.encoder.config.hidden_size,      # 500
        #                               out_features=mlp_arc_size+mlp_rel_size,           # 500 + 500 = 1000
        #                               activation=nn.LeakyReLU(0.1))
        # # 两个MLP输出的向量的形状都是 (k * 1) (k是out_features)

        # # 这里两个MLP分别是dep专用和head专用。
        # # k的量级通常更小，压缩encode的另一个好处是去除多余的信息。
        # # 因为原始ri中含有预测依存弧标签的信息，这对预测head来讲是赘余的。

        # # 特征降维了之后，权值矩阵和偏置也必须做出调整。作者提出用两个矩阵连乘（两次仿射变换biaffine）输入向量:
        # # s_arc = H_arc-head * U1 * h_arc−dep + H_arc−head * u2 (bias)
        # # 其中矩阵H_arc-head是d个 token的特征 经过MLP二次encode 出来的特征向量的stack形式(就是d个h_arc-head叠加在一起)
        # # 上式维度变化是 (d × k)(k × k)(k × 1) + (d × k)(k × 1) = (d × 1)
        # # 结果是拿到了d个token的分数R(d×1)，同时分类器又不需要维护多个不同大小的权值矩阵（只需一个R(k×k)的矩阵和两个MLP），漂亮地实现了可变类别分类。
        # # 将bias放入U1, 得 (H_arc-head ⊙ 1) * U-arc * H_arc-head = S_arc
##
        # BiLSTM TEST
        self.plm = AutoModel.from_pretrained(cfg.plm)
        
        # 创建 BiLSTM 编码器
        self.encoder = BiLSTMEncoder(
            embeddings=[(len(cfg.tokenizer), self.plm.config.embedding_size)],  # 假设使用了词嵌入，需要传入词汇表大小和词嵌入维度
            lstm_hidden_size=cfg.hidden_size,
            embedding_dropout=0.33,
            lstm_dropout=0.33,
            recurrent_dropout=0.33
        )
        self.mlp_arc_dep = NonLinear(in_features=self.encoder.out_size, 
                                     out_features=mlp_arc_size+mlp_rel_size, 
                                     activation=nn.LeakyReLU(0.1))
        # h_arc-head = self.mlp_arc_head(rj)
        self.mlp_arc_head = NonLinear(in_features=self.encoder.out_size,      # 500
                                      out_features=mlp_arc_size+mlp_rel_size,           # 500 + 500 = 1000
                                      activation=nn.LeakyReLU(0.1))
        # BiLSTM TEST 


        # # MLP输出的特征向量拆分成多少份
        # self.total_num = int((mlp_arc_size + mlp_rel_size) / 100)
        # # MLP输出的特征向量中与弧（arc）相关的部分应该被拆分成多少份
        self.arc_num = int(mlp_arc_size / 100)
        # # MLP输出的特征向量中与关系（relation）相关的部分应该被拆分成多少份
        # self.rel_num = int(mlp_rel_size / 100)
        
        # 依存弧
        self.arc_biaffine = Biaffine(mlp_arc_size, mlp_arc_size, 1)
        # 关系
        self.rel_biaffine = Biaffine(mlp_rel_size, mlp_rel_size, cfg.num_labels)

        # Dropout 层会随机将输入张量中的一部分元素设置为零，这样可以减少神经网络对特定输入的依赖，从而增强模型的泛化能力。
        self.dropout = nn.Dropout(dropout)
     
    # 这个方法用于提取输入序列的特征。它接收一个输入字典 inputs，包含模型的输入数据。
    # 通过 encoder 对输入进行编码，获取输入序列的特征表示。
    # 提取特征时，移除了特殊token [CLS] 和 [SEP]，并返回 [CLS] 对应的特征向量、词级别的特征表示以及每个词的长度。 
    # feat里做的是啥处理? 这个encoder到底做了啥? 怎么编码成特征的?   
    def feat(self, inputs):
        # 对于每个篇章进行mask求和然后-2(首尾), 计算出每个篇章的有效长度. 
        length = torch.sum(inputs["attention_mask"], dim=-1) - 2 
        # 这个length张量就是当前输入的每个篇章的有效长度. tensor([28, 13,  7, 35, 12,  3,  8,  6,  8,  9,  9,  5,  4,  2, 14, 16,  8,  1, 6, 12, 13, 12, 12, 12, 14, 10, 16, 16, 21, 19, 10, 16], device = 'cuda:0' ""
        # length长度为32

        # 对之前的inputs进行编码
        feats, *_ = self.encoder(**inputs, return_dict=False)  
        # batch_size, seq_len (tokenized), plm_hidden_size
        # feats torch.Size([32, 160, 768])
        # batch_size: 本批次有多少个对话篇章.
        # seq_len: 一个对话篇章(word_lst)被编码成多少个词(值)?
        # hidden_size: 最后一个维度是词编码的维度? 把每一个词编码成768维度的向量.
        # 隐藏状态的维度，通常是神经网络的隐藏单元数量或者特征向量的维度?

        # remove [CLS] [SEP]
        word_cls = feats[:, :1]
        # :代表选取所有第一维度; :1代表选取第二维度的第一个元素; 第三维度元素全部选取
        # word_cls torch.Size([32, 1, 768])
        
        # 从feats的第dim维度的第start位置, 选取len个元素. (这里维度和位置都从0计数)
        # 我也不到为啥要这么选
        char_input = torch.narrow(feats, 1, 1, feats.size(1) - 2)
        # char_input torch.Size([32, 158, 768])

        return word_cls, char_input, length
    
    # 传入的heads和rels是真实的弧和关系
        # arc 弧: 在一个关系中, 谁是头部单词, 谁是尾部单词
        # label 关系: 就是对于每一条弧, 其所表示的依存关系类型
    # 真实的弧和关系是从输入文件里解析出来的
    def forward(self, inputs, offsets, heads, rels, masks, evaluate=False):  # inputs: batch_size, seq_len
        # inputs = [batch size, seq_len] 输入（token id）
        # offsets = [batch size, seq_len] 每个词的第一个字在整个句子中的索引位置
        # heads = [batch size, seq_len] # 每个词依赖的head词的第一个字的索引位置
        # rels = [batch size, seq_len] # 每一个词对应的依赖关系的编码
        # masks = [batch size, seq_len] # 主要用于loss计算中将pad位置的loss值屏蔽掉

        # cls shape = [bs,1,768]
        # char feat = [bs,158,768]
        # word_len = [bs,1]
        cls_feat, char_feat, word_len = self.feat(inputs)
            # cls_feat torch.Size([32, 1, 768])
            # char_feat torch.Size([32, 158, 768])
            # print('cls_feat', cls_feat.size())
            # print('char_feat', char_feat.size())

        # word_idx [bs,seq_len-1,768]
        # word_feat [bs,seq_len-1,768]
        word_idx = offsets.unsqueeze(-1).expand(-1, -1, char_feat.shape[-1])  # expand to the size of char feat
        word_feat = torch.gather(char_feat, dim=1, index=word_idx)  # embeddings of first char in each word
        # print('word feat size', word_feat.size())
        # word_feat size torch.Size([32, 159, 768])

        # 拼接上cls位置的向量，可能原因是PLM的原因，因为PLM的输入有[CLS]，为了保证句子表示的完整性
        # word_cls_feat [bs,seq_len,768]
        # feats [bs,seq_len,768]
        word_cls_feat = torch.cat([cls_feat, word_feat], dim=1)
        # print('word_cls_feat size', word_cls_feat.size())
        # word_cls_feat size torch.Size([32, 160, 768])

        feats = self.dropout(word_cls_feat)
        # print('feats size', feats.size())
        # feats size torch.Size([32, 160, 768]

        # all_dep [bs,seq_len,800]
        # all_head [bs,seq_len,800]
        # 通过MLP二次encode
        # h_arc-dep = self.mlp_arc_dep(ri)
        # h_arc-head = self.mlp_arc_head(rj)
        all_dep = self.dropout(self.mlp_arc_dep(feats))
        all_head = self.dropout(self.mlp_arc_head(feats))
        # all_dep_splits [8,bs,seq_len,100]
        # all_head_splits [8,bs,seq_len,100]
        # all_dep size torch.Size([32, 160, 800])
        # all_head size torch.Size([32, 160, 800])
        # print('all_dep size', all_dep.size())
        # print('all_head size', all_head.size())
        # print(f'diff {(all_dep != all_head).sum().item()}')     # 3931162
        # print(f'diff2 {(self.mlp_arc_dep(feats) != self.mlp_arc_head(feats)).sum().item()}')    # 4096000 不理解. 为啥输入和模型都相同, 输出还不同.

        all_dep_splits = torch.split(all_dep, split_size_or_sections=100, dim=2)
        all_head_splits = torch.split(all_head, split_size_or_sections=100, dim=2)
        # print(f'num of all_dep_splits, {len(all_dep_splits)} sizeof ele: {all_dep_splits[0].size()}')
        # num of all_dep_splits, 8 sizeof ele: torch.Size([32, 160, 100])
        # print(f'num of all_head_splits, {len(all_head_splits)} sizeof ele: {all_head_splits[0].size()}')
        # num of all_head_splits, 8 sizeof ele: torch.Size([32, 160, 100])

        # arc_dep [bs,seq_len,400]
        # arc_head [bs,seq_len,400]
        arc_dep = torch.cat(all_dep_splits[:self.arc_num], dim=2)   # 在all_dep_splits中的每个元素(向量torch.Size([32, 160, 100]))沿着第2个维度拼接起来. 拼接4个.
        arc_head = torch.cat(all_head_splits[:self.arc_num], dim=2)
        # print(f'self.arc_num {self.arc_num}')
        # self.arc_num = 4
        # arc_dep torch.Size([32, 160, 400])
        # arc_head torch.Size([32, 160, 400])
        # print('arc_dep', arc_dep.size())
        # print('arc_head', arc_head.size())

        # 模型推理出的弧的结果
            # 弧(Arc)预测：预测每个词与句子中其他词之间的依存关系，即句法树中的边。
            # 形状为 (batch_size, seq_len, seq_len)
            # arc_logit[i, j, k] 表示第 i 个句子中第 j 个词作为第 k 个词的父节点的概率得分
        # arc_logit = [bs,seq_len,seq_len]
        # 例子：“我 是 中国人” 在训练时我们并不知道“我”这个字的head是谁，所以就全猜，给每个pair对都赋值一个值:（我,我,value）,（我,是,value）,（我,中国人,value）
        # 对于“是”，“中国人”同样，所以总共需要9个值，这个句子的seq_len = 3所以用矩阵形式表示就是[3(seq_len)x3(seq_len)]，训练时再加上批维度，就是[bs,seq_len,seq_len]
        arc_logit = self.arc_biaffine(arc_dep, arc_head)   # batch_size, seq_len, seq_len
        # print('arc_logit', arc_logit.size())
        # arc_logit torch.Size([32, 160, 160, 1])
        arc_logit = arc_logit.squeeze(3)
        # print('arc_logit', arc_logit.size())
        # arc_logit torch.Size([32, 160, 160])
        # print(f'arc_logit size {arc_logit.size()}, arc_logit[i, j, k] 表示第 i 个句子中第 j 个词作为第 k 个词的父节点的概率得分')
        # print(arc_logit[0])

        # rel_dep [bs,seq_len,400]
        # rel_head [bs,seq_len,400]
        rel_dep = torch.cat(all_dep_splits[self.arc_num:], dim=2)
        rel_head = torch.cat(all_head_splits[self.arc_num:], dim=2)
        # print('rel_dep', rel_dep.size())
        # rel_dep torch.Size([32, 160, 400])
        # print('rel_head', rel_head.size())
        # rel_head torch.Size([32, 160, 400])

        # 模型推理出的关系的结果
            # 关系(Relation)预测: 对于每一条弧，预测其所表示的依存关系类型。
            # 形状为 (batch_size, seq_len, seq_len, num_rels)
            # rel_logit_cond[i, j, k, l] 表示第 i 个句子中第 j 个词和第 k 个词之间的关系标签 l 的条件概率得分。
        # rel_logit_cond [bs,seq_len,seq_len,num_rels]
        # 这个的解释同arc_logit的解释类似，例如：“我 是 中国人” 在训练时我不知道“我”这个词到底与那个头具有何种关系，那就全猜，就有:
        # (我，我，关系1，value),(我，我，关系2，value),(我，我，关系3，value),.....
        # (我，是，关系1，value),(我，是，关系2，value),(我，是，关系3，value),.....
        # (我，中国人”，关系1，value),(我，中国人”，关系2，value),(我，中国人”，关系3，value),.....
        # ......
        # “是”和“中国人”这两个词依次类推，加上批维度，用张量表示就是[bs,seq_len,seq_len,num_rels]
        rel_logit_cond = self.rel_biaffine(rel_dep, rel_head)  # batch_size, seq_len, seq_len, num_rels
        # print('rel_logit_cond', rel_logit_cond.size())
        # print(f'rel_logit_cond size {rel_logit_cond.size()}, rel_logit_cond[i, j, k, l] 表示第 i 个句子中第 j 个词和第 k 个词之间的关系标签')
        # print(rel_logit_cond[0])
        # rel_logit_cond torch.Size([32, 160, 160, 35])
        
        if evaluate:
            # 一个词可以作为多个词的head，但是一个词的head只能有一个，所以取Max就行，如果多对多这里需要取topk
            # 如果是评估阶段没有heads的标签，所以只能用predict的heads来进行依存关系的输入
            _, heads = arc_logit.max(2)  # change golden heads to the predicted
        
        # index = [batch_size, seq_len, seq_len, num_rels]
        # rel_logit = [batch_size, seq_len, num_rels]
        # 为什这里要按照index取，因为在上一步中我们已经知道每个词的head是谁了，就是知道“# (我，我，关系1，value),(我，我，关系2，value),(我，我，关系3，value),.....”
        # 这一行中“我”的真实head是谁，其他的pair对就可以丢弃掉，因此这里需要按照index去取值。
        index = heads.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, rel_logit_cond.shape[-1]) 
        # print('index', index.size())
        # index torch.Size([32, 160, 1, 35])
        rel_logit = torch.gather(rel_logit_cond, dim=2, index=index).squeeze(2)
        # print('rel_logit', rel_logit.size())
        # rel_logit torch.Size([32, 160, 35])

        # sys.exit()

        if evaluate:
            return arc_logit, rel_logit
        else:
            # arc_logit 是模型推理出的弧的结果
            # rel_logit 是模型推理出的关系的结果
            # heads 是传入给模型的真实的弧的结果
            # rels 是传入给模型的真实的关系
            loss = arc_rel_loss(arc_logit, rel_logit, heads, rels, masks)
            return arc_logit, rel_logit, loss