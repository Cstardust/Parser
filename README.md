#### Experimental procedure

```cmd
step1: bash download.sh download plm checkpoint
step2: bash train_with_[bert/bge/electra/roberta].sh
```

#### Results

| Model            | LAS | UAS |
|------------------|-----------|-----------|
| Electra+Biaffine | 0.8321    | 0.8731    |
| Roberta+Biaffine | 0.8213    | 0.8320    |
| Bert+Biaffine    | 0.8019    | 0.8243    |
| Bge+Biaffine     | 0.8710    | 0.9012    |

v0之后都是弧0.8, 关系1.2.

v1: loss 加权, 弧0.8, 关系1.2.
v2: BiLSTM用作编码器.
v3: BiLSTM编码器, Adam优化器, plm_lr = 2e-4, drop_out 0.33, weight_decay 0.v11: train.conll不shuffle. 弧0.8, 关系1.2.
v11: train2.conll 不知道为啥效果很差. 猜测是不是模型不小心改了哪里. 再用train.conll训练对比一下. 
确定是因为优化器的原因, 改回来了.
v111: train.conll 训练对比. v111 batch_size改回32了. 重新训练一遍.

v11: train2.conll; batch_size = 32; 增添了val集且train:val:test为8:1:1; fix log uas las bug; fix twice bug
    采用早停: num_early_stop = 5
    这些都是之前没有的.
    我认为可能造成性能下降的就是早停 / 训练集变少 / 验证集作用
    其实我完全可以说是这么说，但是模型用v0版本的, 如果现在变差的话.