
from typing import *

from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from utils import to_cuda
import sys

class BasicTrainer():
    def __init__(self, 
                 model,             # 神经网络模型 DepParser
                 trainset_size,     # 训练集大小
                 loss_fn: Callable, # 
                 metrics_fn: Callable, 
                 logger,
                 config: Dict) -> None:
        self.loss_fn = loss_fn          # 损失函数 arc_rel_loss
        self.metrics_fn = metrics_fn    # 评估指标函数 uas_las
        
        self.logger = logger

        # model.named_parameters() : 返回模型所有{参数, 对应名称}的迭代器
            # 返回的应该有2个NonLinear的weight和2个Biaffine的weight?
        # for n, p in ...: n: 参数, p: 参数名称
        # Pre-trained Language Model
        plm_params = [p for n,p in model.named_parameters() if 'encoder' in n]
        head_params = [p for n,p in model.named_parameters() if 'encoder' not in n]

        # 需要给优化器提供网络的所有参数
        # 训练的时候就是通过这个Adam优化器来更新参数
        self.optim = AdamW([{'params': plm_params, 'lr':config.plm_lr}, 
                            {'params': head_params, 'lr':config.head_lr}], 
                            lr=config.plm_lr,
                            weight_decay=config.weight_decay
                          )
    
        # 计算训练步数和预热步数，用于学习率调度
        training_step = int(config.num_epochs * (trainset_size / config.batch_size))
        warmup_step = int(config.warmup_ratio * training_step)  
    
        # 学习率调度器：Linear Warmup
        # 用于调整learning rate
        self.optim_schedule = get_linear_schedule_with_warmup(optimizer=self.optim, 
                                                              num_warmup_steps=warmup_step, 
                                                              num_training_steps=training_step)
        # 初始化混合精度训练的 GradScaler 对象
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)
        
        self.print_every = int(config.print_every_ratio * trainset_size / config.batch_size)

        self.config = config

    def train(self, 
              model: nn.Module, 
              train_iter: DataLoader, 
              val_iter: DataLoader):
        # 设置为训练模式
        model.train()
        if self.config.cuda and torch.cuda.is_available():
            model.cuda()
            pass
        
        best_res = [0, 0, 0]
        early_stop_cnt = 0
        best_state_dict = None
        step = 0

        print(f'start {self.config.num_epochs} train')
        # 多轮训练
        for epoch in tqdm(range(self.config.num_epochs)):
            for batch in train_iter:
                # 一次train_iter迭代器会多次调用__getitem__, 将他们纵向拼接成一个矩阵 
                inputs, offsets, heads, rels, masks = batch
                        # inputs: 输入的句子拆分成词, 并得到的所有词的编码：
            # [ {[input_ids],[token_type_ids],[attention_mask]}, {[input_ids],[token_type_ids],[attention_mask]}]
            # "input_ids": 对于word_lst的编码.
            # "token_type_ids": 第一个句子和特殊符号的位置是0. 第二个句子位置是1.
            # "attention_mask": pad的位置是0, 其他位置是1
        # offsets: 包含了每个句子中单词的偏移量. [[tensor],[tensor],[tensor]]
        # heads: 依存关系中的头部单词索引. [[tensor],[tensor],[tensor]]
        # rels: 以当前单词作为尾部词语的依赖关系类型. [[tensor],[tensor],[tensor]]
        # masks: 每个句子中每个单词的掩码信息。掩码信息用于指示模型在处理输入时哪些单词是有效的，哪些是填充的。这个列表记录了每个单词的掩码值. [[tensor],[tensor],[tensor]]
                # 也就是说, 下面是多个篇章对话的输入  
        #  inputs {'input_ids': tensor([[ 101, 2644, 1962,  ...,    0,    0,    0],
        # [ 101, 1305, 3221,  ...,    0,    0,    0],
        # [ 101, 2644, 2575,  ...,    0,    0,    0],
        # ...,
        # [ 101, 6929, 6821,  ...,    0,    0,    0],
        # [ 101, 2769, 3221,  ...,    0,    0,    0],
        # [ 101,  683, 1447,  ...,    0,    0,    0]]), 'token_type_ids': tensor([[0, 0, 0,  ..., 0, 0, 0],
        # [0, 0, 0,  ..., 0, 0, 0],
        # [0, 0, 0,  ..., 0, 0, 0],
        # ...,
        # [0, 0, 0,  ..., 0, 0, 0],
        # [0, 0, 0,  ..., 0, 0, 0],
        # [0, 0, 0,  ..., 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],
        # [1, 1, 1,  ..., 0, 0, 0],
        # [1, 1, 1,  ..., 0, 0, 0],
        # ...,
        # [1, 1, 1,  ..., 0, 0, 0],
        # [1, 1, 1,  ..., 0, 0, 0],
        # [1, 1, 1,  ..., 0, 0, 0]])}
        # offsets tensor([[0, 1, 2,  ..., 0, 0, 0],
        #         [0, 2, 4,  ..., 0, 0, 0],
        #         [0, 1, 3,  ..., 0, 0, 0],
        #         ...,
        #         [0, 1, 3,  ..., 0, 0, 0],
        #         [0, 1, 2,  ..., 0, 0, 0],
        #         [0, 2, 3,  ..., 0, 0, 0]])
        # heads tensor([[ 0,  2,  4,  ...,  0,  0,  0],
        #         [ 0,  0,  3,  ...,  0,  0,  0],
        #         [ 0,  2,  4,  ...,  0,  0,  0],
        #         ...,
        #         [ 0,  3,  3,  ...,  0,  0,  0],
        #         [ 0,  2,  0,  ...,  0,  0,  0],
        #         [ 0, 10, 10,  ...,  0,  0,  0]])
        # rels tensor([[ 0,  4, 27,  ...,  0,  0,  0],
        #         [ 0,  0,  9,  ...,  0,  0,  0],
        #         [ 0,  4,  4,  ...,  0,  0,  0],
        #         ...,
        #         [ 0,  9,  9,  ...,  0,  0,  0],
        #         [ 0,  4,  0,  ...,  0,  0,  0],
        #         [ 0,  4,  9,  ...,  0,  0,  0]])
        # masks tensor([[0, 1, 1,  ..., 0, 0, 0],
        #         [0, 1, 1,  ..., 0, 0, 0],
        #         [0, 1, 1,  ..., 0, 0, 0],
        #         ...,
        #         [0, 1, 1,  ..., 0, 0, 0],
        #         [0, 1, 1,  ..., 0, 0, 0],
        #         [0, 1, 1,  ..., 0, 0, 0]])
                if self.config.cuda and torch.cuda.is_available():
                    inputs_cuda = {}
                    for key, value in inputs.items():
                        inputs_cuda[key] = value.cuda()
                    inputs = inputs_cuda
    
                    offsets, heads, rels, masks = to_cuda(data=(offsets, heads, rels, masks))
                
                # DepParser forward() 获取模型output和loss
                arc_logits, rel_logits, loss = model(inputs, offsets, heads, rels, masks)
                # 清空梯度
                self.optim.zero_grad()
                # 反向传播
                    # 如果配置为使用混合精度训练，则使用梯度缩放器缩放损失值
                if self.config.cuda and self.config.fp16:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optim)
                else:
                    loss.backward()
                # 裁剪梯度, 防止梯度爆炸
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=self.config.grad_clip)

                # 更新参数!!!!
                    # 根据是否使用混合精度训练来更新模型参数
                if self.config.fp16:
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    self.optim.step()

                # 更新学习率
                self.optim_schedule.step()
                # 计算 UAS 和 LAS
                metrics = self.metrics_fn(arc_logits, rel_logits, heads, rels, masks)

                if (step) % self.print_every == 0:
                    self.logger.info(f"--epoch {epoch}, step {step}, loss {loss}")
                    self.logger.info(f"  {metrics}")
    
                # 间隔一定step, 进行评估
                if val_iter is not None and self.config.eval_strategy == 'step' and (step + 1) % self.config.eval_every == 0:
                    avg_loss, uas, las = self.eval(model, val_iter)
                    res = [avg_loss, uas, las]
                    # 以LAS评判是否是最好结果
                    if las > best_res[2]:  # las
                        best_res = res
                        # {{训练参数名称, 张量}, {}, ...}
                        best_state_dict = model.state_dict()
                        early_stop_cnt = 0
                    else:
                        early_stop_cnt += 1
                    # 更新最好估计
                    self.logger.info("--Best Evaluation: ")
                    self.logger.info("-loss: {}  UAS: {}  LAS: {} \n".format(*best_res))
                    # back to train mode
                    model.train()
                
                step += 1
            
            # 每epoch进行一次评估
            if val_iter is not None and self.config.eval_strategy == 'epoch':
                avg_loss, uas, las = self.eval(model, val_iter)
                res = [avg_loss, uas, las]
                if las > best_res[2]:  # las
                    best_res = res
                    best_state_dict = model.state_dict()
                    early_stop_cnt = 0
                else:
                    early_stop_cnt += 1
                self.logger.info("--Best Evaluation: ")
                self.logger.info("-loss: {}  UAS: {}  LAS: {} \n".format(*best_res))
                # back to train mode
                model.train()
            
            # 如果在验证集上的性能有提升（LAS提高），则继续训练，否则，提前结束.
                # 早停有助于避免模型过度拟合训练数据，并且可以减少训练时间。
                # return (AVG_LOSS, UAS, LAS), best_state(最好参数)
            if early_stop_cnt >= self.config.num_early_stop:
                self.logger.info("--early stopping, training finished.")
                return best_res, best_state_dict

        self.logger.info("--training finished.")
        if best_state_dict is None:
            return 0.0, model.state_dict()
        
        return best_res, best_state_dict

    # eval func
    # 在验证集上进行推理
    def eval(self, model: nn.Module, eval_iter: DataLoader, save_file: str = "", save_title: str = ""):
        # 设置为评估模式
        model.eval()
        
        # 初始化存储整个验证集的数据
        head_whole, rel_whole, mask_whole = torch.Tensor(), torch.Tensor(), torch.Tensor()
        arc_logit_whole, rel_logit_whole = torch.Tensor(), torch.Tensor()
        avg_loss = 0.0

        # 遍历验证集
        for step, batch in enumerate(eval_iter):
            # 打印一下这几个张量
            inputs, offsets, heads, rels, masks = batch

            if self.config.cuda and torch.cuda.is_available():
                inputs_cuda = {}
                for key, value in inputs.items():
                    inputs_cuda[key] = value.cuda()
                inputs = inputs_cuda

                offsets, heads, rels, masks = to_cuda(data=(offsets, heads, rels, masks))
            
            # 在不更新梯度的情况下进行前向传播(forward, 即推理)
            # 因为不需要更新参数, 也就无需梯度下降
            with torch.no_grad():
                arc_logits, rel_logits = model(inputs, offsets, heads, rels, masks, evaluate=True)
            
            # 计算损失
            loss = self.loss_fn(arc_logits, rel_logits, heads, rels, masks)
            # 保存本轮弧和关系的预测结果
            arc_logit_whole = torch.cat([arc_logit_whole, arc_logits.cpu()], dim=0)
            rel_logit_whole = torch.cat([rel_logit_whole, rel_logits.cpu()], dim=0)
            # 弧的正确结果, 关系的正确结果
            head_whole, rel_whole = torch.cat([head_whole, heads.cpu()], dim=0), torch.cat([rel_whole, rels.cpu()], dim=0)
            mask_whole = torch.cat([mask_whole, masks.cpu()], dim=0)
            # 计算累计损失
            avg_loss += loss.item() * len(heads)  # times the batch size of data. 乘以当前批次的数据量

        # 计算对于整个验证集的UAS和LAS
        metrics = self.metrics_fn(arc_logit_whole, rel_logit_whole, head_whole, rel_whole, mask_whole)
        uas, las = metrics['UAS'], metrics['LAS']
        # 平均损失
        avg_loss /= len(eval_iter.dataset)  # type: ignore
        
        # 评估结果
        self.logger.info("--Evaluation:")
        self.logger.info("Avg Loss: {}  UAS: {}  LAS: {} \n".format(avg_loss, uas, las))

        # 保存评估结果: AVG_LOSS, UAS, LAS
        if save_file != "":
            results = [save_title, avg_loss, uas, las]  # type: ignore
            results = [str(x) for x in results]
            with open(save_file, "a+") as f:
                f.write(",".join(results) + "\n")  # type: ignore

        return avg_loss, uas, las  # type: ignore
    
    def save_results(self, save_file, save_title, results):
        saves = [save_title] + results
        saves = [str(x) for x in saves]
        with open(save_file, "a+") as f:
            f.write(",".join(saves) + "\n")  # type: ignore