import torch
from torch import nn


class MLP(nn.Module):
    """
    多层感知机,在Transformer解码器层中作为前馈神经网络（FFN）部分
    Qwen3的场景下是一个门控前馈网络
    """

    def __init__(
            self,
            hidden_size,
            intermediate_size: int = 3072,
    ):
        super().__init__()
        # 这是输入和输出的维度
        self.hidden_size = hidden_size
        # 中间层大小，MLP 内部的扩展维度
        self.intermediate_size = intermediate_size
        # 门控投影层
        # 从 hidden_size → intermediate_size 的线性变换
        # 都不使用偏置项
        # 生成"门控信号"，决定哪些信息需要通过
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # 上投影层（"up" 分支）
        # 从 hidden_size → intermediate_size
        # 作用：生成"值信号"，携带实际的信息内容
        # up_output = x @ W_up
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # 下投影层（降维回原始大小）
        # 从 intermediate_size → hidden_size 也就是将扩展后的特征压缩回原始维度
        # 作用：将处理后的信息投影回模型的主要表示空间
        # output = intermediate @ W_down
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # 从激活函数字典 ACT2FN 中获取激活函数
        # 默认是 "silu"（Swish 激活函数）
        # SiLU/Swish：silu(x) = x * sigmoid(x)，平滑的非线性变换
        # 优于ReLU
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(
            self.act_fn(
                self.gate_proj(x)
            )
            * self.up_proj(x)
        )
        # 最终输出 [seq_len, hidden_size]
        # 注意和源码相比这里少了batch
        return down_proj
