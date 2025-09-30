import torch
from torch import nn
from typing import Optional, Callable

from tests.learn_embedding.attention_forward import sdpa_attention_forward
from tests.learn_embedding.rms_norm import RMSNorm
from tests.learn_embedding.rope import apply_rotary_pos_emb


class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            layer_idx: int,
            layer_type:str = 'full_attention',
            hidden_size: int = 1024,
            attention_bias: bool = False,
            sliding_window = None,
            head_dim: int = 128,
            num_key_value_heads: int = 8,
            num_attention_heads: int = 16,
            attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        # 每个注意力头的维度, 如果没有可以用隐藏层大小除以注意力头数量
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_attention_heads = num_attention_heads
        # 计算 GQA（Grouped Query Attention）的分组数
        # 这是一种优化技术，Key 和 Value 的头数可以少于 Query 的头数
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        # 注意力权重的 dropout 概率，用于正则化
        # 标记为因果注意力，意味着只能看到当前位置之前的 token（自回归特性）
        self.attention_dropout = attention_dropout
        self.is_causal = True
        # 注意力分数的缩放因子 = 1/√(head_dim)
        # 用于缩放点积注意力，防止数值过大导致 softmax 梯度消失
        # 这是标准的缩放点积注意力（Scaled Dot-Product Attention）
        self.scaling = self.head_dim ** -0.5
        self.num_key_value_heads = num_key_value_heads
        # Query 投影层(线性层)
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_attention_heads * self.head_dim, bias=attention_bias
        )
        # Key 投影层
        # num_key_value_heads × head_dim（注意是KV头数，不是注意力头数）
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias
        )
        # Value 投影层
        # 与 Key 投影层对称
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias
        )
        # 输出投影层
        # 将多头注意力的输出映射回 hidden_size
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, self.hidden_size, bias=attention_bias
        )

        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)

        self.sliding_window = sliding_window if layer_type == "sliding_attention" else None

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: 输入的隐藏状态，形状为 [batch_size, seq_len, hidden_size]
            position_embeddings: 位置编码（cos, sin），用于旋转位置编码（RoPE）
            attention_mask: 注意力掩码，控制哪些位置可以被关注
            past_key_values: KV 缓存，用于生成时加速
            cache_position: 缓存位置索引
        """
        # 提取 [batch_size, seq_len]
        # 不考虑batch，提取seq_len即可进来是(7,1024), 要变成(7,)
        input_shape = hidden_states.shape[:-1]
        # 构造多头形状 [seq_len, num_heads, head_dim] ，要变成(7, -1, 128)
        hidden_shape = (*input_shape, -1, self.head_dim)

        # 下面是Query计算流程
        # RMSNorm 归一化（在 head_dim 上）
        # 算完是(16, 7, 128)
        query_states = self.q_norm(
            # 线性投影
            self.q_proj(hidden_states)
            # 重塑为多头形状
            .view(hidden_shape)).transpose(0, 1)  # 转置变为 [num_heads, seq_len, head_dim]
        # Key计算，流程相同
        # 算完是(8, 7, 128)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(0, 1)
        # Value计算，除了归一化以外流程相同
        # 算完是(8, 7, 128)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(0, 1)
        # 解包位置编码的余弦和正弦部分
        # 算完是(7, 128)的两个矩阵
        cos, sin = position_embeddings
        # 应用 RoPE（Rotary Position Embedding）
        # 将位置信息编码到 Query 和 Key 中
        # 通过旋转变换实现相对位置编码
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 这里源码中有缓存过程，要关注是否有影响

        # 选择注意力实现方式 默认是eage，但是Qwen3 .默认使用的是 sdpa
        # attention_interface: Callable = eager_attention_forward
        attention_interface: Callable = sdpa_attention_forward

        # 执行注意力计算
        # 传入 Q、K、V
        # 训练时应用 dropout，推理时不用
        # 传入缩放因子和滑动窗口参数
        # 返回注意力输出和权重
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
        )
        # 将多头输出重塑回 [batch_size, seq_len, hidden_size].contiguous() 确保内存连续，提高后续操作效率
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        # 通过输出投影层，完成最终变换
        # 算出来应该是 (7, 1024)
        attn_output = self.o_proj(attn_output)
        # 返回注意力输出和权重（权重可用于可视化）
        return attn_output, attn_weights