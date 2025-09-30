from typing import Optional

import torch
from torch import nn

from tests.learn_embedding.attention import Attention
from tests.learn_embedding.mlp import MLP
from tests.learn_embedding.rms_norm import RMSNorm


class DecoderLayer(nn.Module):

    def __init__(
            self,
            layer_idx: int,
            layer_type: str = "full_attention",
            hidden_size: int = 1024,
            attention_bias: bool = False,
            sliding_window=None,
            head_dim: int = 128,
            num_key_value_heads: int = 8,
            num_attention_heads: int = 16,
            attention_dropout: float = 0.0,
            intermediate_size: int = 3072,
            rms_norm_eps: float = 1e-06,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        # 自注意力层
        self.self_attn = Attention(
            layer_idx,
            layer_type=layer_type,
            hidden_size=hidden_size,
            attention_bias=attention_bias,
            sliding_window=sliding_window,
            head_dim=head_dim,
            num_key_value_heads=num_key_value_heads,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
        )
        # 前馈网络层
        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        # 层归一化：
        # 输入层归一化
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        # 注意力后的层归一化
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        # 设置注意力类型
        self.attention_type = layer_type if layer_type is not None else 'full_attention'

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # 保存残差连接
        residual = hidden_states
        # 输入层归一化
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention 自注意力计算
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        # 残差连接
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        # 注意力后的层归一化
        hidden_states = self.post_attention_layernorm(hidden_states)
        # 前馈网络
        hidden_states = self.mlp(hidden_states)
        # 残差链接
        hidden_states = residual + hidden_states
        # 这里出去是(7, 1024)
        return hidden_states
