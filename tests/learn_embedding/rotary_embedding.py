import torch

from tests.learn_embedding.modeling_rope_utils import _compute_default_rope_parameters


class RotaryEmbedding(torch.nn.Module):
    inv_freq: torch.Tensor

    """
    

    """
    def __init__(self, device="cpu"):
        super().__init__()
        # BC: "rope_type" was originally "type"
        self.rope_type = "default"
        self.max_seq_len_cached = 32768
        self.original_max_seq_len = 32768
        # 策略模式选择选择旋转的函数
        self.rope_init_fn = None # ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = _compute_default_rope_parameters(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq


    @torch.no_grad()
    def forward(self, x, position_ids):
        """
        前向处理RoPE过程
        Args:
            x: hiddent_states
            position_ids:  位置id

        Returns:

        """
        # 扩展逆频率张量
        # 原始形状: [head_dim//2]
        # 扩展后: [head_dim//2, 1]
        # 做一点不考虑batch的修改
        inv_freq_expanded = self.inv_freq[:, None].float().to(x.device)
        # 扩展位置 ID
        # 原始形状: [batch_size, seq_len]
        # 扩展后: [batch_size, 1, seq_len]
        # 不考虑batch
        position_ids_expanded = position_ids[None, :].float()

        with torch.autocast(device_type="cpu", enabled=False):  # Force float32
            # 计算频率矩阵
            freqs = (inv_freq_expanded @ position_ids_expanded).t()
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)