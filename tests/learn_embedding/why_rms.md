# 第一阶段：原始 Transformer (2017) - Post-Norm
论文："Attention Is All You Need" (Vaswani et al., 2017)
https://arxiv.org/pdf/1706.03762
原始结构：
```
# 伪代码 - 原始 Transformer Decoder Layer
def original_transformer_layer(x):
    # 子层 1：Self-Attention
    attn_output = self_attention(x)
    x = x + attn_output  # 残差连接
    x = layer_norm(x)    # ← Post-Norm：在残差之后

    # 子层 2：FFN
    ffn_output = feed_forward(x)
    x = x + ffn_output   # 残差连接
    x = layer_norm(x)    # ← Post-Norm：在残差之后

    return x
```
数学公式：
```
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```
问题：
❌ 训练不稳定：深层网络难以收敛
❌ 梯度问题：需要仔细的学习率预热（warmup）
❌ 难以扩展：层数增加时训练困难

# 第二阶段：Pre-Norm 架构 (2018-2019)
论文：
1. "Learning Deep Transformer Models for Machine Translation" (Wang et al., 2019)
https://arxiv.org/abs/1906.01787
2. "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
https://arxiv.org/pdf/2002.04745

## Pre-Norm 结构：
```
# Pre-Norm：归一化在子层之前
def pre_norm_transformer_layer(x):
    # 子层 1：Self-Attention
    residual = x
    x = layer_norm(x)           # ← Pre-Norm：在子层之前
    attn_output = self_attention(x)
    x = residual + attn_output  # 残差连接

    # 子层 2：FFN
    residual = x
    x = layer_norm(x)           # ← Pre-Norm：在子层之前
    ffn_output = feed_forward(x)
    x = residual + ffn_output   # 残差连接

    return x
```

##  数学公式
```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```
## 优势：
✅ 训练更稳定：梯度流动更顺畅
✅ 更容易扩展：可以堆叠更多层
✅ 不需要 warmup：或只需要很少的 warmup

# 第三阶段：RMSNorm 替代 LayerNorm (2019-2020)
## 论文
"Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
https://arxiv.org/abs/1910.07467

## LayerNorm vs RMSNorm 对比：
LayerNorm（原始）：
计算步骤：
1. 计算均值
2. 减去均值（中心化）
3. 计算方差
4. 归一化
5. 仿射变换（仿射变换 = 线性变换 + 平移。矩阵相乘和相加）

RMSNorm（简化版）：
```
def rms_norm(x):
    variance = (x ** 2).mean(dim=-1, keepdim=True)
    x = x / sqrt(variance + eps)
    return gamma * x  # 只有缩放，没有偏移
```

计算步骤：
1. ~~计算均值~~（省略）
2. ~~减去均值~~（省略）
3. 计算均方根
4. 归一化
5. 只缩放，不偏移

### RMSNorm 的优势：
✅ 计算更快：省略了均值计算和中心化
✅ 参数更少：没有 beta 偏移参数
✅ 效果相当：在 LLM 中性能与 LayerNorm 相当
✅ 更简单：实现和硬件优化更容易

# 第四阶段：现代 LLM 架构 (2020-至今)
以Qwen为例
```
# Qwen3DecoderLayer - 现代 Pre-Norm + RMSNorm 架构
def forward(self, hidden_states):
    # ========== 第一个子层：Self-Attention ==========
    residual = hidden_states  # 保存原始输入

    # Pre-Norm：归一化在 Attention 之前
    hidden_states = self.input_layernorm(hidden_states)  # RMSNorm

    # Self-Attention
    hidden_states, _ = self.self_attn(hidden_states, ...)

    # 残差连接
    hidden_states = residual + hidden_states

    # ========== 第二个子层：FFN/MLP ==========
    residual = hidden_states  # 保存中间结果

    # Pre-Norm：归一化在 FFN 之前
    hidden_states = self.post_attention_layernorm(hidden_states)  # RMSNorm

    # MLP/FFN
    hidden_states = self.mlp(hidden_states)

    # 残差连接
    hidden_states = residual + hidden_states

    return hidden_states
```

#  总结
演变路径
```
2017: Post-Norm + LayerNorm (原始 Transformer)
         ↓
2019: Pre-Norm + LayerNorm (提升稳定性)
         ↓
2020: Pre-Norm + RMSNorm (提升效率)
         ↓
2024: Pre-Norm + RMSNorm + 其他优化 (Qwen3)
```

两个归一化层的必要性：
- ✅ 独立归一化：Attention 和 FFN 是两个独立的子层，各需要稳定的输入
- ✅ Pre-Norm 设计：每个子层之前都要归一化，才能获得最佳训练稳定性
- ✅ 已验证有效：这是所有现代 LLM 的标准设计
为什么不共用一个？
如果只用一个归一化层：
- ❌ FFN 会直接接收 Attention 的输出（未归一化）
- ❌ 数值分布不稳定
- ❌ 训练更难收敛


