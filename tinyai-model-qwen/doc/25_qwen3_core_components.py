"""
Qwen3 模型核心组件实现
包含 RMSNorm、旋转位置编码(RoPE)、多头注意力机制、SwiGLU 激活的前馈网络等核心组件
基于 PyTorch 实现，采用解码器-only架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class Qwen3Config:
    """Qwen3 模型配置类"""
    vocab_size: int = 32000          # 词汇表大小
    hidden_size: int = 2048          # 隐藏层维度
    intermediate_size: int = 5632    # 前馈网络中间层维度
    num_hidden_layers: int = 24      # 隐藏层数量
    num_attention_heads: int = 16    # 注意力头数量
    num_key_value_heads: int = 16    # 键值头数量（用于分组查询注意力）
    max_position_embeddings: int = 8192  # 最大位置编码长度
    rope_theta: float = 10000.0      # RoPE 基础频率
    rms_norm_eps: float = 1e-6       # RMSNorm 的 epsilon
    pad_token_id: int = 0            # 填充标记ID
    bos_token_id: int = 1            # 开始标记ID
    eos_token_id: int = 2            # 结束标记ID
    tie_word_embeddings: bool = False  # 是否共享输入输出嵌入权重


class RMSNorm(nn.Module):
    """
    RMS归一化层
    相比LayerNorm，RMSNorm去除了重新中心化的步骤，只进行重新缩放
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        # 计算均方根
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RotaryPositionalEmbedding(nn.Module):
    """
    旋转位置编码 (RoPE)
    通过旋转查询和键向量来编码位置信息
    """
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # 计算逆频率
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None):
        # x: [batch_size, num_heads, seq_len, head_dim]
        if seq_len is None:
            seq_len = x.shape[-2]
        
        # 生成位置索引
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        
        # 计算频率矩阵
        freqs = torch.outer(t, self.inv_freq)
        # 复制频率以匹配头部维度
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()
        sin = emb.sin()
        
        return cos, sin

    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """应用旋转位置编码到查询和键向量"""
        def rotate_half(x):
            """旋转输入的一半特征"""
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        
        return q_embed, k_embed


class Qwen3Attention(nn.Module):
    """
    Qwen3多头注意力机制
    支持分组查询注意力 (Grouped Query Attention)
    """
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size 必须能被 num_heads 整除 (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        # 线性投影层
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # 旋转位置编码
        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """重塑张量形状用于多头注意力"""
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()

        # 计算查询、键、值投影
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 重塑为多头形式
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 应用旋转位置编码
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        query_states, key_states = self.rotary_emb.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 处理KV缓存
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # 重复键值头以匹配查询头数量（分组查询注意力）
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # 计算注意力分数
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 应用注意力掩码
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # 应用softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # 应用注意力权重到值
        attn_output = torch.matmul(attn_weights, value_states)

        # 重塑输出
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # 输出投影
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


class Qwen3MLP(nn.Module):
    """
    Qwen3 多层感知机 (MLP)
    使用 SwiGLU 激活函数
    """
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # SwiGLU 需要两个线性层用于门控
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU 激活: SwiGLU(x) = Swish(xW1) ⊙ (xW2)
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # Swish(x) = x * sigmoid(x)
        gate = gate * torch.sigmoid(gate)
        # 元素级别乘法
        intermediate = gate * up
        # 下投影
        output = self.down_proj(intermediate)
        return output


class Qwen3DecoderLayer(nn.Module):
    """
    Qwen3 解码器层
    包含自注意力、前馈网络和残差连接
    """
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # 自注意力层
        self.self_attn = Qwen3Attention(config)
        
        # 前馈网络
        self.mlp = Qwen3MLP(config)
        
        # RMS归一化层
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        residual = hidden_states

        # 预归一化 + 自注意力
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        
        # 残差连接
        hidden_states = residual + hidden_states

        # 预归一化 + 前馈网络
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # 残差连接
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value


def create_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    创建因果注意力掩码
    确保每个位置只能关注之前的位置
    """
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]


def prepare_attention_mask(attention_mask: torch.Tensor, input_shape: Tuple[int, int], 
                          dtype: torch.dtype) -> torch.Tensor:
    """
    准备注意力掩码，结合填充掩码和因果掩码
    """
    batch_size, seq_length = input_shape
    device = attention_mask.device
    
    # 创建因果掩码
    causal_mask = create_causal_mask(seq_length, device, dtype)
    
    # 扩展填充掩码
    if attention_mask.dim() == 2:
        # 从 [batch_size, seq_len] 扩展到 [batch_size, 1, seq_len, seq_len]
        expanded_mask = attention_mask[:, None, None, :]
        expanded_mask = expanded_mask.expand(batch_size, 1, seq_length, seq_length)
        
        # 将填充位置设为 -inf
        inverted_mask = (1.0 - expanded_mask) * torch.finfo(dtype).min
        
        # 结合因果掩码和填充掩码
        combined_mask = torch.maximum(causal_mask, inverted_mask)
        
        return combined_mask
    
    return causal_mask


if __name__ == "__main__":
    # 测试核心组件
    print("测试 Qwen3 核心组件...")
    
    # 创建配置
    config = Qwen3Config(
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=1024,
        vocab_size=1000,
        max_position_embeddings=1024
    )
    
    # 测试 RMSNorm
    print("\n测试 RMSNorm...")
    rms_norm = RMSNorm(config.hidden_size)
    test_input = torch.randn(2, 10, config.hidden_size)
    normalized = rms_norm(test_input)
    print(f"输入形状: {test_input.shape}, 输出形状: {normalized.shape}")
    
    # 测试 RoPE
    print("\n测试旋转位置编码...")
    rope = RotaryPositionalEmbedding(config.hidden_size // config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    q = torch.randn(2, config.num_attention_heads, 10, head_dim)
    k = torch.randn(2, config.num_attention_heads, 10, head_dim)
    cos, sin = rope(q, seq_len=10)
    q_rot, k_rot = rope.apply_rotary_pos_emb(q, k, cos, sin)
    print(f"旋转后查询形状: {q_rot.shape}, 键形状: {k_rot.shape}")
    
    # 测试注意力层
    print("\n测试注意力层...")
    attention = Qwen3Attention(config)
    hidden_states = torch.randn(2, 10, config.hidden_size)
    attn_mask = torch.ones(2, 10)
    prepared_mask = prepare_attention_mask(attn_mask, (2, 10), torch.float32)
    
    attn_output, _ = attention(hidden_states, attention_mask=prepared_mask)
    print(f"注意力输出形状: {attn_output.shape}")
    
    # 测试 MLP
    print("\n测试 MLP...")
    mlp = Qwen3MLP(config)
    mlp_output = mlp(hidden_states)
    print(f"MLP输出形状: {mlp_output.shape}")
    
    # 测试解码器层
    print("\n测试解码器层...")
    decoder_layer = Qwen3DecoderLayer(config)
    layer_output = decoder_layer(hidden_states, attention_mask=prepared_mask)
    print(f"解码器层输出形状: {layer_output[0].shape}")
    
    print("\n✅ 所有核心组件测试通过！")