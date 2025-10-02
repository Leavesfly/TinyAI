"""
Qwen3 大语言模型实现
基于Transformer架构的中文大语言模型
作者：山泽
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union
from dataclasses import dataclass
import json


@dataclass
class Qwen3Config:
    """Qwen3模型配置类"""
    vocab_size: int = 151936  # Qwen3词汇表大小
    hidden_size: int = 4096   # 隐藏层维度
    num_hidden_layers: int = 32  # Transformer层数
    num_attention_heads: int = 32  # 注意力头数
    num_key_value_heads: int = 32  # KV头数（用于GQA）
    intermediate_size: int = 11008  # FFN中间层维度
    max_position_embeddings: int = 32768  # 最大位置编码长度
    rms_norm_eps: float = 1e-6  # RMSNorm的epsilon
    rope_theta: float = 1000000.0  # RoPE的theta参数
    attention_dropout: float = 0.0  # 注意力dropout
    hidden_dropout: float = 0.0  # 隐藏层dropout
    use_cache: bool = True  # 是否使用KV缓存
    pad_token_id: int = 151643  # padding token id
    bos_token_id: int = 151643  # 开始token id
    eos_token_id: int = 151645  # 结束token id


class RMSNorm(nn.Module):
    """RMS归一化层"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RotaryPositionalEmbedding(nn.Module):
    """旋转位置编码(RoPE)"""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # 计算旋转角度
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None):
        """
        应用旋转位置编码

        Args:
            x: 输入张量，形状为 [batch_size, num_heads, seq_len, head_dim]
            seq_len: 序列长度
        """
        if seq_len is None:
            seq_len = x.shape[-2]

        # 生成位置索引
        position_ids = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)

        # 计算旋转矩阵
        freqs = torch.outer(position_ids, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        return self.apply_rotary_pos_emb(x, cos, sin)

    def apply_rotary_pos_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """应用旋转位置编码到输入张量"""
        # 将x分割为前半部分和后半部分
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]

        # 应用旋转变换
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim//2]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim//2]

        rotated_x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated_x


class Qwen3Attention(nn.Module):
    """Qwen3多头注意力机制"""

    def __init__(self, config: Qwen3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size必须能被num_heads整除 (得到 `{self.hidden_size}` 和 `{self.num_heads}`)."
            )

        # Q、K、V投影层
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # 旋转位置编码
        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=config.rope_theta,
        )

        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        # 计算Q、K、V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 重塑张量形状
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 应用旋转位置编码
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        query_states = self.rotary_emb(query_states, kv_seq_len)
        key_states = self.rotary_emb(key_states, kv_seq_len)

        # 处理KV缓存
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # 重复K、V以匹配Q的头数（用于分组查询注意力）
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # 计算注意力分数
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # 应用softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attention_dropout(attn_weights)

        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, value_states)

        # 重塑输出张量
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # 最终投影
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """重复K、V张量以匹配查询头数"""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen3MLP(nn.Module):
    """Qwen3前馈神经网络"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # 门控线性单元
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # 激活函数：SwiGLU
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU激活：SiLU(gate_proj(x)) * up_proj(x)
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class Qwen3DecoderLayer(nn.Module):
    """Qwen3解码器层"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # 自注意力层
        self.self_attn = Qwen3Attention(config, layer_idx)

        # 前馈网络
        self.mlp = Qwen3MLP(config)

        # 层归一化
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        # 前归一化 + 自注意力
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        # 残差连接
        hidden_states = residual + hidden_states

        # 前归一化 + FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        # 残差连接
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Qwen3Model(nn.Module):
    """Qwen3模型主体"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # 词嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Transformer解码器层
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])

        # 最终层归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, ...]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # 获取输入嵌入
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("不能同时指定input_ids和inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("必须指定input_ids或inputs_embeds中的一个")

        # 处理位置编码
        if position_ids is None:
            seq_length_with_past = seq_length
            if past_key_values is not None:
                seq_length_with_past = seq_length + past_key_values[0][0].shape[2]
            position_ids = torch.arange(
                seq_length_with_past - seq_length,
                seq_length_with_past,
                dtype=torch.long,
                device=inputs_embeds.device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        # 处理注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=inputs_embeds.device
            )

        # 创建因果掩码
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, seq_length
        )

        hidden_states = inputs_embeds

        # 初始化输出容器
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # 通过每个解码器层
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # 最终归一化
        hidden_states = self.norm(hidden_states)

        # 添加最后的隐藏状态
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)
        if use_cache:
            outputs += (next_decoder_cache,)
        if output_hidden_states:
            outputs += (all_hidden_states,)
        if output_attentions:
            outputs += (all_self_attns,)

        return outputs

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        """准备解码器的注意力掩码"""
        # 创建因果掩码
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = self._make_causal_mask(
                input_shape, inputs_embeds.dtype, device=inputs_embeds.device, past_key_values_length=past_key_values_length
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = self._expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def _make_causal_mask(self, input_ids_shape, dtype, device, past_key_values_length=0):
        """创建因果掩码以确保位置i只能注意到位置j<=i"""
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    def _expand_mask(self, mask, dtype, tgt_len=None):
        """将2D掩码扩展为4D掩码"""
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class Qwen3ForCausalLM(nn.Module):
    """Qwen3因果语言模型"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config

        # 主要的Transformer模型
        self.model = Qwen3Model(config)

        # 语言模型头（输出投影层）
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
    ):

        # 调用主要模型
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs[0]

        # 计算logits
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # 计算语言模型损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            # 启用模型并行时
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """为生成准备输入"""
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # 如果传递了`inputs_embeds`，只在第一个生成步骤中使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @torch.no_grad()
    def generate(
            self,
            input_ids: torch.LongTensor,
            max_length: int = 100,
            temperature: float = 1.0,
            top_k: int = 50,
            top_p: float = 0.9,
            do_sample: bool = True,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """文本生成方法"""

        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        batch_size = input_ids.shape[0]
        cur_len = input_ids.shape[1]

        # 初始化past_key_values
        past_key_values = None

        while cur_len < max_length:
            # 前向传播
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past_key_values=past_key_values, use_cache=True
            )

            outputs = self.forward(**model_inputs)
            logits = outputs[0][:, -1, :]  # 只取最后一个时间步的logits
            past_key_values = outputs[1]

            # 应用温度
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k筛选
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(-1, top_k_indices, top_k_logits)

            # Top-p筛选
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # 删除累积概率高于阈值的token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # 采样下一个token
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)

            # 更新input_ids
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            cur_len += 1

            # 检查是否遇到结束token
            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break

        return input_ids


def create_qwen3_model(config_dict: dict = None) -> Qwen3ForCausalLM:
    """创建Qwen3模型实例"""
    if config_dict is None:
        config = Qwen3Config()
    else:
        config = Qwen3Config(**config_dict)

    model = Qwen3ForCausalLM(config)
    return model


def demo_qwen3():
    """Qwen3模型演示"""
    print("=== Qwen3 大语言模型演示 ===")

    # 创建小型配置用于演示
    config_dict = {
        "vocab_size": 1000,
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "intermediate_size": 512,
        "max_position_embeddings": 512,
    }

    # 创建模型
    model = create_qwen3_model(config_dict)
    model.eval()

    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    #