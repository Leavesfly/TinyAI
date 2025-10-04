"""
Qwen3 完整模型实现
包含词嵌入、多层解码器、语言模型头等完整架构
支持文本生成、KV缓存等功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import json
import math

# 导入核心组件
try:
    from qwen3_core_components import (
        Qwen3Config, 
        Qwen3DecoderLayer, 
        RMSNorm,
        prepare_attention_mask,
        create_causal_mask
    )
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    
    from qwen3_core_components import (
        Qwen3Config, 
        Qwen3DecoderLayer, 
        RMSNorm,
        prepare_attention_mask,
        create_causal_mask
    )


class Qwen3Model(nn.Module):
    """
    Qwen3 模型主体
    包含词嵌入层和多层解码器
    """
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        
        # 词嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # 解码器层
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # 最终的归一化层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化线性层
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化嵌入层
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 设置默认值
        use_cache = use_cache if use_cache is not None else self.config.use_cache if hasattr(self.config, 'use_cache') else False
        return_dict = return_dict if return_dict is not None else True
        
        if input_ids is None:
            raise ValueError("必须提供 input_ids")
        
        batch_size, seq_length = input_ids.shape
        
        # 处理过去的键值对
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        
        # 确保past_key_values的长度正确
        if len(past_key_values) != len(self.layers):
            past_key_values = [None] * len(self.layers)
        
        # 词嵌入
        hidden_states = self.embed_tokens(input_ids)
        
        # 准备注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=input_ids.device)
        
        attention_mask = prepare_attention_mask(
            attention_mask, 
            (batch_size, seq_length), 
            hidden_states.dtype
        )
        
        # 存储新的键值对
        next_cache = [] if use_cache else None
        
        # 通过所有解码器层
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            layer_outputs = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache and next_cache is not None:
                next_cache.append(layer_outputs[1])
        
        # 最终归一化
        hidden_states = self.norm(hidden_states)
        
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache] if v is not None)
        
        return {
            'last_hidden_state': hidden_states,
            'past_key_values': next_cache,
        }


class Qwen3ForCausalLM(nn.Module):
    """
    Qwen3 因果语言模型
    在 Qwen3Model 基础上添加语言模型头
    """
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        
        # 语言模型头 - 输出词汇表概率
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 如果配置要求，共享嵌入权重
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else True
        
        # 前向传播主模型
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        
        if return_dict:
            hidden_states = transformer_outputs['last_hidden_state']
            past_key_values = transformer_outputs.get('past_key_values')
        else:
            hidden_states = transformer_outputs[0]
            past_key_values = transformer_outputs[1] if len(transformer_outputs) > 1 else None
        
        # 语言模型头输出logits
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # 计算交叉熵损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        if not return_dict:
            output = (logits,)
            if past_key_values is not None:
                output += (past_key_values,)
            return ((loss,) + output) if loss is not None else output
        
        return {
            'loss': loss,
            'logits': logits,
            'past_key_values': past_key_values,
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        生成文本序列
        
        Args:
            input_ids: 输入token序列 [batch_size, seq_len]
            max_new_tokens: 最大生成token数量
            temperature: 温度参数，控制随机性
            top_p: nucleus sampling参数
            top_k: top-k sampling参数
            do_sample: 是否使用采样
            pad_token_id: 填充token ID
            eos_token_id: 结束token ID
        
        Returns:
            generated_ids: 生成的完整序列 [batch_size, seq_len + new_tokens]
        """
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        
        batch_size, cur_len = input_ids.shape
        device = input_ids.device
        
        # 用于存储生成的序列
        generated = input_ids.clone()
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_new_tokens):
            # 如果所有序列都已完成，停止生成
            if finished.all():
                break
            
            # 前向传播
            outputs = self.forward(
                input_ids=generated if past_key_values is None else generated[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            
            logits = outputs['logits']
            past_key_values = outputs['past_key_values']
            
            # 获取下一个token的logits
            next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
            
            if do_sample:
                # 应用温度
                next_token_logits = next_token_logits / temperature
                
                # Top-k过滤
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Top-p (nucleus) 过滤
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 移除累积概率超过top_p的token
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # 采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # 贪心解码
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # 对已完成的序列，使用pad_token_id
            next_tokens = torch.where(finished, pad_token_id, next_tokens)
            
            # 添加到生成序列
            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
            
            # 检查是否遇到结束token
            if eos_token_id is not None:
                finished = finished | (next_tokens == eos_token_id)
        
        return generated
    
    def prepare_inputs_for_generation(
        self, 
        input_ids: torch.Tensor, 
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        **kwargs
    ):
        """准备生成时的输入"""
        if past_key_values is not None:
            # 如果有过去的键值对，只需要最后一个token
            input_ids = input_ids[:, -1:]
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }


def create_qwen3_model(config_dict: Dict[str, Any]) -> Qwen3ForCausalLM:
    """
    从配置字典创建Qwen3模型
    
    Args:
        config_dict: 包含模型配置的字典
    
    Returns:
        Qwen3ForCausalLM: 初始化的Qwen3模型
    """
    config = Qwen3Config(**config_dict)
    model = Qwen3ForCausalLM(config)
    return model


def get_model_size(model: nn.Module) -> int:
    """
    计算模型参数数量
    
    Args:
        model: PyTorch模型
    
    Returns:
        int: 模型参数总数
    """
    return sum(p.numel() for p in model.parameters())


def get_trainable_parameters(model: nn.Module) -> int:
    """
    计算可训练参数数量
    
    Args:
        model: PyTorch模型
    
    Returns:
        int: 可训练参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试完整模型
    print("测试 Qwen3 完整模型...")
    
    # 创建小型配置用于测试
    config = Qwen3Config(
        vocab_size=1000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=1024,
    )
    
    print(f"模型配置: {config}")
    
    # 创建模型
    model = Qwen3ForCausalLM(config)
    
    # 计算模型大小
    total_params = get_model_size(model)
    trainable_params = get_trainable_parameters(model)
    
    print(f"\n模型参数统计:")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (假设 FP32)")
    
    # 测试前向传播
    print(f"\n测试前向传播...")
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        print(f"输出logits形状: {outputs['logits'].shape}")
        print(f"预期形状: ({batch_size}, {seq_len}, {config.vocab_size})")
    
    # 测试生成功能
    print(f"\n测试文本生成...")
    input_ids = torch.randint(0, config.vocab_size, (1, 5))  # 单个序列
    
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=10,
            temperature=1.0,
            do_sample=True,
        )
        print(f"输入序列长度: {input_ids.shape[1]}")
        print(f"生成序列长度: {generated.shape[1]}")
        print(f"输入tokens: {input_ids[0].tolist()}")
        print(f"完整序列: {generated[0].tolist()}")
    
    print(f"\n✅ Qwen3完整模型测试通过！")