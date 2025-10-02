import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import math
import json
import time
from dataclasses import dataclass

@dataclass
class GPT3Config:
    """GPT-3 模型配置"""
    vocab_size: int = 50257
    n_positions: int = 2048  # GPT-3 支持更长的上下文
    n_embd: int = 12288      # GPT-3-175B 的隐藏层维度
    n_layer: int = 96        # GPT-3-175B 的层数
    n_head: int = 96         # GPT-3-175B 的注意力头数
    n_inner: Optional[int] = None  # 4 * n_embd
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    # GPT-3 特有配置
    sparse_attention: bool = False  # 稀疏注意力（用于超大模型）
    gradient_checkpointing: bool = True  # 梯度检查点以节省内存
    parallel_attention: bool = True  # 并行注意力和MLP计算
    rotary_pct: float = 0.25  # 旋转位置编码比例
    
    def __post_init__(self):
        if self.n_inner is None:
            self.n_inner = 4 * self.n_embd

class GPT3RotaryEmbedding(nn.Module):
    """旋转位置编码 (RoPE) - GPT-3 的改进之一"""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, seq_len: int, device: torch.device):
        """生成旋转位置编码"""
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x):
    """将张量的后半部分移到前面并取负号"""
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """应用旋转位置编码"""
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GPT3SparseAttention(nn.Module):
    """GPT-3 稀疏注意力机制（模拟）"""
    
    def __init__(self, config: GPT3Config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        max_positions = config.n_positions
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        
        if self.head_dim * self.n_head != self.n_embd:
            raise ValueError(f"n_embd ({self.n_embd}) 必须能被 n_head ({self.n_head}) 整除")
        
        # 查询、键、值投影
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        # 旋转位置编码
        rotary_dim = int(self.head_dim * config.rotary_pct)
        self.rotary_emb = GPT3RotaryEmbedding(rotary_dim, config.n_positions)
        self.rotary_dim = rotary_dim
        
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """将隐藏状态分割为多个注意力头"""
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    
    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """合并多个注意力头"""
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)
    
    def _sparse_attn_mask(self, seq_len: int):
        """创建稀疏注意力掩码（简化版）"""
        if not self.config.sparse_attention:
            return None
            
        # 简单的带状稀疏注意力模式
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        band_width = min(64, seq_len // 4)  # 局部注意力带宽
        
        # 只保留对角线附近的注意力
        for i in range(seq_len):
            start = max(0, i - band_width)
            end = min(seq_len, i + band_width + 1)
            mask[i, :start] = False
            mask[i, end:] = False
            
        return mask
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        batch_size, seq_len = hidden_states.shape[:2]
        
        # 计算查询、键、值
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # 分割注意力头
        q = self._split_heads(q, self.n_head, self.head_dim)
        k = self._split_heads(k, self.n_head, self.head_dim)
        v = self._split_heads(v, self.n_head, self.head_dim)
        
        # 应用旋转位置编码
        if self.rotary_dim > 0:
            cos, sin = self.rotary_emb(seq_len, hidden_states.device)
            q_rot = q[..., :self.rotary_dim]
            q_pass = q[..., self.rotary_dim:]
            k_rot = k[..., :self.rotary_dim]
            k_pass = k[..., self.rotary_dim:]
            
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)
        
        # 使用缓存
        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=-2)
            v = torch.cat([past_value, v], dim=-2)
        
        if use_cache:
            present = (k, v)
        else:
            present = None
            
        # 计算注意力分数
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        
        # 应用因果掩码
        causal_mask = self.bias[:, :, :seq_len, :k.size(-2)]
        attn_weights = torch.where(causal_mask, attn_weights, torch.finfo(attn_weights.dtype).min)
        
        # 应用稀疏注意力掩码
        sparse_mask = self._sparse_attn_mask(k.size(-2))
        if sparse_mask is not None:
            sparse_mask = sparse_mask.to(attn_weights.device)
            attn_weights = torch.where(sparse_mask, attn_weights, torch.finfo(attn_weights.dtype).min)
        
        # 应用attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax和dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, v)
        attn_output = self._merge_heads(attn_output, self.n_head, self.head_dim)
        
        # 输出投影
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output, present

class GPT3MLP(nn.Module):
    """GPT-3 前馈网络"""
    
    def __init__(self, config: GPT3Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd)
        self.dropout = nn.Dropout(config.resid_pdrop)
        
        # 激活函数
        if config.activation_function == "gelu_new":
            self.act = lambda x: 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        elif config.activation_function == "gelu":
            self.act = F.gelu
        elif config.activation_function == "relu":
            self.act = F.relu
        elif config.activation_function == "swish":
            self.act = lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"不支持的激活函数: {config.activation_function}")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class GPT3Block(nn.Module):
    """GPT-3 Transformer块（支持并行计算）"""
    
    def __init__(self, config: GPT3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Pre-LayerNorm 架构
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT3SparseAttention(config, layer_idx)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT3MLP(config)
        
        # 并行注意力和MLP（GPT-3的优化）
        self.parallel_attention = config.parallel_attention
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        if self.parallel_attention:
            # 并行计算注意力和MLP
            ln_1_output = self.ln_1(hidden_states)
            attn_output, present = self.attn(
                ln_1_output,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            
            ln_2_output = self.ln_2(hidden_states)
            mlp_output = self.mlp(ln_2_output)
            
            # 残差连接
            hidden_states = hidden_states + attn_output + mlp_output
        else:
            # 串行计算（传统方式）
            residual = hidden_states
            hidden_states = self.ln_1(hidden_states)
            attn_output, present = self.attn(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            hidden_states = residual + attn_output
            
            residual = hidden_states
            hidden_states = self.ln_2(hidden_states)
            mlp_output = self.mlp(hidden_states)
            hidden_states = residual + mlp_output
        
        return hidden_states, present

class GPT3Model(nn.Module):
    """GPT-3模型主体"""
    
    def __init__(self, config: GPT3Config):
        super().__init__()
        self.config = config
        
        # 词嵌入和位置嵌入
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        
        # 嵌入dropout
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # Transformer层
        self.h = nn.ModuleList([GPT3Block(config, layer_idx=i) for i in range(config.n_layer)])
        
        # 最终层归一化
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # 模型并行设置
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = config.gradient_checkpointing
        
        # 初始化权重
        self.post_init()
        
        print(f"GPT-3模型初始化完成，参数量: {self.get_num_params():,}")
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """计算模型参数数量"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
            n_params -= self.wte.weight.numel()
        return n_params
    
    def post_init(self):
        """初始化权重"""
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        batch_size, seq_length = input_ids.shape
        
        if position_ids is None:
            device = input_ids.device
            past_length = 0 if past_key_values is None else past_key_values[0][0].size(-2)
            position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 嵌入
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        
        # 注意力掩码
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        
        # Transformer层
        presents = () if use_cache else None
        for i, (block, past_key_value) in enumerate(zip(self.h, past_key_values or [None] * len(self.h))):
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点节省内存
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, use_cache=use_cache)
                    return custom_forward
                
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    past_key_value,
                )
            else:
                layer_outputs = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                presents = presents + (layer_outputs[1],)
        
        # 最终归一化
        hidden_states = self.ln_f(hidden_states)
        
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": presents,
        }

class GPT3LMHeadModel(nn.Module):
    """带有语言模型头的GPT-3模型"""
    
    def __init__(self, config: GPT3Config):
        super().__init__()
        self.config = config
        self.transformer = GPT3Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 权重共享（可选）
        # self.lm_head.weight = self.transformer.wte.weight
        
        self.model_parallel = False
        self.device_map = None
        
        self.post_init()
    
    def post_init(self):
        """初始化权重并应用最终处理"""
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """为生成准备输入"""
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        
        # 只传递最后一个token（如果已经有past_key_values）
        if past_key_values:
            input_ids = input_ids[:, -1:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -1:]
        
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1:]
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[Tuple[torch.Tensor]]]]:
        
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
        )
        
        hidden_states = transformer_outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # 下一个token预测的移位损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        return logits, loss, transformer_outputs["past_key_values"]

class GPT3InContextLearning:
    """GPT-3 上下文学习（Few-shot Learning）"""
    
    def __init__(self, model: GPT3LMHeadModel, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
    
    def few_shot_prompt(
        self, 
        task_description: str,
        examples: List[Tuple[str, str]],
        query: str,
        max_examples: int = 5
    ) -> str:
        """构建few-shot学习提示"""
        prompt_parts = [task_description, "\n\n"]
        
        # 添加示例
        for i, (input_text, output_text) in enumerate(examples[:max_examples]):
            prompt_parts.append(f"示例 {i+1}:\n")
            prompt_parts.append(f"输入: {input_text}\n")
            prompt_parts.append(f"输出: {output_text}\n\n")
        
        # 添加查询
        prompt_parts.append(f"查询:\n输入: {query}\n输出:")
        
        return "".join(prompt_parts)
    
    @torch.no_grad()
    def generate_with_context(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """基于上下文的文本生成"""
        input_ids = torch.tensor([self.tokenizer.encode(prompt, add_special_tokens=True)], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        past_key_values = None
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            model_inputs = self.model.prepare_inputs_for_generation(
                input_ids=generated,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
            )
            
            logits, _, past_key_values = self.model(**model_inputs)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            
            # Top-k和Top-p采样
            if top_k > 0:
                top_k_vals, top_k_idx = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)))
                filtered = torch.full_like(next_logits, float('-inf'))
                filtered.scatter_(dim=-1, index=top_k_idx, src=top_k_vals)
                next_logits = filtered
            
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(probs, dim=-1)
                cutoff = cumprobs > top_p
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = False
                sorted_logits[cutoff] = float('-inf')
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
            
            if do_sample:
                probs = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_id], dim=1)
            attention_mask = torch.ones_like(generated)
            
            # EOS终止
            if next_id.item() == self.tokenizer.eos_token_id:
                break
        
        return self.tokenizer.decode(generated[0].tolist())

# 重用GPT-2的简单分词器
class GPT3SimpleTokenizer:
    """简单字符级分词器（用于演示）"""
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        for tok in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
            idx = len(self.vocab)
            self.vocab[tok] = idx
            self.inverse_vocab[idx] = tok

    def train(self, texts: List[str]):
        chars = set()
        for t in texts:
            for ch in t:
                chars.add(ch)
        for ch in sorted(chars):
            if ch not in self.vocab:
                idx = len(self.vocab)
                self.vocab[ch] = idx
                self.inverse_vocab[idx] = ch
        return self

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = []
        if add_special_tokens:
            ids.append(self.vocab[self.bos_token])
        for ch in text:
            ids.append(self.vocab.get(ch, self.vocab[self.unk_token]))
        if add_special_tokens:
            ids.append(self.vocab[self.eos_token])
        return ids

     def decode(self, ids: List[int]) -> str:
        out = []
        for i in ids:
            tok = self.inverse_vocab.get(i, self.unk_token)
            if tok not in [self.pad_token, self.bos_token, self.eos_token]:
                out.append(tok)
        return ''.join(out)

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def pad_token_id(self):
        return self.vocab[self.pad_token]

    @property
    def unk_token_id(self):
        return self.vocab[self.unk_token]

    @property
    def bos_token_id(self):
        return self.vocab[self.bos_token]

    @property
    def eos_token_id(self):
        return self.vocab[self.eos_token]

def create_small_gpt3_config():
    """创建适合演示的小型GPT-3配置"""
    return GPT3Config(
        vocab_size=1000,      # 小词汇表
        n_positions=512,      # 较短的上下文
        n_embd=256,          # 较小的嵌入维度
        n_layer=6,           # 较少的层数
        n_head=8,            # 较少的注意力头
        n_inner=1024,        # 较小的MLP维度
        sparse_attention=False,
        gradient_checkpointing=False,
        parallel_attention=True,
        rotary_pct=0.25,
    )

def gpt3_training_demo():
    """GPT-3训练演示"""
    print("=== GPT-3 训练演示 ===\n")
    
    # 创建小型配置用于演示
    config = create_small_gpt3_config()
    
    # 准备训练数据
    texts = [
        "人工智能是计算机科学的一个分支",
        "机器学习是实现人工智能的重要方法",
        "深度学习基于神经网络模型",
        "Transformer是现代自然语言处理的基础",
        "GPT是基于Transformer的生成式预训练模型",
        "注意力机制让模型能够关注重要信息",
        "自然语言处理包括文本理解和生成",
        "预训练模型在下游任务中表现优异"
    ]
    
    # 训练分词器
    tokenizer = GPT3SimpleTokenizer()
    tokenizer.train(texts)
    print(f"词汇表大小: {tokenizer.vocab_size}")
    
    # 更新配置中的词汇表大小
    config.vocab_size = tokenizer.vocab_size
    
    # 创建模型
    model = GPT3LMHeadModel(config)
    print(f"模型参数量: {model.transformer.get_num_params():,}")
    
    # 准备训练数据
    train_data = []
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) <= config.n_positions:
            train_data.append(tokens)
    
    # 简单的训练循环
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    print("\n开始训练...")
    for epoch in range(10):
        total_loss = 0
        for tokens in train_data:
            input_ids = torch.tensor([tokens[:-1]], dtype=torch.long)
            labels = torch.tensor([tokens[1:]], dtype=torch.long)
            
            optimizer.zero_grad()
            logits, loss, _ = model(input_ids=input_ids, labels=labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_data)
        if epoch % 2 == 0:
            print(f"Epoch {epoch + 1}: 平均损失 = {avg_loss:.4f}")
    
    print("训练完成!\n")
    return model, tokenizer

def gpt3_inference_demo(model, tokenizer):
    """GPT-3推理演示"""
    print("=== GPT-3 推理演示 ===\n")
    
    model.eval()
    
    # 简单文本生成
    def generate_text(prompt, max_length=50):
        input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=True)], dtype=torch.long)
        
        with torch.no_grad():
            for _ in range(max_length):
                logits, _, _ = model(input_ids=input_ids)
                next_logits = logits[0, -1, :]
                
                # 使用温度采样
                next_logits = next_logits / 0.8
                probs = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)
                
                if next_id.item() == tokenizer.eos_token_id:
                    break
        
        return tokenizer.decode(input_ids[0].tolist())
    
    # 测试生成
    prompts = [
        "人工智能",
        "机器学习",
        "深度学习",
        "自然语言"
    ]
    
    for prompt in prompts:
        generated = generate_text(prompt, max_length=30)
        print(f"输入: {prompt}")
        print(f"生成: {generated}")
        print("-" * 50)

def gpt3_few_shot_demo():
    """GPT-3 Few-shot学习演示"""
    print("=== GPT-3 Few-shot学习演示 ===\n")
    
    # 创建较大的配置用于few-shot学习
    config = GPT3Config(
        vocab_size=2000,
        n_positions=1024,
        n_embd=512,
        n_layer=8,
        n_head=8,
        n_inner=2048,
        sparse_attention=False,
        gradient_checkpointing=False,
        parallel_attention=True,
        rotary_pct=0.25,
    )
    
    # 创建扩展的训练数据集
    extended_texts = [
        "这是一个积极的评论：这部电影真的很棒！",
        "这是一个消极的评论：这部电影太糟糕了。",
        "这是一个中性的评论：这部电影还可以。",
        "这是一个积极的评论：我非常喜欢这本书。",
        "这是一个消极的评论：这本书让我失望。",
        "翻译成英文：你好 -> Hello",
        "翻译成英文：谢谢 -> Thank you",
        "翻译成英文：再见 -> Goodbye",
        "问答：北京是中国的首都吗？是的。",
        "问答：地球是平的吗？不是的。",
        "分类：苹果是水果。标签：水果",
        "分类：汽车是交通工具。标签：交通工具",
    ]
    
    # 训练扩展的分词器
    tokenizer = GPT3SimpleTokenizer()
    tokenizer.train(extended_texts)
    config.vocab_size = tokenizer.vocab_size
    
    # 创建模型
    model = GPT3LMHeadModel(config)
    print(f"Few-shot模型参数量: {model.transformer.get_num_params():,}")
    
    # 创建上下文学习器
    icl = GPT3InContextLearning(model, tokenizer)
    
    # 情感分析示例
    sentiment_examples = [
        ("这部电影很好看", "积极"),
        ("我不喜欢这个产品", "消极"),
        ("还不错吧", "中性"),
    ]
    
    prompt = icl.few_shot_prompt(
        task_description="情感分析任务：判断文本的情感倾向",
        examples=sentiment_examples,
        query="这个服务质量很高"
    )
    
    print("Few-shot提示示例:")
    print(prompt)
    print("\n" + "="*60 + "\n")

def gpt3_architecture_analysis():
    """GPT-3架构分析"""
    print("=== GPT-3 架构分析 ===\n")
    
    # 不同规模的GPT-3配置
    configs = {
        "GPT-3 Small (125M)": GPT3Config(
            n_embd=768, n_layer=12, n_head=12, n_positions=2048
        ),
        "GPT-3 Medium (350M)": GPT3Config(
            n_embd=1024, n_layer=24, n_head=16, n_positions=2048
        ),
        "GPT-3 Large (760M)": GPT3Config(
            n_embd=1280, n_layer=36, n_head=20, n_positions=2048
        ),
        "GPT-3 XL (1.3B)": GPT3Config(
            n_embd=2048, n_layer=24, n_head=32, n_positions=2048
        ),
        "GPT-3 2.7B": GPT3Config(
            n_embd=2560, n_layer=32, n_head=32, n_positions=2048
        ),
        "GPT-3 6.7B": GPT3Config(
            n_embd=4096, n_layer=32, n_head=32, n_positions=2048
        ),
        "GPT-3 13B": GPT3Config(
            n_embd=5140, n_layer=40, n_head=40, n_positions=2048
        ),
        "GPT-3 175B": GPT3Config(
            n_embd=12288, n_layer=96, n_head=96, n_positions=2048
        ),
    }
    
    print("GPT-3不同规模模型配置:")
    print(f"{'模型':<20} {'层数':<6} {'隐藏维度':<8} {'注意力头':<8} {'参数量':<15}")
    print("-" * 65)
    
    for name, config in configs.items():
        # 估算参数量（简化计算）
        embed_params = config.vocab_size * config.n_embd
        pos_embed_params = config.n_positions * config.n_embd
        layer_params = config.n_layer * (
            # 注意力层
            4 * config.n_embd * config.n_embd +  # QKV + output projection
            # MLP层
            config.n_embd * config.n_inner + config.n_inner * config.n_embd +
            # LayerNorm
            4 * config.n_embd
        )
        total_params = embed_params + pos_embed_params + layer_params
        
        print(f"{name:<20} {config.n_layer:<6} {config.n_embd:<8} {config.n_head:<8} {total_params/1e6:.1f}M")
    
    print("\nGPT-3关键创新:")
    print("1. 极大的模型规模 (175B参数)")
    print("2. 高质量的训练数据 (约45TB文本)")
    print("3. 强大的Few-shot学习能力")
    print("4. 上下文学习 (In-Context Learning)")
    print("5. 稀疏注意力机制优化")
    print("6. 模型并行和梯度检查点")
    print("7. 旋转位置编码 (RoPE)")
    print("8. 并行注意力和MLP计算")

def main():
    """主函数"""
    print("🚀 GPT-3 模型实现与演示")
    print("=" * 60)
    
    try:
        # 1. 架构分析
        gpt3_architecture_analysis()
        print("\n" + "="*60 + "\n")
        
        # 2. 训练演示
        model, tokenizer = gpt3_training_demo()
        print("="*60 + "\n")
        
        # 3. 推理演示
        gpt3_inference_demo(model, tokenizer)
        print("="*60 + "\n")
        
        # 4. Few-shot学习演示
        gpt3_few_shot_demo()
        
        print("✅ GPT-3演示完成！")
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()