"""
GPT-1 (Generative Pre-trained Transformer 1) 实现

这是OpenAI在2018年发布的第一个GPT模型的实现，包含：
- Transformer Decoder架构
- 多头自注意力机制
- 位置编码
- 文本生成功能
- 简单的训练示例

参考论文：Improving Language Understanding by Generative Pre-Training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple
import json
import re
from collections import Counter

class GPT1Config:
    """GPT-1模型配置类"""
    def __init__(
        self,
        vocab_size: int = 40000,      # 词汇表大小
        n_positions: int = 512,       # 最大序列长度
        n_embd: int = 768,           # 嵌入维度
        n_layer: int = 12,           # Transformer层数
        n_head: int = 12,            # 注意力头数
        n_inner: int = None,         # FFN中间层维度
        activation_function: str = "gelu",  # 激活函数
        resid_pdrop: float = 0.1,    # 残差连接dropout
        embd_pdrop: float = 0.1,     # 嵌入dropout
        attn_pdrop: float = 0.1,     # 注意力dropout
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner if n_inner is not None else 4 * n_embd
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, config: GPT1Config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        
        assert config.n_embd % config.n_head == 0
        
        # 查询、键、值投影
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        # 创建因果掩码（下三角矩阵）
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_positions, config.n_positions))
            .view(1, 1, config.n_positions, config.n_positions)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch_size, seq_len, n_embd
        
        # 计算查询、键、值
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # 重塑为多头形式 (B, T, C) -> (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用因果掩码
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # 应用注意力权重
        y = torch.matmul(att, v)  # (B, nh, T, hs)
        
        # 重新组合头 (B, nh, T, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 输出投影
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """多层感知机（前馈网络）"""
    
    def __init__(self, config: GPT1Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd)
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.activation = self._get_activation(config.activation_function)
    
    def _get_activation(self, activation_function: str):
        if activation_function == "relu":
            return nn.ReLU()
        elif activation_function == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer块（层）"""
    
    def __init__(self, config: GPT1Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 注意力子层（带残差连接）
        x = x + self.attn(self.ln_1(x))
        # 前馈子层（带残差连接）
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT1Model(nn.Module):
    """GPT-1模型主体"""
    
    def __init__(self, config: GPT1Config):
        super().__init__()
        self.config = config
        
        # 词嵌入
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # 位置嵌入
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        # 嵌入dropout
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # Transformer块
        self.h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        
        # 最终层归一化
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # 语言模型头（输出投影）
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 初始化权重
        self.apply(self._init_weights)
        
        print(f"GPT-1模型初始化完成，参数量: {self.get_num_params():,}")
    
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
    
    def get_num_params(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters())
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            input_ids: 输入token ids, shape (batch_size, seq_len)
            position_ids: 位置ids, shape (batch_size, seq_len)
            labels: 标签（用于计算损失）, shape (batch_size, seq_len)
        
        Returns:
            logits: 输出logits, shape (batch_size, seq_len, vocab_size)
            loss: 损失值（如果提供了labels）
        """
        device = input_ids.device
        b, t = input_ids.size()
        
        assert t <= self.config.n_positions, f"序列长度 {t} 超过最大位置数 {self.config.n_positions}"
        
        # 生成位置ids
        if position_ids is None:
            position_ids = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        
        # 词嵌入和位置嵌入
        tok_emb = self.wte(input_ids)  # (b, t, n_embd)
        pos_emb = self.wpe(position_ids)  # (b, t, n_embd)
        
        # 组合嵌入
        x = self.drop(tok_emb + pos_emb)
        
        # 通过Transformer块
        for block in self.h:
            x = block(x)
        
        # 最终层归一化
        x = self.ln_f(x)
        
        # 语言模型头
        logits = self.lm_head(x)  # (b, t, vocab_size)
        
        # 计算损失
        loss = None
        if labels is not None:
            # 移位标签用于下一个token预测
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 计算交叉熵损失
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return logits, loss

class SimpleTokenizer:
    """简单的分词器（用于演示）"""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        self.unk_token = "<UNK>"
        self.pad_token = "<PAD>"
        self.eos_token = "<EOS>"
        self.bos_token = "<BOS>"
        
        # 特殊token
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for i, token in enumerate(special_tokens):
            self.vocab[token] = i
            self.inverse_vocab[i] = token
    
    def train(self, texts: list):
        """从文本语料库中构建词汇表"""
        print("开始构建词汇表...")
        
        # 收集所有单词
        word_counts = Counter()
        for text in texts:
            # 简单的单词分割
            words = re.findall(r'\b\w+\b|[.,!?;]', text.lower())
            word_counts.update(words)
        
        # 选择最频繁的单词
        most_common = word_counts.most_common(self.vocab_size - len(self.vocab))
        
        # 添加到词汇表
        for word, count in most_common:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
                self.inverse_vocab[len(self.inverse_vocab)] = word
        
        print(f"词汇表构建完成，共 {len(self.vocab)} 个token")
        return self
    
    def encode(self, text: str) -> list:
        """将文本编码为token ids"""
        words = re.findall(r'\b\w+\b|[.,!?;]', text.lower())
        token_ids = []
        
        for word in words:
            token_id = self.vocab.get(word, self.vocab[self.unk_token])
            token_ids.append(token_id)
        
        return token_ids
    
    def decode(self, token_ids: list) -> str:
        """将token ids解码为文本"""
        words = []
        for token_id in token_ids:
            word = self.inverse_vocab.get(token_id, self.unk_token)
            if word not in [self.pad_token, self.bos_token, self.eos_token]:
                words.append(word)
        
        return ' '.join(words)
    
    def get_vocab_size(self) -> int:
        return len(self.vocab)

class GPT1TextGenerator:
    """GPT-1文本生成器"""
    
    def __init__(self, model: GPT1Model, tokenizer: SimpleTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        do_sample: bool = True
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示文本
            max_length: 最大生成长度
            temperature: 温度参数（控制随机性）
            top_k: top-k采样
            do_sample: 是否使用采样
        
        Returns:
            生成的文本
        """
        self.model.eval()
        
        # 编码输入
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.long)
        
        generated_ids = input_ids
        
        with torch.no_grad():
            for _ in range(max_length):
                # 前向传播
                logits, _ = self.model(generated_ids)
                
                # 获取最后一个位置的logits
                logits = logits[0, -1, :] / temperature
                
                # Top-k过滤
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits[top_k_indices] = top_k_logits
                
                # 采样或贪心
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token_id = torch.multinomial(probs, 1)
                else:
                    next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
                
                # 添加新token
                generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)
                
                # 检查是否达到最大长度
                if generated_ids.size(1) >= self.model.config.n_positions:
                    break
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(generated_ids[0].tolist())
        return generated_text

def create_sample_dataset():
    """创建示例数据集"""
    sample_texts = [
        "人工智能是计算机科学的一个分支，它试图创造智能机器。",
        "机器学习是人工智能的一个子领域，专注于算法的开发。",
        "深度学习使用神经网络来模拟人脑的工作方式。",
        "自然语言处理帮助计算机理解和生成人类语言。",
        "计算机视觉让机器能够识别和理解图像内容。",
        "强化学习通过奖励和惩罚来训练智能代理。",
        "神经网络是由相互连接的节点组成的计算模型。",
        "数据挖掘从大量数据中发现有用的模式和信息。",
        "云计算提供按需访问计算资源的服务模式。",
        "大数据技术处理传统方法无法处理的大规模数据集。",
        "区块链是一种分布式账本技术，具有去中心化特性。",
        "物联网连接各种设备，使它们能够交换数据。",
        "虚拟现实创造沉浸式的三维体验环境。",
        "增强现实将数字信息叠加到真实世界中。",
        "量子计算利用量子力学原理进行信息处理。"
    ]
    return sample_texts

def train_simple_gpt1():
    """简单的GPT-1训练示例"""
    print("=== GPT-1 训练演示 ===")
    
    # 创建示例数据
    texts = create_sample_dataset()
    
    # 初始化分词器
    tokenizer = SimpleTokenizer(vocab_size=1000)
    tokenizer.train(texts)
    
    # 创建模型配置（小模型用于演示）
    config = GPT1Config(
        vocab_size=tokenizer.get_vocab_size(),
        n_positions=128,
        n_embd=256,
        n_layer=6,
        n_head=8,
        n_inner=1024
    )
    
    # 初始化模型
    model = GPT1Model(config)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # 准备训练数据
    train_data = []
    for text in texts:
        token_ids = tokenizer.encode(text)
        if len(token_ids) > 1:  # 确保至少有输入和标签
            train_data.append(token_ids)
    
    print(f"训练数据准备完成，共 {len(train_data)} 个样本")
    
    # 简单训练循环
    model.train()
    num_epochs = 50
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for token_ids in train_data:
            if len(token_ids) < 2:
                continue
                
            # 准备输入和标签
            max_len = min(len(token_ids), config.n_positions)
            input_ids = torch.tensor([token_ids[:max_len]], dtype=torch.long)
            labels = input_ids.clone()
            
            # 前向传播
            logits, loss = model(input_ids, labels=labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, 平均损失: {avg_loss:.4f}")
    
    print("训练完成！")
    
    # 测试文本生成
    generator = GPT1TextGenerator(model, tokenizer)
    
    test_prompts = [
        "人工智能",
        "机器学习",
        "深度学习"
    ]
    
    print("\n=== 文本生成测试 ===")
    for prompt in test_prompts:
        print(f"\n输入: {prompt}")
        generated = generator.generate(
            prompt, 
            max_length=20, 
            temperature=0.8, 
            top_k=10
        )
        print(f"输出: {generated}")
    
    return model, tokenizer, generator

def demonstrate_gpt1_architecture():
    """演示GPT-1架构的各个组件"""
    print("=== GPT-1 架构演示 ===")
    
    # 创建配置
    config = GPT1Config(
        vocab_size=1000,
        n_positions=128,
        n_embd=256,
        n_layer=4,
        n_head=8
    )
    
    print(f"模型配置:")
    print(f"  词汇表大小: {config.vocab_size}")
    print(f"  最大序列长度: {config.n_positions}")
    print(f"  嵌入维度: {config.n_embd}")
    print(f"  Transformer层数: {config.n_layer}")
    print(f"  注意力头数: {config.n_head}")
    print(f"  前馈网络维度: {config.n_inner}")
    
    # 创建模型
    model = GPT1Model(config)
    
    print(f"\n模型结构:")
    print(f"  总参数量: {model.get_num_params():,}")
    
    # 测试前向传播
    batch_size = 2
    seq_len = 50
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\n测试前向传播:")
    print(f"  输入形状: {input_ids.shape}")
    
    with torch.no_grad():
        logits, _ = model(input_ids)
        print(f"  输出形状: {logits.shape}")
        print(f"  输出范围: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    
    # 演示注意力机制
    print(f"\n=== 注意力机制演示 ===")
    attn_layer = MultiHeadAttention(config)
    
    # 创建示例输入
    x = torch.randn(1, 10, config.n_embd)
    print(f"输入形状: {x.shape}")
    
    with torch.no_grad():
        attn_output = attn_layer(x)
        print(f"注意力输出形状: {attn_output.shape}")
    
    print(f"注意力头数: {config.n_head}")
    print(f"每个头的维度: {config.n_embd // config.n_head}")

if __name__ == "__main__":
    print("GPT-1 (Generative Pre-trained Transformer 1) 实现")
    print("=" * 50)
    
    # 演示模型架构
    demonstrate_gpt1_architecture()
    
    print("\n" + "=" * 50)
    
    # 训练和测试
    model, tokenizer, generator = train_simple_gpt1()
    
    print("\n" + "=" * 50)
    print("GPT-1 实现完成！")
    print("\n主要特性:")
    print("1. 完整的Transformer Decoder架构")
    print("2. 多头自注意力机制")
    print("3. 位置编码和词嵌入")
    print("4. 层归一化和残差连接")
    print("5. 文本生成功能")
    print("6. 简单的训练示例")
    
    print("\n这个实现展示了GPT-1的核心概念，")
    print("是理解现代大语言模型的重要基础！")