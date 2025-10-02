# ... existing code ...
class GPT2Model(nn.Module):
    """GPT-2模型主体"""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        
        # 词嵌入
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # 位置嵌入
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        
        # 嵌入dropout
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # Transformer块
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.n_layer)])
        
        # 最终层归一化
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # 模型并行
        self.model_parallel = False
        self.device_map = None
        # 新增：默认关闭梯度检查点，避免属性缺失错误
        self.gradient_checkpointing = False
        
        # 初始化权重
        self.post_init()
        
        print(f"GPT-2模型初始化完成，参数量: {self.get_num_params():,}")
# ... existing code ...

class GPT2LMHeadModel(nn.Module):
    """带有语言模型头的GPT-2模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 模型并行
        self.model_parallel = False
        self.device_map = None
        
        # 初始化权重并应用最终处理
        self.post_init()
    
    def post_init(self):
        """初始化权重和应用最终处理"""
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
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        
        # 只传递最后一个token（如果已经有past_key_values）
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
        
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        
        # 如果传递了`inputs_embeds`，则只在第一次生成步骤中使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", True),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[Tuple[torch.Tensor]]]]:
        """
        返回:
            logits: (batch, seq_len, vocab)
            loss: 可选
            past_key_values: 用于加速生成
        """
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
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
# ... existing code ...

class GPT2SimpleTokenizer:
    """简单字符级分词器（用于演示，接近GPT-2字节级思想）"""
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
        # 稳定排序，确保一致性
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
            if tok in [self.pad_token, self.bos_token, self.eos_token]:
                continue
            out.append(tok)
        return "".join(out)

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.vocab[self.pad_token]

    @property
    def eos_token_id(self) -> int:
        return self.vocab[self.eos_token]

class GPT2TextGenerator:
    """GPT-2 文本生成器（支持缓存、温度、top-k、top-p）"""
    def __init__(self, model: GPT2LMHeadModel, tokenizer: GPT2SimpleTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
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

            # top-k
            if top_k > 0:
                top_k_vals, top_k_idx = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)))
                filtered = torch.full_like(next_logits, float('-inf'))
                filtered.scatter_(dim=-1, index=top_k_idx, src=top_k_vals)
                next_logits = filtered

            # top-p (核采样)
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(probs, dim=-1)
                cutoff = cumprobs > top_p
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = False
                sorted_logits[cutoff] = float('-inf')
                # 恢复到原索引位置
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

            if do_sample:
                probs = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_id], dim=1)
            attention_mask = torch.ones_like(generated)

            # EOS 终止
            if next_id.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated[0].tolist())

def create_sample_dataset() -> List[str]:
    return [
        "人工智能是计算机科学的一个分支，它试图创造智能机器。",
        "机器学习是人工智能的一个子领域，专注于算法的开发。",
        "深度学习使用神经网络来模拟人脑的工作方式。",
        "自然语言处理帮助计算机理解和生成人类语言。",
        "计算机视觉让机器能够识别和理解图像内容。",
        "强化学习通过奖励和惩罚来训练智能代理。",
        "神经网络是由相互连接的节点组成的计算模型。",
        "数据挖掘从大量数据中发现有用的模式和信息。",
    ]

def train_simple_gpt2():
    """简单的GPT-2训练与生成演示（小模型）"""
    print("=== GPT-2 训练演示（小模型）===")
    texts = create_sample_dataset()

    tokenizer = GPT2SimpleTokenizer().train(texts)
    vocab_size = tokenizer.get_vocab_size()

    # 使用小配置以便快速演示
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=128,
        n_embd=256,
        n_layer=6,
        n_head=8,
        n_inner=4 * 256,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )

    model = GPT2LMHeadModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # 构造训练样本
    train_data = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=True)
        if len(ids) > 1:
            train_data.append(ids)

    model.train()
    num_epochs = 30
    for epoch in range(num_epochs):
        total_loss = 0.0
        steps = 0
        for ids in train_data:
            max_len = min(len(ids), config.n_positions)
            input_ids = torch.tensor([ids[:max_len]], dtype=torch.long)
            labels = input_ids.clone()

            logits, loss, _ = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                labels=labels,
                use_cache=False,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - 平均损失: {total_loss/max(steps,1):.4f}")

    print("训练完成！\n=== 文本生成测试 ===")
    generator = GPT2TextGenerator(model, tokenizer)
    for prompt in ["人工智能", "机器学习", "深度学习"]:
        out = generator.generate(prompt, max_new_tokens=30, temperature=0.8, top_k=20, top_p=0.9)
        print(f"\n输入: {prompt}\n输出: {out}")

    return model, tokenizer

if __name__ == "__main__":
    print("GPT-2 (Generative Pre-trained Transformer 2) 实现")
    print("=" * 50)
    _ = train_simple_gpt2()
    print("\n实现完成：包含模型、训练与生成演示。")