"""
Qwen3 模型演示程序
展示如何使用 Qwen3 模型进行文本生成和对话
包含基本的tokenizer实现和文本生成示例
"""

import torch
import torch.nn.functional as F
import json
import re
from typing import List, Dict, Optional, Union
import math

# 导入 Qwen3 模型组件
try:
    from qwen3_model import Qwen3ForCausalLM, Qwen3Config, create_qwen3_model
    from qwen3_core_components import prepare_attention_mask
except ImportError:
    print("注意：无法导入Qwen3模型组件，某些功能可能不可用")


class SimpleTokenizer:
    """
    简单的分词器实现
    用于演示目的，实际使用中应该使用更完善的分词器如SentencePiece
    """
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        
        # 特殊token
        self.pad_token_id = 0
        self.bos_token_id = 1  # Begin of sequence
        self.eos_token_id = 2  # End of sequence
        self.unk_token_id = 3  # Unknown token
        
        # 特殊token字符串
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        
        # 构建基础词汇表
        self._build_vocab()
    
    def _build_vocab(self):
        """构建基础词汇表"""
        # 特殊token
        special_tokens = [
            self.pad_token, self.bos_token, self.eos_token, self.unk_token
        ]
        
        # 基础字符集（ASCII + 常用中文字符）
        basic_chars = []
        
        # ASCII字符
        for i in range(32, 127):  # 可打印ASCII字符
            basic_chars.append(chr(i))
        
        # 常用中文字符（简化版）
        common_chinese = "的一是不了人我在有他这为之大来以个中上们到说国和地也子时道出而要于就下得可你年生自会那后能对着事其里所去行过家十用发天如然作方成者多日都三小军二无同么经法当起与好看学进种将还分此心前面又定见只主没公从"
        for char in common_chinese:
            if char not in basic_chars:
                basic_chars.append(char)
        
        # 构建token到ID的映射
        self.token_to_id = {}
        self.id_to_token = {}
        
        # 添加特殊token
        for i, token in enumerate(special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        # 添加基础字符
        current_id = len(special_tokens)
        for char in basic_chars:
            if char not in self.token_to_id:
                self.token_to_id[char] = current_id
                self.id_to_token[current_id] = char
                current_id += 1
        
        # 填充到指定词汇表大小
        while current_id < self.vocab_size:
            placeholder_token = f"<unused_{current_id}>"
            self.token_to_id[placeholder_token] = current_id
            self.id_to_token[current_id] = placeholder_token
            current_id += 1
    
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """
        将文本编码为token ID序列
        
        Args:
            text: 输入文本
            add_bos: 是否添加开始token
            add_eos: 是否添加结束token
        
        Returns:
            token ID列表
        """
        # 简单的字符级别tokenization
        tokens = []
        
        if add_bos:
            tokens.append(self.bos_token_id)
        
        for char in text:
            token_id = self.token_to_id.get(char, self.unk_token_id)
            tokens.append(token_id)
        
        if add_eos:
            tokens.append(self.eos_token_id)
        
        return tokens
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """
        将token ID序列解码为文本
        
        Args:
            token_ids: token ID序列
            skip_special_tokens: 是否跳过特殊token
        
        Returns:
            解码后的文本
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        tokens = []
        special_token_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_token_ids:
                continue
            
            token = self.id_to_token.get(token_id, self.unk_token)
            if not (skip_special_tokens and token.startswith("<") and token.endswith(">")):
                tokens.append(token)
        
        return "".join(tokens)
    
    def batch_encode(self, texts: List[str], padding: bool = True, 
                    max_length: Optional[int] = None, add_bos: bool = True, 
                    add_eos: bool = False) -> Dict[str, torch.Tensor]:
        """
        批量编码文本
        
        Args:
            texts: 文本列表
            padding: 是否进行填充
            max_length: 最大长度
            add_bos: 是否添加开始token
            add_eos: 是否添加结束token
        
        Returns:
            包含input_ids和attention_mask的字典
        """
        encoded_batch = []
        
        for text in texts:
            encoded = self.encode(text, add_bos=add_bos, add_eos=add_eos)
            encoded_batch.append(encoded)
        
        if max_length is None:
            max_length = max(len(seq) for seq in encoded_batch)
        
        # 填充或截断
        input_ids = []
        attention_mask = []
        
        for encoded in encoded_batch:
            if len(encoded) > max_length:
                # 截断
                encoded = encoded[:max_length]
                mask = [1] * max_length
            else:
                # 填充
                pad_length = max_length - len(encoded)
                mask = [1] * len(encoded) + [0] * pad_length
                encoded = encoded + [self.pad_token_id] * pad_length
            
            input_ids.append(encoded)
            attention_mask.append(mask)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }


class Qwen3ChatBot:
    """
    基于Qwen3模型的简单聊天机器人
    """
    
    def __init__(self, model: 'Qwen3ForCausalLM', tokenizer: SimpleTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.conversation_history = []
        
        # 生成参数
        self.max_new_tokens = 100
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 50
        self.do_sample = True
    
    def chat(self, user_input: str, system_prompt: str = "你是一个有用的AI助手。") -> str:
        """
        与用户进行对话
        
        Args:
            user_input: 用户输入
            system_prompt: 系统提示
        
        Returns:
            AI回复
        """
        # 构建对话提示
        if not self.conversation_history:
            # 首次对话，添加系统提示
            prompt = f"{system_prompt}\n\n用户: {user_input}\nAI:"
        else:
            # 继续对话
            prompt = f"用户: {user_input}\nAI:"
        
        # 编码输入
        input_ids = torch.tensor([self.tokenizer.encode(prompt, add_bos=True)], dtype=torch.long)
        
        # 生成回复
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # 提取AI回复部分
        ai_response = generated_text[len(self.tokenizer.decode(input_ids[0], skip_special_tokens=True)):]
        
        # 清理回复
        ai_response = ai_response.strip()
        if ai_response.startswith("AI:"):
            ai_response = ai_response[3:].strip()
        
        # 在遇到新的用户输入前停止
        if "用户:" in ai_response:
            ai_response = ai_response.split("用户:")[0].strip()
        
        # 更新对话历史
        self.conversation_history.append({
            "user": user_input,
            "ai": ai_response
        })
        
        return ai_response
    
    def clear_history(self):
        """清除对话历史"""
        self.conversation_history = []
    
    def set_generation_params(self, **kwargs):
        """设置生成参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


def create_demo_model() -> 'Qwen3ForCausalLM':
    """
    创建用于演示的Qwen3模型
    使用较小的配置以便快速测试
    """
    config = Qwen3Config(
        vocab_size=32000,        # 词汇表大小
        hidden_size=512,         # 隐藏层维度  
        intermediate_size=1024,  # 前馈网络中间层维度
        num_hidden_layers=6,     # 隐藏层数量
        num_attention_heads=8,   # 注意力头数量
        num_key_value_heads=8,   # 键值头数量
        max_position_embeddings=2048,  # 最大位置编码长度
        rope_theta=10000.0,      # RoPE基础频率
        rms_norm_eps=1e-6,       # RMSNorm的epsilon
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False
    )
    
    model = Qwen3ForCausalLM(config)
    return model


def text_generation_demo():
    """文本生成演示"""
    print("=== Qwen3 文本生成演示 ===\n")
    
    try:
        # 创建模型和分词器
        print("正在创建模型...")
        model = create_demo_model()
        tokenizer = SimpleTokenizer(vocab_size=32000)
        
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"词汇表大小: {tokenizer.vocab_size}")
        
        # 测试文本生成
        test_prompts = [
            "今天天气",
            "人工智能",
            "Python编程",
            "机器学习是"
        ]
        
        print(f"\n开始文本生成测试...")
        
        for prompt in test_prompts:
            print(f"\n输入提示: '{prompt}'")
            
            # 编码输入
            input_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long)
            print(f"输入token数: {input_ids.shape[1]}")
            
            # 生成文本
            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=20,  # 生成较少token以便观察
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # 解码生成的文本
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            new_text = generated_text[len(prompt):]  # 只显示新生成的部分
            
            print(f"生成结果: '{generated_text}'")
            print(f"新增文本: '{new_text}'")
            print(f"总token数: {generated.shape[1]}")
    
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        print("这可能是由于PyTorch未安装或其他依赖问题导致的")


def chat_demo():
    """聊天演示"""
    print("\n=== Qwen3 聊天演示 ===\n")
    
    try:
        # 创建聊天机器人
        print("正在初始化聊天机器人...")
        model = create_demo_model()
        tokenizer = SimpleTokenizer(vocab_size=32000)
        chatbot = Qwen3ChatBot(model, tokenizer)
        
        # 设置生成参数
        chatbot.set_generation_params(
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        print("聊天机器人已准备就绪！")
        print("(输入 'quit' 退出，输入 'clear' 清除历史)\n")
        
        # 模拟对话
        demo_conversations = [
            "你好，请介绍一下自己",
            "你能做什么？", 
            "解释一下机器学习",
            "quit"
        ]
        
        for user_input in demo_conversations:
            print(f"用户: {user_input}")
            
            if user_input.lower() == 'quit':
                print("对话结束！")
                break
            elif user_input.lower() == 'clear':
                chatbot.clear_history()
                print("对话历史已清除！")
                continue
            
            # 获取AI回复
            try:
                ai_response = chatbot.chat(user_input)
                print(f"AI: {ai_response}\n")
            except Exception as e:
                print(f"生成回复时出错: {e}\n")
        
        # 显示对话历史
        print("对话历史:")
        for i, conv in enumerate(chatbot.conversation_history, 1):
            print(f"{i}. 用户: {conv['user']}")
            print(f"   AI: {conv['ai']}")
    
    except Exception as e:
        print(f"聊天演示过程中发生错误: {e}")
        print("这可能是由于PyTorch未安装或其他依赖问题导致的")


def model_info_demo():
    """模型信息演示"""
    print("\n=== Qwen3 模型信息 ===\n")
    
    try:
        # 创建模型
        model = create_demo_model()
        tokenizer = SimpleTokenizer()
        
        # 模型配置信息
        config = model.config
        print("模型配置:")
        print(f"  词汇表大小: {config.vocab_size:,}")
        print(f"  隐藏层维度: {config.hidden_size}")
        print(f"  隐藏层数量: {config.num_hidden_layers}")
        print(f"  注意力头数: {config.num_attention_heads}")
        print(f"  中间层维度: {config.intermediate_size}")
        print(f"  最大序列长度: {config.max_position_embeddings}")
        
        # 参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n参数统计:")
        print(f"  总参数数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
        
        # 分词器信息
        print(f"\n分词器信息:")
        print(f"  词汇表大小: {tokenizer.vocab_size}")
        print(f"  特殊token: {tokenizer.pad_token}, {tokenizer.bos_token}, {tokenizer.eos_token}, {tokenizer.unk_token}")
        
        # 测试编码解码
        test_text = "你好，世界！Hello, World!"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        
        print(f"\n编码解码测试:")
        print(f"  原文: {test_text}")
        print(f"  编码: {encoded}")
        print(f"  解码: {decoded}")
        print(f"  长度: {len(encoded)} tokens")
    
    except Exception as e:
        print(f"模型信息演示过程中发生错误: {e}")


if __name__ == "__main__":
    print("🤖 Qwen3 模型演示程序")
    print("=" * 50)
    
    # 运行各种演示
    model_info_demo()
    text_generation_demo()
    chat_demo()
    
    print("\n✅ 演示完成！")
    print("\n说明：")
    print("- 这是一个简化的Qwen3实现，用于学习和演示目的")
    print("- 实际使用中需要预训练的权重和更完善的分词器")
    print("- 生成的文本质量取决于模型训练程度")
    print("- 如果遇到PyTorch相关错误，请确保已正确安装PyTorch")