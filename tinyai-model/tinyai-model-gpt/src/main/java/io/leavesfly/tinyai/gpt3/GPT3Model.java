package io.leavesfly.tinyai.gpt3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.ArrayList;
import java.util.List;

/**
 * GPT-3模型类
 * 
 * GPT-3是OpenAI于2020年发布的大规模语言模型,引入了多项创新:
 * 1. 并行注意力计算 - 提升计算效率
 * 2. 超大规模 - 最大175B参数
 * 3. Few-shot学习能力 - 无需微调即可执行新任务
 * 4. 上下文学习 - 从示例中快速理解任务模式
 * 
 * 本实现基于TinyAI框架,提供多规模配置:
 * - 小型: 125M参数 (学习测试)
 * - 中型: 350M参数 (实用应用)
 * - 大型: 1.3B参数 (高质量生成)
 * - 超大型: 175B参数 (顶级性能)
 * 
 * @author leavesfly
 * @version 1.0
 */
public class GPT3Model extends Model {
    
    private final GPT3Config config;
    private final GPT3MainBlock gpt3Block;
    
    /**
     * 构造GPT-3模型
     * 
     * @param name 模型名称
     * @param config GPT-3配置
     */
    public GPT3Model(String name, GPT3Config config) {
        super(name, new GPT3MainBlock(name + "_main", config));
        this.config = config;
        this.gpt3Block = (GPT3MainBlock) getModule();

        // 校验配置合法性
        config.validate();

        // 设置模型描述
        setDescription(buildDescription());
    }
    
    /**
     * 构建模型描述
     */
    private String buildDescription() {
        return String.format(
            "GPT-3语言模型 | 参数量: %s | 层数: %d | 维度: %d | 注意力头: %d | " +
            "架构: %s | 特性: %s",
            formatParamCount(config.estimateParameterCount()),
            config.getNLayer(),
            config.getNEmbd(),
            config.getNHead(),
            config.isParallelAttention() ? "并行计算" : "串行（GPT-2风格）",
            buildFeatureList()
        );
    }
    
    /**
     * 构建特性列表字符串
     */
    private String buildFeatureList() {
        StringBuilder features = new StringBuilder();
        if (config.isParallelAttention()) {
            features.append("并行Attn+MLP");
        }
        if (config.isUseRotaryEmbedding()) {
            if (features.length() > 0) features.append(", ");
            features.append("RoPE");
        }
        if (config.isSparseAttention()) {
            if (features.length() > 0) features.append(", ");
            features.append("稀疏注意力");
        }
        if (config.isGradientCheckpointing()) {
            if (features.length() > 0) features.append(", ");
            features.append("梯度检查点");
        }
        if (features.length() == 0) {
            features.append("标准");
        }
        return features.toString();
    }
    
    /**
     * 格式化参数数量
     */
    private String formatParamCount(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.2fB", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.2fM", count / 1_000_000.0);
        } else {
            return String.format("%,d", count);
        }
    }
    
    // ==================== 工厂方法 ====================
    
    /**
     * 创建小型GPT-3模型（125M参数）
     * 配置: 768维, 12层, 12头
     * 适用: 学习测试、快速实验
     * 
     * @param name 模型名称
     * @return GPT-3模型实例
     */
    public static GPT3Model createSmallModel(String name) {
        GPT3Config config = GPT3Config.createSmallConfig();
        return new GPT3Model(name, config);
    }
    
    /**
     * 创建中型GPT-3模型（350M参数）
     * 配置: 1024维, 24层, 16头
     * 适用: 实用应用、生产环境
     * 
     * @param name 模型名称
     * @return GPT-3模型实例
     */
    public static GPT3Model createMediumModel(String name) {
        GPT3Config config = GPT3Config.createMediumConfig();
        return new GPT3Model(name, config);
    }
    
    /**
     * 创建大型GPT-3模型（1.3B参数）
     * 配置: 2048维, 24层, 32头
     * 适用: 高质量生成、复杂任务
     * 特性: 启用RoPE、稀疏注意力、梯度检查点
     * 
     * @param name 模型名称
     * @return GPT-3模型实例
     */
    public static GPT3Model createLargeModel(String name) {
        GPT3Config config = GPT3Config.createLargeConfig();
        return new GPT3Model(name, config);
    }
    
    /**
     * 创建超大型GPT-3模型（175B参数）
     * 配置: 12288维, 96层, 96头
     * 适用: 顶级性能、研究实验
     * 特性: 全部优化特性启用
     * 
     * @param name 模型名称
     * @return GPT-3模型实例
     */
    public static GPT3Model createXLModel(String name) {
        GPT3Config config = GPT3Config.createXLConfig();
        return new GPT3Model(name, config);
    }
    
    // ==================== 推理方法 ====================
    
    /**
     * 预测下一个token的概率分布
     * 
     * @param tokenIds 输入token序列 (batch_size, seq_len)
     * @return logits (batch_size, seq_len, vocab_size)
     */
    public Variable predict(Variable tokenIds) {
        return forward(tokenIds);
    }
    
    /**
     * 生成文本序列（简化实现：贪婪解码，不使用 KV Cache）
     *
     * @param promptIds    提示序列 (batch_size, prompt_len)
     * @param maxNewTokens 最大生成 Token 数
     * @return 生成的完整序列（含提示）
     */
    public NdArray generateSequence(NdArray promptIds, int maxNewTokens) {
        int batchSize = promptIds.getShape().getDimension(0);
        int promptLen = promptIds.getShape().getDimension(1);

        float[][] generatedSeq = new float[batchSize][promptLen + maxNewTokens];

        // 复制提示序列
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < promptLen; t++) {
                generatedSeq[b][t] = promptIds.get(b, t);
            }
        }

        // 逐步生成（无 KV Cache，每步全量计算）
        for (int i = 0; i < maxNewTokens; i++) {
            int currentLen = promptLen + i;
            float[][] currentInput = new float[batchSize][currentLen];
            for (int b = 0; b < batchSize; b++) {
                System.arraycopy(generatedSeq[b], 0, currentInput[b], 0, currentLen);
            }
            Variable logits = predict(new Variable(NdArray.of(currentInput)));
            NdArray logitsArray = logits.getValue();
            for (int b = 0; b < batchSize; b++) {
                int nextToken = argmax(logitsArray, b, currentLen - 1);
                generatedSeq[b][currentLen] = nextToken;
            }
        }

        return NdArray.of(generatedSeq);
    }

    /**
     * 带 KV Cache 的自回归文本生成（加速推理）
     *
     * 利用 KV Cache 避免重复计算历史 Token 的 K/V，每步只计算最新一个 Token：
     * 1. 处理 Prompt：全量计算，填充 KV Cache
     * 2. 逐步生成：每步仅输入上一步生成的 Token，利用 Cache 快速计算
     *
     * 计算效率：O(n) 每步 vs 无 Cache 时的 O(n²)
     *
     * @param promptIds    提示序列 (batch_size, prompt_len)
     * @param maxNewTokens 最大生成新 Token 数
     * @return 生成的完整序列（含提示），Shape: (batch_size, prompt_len + maxNewTokens)
     */
    public NdArray generateWithCache(NdArray promptIds, int maxNewTokens) {
        int batchSize = promptIds.getShape().getDimension(0);
        int promptLen = promptIds.getShape().getDimension(1);

        // 限制生成长度不超过位置编码上限
        int maxPositions = config.getNPositions();
        int maxAllowed = maxPositions - promptLen;
        if (maxAllowed <= 0) {
            return promptIds;
        }
        if (maxNewTokens > maxAllowed) {
            maxNewTokens = maxAllowed;
        }

        // 为每层 Transformer 块创建独立的 KV Cache
        List<GPT3KVCache> kvCaches = new ArrayList<>();
        for (int i = 0; i < config.getNLayer(); i++) {
            kvCaches.add(new GPT3KVCache(
                    batchSize,
                    config.getNHead(),
                    config.getNEmbd() / config.getNHead(),
                    config.getNPositions(),
                    config.isUseRotaryEmbedding()
            ));
        }

        float[][] generatedSeq = new float[batchSize][promptLen + maxNewTokens];
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < promptLen; t++) {
                generatedSeq[b][t] = promptIds.get(b, t);
            }
        }

        // 阶段1：处理完整 Prompt，填充 KV Cache
        Variable promptVar = new Variable(promptIds);
        Variable promptLogits = gpt3Block.forwardWithCache(promptVar, kvCaches, 0);
        NdArray promptLogitsArr = promptLogits.getValue();

        // 从 Prompt 最后一个位置采样第一个生成 Token
        for (int b = 0; b < batchSize; b++) {
            int nextToken = argmax(promptLogitsArr, b, promptLen - 1);
            generatedSeq[b][promptLen] = nextToken;
        }

        // 阶段2：增量生成（每步只输入1个 Token，利用 KV Cache）
        int eosTokenId = config.getVocabSize() - 1;
        int actualGenLen = maxNewTokens;

        for (int step = 1; step < maxNewTokens; step++) {
            int currentPos = promptLen + step - 1;

            // 检查上一步是否所有 batch 都已生成 EOS
            boolean allEos = true;
            for (int b = 0; b < batchSize; b++) {
                if ((int) generatedSeq[b][currentPos] != eosTokenId) {
                    allEos = false;
                    break;
                }
            }
            if (allEos) {
                actualGenLen = step;
                break;
            }

            // 取上一步生成的 Token 作为输入（shape: batch_size × 1）
            float[][] singleToken = new float[batchSize][1];
            for (int b = 0; b < batchSize; b++) {
                singleToken[b][0] = generatedSeq[b][currentPos];
            }

            Variable tokenVar = new Variable(NdArray.of(singleToken));
            Variable logits = gpt3Block.forwardWithCache(tokenVar, kvCaches, currentPos);
            NdArray logitsArr = logits.getValue();

            // 贪婪解码：选最大概率 Token
            for (int b = 0; b < batchSize; b++) {
                int nextToken = argmax(logitsArr, b, 0);
                generatedSeq[b][currentPos + 1] = nextToken;
            }
        }

        // 截断到实际生成长度
        int finalLen = promptLen + actualGenLen;
        float[][] trimmedSeq = new float[batchSize][finalLen];
        for (int b = 0; b < batchSize; b++) {
            System.arraycopy(generatedSeq[b], 0, trimmedSeq[b], 0, finalLen);
        }
        return NdArray.of(trimmedSeq);
    }
    
    /**
     * 找到 logits 张量中指定 batch 和序列位置上概率最大的 token 索引。
     * 此方法为静态公共方法，供 GPT3Inference 等外部类复用，避免重复定义。
     *
     * @param logits   logits 张量，Shape: (batch, seqLen, vocabSize)
     * @param batchIdx 批次索引
     * @param seqIdx   序列位置索引
     * @return 概率最大的 token 索引
     */
    public static int argmax(NdArray logits, int batchIdx, int seqIdx) {
        int vocabSize = logits.getShape().getDimension(2);
        int maxIdx = 0;
        float maxVal = logits.get(batchIdx, seqIdx, 0);
        
        for (int i = 1; i < vocabSize; i++) {
            float val = logits.get(batchIdx, seqIdx, i);
            if (val > maxVal) {
                maxVal = val;
                maxIdx = i;
            }
        }
        
        return maxIdx;
    }
    
    // ==================== 信息展示方法 ====================
    
    /**
     * 打印详细的模型信息
     */
    @Override
    public void printModelInfo() {
        System.out.println("=".repeat(70));
        System.out.println("GPT-3 模型详细信息");
        System.out.println("=".repeat(70));
        System.out.println("模型名称: " + getName());
        System.out.println("模型描述: " + buildDescription());
        System.out.println("-".repeat(70));
        System.out.println(config);
        System.out.println("-".repeat(70));
        
        // 打印架构细节
        if (gpt3Block != null) {
            gpt3Block.printArchitecture();
        }
        
        System.out.println("=".repeat(70));
    }
    
    /**
     * 获取模型配置摘要
     * 
     * @return 配置摘要字符串
     */
    public String getConfigSummary() {
        return String.format(
            "GPT-3配置摘要:\n" +
            "  - 词汇表大小: %,d\n" +
            "  - 嵌入维度: %d\n" +
            "  - Transformer层数: %d\n" +
            "  - 注意力头数: %d\n" +
            "  - 前馈网络维度: %d\n" +
            "  - 最大序列长度: %d\n" +
            "  - 并行架构: %s\n" +
            "  - 稀疏注意力: %s\n" +
            "  - 估算参数量: %s",
            config.getVocabSize(),
            config.getNEmbd(),
            config.getNLayer(),
            config.getNHead(),
            config.getNInner(),
            config.getNPositions(),
            config.isParallelAttention() ? "是" : "否",
            config.isSparseAttention() ? "是" : "否",
            formatParamCount(config.estimateParameterCount())
        );
    }
    
    // ==================== Getter方法 ====================
    
    public GPT3Config getConfig() {
        return config;
    }
    
    public GPT3MainBlock getGPT3Block() {
        return gpt3Block;
    }
    
    /**
     * 获取指定的Transformer块
     * 
     * @param index 块索引
     * @return Transformer块
     */
    public GPT3TransformerBlock getTransformerBlock(int index) {
        return gpt3Block.getTransformerBlock(index);
    }
    
    @Override
    public String toString() {
        return String.format("GPT3Model{name='%s', params=%s, nLayer=%d, nEmbd=%d}",
            getName(), formatParamCount(config.estimateParameterCount()), 
            config.getNLayer(), config.getNEmbd());
    }
}
