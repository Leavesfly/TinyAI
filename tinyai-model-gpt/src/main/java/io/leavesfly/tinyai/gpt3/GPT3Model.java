package io.leavesfly.tinyai.gpt3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.layer.transformer.LayerNorm;
import io.leavesfly.tinyai.gpt2.GPT2TokenEmbedding;
import io.leavesfly.tinyai.gpt2.GPT2OutputHead;
import io.leavesfly.tinyai.gpt2.GPT2Config;

import java.util.ArrayList;
import java.util.List;

/**
 * GPT-3模型主体
 * 
 * 继承自Model类，实现完整的GPT-3语言模型
 * 采用解码器-only的Transformer架构
 * 
 * 主要特性：
 * 1. 更大的模型规模（最大175B参数）
 * 2. 并行注意力和MLP计算
 * 3. 支持稀疏注意力机制
 * 4. 梯度检查点和内存优化
 * 5. 强大的Few-shot学习能力
 * 
 * @author 山泽
 * @version 1.0
 */
public class GPT3Model extends Model {
    
    /** GPT-3配置 */
    private GPT3Config config;
    
    /** GPT3主块实例 */
    private GPT3MainBlock gpt3Block;
    
    /**
     * 构造GPT-3模型
     * 
     * @param name 模型名称
     * @param config GPT-3配置
     */
    public GPT3Model(String name, GPT3Config config) {
        super(name, new GPT3MainBlock(name + "_transformer", config));
        this.config = config;
        this.gpt3Block = (GPT3MainBlock) getBlock();
        
        setDescription("GPT-3语言模型 - " + config.toString());
        updateModelInfo();
    }
    
    /**
     * 使用默认配置的构造函数
     */
    public GPT3Model(String name) {
        this(name, new GPT3Config());
    }
    
    /**
     * 创建小型GPT-3模型（125M参数）
     */
    public static GPT3Model createSmallModel(String name) {
        return new GPT3Model(name, GPT3Config.createSmallConfig());
    }
    
    /**
     * 创建中型GPT-3模型（350M参数）
     */
    public static GPT3Model createMediumModel(String name) {
        return new GPT3Model(name, GPT3Config.createMediumConfig());
    }
    
    /**
     * 创建大型GPT-3模型（1.3B参数）
     */
    public static GPT3Model createLargeModel(String name) {
        return new GPT3Model(name, GPT3Config.createLargeConfig());
    }
    
    /**
     * 创建超大型GPT-3模型（175B参数）
     */
    public static GPT3Model createXLModel(String name) {
        return new GPT3Model(name, GPT3Config.createXLConfig());
    }
    
    /**
     * 更新模型信息
     */
    private void updateModelInfo() {
        if (getModelInfo() != null) {
            getModelInfo().setArchitectureType("GPT-3");
            addMetric("vocabulary_size", config.getVocabSize());
            addMetric("embedding_dimension", config.getNEmbd());
            addMetric("num_layers", config.getNLayer());
            addMetric("num_heads", config.getNHead());
            addMetric("sparse_attention", config.isSparseAttention() ? 1 : 0);
            addMetric("parallel_attention", config.isParallelAttention() ? 1 : 0);
            
            long totalParams = gpt3Block.getParameterCount();
            getModelInfo().setTotalParameters(totalParams);
        }
    }
    
    /**
     * 模型前向传播
     */
    public Variable predict(NdArray tokenIds) {
        return forward(new Variable(tokenIds));
    }
    
    /**
     * 预测下一个token
     */
    public int predictNextToken(NdArray tokenIds) {
        return gpt3Block.predictNextToken(tokenIds);
    }
    
    /**
     * 生成文本序列
     */
    public NdArray generateSequence(NdArray startTokenIds, int maxLength) {
        return gpt3Block.generateSequence(startTokenIds, maxLength);
    }
    
    /**
     * Few-shot学习生成
     * 基于提供的示例进行上下文学习
     */
    public NdArray fewShotGenerate(NdArray contextTokenIds, int maxNewTokens) {
        return gpt3Block.generateWithContext(contextTokenIds, maxNewTokens);
    }
    
    /**
     * 验证输入序列的有效性
     */
    public void validateInput(NdArray tokenIds) {
        Shape shape = tokenIds.getShape();
        
        if (shape.getDimNum() != 2) {
            throw new IllegalArgumentException("输入必须是二维数组 (batch_size, seq_len)");
        }
        
        int seqLen = shape.getDimension(1);
        if (seqLen > config.getNPositions()) {
            throw new IllegalArgumentException(
                String.format("序列长度(%d)超过最大支持长度(%d)", seqLen, config.getNPositions())
            );
        }
    }
    
    /**
     * 获取模型配置信息摘要
     */
    public String getConfigSummary() {
        return String.format(
            "GPT-3模型配置摘要:\n" +
            "- 词汇表大小: %,d\n" +
            "- 嵌入维度: %d\n" +
            "- Transformer层数: %d\n" +
            "- 注意力头数: %d\n" +
            "- 前馈网络维度: %d\n" +
            "- 最大序列长度: %d\n" +
            "- 并行注意力: %s\n" +
            "- 稀疏注意力: %s\n" +
            "- 总参数数量: %,d\n",
            config.getVocabSize(),
            config.getNEmbd(),
            config.getNLayer(),
            config.getNHead(),
            config.getNInner(),
            config.getNPositions(),
            config.isParallelAttention() ? "启用" : "禁用",
            config.isSparseAttention() ? "启用" : "禁用",
            gpt3Block.getParameterCount()
        );
    }
    
    @Override
    public void printModelInfo() {
        System.out.println("=== GPT-3 模型详细信息 ===");
        System.out.println(getConfigSummary());
        super.printModelInfo();
        System.out.println("========================");
    }
    
    // ==================== Getter方法 ====================
    
    /**
     * 获取GPT-3配置
     */
    public GPT3Config getConfig() {
        return config;
    }
    
    /**
     * 获取GPT-3主块
     */
    public GPT3MainBlock getGPT3Block() {
        return gpt3Block;
    }
    
    /**
     * 获取Token嵌入层
     */
    public GPT2TokenEmbedding getTokenEmbedding() {
        return gpt3Block.getTokenEmbedding();
    }
    
    /**
     * 获取所有Transformer块
     */
    public List<GPT3TransformerBlock> getTransformerBlocks() {
        return gpt3Block.getTransformerBlocks();
    }
    
    /**
     * 获取指定索引的Transformer块
     */
    public GPT3TransformerBlock getTransformerBlock(int index) {
        return gpt3Block.getTransformerBlock(index);
    }
    
    /**
     * 获取最终层归一化
     */
    public LayerNorm getFinalLayerNorm() {
        return gpt3Block.getFinalLayerNorm();
    }
    
    /**
     * 获取输出头
     */
    public GPT2OutputHead getOutputHead() {
        return gpt3Block.getOutputHead();
    }
}

