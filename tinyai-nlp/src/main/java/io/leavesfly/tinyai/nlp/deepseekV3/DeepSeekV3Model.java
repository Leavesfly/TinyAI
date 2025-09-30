package io.leavesfly.tinyai.nlp.deepseekV3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.layer.transformer.GPT2TokenEmbedding;
import io.leavesfly.tinyai.nnet.layer.transformer.GPT2OutputHead;
import io.leavesfly.tinyai.nnet.layer.transformer.LayerNorm;

import java.util.ArrayList;
import java.util.List;

/**
 * DeepSeek-V3 主模型实现
 * 
 * DeepSeek-V3是一个大规模的Mixture of Experts(MoE)语言模型，
 * 采用Multi-head Latent Attention (MLA)和DeepSeekMoE架构。
 * 
 * 模型架构：
 * Token Embedding + Position Embedding
 * → N × DeepSeekV3TransformerBlock
 * → Final LayerNorm
 * → Output Head
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3Model extends Block {
    
    // ========== 配置参数 ==========
    private DeepSeekV3Config config;
    
    // ========== 模型组件 ==========
    private GPT2TokenEmbedding tokenEmbedding;
    private List<DeepSeekV3TransformerBlock> transformerBlocks;
    private LayerNorm finalLayerNorm;
    private GPT2OutputHead outputHead;
    
    // ========== 模型状态 ==========
    private boolean isTraining;
    private long totalTokensProcessed;
    
    /**
     * 使用配置构造DeepSeek-V3模型
     */
    public DeepSeekV3Model(String name, DeepSeekV3Config config) {
        super(name, 
              Shape.of(-1, config.getMaxSeqLength()), 
              Shape.of(-1, config.getMaxSeqLength(), config.getVocabSize()));
        
        this.config = config;
        this.isTraining = false;
        this.totalTokensProcessed = 0;
        
        init();
    }
    
    /**
     * 使用基本参数构造DeepSeek-V3模型
     */
    public DeepSeekV3Model(String name, int vocabSize, int dModel, int numLayers,
                          int numHeads, int numExperts, int topK, int maxSeqLength) {
        this(name, new DeepSeekV3Config(vocabSize, dModel, numLayers, numHeads,
                                       maxSeqLength, numExperts, topK));
    }
    
    /**
     * 兼容原有构造函数
     */
    public DeepSeekV3Model(String _name, Shape _inputShape) {
        super(_name, _inputShape);
        // 使用默认配置
        this.config = DeepSeekV3Config.createStandardConfig();
        this.isTraining = false;
        this.totalTokensProcessed = 0;
        
        init();
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 1. 初始化Token嵌入层
            tokenEmbedding = new GPT2TokenEmbedding(
                name + "_token_embedding",
                config.getVocabSize(),
                config.getDModel(),
                config.getMaxSeqLength(),
                true, // 使用位置嵌入
                config.getDropoutRate()
            );
            addLayer(tokenEmbedding);
            
            // 2. 初始化Transformer块列表
            transformerBlocks = new ArrayList<DeepSeekV3TransformerBlock>();
            for (int i = 0; i < config.getNumLayers(); i++) {
                DeepSeekV3TransformerBlock block = new DeepSeekV3TransformerBlock(
                    name + "_block_" + i,
                    config
                );
                transformerBlocks.add(block);
                addLayer(block);
            }
            
            // 3. 初始化最终层归一化
            finalLayerNorm = new LayerNorm(name + "_final_ln", config.getDModel());
            addLayer(finalLayerNorm);
            
            // 4. 初始化输出头
            outputHead = new GPT2OutputHead(
                name + "_output_head",
                config.getDModel(),
                config.getVocabSize(),
                false // 不使用偏置
            );
            addLayer(outputHead);
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];  // shape: [batchSize, seqLen]
        NdArray inputData = input.getValue();
        
        int batchSize = inputData.getShape().getDimension(0);
        int seqLen = inputData.getShape().getDimension(1);
        
        // 验证输入序列长度
        if (seqLen > config.getMaxSeqLength()) {
            throw new IllegalArgumentException(
                String.format("Input sequence length %d exceeds maximum %d", 
                             seqLen, config.getMaxSeqLength())
            );
        }
        
        // 更新处理的token数
        totalTokensProcessed += (long) batchSize * seqLen;
        
        // 1. Token嵌入 + 位置嵌入
        Variable x = tokenEmbedding.layerForward(input);
        
        // 2. 通过所有DeepSeek-V3 Transformer块
        for (DeepSeekV3TransformerBlock block : transformerBlocks) {
            x = block.layerForward(x);
        }
        
        // 3. 最终层归一化
        x = finalLayerNorm.layerForward(x);
        
        // 4. 输出头得到词汇表概率分布
        Variable output = outputHead.layerForward(x);
        
        return output;
    }
    
    /**
     * 计算总负载均衡损失
     */
    public double computeTotalLoadBalancingLoss() {
        double totalLoss = 0.0;
        for (DeepSeekV3TransformerBlock block : transformerBlocks) {
            totalLoss += block.computeLoadBalancingLoss();
        }
        return totalLoss;
    }
    
    /**
     * 重置所有层的MoE统计信息
     */
    public void resetAllMoEStats() {
        for (DeepSeekV3TransformerBlock block : transformerBlocks) {
            block.resetMoEStats();
        }
    }
    
    /**
     * 获取所有层的专家使用统计
     */
    public List<long[]> getAllLayersExpertUsage() {
        List<long[]> allUsage = new ArrayList<long[]>();
        for (DeepSeekV3TransformerBlock block : transformerBlocks) {
            allUsage.add(block.getExpertUsageCount());
        }
        return allUsage;
    }
    
    /**
     * 计算模型总参数数量
     */
    public long getTotalParameterCount() {
        if (config != null) {
            return config.getTotalParameterCount();
        }
        
        // 手动计算
        long totalParams = 0;
        
        // Token嵌入参数
        totalParams += (long) config.getVocabSize() * config.getDModel();
        totalParams += (long) config.getMaxSeqLength() * config.getDModel();
        
        // Transformer块参数
        for (DeepSeekV3TransformerBlock block : transformerBlocks) {
            totalParams += block.getTotalParameterCount();
        }
        
        // 最终LayerNorm参数
        totalParams += 2L * config.getDModel();
        
        // 输出头参数
        totalParams += (long) config.getDModel() * config.getVocabSize();
        
        return totalParams;
    }
    
    /**
     * 计算激活参数数量
     */
    public long getActiveParameterCount() {
        if (config != null) {
            return config.getActiveParameterCount();
        }
        
        // 手动计算
        long activeParams = 0;
        
        // Token嵌入参数（全部激活）
        activeParams += (long) config.getVocabSize() * config.getDModel();
        activeParams += (long) config.getMaxSeqLength() * config.getDModel();
        
        // Transformer块参数
        for (DeepSeekV3TransformerBlock block : transformerBlocks) {
            activeParams += block.getActiveParameterCount();
        }
        
        // 最终LayerNorm参数
        activeParams += 2L * config.getDModel();
        
        // 输出头参数
        activeParams += (long) config.getDModel() * config.getVocabSize();
        
        return activeParams;
    }
    
    /**
     * 获取模型配置信息
     */
    public String getModelInfo() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== DeepSeek-V3 Model Information ===\n");
        sb.append(config.getSummary()).append("\n");
        sb.append("Model Statistics:\n");
        sb.append(String.format("  - Total Tokens Processed: %,d\n", totalTokensProcessed));
        sb.append(String.format("  - Training Mode: %s\n", isTraining ? "ON" : "OFF"));
        sb.append(String.format("  - Parameter Efficiency: %.1f%% active\n", 
                 (double) getActiveParameterCount() / getTotalParameterCount() * 100));
        sb.append("==========================================");
        return sb.toString();
    }
    
    /**
     * 打印模型信息
     */
    public void printModelInfo() {
        System.out.println(getModelInfo());
    }
    
    /**
     * 设置训练模式
     */
    public void setTraining(boolean training) {
        this.isTraining = training;
    }
    
    /**
     * 启用KV缓存（用于推理优化）
     */
    public void enableKVCache() {
        for (DeepSeekV3TransformerBlock block : transformerBlocks) {
            block.getMlaAttention().enableKVCache();
        }
    }
    
    /**
     * 禁用KV缓存
     */
    public void disableKVCache() {
        for (DeepSeekV3TransformerBlock block : transformerBlocks) {
            block.getMlaAttention().disableKVCache();
        }
    }
    
    /**
     * 清空所有KV缓存
     */
    public void clearAllKVCache() {
        for (DeepSeekV3TransformerBlock block : transformerBlocks) {
            block.getMlaAttention().clearKVCache();
        }
    }
    
    /**
     * 计算MLA缓存的总内存节省量
     */
    public long getTotalMemorySavings(int seqLen) {
        long totalSavings = 0;
        for (DeepSeekV3TransformerBlock block : transformerBlocks) {
            totalSavings += block.getMlaAttention().getMemorySavingsBytes(seqLen);
        }
        return totalSavings;
    }
    
    /**
     * 获取专家负载均衡报告
     */
    public String getLoadBalancingReport() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== DeepSeek-V3 Load Balancing Report ===\n");
        
        for (int i = 0; i < transformerBlocks.size(); i++) {
            DeepSeekV3TransformerBlock block = transformerBlocks.get(i);
            sb.append(String.format("Layer %d:\n", i));
            sb.append(String.format("  Total Tokens: %d\n", block.getTotalTokens()));
            sb.append(String.format("  Load Balance Loss: %.6f\n", block.computeLoadBalancingLoss()));
            
            long[] usage = block.getExpertUsageCount();
            sb.append("  Expert Usage: [");
            for (int j = 0; j < usage.length; j++) {
                if (j > 0) sb.append(", ");
                sb.append(usage[j]);
            }
            sb.append("]\n");
        }
        
        sb.append("==========================================");
        return sb.toString();
    }
    
    /**
     * 打印负载均衡报告
     */
    public void printLoadBalancingReport() {
        System.out.println(getLoadBalancingReport());
    }
    
    // ========== 工厂方法 ==========
    
    /**
     * 创建小型DeepSeek-V3模型
     */
    public static DeepSeekV3Model createTinyModel(String name, int vocabSize) {
        return new DeepSeekV3Model(name, DeepSeekV3Config.createTinyConfig());
    }
    
    /**
     * 创建小型DeepSeek-V3模型
     */
    public static DeepSeekV3Model createSmallModel(String name, int vocabSize) {
        return new DeepSeekV3Model(name, DeepSeekV3Config.createSmallConfig());
    }
    
    /**
     * 创建标准DeepSeek-V3模型
     */
    public static DeepSeekV3Model createStandardModel(String name, int vocabSize) {
        return new DeepSeekV3Model(name, DeepSeekV3Config.createStandardConfig());
    }
    
    // ========== Getter Methods ==========
    public DeepSeekV3Config getConfig() { return config; }
    public boolean isTraining() { return isTraining; }
    public long getTotalTokensProcessed() { return totalTokensProcessed; }
    
    public GPT2TokenEmbedding getTokenEmbedding() { return tokenEmbedding; }
    public List<DeepSeekV3TransformerBlock> getTransformerBlocks() { return transformerBlocks; }
    public DeepSeekV3TransformerBlock getTransformerBlock(int index) {
        if (index < 0 || index >= transformerBlocks.size()) {
            throw new IndexOutOfBoundsException("Block index out of range: " + index);
        }
        return transformerBlocks.get(index);
    }
    public LayerNorm getFinalLayerNorm() { return finalLayerNorm; }
    public GPT2OutputHead getOutputHead() { return outputHead; }
}