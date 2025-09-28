package io.leavesfly.tinyai.modality.nlp;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.block.transformer.MoETransformerBlock;
import io.leavesfly.tinyai.nnet.layer.moe.MoELayer;
import io.leavesfly.tinyai.nnet.layer.transformer.GPT2OutputHead;
import io.leavesfly.tinyai.nnet.layer.transformer.GPT2TokenEmbedding;
import io.leavesfly.tinyai.nnet.layer.transformer.LayerNorm;

import java.util.ArrayList;
import java.util.List;

/**
 * 基于Mixture of Experts (MoE) 的GPT模型实现
 * <p>
 * 这个模型将传统GPT-2中的FeedForward层替换为MoE层，
 * 从而大幅增加模型容量而不显著增加计算开销。
 * <p>
 * MoE-GPT模型的核心优势：
 * 1. 大幅增加模型参数量而保持合理的计算成本
 * 2. 每个token只激活部分专家，实现稀疏计算
 * 3. 不同专家可以专门处理不同类型的语言模式
 * 4. 可以通过增加专家数量来扩展模型容量
 * <p>
 * 模型结构：
 * Token Embedding + Position Embedding
 * → N × MoETransformerBlock  (替换标准TransformerBlock)
 * → Final LayerNorm
 * → Output Head
 *
 * @author leavesfly
 * @version 1.0
 */
public class MoEGPTModel extends Block {
    
    // 模型超参数
    private int vocabSize;              // 词汇表大小
    private int dModel;                 // 模型维度
    private int numLayers;              // MoE Transformer块数量
    private int numHeads;               // 注意力头数量
    private int numExperts;             // 每层的专家数量
    private int dExpert;                // 专家隐藏层维度
    private int topK;                   // Top-K专家选择
    private int maxSeqLength;           // 最大序列长度
    private double dropoutRate;         // Dropout比率
    private boolean useNoise;           // 门控是否使用噪声
    private double noiseEpsilon;        // 噪声强度
    
    // 模型组件
    private GPT2TokenEmbedding tokenEmbedding;          // Token嵌入层
    private List<MoETransformerBlock> moeTransformerBlocks; // MoE Transformer块列表
    private LayerNorm finalLayerNorm;                   // 最终层归一化
    private GPT2OutputHead outputHead;                  // 输出头
    
    // 统计信息
    private boolean collectStats;                       // 是否收集统计信息
    
    /**
     * 构造MoE-GPT模型
     * 
     * @param name 模型名称
     * @param vocabSize 词汇表大小
     * @param dModel 模型维度
     * @param numLayers MoE Transformer块数量
     * @param numHeads 注意力头数量
     * @param numExperts 每层的专家数量
     * @param dExpert 专家隐藏层维度
     * @param topK Top-K专家选择
     * @param maxSeqLength 最大序列长度
     * @param dropoutRate Dropout比率
     * @param useNoise 门控是否使用噪声
     * @param noiseEpsilon 噪声强度
     */
    public MoEGPTModel(String name, int vocabSize, int dModel, int numLayers, 
                      int numHeads, int numExperts, int dExpert, int topK, 
                      int maxSeqLength, double dropoutRate, 
                      boolean useNoise, double noiseEpsilon) {
        super(name, Shape.of(-1, maxSeqLength), Shape.of(-1, maxSeqLength, vocabSize));
        
        // 参数验证
        if (vocabSize <= 0 || dModel <= 0 || numLayers <= 0 || numHeads <= 0 || numExperts <= 0) {
            throw new IllegalArgumentException("所有参数必须大于0");
        }
        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException("dModel必须能被numHeads整除");
        }
        if (topK <= 0 || topK > numExperts) {
            throw new IllegalArgumentException("topK必须在1到numExperts之间");
        }
        
        this.vocabSize = vocabSize;
        this.dModel = dModel;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.numExperts = numExperts;
        this.dExpert = dExpert;
        this.topK = topK;
        this.maxSeqLength = maxSeqLength;
        this.dropoutRate = dropoutRate;
        this.useNoise = useNoise;
        this.noiseEpsilon = noiseEpsilon;
        this.collectStats = true;
        
        init();
    }
    
    /**
     * 使用默认参数的构造函数
     */
    public MoEGPTModel(String name, int vocabSize, int dModel, int numLayers, 
                      int numHeads, int numExperts, int topK, int maxSeqLength) {
        this(name, vocabSize, dModel, numLayers, numHeads, numExperts, 
             dModel * 4, topK, maxSeqLength, 0.1, true, 0.1);
    }
    
    /**
     * 中等规模MoE-GPT配置的构造函数
     */
    public MoEGPTModel(String name, int vocabSize, int maxSeqLength) {
        this(name, vocabSize, 768, 12, 12, 8, 2, maxSeqLength);
    }
    
    /**
     * 兼容原有构造函数
     */
    public MoEGPTModel(String _name, Shape _inputShape) {
        super(_name, _inputShape);
        // 使用默认配置
        this.vocabSize = 50257;  // GPT-2默认词汇表大小
        this.dModel = 768;
        this.numLayers = 12;
        this.numHeads = 12;
        this.numExperts = 8;
        this.dExpert = 3072;
        this.topK = 2;
        this.maxSeqLength = 1024;
        this.dropoutRate = 0.1;
        this.useNoise = true;
        this.noiseEpsilon = 0.1;
        this.collectStats = true;
        
        init();
    }

    @Override
    public void init() {
        if (!alreadyInit) {
            // 1. 初始化Token嵌入层
            tokenEmbedding = new GPT2TokenEmbedding(
                name + "_token_embedding", 
                vocabSize, 
                dModel, 
                maxSeqLength, 
                true,  // 使用位置嵌入
                dropoutRate
            );
            addLayer(tokenEmbedding);
            
            // 2. 初始化MoE Transformer块列表
            moeTransformerBlocks = new ArrayList<>();
            for (int i = 0; i < numLayers; i++) {
                MoETransformerBlock block = new MoETransformerBlock(
                    name + "_moe_block_" + i,
                    dModel,
                    numHeads,
                    numExperts,
                    dExpert,
                    topK,
                    dropoutRate,
                    useNoise,
                    noiseEpsilon
                );
                moeTransformerBlocks.add(block);
                addLayer(block);
            }
            
            // 3. 初始化最终层归一化
            finalLayerNorm = new LayerNorm(name + "_final_ln", dModel);
            addLayer(finalLayerNorm);
            
            // 4. 初始化输出头
            outputHead = new GPT2OutputHead(
                name + "_output_head",
                dModel,
                vocabSize,
                false  // MoE-GPT通常不使用输出偏置
            );
            addLayer(outputHead);
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];  // shape: (batch_size, seq_len)
        
        // 验证输入序列长度
        int seqLen = input.getValue().getShape().getDimension(1);
        if (seqLen > maxSeqLength) {
            throw new IllegalArgumentException(
                String.format("输入序列长度 %d 超过最大长度 %d", seqLen, maxSeqLength)
            );
        }
        
        // 1. Token嵌入 + 位置嵌入
        Variable x = tokenEmbedding.layerForward(input);  // shape: (batch_size, seq_len, dModel)
        
        // 2. 通过所有MoE Transformer块
        for (MoETransformerBlock block : moeTransformerBlocks) {
            x = block.layerForward(x);
        }
        
        // 3. 最终层归一化
        x = finalLayerNorm.layerForward(x);
        
        // 4. 输出头得到词汇表概率分布
        Variable output = outputHead.layerForward(x);  // shape: (batch_size, seq_len, vocab_size)
        
        return output;
    }
    
    /**
     * 获取所有MoE层的负载均衡统计信息
     */
    public List<MoELayer.LoadBalancingStats> getAllMoEStats() {
        List<MoELayer.LoadBalancingStats> statsList = new ArrayList<>();
        for (MoETransformerBlock block : moeTransformerBlocks) {
            statsList.add(block.getMoEStats());
        }
        return statsList;
    }
    
    /**
     * 重置所有MoE层的统计信息
     */
    public void resetAllMoEStats() {
        for (MoETransformerBlock block : moeTransformerBlocks) {
            block.resetMoEStats();
        }
    }
    
    /**
     * 获取模型配置信息
     */
    public String getModelConfig() {
        StringBuilder sb = new StringBuilder();
        sb.append("MoE-GPT Model Config:\n");
        sb.append(String.format("  - Vocab Size: %d\n", vocabSize));
        sb.append(String.format("  - Model Dim: %d\n", dModel));
        sb.append(String.format("  - Num Layers: %d\n", numLayers));
        sb.append(String.format("  - Num Heads: %d\n", numHeads));
        sb.append(String.format("  - Num Experts: %d\n", numExperts));
        sb.append(String.format("  - Expert Hidden Dim: %d\n", dExpert));
        sb.append(String.format("  - Top-K: %d\n", topK));
        sb.append(String.format("  - Max Seq Length: %d\n", maxSeqLength));
        sb.append(String.format("  - Dropout Rate: %.2f\n", dropoutRate));
        sb.append(String.format("  - Use Noise: %s\n", useNoise));
        sb.append(String.format("  - Noise Epsilon: %.3f\n", noiseEpsilon));
        sb.append(String.format("  - Total Parameters: %,d", getTotalParameterCount()));
        return sb.toString();
    }
    
    /**
     * 计算模型总参数数量
     */
    public long getTotalParameterCount() {
        long totalParams = 0;
        
        // Token嵌入层参数
        totalParams += vocabSize * dModel;  // token embedding
        totalParams += maxSeqLength * dModel;  // position embedding
        
        // MoE Transformer块参数
        for (MoETransformerBlock block : moeTransformerBlocks) {
            totalParams += block.getTotalParameterCount();
        }
        
        // 最终LayerNorm参数
        totalParams += 2 * dModel;  // gamma和beta
        
        // 输出头参数
        totalParams += dModel * vocabSize;
        
        return totalParams;
    }
    
    /**
     * 计算相比传统GPT-2的参数增加比例
     */
    public double getParameterIncreaseRatio() {
        // 传统GPT-2的MLP参数：2 * dModel * (dModel * 4) = 8 * dModel^2 per layer
        long traditionalMLPParams = numLayers * 8L * dModel * dModel;
        
        // MoE层的参数
        long moeParams = 0;
        for (MoETransformerBlock block : moeTransformerBlocks) {
            moeParams += block.getMoEParameterCount();
        }
        
        return (double) moeParams / traditionalMLPParams;
    }
    
    /**
     * 获取负载均衡报告
     */
    public String getLoadBalancingReport() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== MoE Load Balancing Report ===\n");
        
        List<MoELayer.LoadBalancingStats> allStats = getAllMoEStats();
        for (int i = 0; i < allStats.size(); i++) {
            MoELayer.LoadBalancingStats stats = allStats.get(i);
            sb.append(String.format("Layer %d:\n", i));
            sb.append(String.format("  总tokens: %d\n", stats.totalTokens));
            sb.append(String.format("  平均使用率: %.2f\n", stats.averageUsage));
            sb.append(String.format("  负载不均衡系数: %.4f\n", stats.loadImbalance));
            sb.append("  专家使用次数: [");
            for (int j = 0; j < stats.expertUsageCount.length; j++) {
                if (j > 0) sb.append(", ");
                sb.append(stats.expertUsageCount[j]);
            }
            sb.append("]\n");
        }
        
        return sb.toString();
    }
    
    // Getter方法
    public int getVocabSize() { return vocabSize; }
    public int getDModel() { return dModel; }
    public int getNumLayers() { return numLayers; }
    public int getNumHeads() { return numHeads; }
    public int getNumExperts() { return numExperts; }
    public int getDExpert() { return dExpert; }
    public int getTopK() { return topK; }
    public int getMaxSeqLength() { return maxSeqLength; }
    public double getDropoutRate() { return dropoutRate; }
    public boolean isUseNoise() { return useNoise; }
    public double getNoiseEpsilon() { return noiseEpsilon; }
    public boolean isCollectStats() { return collectStats; }
    
    public GPT2TokenEmbedding getTokenEmbedding() { return tokenEmbedding; }
    public List<MoETransformerBlock> getMoeTransformerBlocks() { return moeTransformerBlocks; }
    public MoETransformerBlock getMoeTransformerBlock(int index) {
        if (index < 0 || index >= moeTransformerBlocks.size()) {
            throw new IndexOutOfBoundsException("MoE Transformer块索引超出范围: " + index);
        }
        return moeTransformerBlocks.get(index);
    }
    public LayerNorm getFinalLayerNorm() { return finalLayerNorm; }
    public GPT2OutputHead getOutputHead() { return outputHead; }
    
    public void setCollectStats(boolean collectStats) { this.collectStats = collectStats; }
}