package io.leavesfly.tinyai.gpt1;

/**
 * GPT-1模型配置类
 * 用于管理GPT-1模型的超参数和配置信息
 * 
 * 基于OpenAI 2018年发布的原始GPT-1论文：
 * "Improving Language Understanding by Generative Pre-Training"
 * 
 * @author 山泽
 * @version 1.0
 */
public class GPT1Config {
    
    // ==================== 核心模型参数 ====================
    
    /** 词汇表大小，默认40000（与原论文一致） */
    private int vocabSize;
    
    /** 最大序列长度，默认512（与原论文一致） */
    private int maxSequenceLength;
    
    /** 模型嵌入维度，默认768（与原论文一致） */
    private int hiddenSize;
    
    /** Transformer层数，默认12（与原论文一致） */
    private int numLayers;
    
    /** 注意力头数，默认12（与原论文一致） */
    private int numAttentionHeads;
    
    /** 前馈网络中间层维度，默认为hiddenSize的4倍 */
    private int intermediateSize;
    
    // ==================== 训练相关参数 ====================
    
    /** 残差连接的dropout概率 */
    private double residualDropoutProb;
    
    /** 嵌入层的dropout概率 */
    private double embeddingDropoutProb;
    
    /** 注意力层的dropout概率 */
    private double attentionDropoutProb;
    
    /** 层归一化的epsilon值 */
    private double layerNormEpsilon;
    
    /** 参数初始化的标准差 */
    private double initializerRange;
    
    // ==================== 激活函数配置 ====================
    
    /** 激活函数类型 */
    private String activationFunction;
    
    /**
     * 默认构造函数，使用GPT-1原论文的默认配置
     */
    public GPT1Config() {
        this(40000, 512, 768, 12, 12);
    }
    
    /**
     * 基本构造函数
     * 
     * @param vocabSize 词汇表大小
     * @param maxSequenceLength 最大序列长度
     * @param hiddenSize 隐藏层维度
     * @param numLayers Transformer层数
     * @param numAttentionHeads 注意力头数
     */
    public GPT1Config(int vocabSize, int maxSequenceLength, int hiddenSize, 
                      int numLayers, int numAttentionHeads) {
        this.vocabSize = vocabSize;
        this.maxSequenceLength = maxSequenceLength;
        this.hiddenSize = hiddenSize;
        this.numLayers = numLayers;
        this.numAttentionHeads = numAttentionHeads;
        
        // 设置默认值
        this.intermediateSize = hiddenSize * 4;
        this.residualDropoutProb = 0.1;
        this.embeddingDropoutProb = 0.1;
        this.attentionDropoutProb = 0.1;
        this.layerNormEpsilon = 1e-5;
        this.initializerRange = 0.02;
        this.activationFunction = "gelu";
    }
    
    /**
     * 完整构造函数
     */
    public GPT1Config(int vocabSize, int maxSequenceLength, int hiddenSize,
                      int numLayers, int numAttentionHeads, int intermediateSize,
                      double residualDropoutProb, double embeddingDropoutProb,
                      double attentionDropoutProb, double layerNormEpsilon,
                      double initializerRange, String activationFunction) {
        this.vocabSize = vocabSize;
        this.maxSequenceLength = maxSequenceLength;
        this.hiddenSize = hiddenSize;
        this.numLayers = numLayers;
        this.numAttentionHeads = numAttentionHeads;
        this.intermediateSize = intermediateSize;
        this.residualDropoutProb = residualDropoutProb;
        this.embeddingDropoutProb = embeddingDropoutProb;
        this.attentionDropoutProb = attentionDropoutProb;
        this.layerNormEpsilon = layerNormEpsilon;
        this.initializerRange = initializerRange;
        this.activationFunction = activationFunction;
    }
    
    /**
     * 创建小型GPT-1配置（用于测试和演示）
     * 
     * @param vocabSize 词汇表大小
     * @param maxSequenceLength 最大序列长度
     * @return 小型配置
     */
    public static GPT1Config createTinyConfig(int vocabSize, int maxSequenceLength) {
        return new GPT1Config(vocabSize, maxSequenceLength, 256, 6, 8);
    }
    
    /**
     * 创建中型GPT-1配置
     * 
     * @param vocabSize 词汇表大小
     * @param maxSequenceLength 最大序列长度
     * @return 中型配置
     */
    public static GPT1Config createMediumConfig(int vocabSize, int maxSequenceLength) {
        return new GPT1Config(vocabSize, maxSequenceLength, 512, 8, 8);
    }
    
    /**
     * 验证配置的有效性
     * 
     * @throws IllegalArgumentException 如果配置无效
     */
    public void validate() {
        if (vocabSize <= 0) {
            throw new IllegalArgumentException("词汇表大小必须大于0");
        }
        if (maxSequenceLength <= 0) {
            throw new IllegalArgumentException("最大序列长度必须大于0");
        }
        if (hiddenSize <= 0) {
            throw new IllegalArgumentException("隐藏层维度必须大于0");
        }
        if (numLayers <= 0) {
            throw new IllegalArgumentException("Transformer层数必须大于0");
        }
        if (numAttentionHeads <= 0) {
            throw new IllegalArgumentException("注意力头数必须大于0");
        }
        if (hiddenSize % numAttentionHeads != 0) {
            throw new IllegalArgumentException("隐藏层维度必须能被注意力头数整除");
        }
        if (residualDropoutProb < 0 || residualDropoutProb > 1) {
            throw new IllegalArgumentException("dropout概率必须在[0,1]范围内");
        }
    }
    
    // ==================== Getter和Setter方法 ====================
    
    public int getVocabSize() {
        return vocabSize;
    }
    
    public void setVocabSize(int vocabSize) {
        this.vocabSize = vocabSize;
    }
    
    public int getMaxSequenceLength() {
        return maxSequenceLength;
    }
    
    public void setMaxSequenceLength(int maxSequenceLength) {
        this.maxSequenceLength = maxSequenceLength;
    }
    
    public int getHiddenSize() {
        return hiddenSize;
    }
    
    public void setHiddenSize(int hiddenSize) {
        this.hiddenSize = hiddenSize;
    }
    
    public int getNumLayers() {
        return numLayers;
    }
    
    public void setNumLayers(int numLayers) {
        this.numLayers = numLayers;
    }
    
    public int getNumAttentionHeads() {
        return numAttentionHeads;
    }
    
    public void setNumAttentionHeads(int numAttentionHeads) {
        this.numAttentionHeads = numAttentionHeads;
    }
    
    public int getIntermediateSize() {
        return intermediateSize;
    }
    
    public void setIntermediateSize(int intermediateSize) {
        this.intermediateSize = intermediateSize;
    }
    
    public double getResidualDropoutProb() {
        return residualDropoutProb;
    }
    
    public void setResidualDropoutProb(double residualDropoutProb) {
        this.residualDropoutProb = residualDropoutProb;
    }
    
    public double getEmbeddingDropoutProb() {
        return embeddingDropoutProb;
    }
    
    public void setEmbeddingDropoutProb(double embeddingDropoutProb) {
        this.embeddingDropoutProb = embeddingDropoutProb;
    }
    
    public double getAttentionDropoutProb() {
        return attentionDropoutProb;
    }
    
    public void setAttentionDropoutProb(double attentionDropoutProb) {
        this.attentionDropoutProb = attentionDropoutProb;
    }
    
    public double getLayerNormEpsilon() {
        return layerNormEpsilon;
    }
    
    public void setLayerNormEpsilon(double layerNormEpsilon) {
        this.layerNormEpsilon = layerNormEpsilon;
    }
    
    public double getInitializerRange() {
        return initializerRange;
    }
    
    public void setInitializerRange(double initializerRange) {
        this.initializerRange = initializerRange;
    }
    
    public String getActivationFunction() {
        return activationFunction;
    }
    
    public void setActivationFunction(String activationFunction) {
        this.activationFunction = activationFunction;
    }
    
    /**
     * 计算每个注意力头的维度
     * 
     * @return 注意力头维度
     */
    public int getAttentionHeadSize() {
        return hiddenSize / numAttentionHeads;
    }
    
    @Override
    public String toString() {
        return String.format(
            "GPT1Config{" +
            "vocabSize=%d, " +
            "maxSequenceLength=%d, " +
            "hiddenSize=%d, " +
            "numLayers=%d, " +
            "numAttentionHeads=%d, " +
            "intermediateSize=%d, " +
            "activationFunction='%s'" +
            "}",
            vocabSize, maxSequenceLength, hiddenSize, 
            numLayers, numAttentionHeads, intermediateSize, 
            activationFunction
        );
    }
}