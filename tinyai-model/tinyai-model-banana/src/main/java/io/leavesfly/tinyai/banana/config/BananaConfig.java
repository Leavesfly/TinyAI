package io.leavesfly.tinyai.banana.config;

import java.io.Serializable;

/**
 * Gemini Nano Banana模型配置类
 * 
 * Banana是一个多模态图像生成模型,支持文本-图像融合和高质量图像生成。
 * 
 * 核心特点:
 * 1. 多模态融合 - 支持文本和图像的统一编码和交互
 * 2. Vision Transformer - 基于Patch嵌入的图像编码
 * 3. 跨模态注意力 - 文本-图像特征融合
 * 4. 轻量化设计 - 针对教育和实验场景优化
 * 
 * 本实现完全基于TinyAI框架的V2 API。
 * 
 * @author leavesfly
 * @version 1.0
 */
public class BananaConfig implements Serializable {

    private static final long serialVersionUID = 1L;
    
    // ==================== 基础模型配置 ====================
    
    /** 文本词汇表大小，默认32000 */
    private int vocabSize = 32000;
    
    /** 最大文本序列长度，默认512 */
    private int maxTextLength = 512;
    
    /** 隐藏层维度（文本和图像统一），默认512 */
    private int hiddenSize = 512;
    
    /** Transformer层数，默认8层 */
    private int numLayers = 8;
    
    /** 注意力头数，默认8头 */
    private int numHeads = 8;
    
    /** 前馈网络中间层维度，默认2048 */
    private int ffnHiddenSize = 2048;
    
    /** 激活函数类型，默认"gelu" */
    private String activationFunction = "gelu";
    
    // ==================== 图像编码器配置 ====================
    
    /** 图像尺寸（高度和宽度），默认256x256 */
    private int imageSize = 256;
    
    /** Patch尺寸（每个patch的高度和宽度），默认16x16 */
    private int patchSize = 16;
    
    /** 图像通道数，默认3（RGB） */
    private int imageChannels = 3;
    
    /** Patch数量（自动计算） = (imageSize / patchSize)^2 */
    private int numPatches;
    
    /** 图像编码器层数，默认6层 */
    private int numEncoderLayers = 6;
    
    // ==================== 多模态配置 ====================
    
    /** 是否启用跨模态注意力，默认启用 */
    private boolean enableCrossModalAttention = true;
    
    /** 跨模态注意力头数，默认8头 */
    private int crossModalHeads = 8;
    
    /** 模态融合方式: "concat", "add", "cross_attn"，默认"cross_attn" */
    private String modalityFusionType = "cross_attn";
    
    // ==================== 图像解码器配置（简化版） ====================
    
    /** 图像码本大小（VQVAE词汇表），默认8192 */
    private int imageVocabSize = 8192;
    
    /** 图像Token序列长度，默认256 (16x16) */
    private int imageTokenLength = 256;
    
    /** 是否启用自回归图像生成，默认启用 */
    private boolean enableAutoRegressiveGeneration = true;
    
    // ==================== Dropout配置 ====================
    
    /** 残差dropout概率，默认0.1 */
    private float dropoutRate = 0.1f;
    
    /** 注意力dropout概率，默认0.1 */
    private float attentionDropout = 0.1f;
    
    /** 嵌入dropout概率，默认0.1 */
    private float embeddingDropout = 0.1f;
    
    // ==================== 初始化配置 ====================
    
    /** 层归一化epsilon，默认1e-5 */
    private float layerNormEpsilon = 1e-5f;
    
    /** 权重初始化范围，默认0.02 */
    private float initializerRange = 0.02f;
    
    /**
     * 默认构造函数，创建Tiny配置
     */
    public BananaConfig() {
        // 自动计算patch数量
        this.numPatches = (imageSize / patchSize) * (imageSize / patchSize);
    }
    
    // ==================== 预设配置工厂方法 ====================
    
    /**
     * 创建Nano配置（演示用，极小规模，低内存占用）
     * 配置：128维, 2层, 4头, 64x64图像, 16x16 patch → 16个patches
     */
    public static BananaConfig createNanoConfig() {
        BananaConfig config = new BananaConfig();
        config.setHiddenSize(128);
        config.setNumLayers(2);
        config.setNumHeads(4);
        config.setFfnHiddenSize(256);
        config.setImageSize(64);
        config.setPatchSize(16);
        config.setNumEncoderLayers(2);
        config.setCrossModalHeads(4);
        config.setDropoutRate(0.0f);
        config.setAttentionDropout(0.0f);
        config.setEmbeddingDropout(0.0f);
        config.updateNumPatches();
        return config;
    }

    /**
     * 创建Tiny配置（教学用，最小规模）
     * 配置：512维, 8层, 8头, 256x256图像, 16x16 patch
     */
    public static BananaConfig createTinyConfig() {
        BananaConfig config = new BananaConfig();
        config.setHiddenSize(512);
        config.setNumLayers(8);
        config.setNumHeads(8);
        config.setFfnHiddenSize(2048);
        config.setImageSize(256);
        config.setPatchSize(16);
        config.setNumEncoderLayers(6);
        config.updateNumPatches();
        return config;
    }
    
    /**
     * 创建Small配置（实验用）
     * 配置：768维, 12层, 12头, 384x384图像, 16x16 patch
     */
    public static BananaConfig createSmallConfig() {
        BananaConfig config = new BananaConfig();
        config.setHiddenSize(768);
        config.setNumLayers(12);
        config.setNumHeads(12);
        config.setFfnHiddenSize(3072);
        config.setImageSize(384);
        config.setPatchSize(16);
        config.setNumEncoderLayers(8);
        config.updateNumPatches();
        return config;
    }
    
    /**
     * 创建Base配置（标准规模）
     * 配置：1024维, 16层, 16头, 512x512图像, 16x16 patch
     */
    public static BananaConfig createBaseConfig() {
        BananaConfig config = new BananaConfig();
        config.setHiddenSize(1024);
        config.setNumLayers(16);
        config.setNumHeads(16);
        config.setFfnHiddenSize(4096);
        config.setImageSize(512);
        config.setPatchSize(16);
        config.setNumEncoderLayers(12);
        config.updateNumPatches();
        return config;
    }
    
    // ==================== 辅助方法 ====================
    
    /**
     * 更新patch数量（当imageSize或patchSize变化时调用）
     */
    public void updateNumPatches() {
        this.numPatches = (imageSize / patchSize) * (imageSize / patchSize);
        this.imageTokenLength = numPatches;
    }
    
    /**
     * 获取每个注意力头的维度
     */
    public int getHeadDim() {
        return hiddenSize / numHeads;
    }
    
    /**
     * 估算模型参数量
     * 
     * 计算公式：
     * - 文本嵌入层: vocabSize * hiddenSize
     * - 图像Patch嵌入(Conv2D): patchSize * patchSize * imageChannels * hiddenSize + hiddenSize(bias)
     * - 位置编码: numPatches * hiddenSize
     * - 每个Transformer层: 4 * hiddenSize^2 (QKV+O投影) + 2 * hiddenSize * ffnHiddenSize (FFN) + 4 * hiddenSize (LayerNorm)
     * - 跨模态注意力层: 4 * hiddenSize^2 (QKV+O投影) + 2 * hiddenSize (LayerNorm)
     * - 图像编码器层: 与Transformer层相同
     */
    public long estimateParameters() {
        // 文本嵌入: vocabSize * hiddenSize
        long textEmbedding = (long) vocabSize * hiddenSize;
        
        // 图像Patch嵌入(Conv2D权重 + bias): patchSize^2 * channels * hiddenSize + hiddenSize
        long patchEmbedding = (long) (patchSize * patchSize * imageChannels) * hiddenSize + hiddenSize;
        
        // 位置编码: numPatches * hiddenSize
        long positionEmbedding = (long) numPatches * hiddenSize;
        
        // 每个Transformer层的参数量:
        // - 自注意力QKV投影: 3 * hiddenSize * hiddenSize + 3 * hiddenSize (bias)
        // - 输出投影: hiddenSize * hiddenSize + hiddenSize (bias)
        // - FFN: hiddenSize * ffnHiddenSize + ffnHiddenSize + ffnHiddenSize * hiddenSize + hiddenSize
        // - LayerNorm: 2 * (hiddenSize + hiddenSize)
        long selfAttnParams = 4L * hiddenSize * hiddenSize + 4L * hiddenSize;
        long ffnParams = 2L * hiddenSize * ffnHiddenSize + hiddenSize + ffnHiddenSize;
        long layerNormParams = 4L * hiddenSize;
        long perLayerParams = selfAttnParams + ffnParams + layerNormParams;
        
        long transformerParams = (long) numLayers * perLayerParams;
        long encoderParams = (long) numEncoderLayers * perLayerParams;
        
        // 跨模态注意力层: QKV投影 + 输出投影 + LayerNorm
        long crossModalParams = enableCrossModalAttention ? (selfAttnParams + 2L * hiddenSize) : 0;
        
        return textEmbedding + patchEmbedding + positionEmbedding 
             + transformerParams + encoderParams + crossModalParams;
    }
    
    /**
     * 格式化参数量显示
     */
    public String formatParameters() {
        long params = estimateParameters();
        if (params >= 1_000_000_000) {
            return String.format("%.2fB", params / 1_000_000_000.0);
        } else if (params >= 1_000_000) {
            return String.format("%.2fM", params / 1_000_000.0);
        } else {
            return String.format("%,d", params);
        }
    }
    
    /**
     * 验证配置有效性
     */
    public void validate() {
        if (hiddenSize % numHeads != 0) {
            throw new IllegalArgumentException(
                "hiddenSize必须能被numHeads整除: " + hiddenSize + " % " + numHeads + " != 0"
            );
        }
        
        if (imageSize % patchSize != 0) {
            throw new IllegalArgumentException(
                "imageSize必须能被patchSize整除: " + imageSize + " % " + patchSize + " != 0"
            );
        }
        
        if (numPatches != (imageSize / patchSize) * (imageSize / patchSize)) {
            throw new IllegalArgumentException(
                "numPatches计算错误,请调用updateNumPatches()"
            );
        }
    }
    
    @Override
    public String toString() {
        return String.format(
            "BananaConfig{\n" +
            "  基础配置: hiddenSize=%d, numLayers=%d, numHeads=%d, ffnHiddenSize=%d\n" +
            "  文本配置: vocabSize=%d, maxTextLength=%d\n" +
            "  图像配置: imageSize=%dx%d, patchSize=%dx%d, numPatches=%d\n" +
            "  编码器: numEncoderLayers=%d\n" +
            "  多模态: enableCrossModal=%b, fusionType=%s\n" +
            "  参数量: %s\n" +
            "}",
            hiddenSize, numLayers, numHeads, ffnHiddenSize,
            vocabSize, maxTextLength,
            imageSize, imageSize, patchSize, patchSize, numPatches,
            numEncoderLayers,
            enableCrossModalAttention, modalityFusionType,
            formatParameters()
        );
    }
    
    // ==================== Getter和Setter方法 ====================
    
    public int getVocabSize() {
        return vocabSize;
    }
    
    public void setVocabSize(int vocabSize) {
        this.vocabSize = vocabSize;
    }
    
    public int getMaxTextLength() {
        return maxTextLength;
    }
    
    public void setMaxTextLength(int maxTextLength) {
        this.maxTextLength = maxTextLength;
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
    
    public int getNumHeads() {
        return numHeads;
    }
    
    public void setNumHeads(int numHeads) {
        this.numHeads = numHeads;
    }
    
    public int getFfnHiddenSize() {
        return ffnHiddenSize;
    }
    
    public void setFfnHiddenSize(int ffnHiddenSize) {
        this.ffnHiddenSize = ffnHiddenSize;
    }
    
    public String getActivationFunction() {
        return activationFunction;
    }
    
    public void setActivationFunction(String activationFunction) {
        this.activationFunction = activationFunction;
    }
    
    public int getImageSize() {
        return imageSize;
    }
    
    public void setImageSize(int imageSize) {
        this.imageSize = imageSize;
        updateNumPatches();
    }
    
    public int getPatchSize() {
        return patchSize;
    }
    
    public void setPatchSize(int patchSize) {
        this.patchSize = patchSize;
        updateNumPatches();
    }
    
    public int getImageChannels() {
        return imageChannels;
    }
    
    public void setImageChannels(int imageChannels) {
        this.imageChannels = imageChannels;
    }
    
    public int getNumPatches() {
        return numPatches;
    }
    
    public int getNumEncoderLayers() {
        return numEncoderLayers;
    }
    
    public void setNumEncoderLayers(int numEncoderLayers) {
        this.numEncoderLayers = numEncoderLayers;
    }
    
    public boolean isEnableCrossModalAttention() {
        return enableCrossModalAttention;
    }
    
    public void setEnableCrossModalAttention(boolean enableCrossModalAttention) {
        this.enableCrossModalAttention = enableCrossModalAttention;
    }
    
    public int getCrossModalHeads() {
        return crossModalHeads;
    }
    
    public void setCrossModalHeads(int crossModalHeads) {
        this.crossModalHeads = crossModalHeads;
    }
    
    public String getModalityFusionType() {
        return modalityFusionType;
    }
    
    public void setModalityFusionType(String modalityFusionType) {
        this.modalityFusionType = modalityFusionType;
    }
    
    public int getImageVocabSize() {
        return imageVocabSize;
    }
    
    public void setImageVocabSize(int imageVocabSize) {
        this.imageVocabSize = imageVocabSize;
    }
    
    public int getImageTokenLength() {
        return imageTokenLength;
    }
    
    public boolean isEnableAutoRegressiveGeneration() {
        return enableAutoRegressiveGeneration;
    }
    
    public void setEnableAutoRegressiveGeneration(boolean enableAutoRegressiveGeneration) {
        this.enableAutoRegressiveGeneration = enableAutoRegressiveGeneration;
    }
    
    public float getDropoutRate() {
        return dropoutRate;
    }
    
    public void setDropoutRate(float dropoutRate) {
        this.dropoutRate = dropoutRate;
    }
    
    public float getAttentionDropout() {
        return attentionDropout;
    }
    
    public void setAttentionDropout(float attentionDropout) {
        this.attentionDropout = attentionDropout;
    }
    
    public float getEmbeddingDropout() {
        return embeddingDropout;
    }
    
    public void setEmbeddingDropout(float embeddingDropout) {
        this.embeddingDropout = embeddingDropout;
    }
    
    public float getLayerNormEpsilon() {
        return layerNormEpsilon;
    }
    
    public void setLayerNormEpsilon(float layerNormEpsilon) {
        this.layerNormEpsilon = layerNormEpsilon;
    }
    
    public float getInitializerRange() {
        return initializerRange;
    }
    
    public void setInitializerRange(float initializerRange) {
        this.initializerRange = initializerRange;
    }
}
