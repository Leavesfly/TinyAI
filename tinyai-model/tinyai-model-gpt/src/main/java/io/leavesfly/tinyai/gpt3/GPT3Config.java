package io.leavesfly.tinyai.gpt3;

import java.io.Serializable;

/**
 * GPT-3模型配置类（完全独立实现）
 * <p>
 * 包含GPT-3模型的所有超参数配置：
 * 1. 基础配置：词汇表、序列长度、嵌入维度等
 * 2. Transformer配置：层数、注意力头数、前馈维度
 * 3. GPT-3特有配置：并行注意力、RoPE、稀疏注意力等
 * 4. 训练配置：dropout、初始化范围等
 *
 * @author leavesfly
 * @version 2.0 - 完全独立实现
 */
public class GPT3Config implements Serializable {

    private static final long serialVersionUID = 1L;

    // ==================== 基础模型配置 ====================

    /**
     * 词汇表大小，默认50257 (GPT-3默认值)
     */
    private int vocabSize = 50257;

    /**
     * 最大位置数（序列长度），默认2048
     */
    private int nPositions = 2048;

    /**
     * 嵌入维度，默认768
     */
    private int nEmbd = 768;

    /**
     * Transformer层数，默认12
     */
    private int nLayer = 12;

    /**
     * 注意力头数，默认12
     */
    private int nHead = 12;

    /**
     * 前馈网络中间层维度，默认4倍嵌入维度
     */
    private int nInner = 3072;

    /**
     * 激活函数类型，默认"gelu"
     */
    private String activationFunction = "gelu";

    // ==================== Dropout配置 ====================

    /**
     * 残差dropout概率，默认0.1
     */
    private double residPdrop = 0.1;

    /**
     * 嵌入dropout概率，默认0.1
     */
    private double embdPdrop = 0.1;

    /**
     * 注意力dropout概率，默认0.1
     */
    private double attnPdrop = 0.1;

    // ==================== 初始化配置 ====================

    /**
     * 层归一化epsilon，默认1e-5
     */
    private double layerNormEpsilon = 1e-5;

    /**
     * 权重初始化范围，默认0.02
     */
    private double initializerRange = 0.02;

    /**
     * 是否使用并行注意力和MLP计算，默认true（GPT-3核心特性）
     */
    private boolean parallelAttention = true;

    /**
     * 是否使用旋转位置编码(RoPE)，默认false（保持兼容性）
     */
    private boolean useRotaryEmbedding = false;

    /**
     * 旋转位置编码的维度比例，默认0.25（25%维度使用RoPE）
     */
    private double rotaryPct = 0.25;

    /**
     * RoPE的base参数，用于计算频率，默认10000
     */
    private double rotaryBase = 10000.0;

    /**
     * 是否使用稀疏注意力机制，默认false
     */
    private boolean sparseAttention = false;

    /**
     * 稀疏注意力的局部窗口大小，默认256
     */
    private int sparseLocalWindow = 256;

    /**
     * 稀疏注意力的步长大小，默认128
     */
    private int sparseStrideSize = 128;

    /**
     * 是否启用梯度检查点，默认false
     */
    private boolean gradientCheckpointing = false;

    /**
     * 是否启用KV缓存（用于加速推理），默认true
     */
    private boolean useCache = true;

    /**
     * Few-shot学习的最大示例数，默认32
     */
    private int maxFewShotExamples = 32;

    /**
     * 默认构造函数，创建标准GPT-3配置
     */
    public GPT3Config() {
        // 使用默认值
    }

    /**
     * 完整配置构造函数
     */
    public GPT3Config(int vocabSize, int nPositions, int nEmbd, int nLayer,
                      int nHead, int nInner, String activationFunction,
                      double residPdrop, double embdPdrop, double attnPdrop,
                      double layerNormEpsilon, double initializerRange,
                      boolean parallelAttention, boolean useRotaryEmbedding,
                      double rotaryPct, double rotaryBase,
                      boolean sparseAttention, int sparseLocalWindow, int sparseStrideSize,
                      boolean gradientCheckpointing, boolean useCache, int maxFewShotExamples) {
        // 基础配置
        this.vocabSize = vocabSize;
        this.nPositions = nPositions;
        this.nEmbd = nEmbd;
        this.nLayer = nLayer;
        this.nHead = nHead;
        this.nInner = nInner;
        this.activationFunction = activationFunction;

        // Dropout配置
        this.residPdrop = residPdrop;
        this.embdPdrop = embdPdrop;
        this.attnPdrop = attnPdrop;

        // 初始化配置
        this.layerNormEpsilon = layerNormEpsilon;
        this.initializerRange = initializerRange;

        // GPT-3特有配置
        this.parallelAttention = parallelAttention;
        this.useRotaryEmbedding = useRotaryEmbedding;
        this.rotaryPct = rotaryPct;
        this.rotaryBase = rotaryBase;
        this.sparseAttention = sparseAttention;
        this.sparseLocalWindow = sparseLocalWindow;
        this.sparseStrideSize = sparseStrideSize;
        this.gradientCheckpointing = gradientCheckpointing;
        this.useCache = useCache;
        this.maxFewShotExamples = maxFewShotExamples;
    }

    // ==================== 预设配置工厂方法 ====================


    /**
     * 创建nano配置（极小参数量，适合演示和单元测试）
     * 嵌入维度64, 2层, 2头, 序列长度64
     */
    public static GPT3Config createNanoConfig() {
        GPT3Config config = new GPT3Config();
        config.setNEmbd(32);
        config.setNLayer(2);
        config.setNHead(2);
        config.setNInner(8);
        config.setNPositions(32);
        config.setVocabSize(100);     // 演示用小词汇表
        config.setParallelAttention(true);
        config.setUseRotaryEmbedding(false);
        config.setSparseAttention(false);
        config.setGradientCheckpointing(false);
        config.setUseCache(true);
        return config;
    }

    /**
     * 创建小型GPT-3配置（125M参数，用于学习和测试）
     * 配置：768维, 12层, 12头
     */
    public static GPT3Config createSmallConfig() {
        GPT3Config config = new GPT3Config();
        config.setNEmbd(768);
        config.setNLayer(12);
        config.setNHead(12);
        config.setNInner(3072);
        config.setNPositions(2048);
        config.setParallelAttention(true);
        config.setUseRotaryEmbedding(false);  // 小模型不使用RoPE
        config.setSparseAttention(false);
        return config;
    }

    /**
     * 创建中型GPT-3配置（350M参数，用于实用应用）
     * 配置：1024维, 24层, 16头
     */
    public static GPT3Config createMediumConfig() {
        GPT3Config config = new GPT3Config();
        config.setNEmbd(1024);
        config.setNLayer(24);
        config.setNHead(16);
        config.setNInner(4096);
        config.setNPositions(2048);
        config.setParallelAttention(true);
        config.setUseRotaryEmbedding(false);
        config.setSparseAttention(false);
        return config;
    }

    /**
     * 创建大型GPT-3配置（1.3B参数，高质量生成）
     * 配置：2048维, 24层, 32头，启用稀疏注意力
     */
    public static GPT3Config createLargeConfig() {
        GPT3Config config = new GPT3Config();
        config.setNEmbd(2048);
        config.setNLayer(24);
        config.setNHead(32);
        config.setNInner(8192);
        config.setNPositions(2048);
        config.setParallelAttention(true);
        config.setUseRotaryEmbedding(true);   // 大模型使用RoPE
        config.setRotaryPct(0.25);
        config.setSparseAttention(true);       // 启用稀疏注意力
        config.setGradientCheckpointing(true); // 启用梯度检查点
        return config;
    }

    /**
     * 创建超大型GPT-3配置（175B参数，顶级性能）
     * 配置：12288维, 96层, 96头，全部优化特性
     */
    public static GPT3Config createXLConfig() {
        GPT3Config config = new GPT3Config();
        config.setNEmbd(12288);
        config.setNLayer(96);
        config.setNHead(96);
        config.setNInner(49152);
        config.setNPositions(2048);
        config.setParallelAttention(true);
        config.setUseRotaryEmbedding(true);
        config.setRotaryPct(0.25);
        config.setSparseAttention(true);
        config.setGradientCheckpointing(true);
        return config;
    }

    /**
     * 估算模型参数数量
     *
     * @return 估算的参数数量
     */
    public long estimateParameterCount() {
        long vocabParams = (long) getVocabSize() * getNEmbd();
        long positionParams = (long) getNPositions() * getNEmbd();

        // 每个Transformer块的参数
        long attentionParams = 4L * getNEmbd() * getNEmbd();  // Q, K, V, O投影
        long ffnParams = 2L * getNEmbd() * getNInner();       // 两个线性层
        long normParams = 2L * getNEmbd();                     // 两个LayerNorm
        long blockParams = attentionParams + ffnParams + normParams;

        long transformerParams = getNLayer() * blockParams;
        long outputParams = (long) getNEmbd() * getVocabSize();

        return vocabParams + positionParams + transformerParams + outputParams;
    }

    public void validate() {
        // 验证基础配置
        if (vocabSize <= 0) {
            throw new IllegalArgumentException("词汇表大小必须大于0");
        }
        if (nPositions <= 0) {
            throw new IllegalArgumentException("最大位置数必须大于0");
        }
        if (nEmbd <= 0) {
            throw new IllegalArgumentException("嵌入维度必须大于0");
        }
        if (nLayer <= 0) {
            throw new IllegalArgumentException("Transformer层数必须大于0");
        }
        if (nHead <= 0) {
            throw new IllegalArgumentException("注意力头数必须大于0");
        }
        if (nEmbd % nHead != 0) {
            throw new IllegalArgumentException("嵌入维度必须能被注意力头数整除");
        }
        if (nInner <= 0) {
            throw new IllegalArgumentException("前馈网络维度必须大于0");
        }

        // 验证Dropout配置
        if (residPdrop < 0 || residPdrop >= 1) {
            throw new IllegalArgumentException("残差dropout概率必须在[0,1)范围内");
        }
        if (embdPdrop < 0 || embdPdrop >= 1) {
            throw new IllegalArgumentException("嵌入dropout概率必须在[0,1)范围内");
        }
        if (attnPdrop < 0 || attnPdrop >= 1) {
            throw new IllegalArgumentException("注意力dropout概率必须在[0,1)范围内");
        }

        // 验证初始化配置
        if (layerNormEpsilon <= 0) {
            throw new IllegalArgumentException("层归一化epsilon必须大于0");
        }
        if (initializerRange <= 0) {
            throw new IllegalArgumentException("初始化范围必须大于0");
        }

        // 验证GPT-3特有配置
        if (rotaryPct < 0 || rotaryPct > 1) {
            throw new IllegalArgumentException("rotaryPct必须在[0,1]范围内");
        }
        if (rotaryBase <= 0) {
            throw new IllegalArgumentException("rotaryBase必须大于0");
        }
        if (sparseLocalWindow <= 0) {
            throw new IllegalArgumentException("稀疏注意力局部窗口大小必须大于0");
        }
        if (sparseStrideSize <= 0) {
            throw new IllegalArgumentException("稀疏注意力步长大小必须大于0");
        }
        if (maxFewShotExamples < 0) {
            throw new IllegalArgumentException("Few-shot最大示例数不能为负");
        }
    }


    // ==================== 基础配置Getter和Setter ====================

    public int getVocabSize() {
        return vocabSize;
    }

    public void setVocabSize(int vocabSize) {
        this.vocabSize = vocabSize;
    }

    public int getNPositions() {
        return nPositions;
    }

    public void setNPositions(int nPositions) {
        this.nPositions = nPositions;
    }

    public int getNEmbd() {
        return nEmbd;
    }

    public void setNEmbd(int nEmbd) {
        this.nEmbd = nEmbd;
    }

    public int getNLayer() {
        return nLayer;
    }

    public void setNLayer(int nLayer) {
        this.nLayer = nLayer;
    }

    public int getNHead() {
        return nHead;
    }

    public void setNHead(int nHead) {
        this.nHead = nHead;
    }

    public int getNInner() {
        return nInner;
    }

    public void setNInner(int nInner) {
        this.nInner = nInner;
    }

    public String getActivationFunction() {
        return activationFunction;
    }

    public void setActivationFunction(String activationFunction) {
        this.activationFunction = activationFunction;
    }

    // ==================== Dropout配置Getter和Setter ====================

    public double getResidPdrop() {
        return residPdrop;
    }

    public void setResidPdrop(double residPdrop) {
        this.residPdrop = residPdrop;
    }

    public double getEmbdPdrop() {
        return embdPdrop;
    }

    public void setEmbdPdrop(double embdPdrop) {
        this.embdPdrop = embdPdrop;
    }

    public double getAttnPdrop() {
        return attnPdrop;
    }

    public void setAttnPdrop(double attnPdrop) {
        this.attnPdrop = attnPdrop;
    }

    // ==================== 初始化配置Getter和Setter ====================

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

    // ==================== GPT-3特有配置Getter和Setter ====================

    public boolean isParallelAttention() {
        return parallelAttention;
    }

    public void setParallelAttention(boolean parallelAttention) {
        this.parallelAttention = parallelAttention;
    }

    public boolean isUseRotaryEmbedding() {
        return useRotaryEmbedding;
    }

    public void setUseRotaryEmbedding(boolean useRotaryEmbedding) {
        this.useRotaryEmbedding = useRotaryEmbedding;
    }

    public double getRotaryPct() {
        return rotaryPct;
    }

    public void setRotaryPct(double rotaryPct) {
        this.rotaryPct = rotaryPct;
    }

    public double getRotaryBase() {
        return rotaryBase;
    }

    public void setRotaryBase(double rotaryBase) {
        this.rotaryBase = rotaryBase;
    }

    public boolean isSparseAttention() {
        return sparseAttention;
    }

    public void setSparseAttention(boolean sparseAttention) {
        this.sparseAttention = sparseAttention;
    }

    public int getSparseLocalWindow() {
        return sparseLocalWindow;
    }

    public void setSparseLocalWindow(int sparseLocalWindow) {
        this.sparseLocalWindow = sparseLocalWindow;
    }

    public int getSparseStrideSize() {
        return sparseStrideSize;
    }

    public void setSparseStrideSize(int sparseStrideSize) {
        this.sparseStrideSize = sparseStrideSize;
    }

    public boolean isGradientCheckpointing() {
        return gradientCheckpointing;
    }

    public void setGradientCheckpointing(boolean gradientCheckpointing) {
        this.gradientCheckpointing = gradientCheckpointing;
    }

    public boolean isUseCache() {
        return useCache;
    }

    public void setUseCache(boolean useCache) {
        this.useCache = useCache;
    }

    public int getMaxFewShotExamples() {
        return maxFewShotExamples;
    }

    public void setMaxFewShotExamples(int maxFewShotExamples) {
        this.maxFewShotExamples = maxFewShotExamples;
    }

    @Override
    public String toString() {
        return String.format("GPT3Config{" +
                        "vocabSize=%d, nPositions=%d, nEmbd=%d, nLayer=%d, nHead=%d, nInner=%d, " +
                        "parallelAttention=%b, useRotaryEmbedding=%b, rotaryPct=%.2f, " +
                        "sparseAttention=%b, gradientCheckpointing=%b, estimatedParams=%s}",
                getVocabSize(), getNPositions(), getNEmbd(), getNLayer(), getNHead(), getNInner(),
                parallelAttention, useRotaryEmbedding, rotaryPct,
                sparseAttention, gradientCheckpointing, formatParamCount(estimateParameterCount()));
    }

    /**
     * 格式化参数数量为可读字符串
     */
    private String formatParamCount(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.1fB", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.1fM", count / 1_000_000.0);
        } else {
            return String.format("%d", count);
        }
    }
}
