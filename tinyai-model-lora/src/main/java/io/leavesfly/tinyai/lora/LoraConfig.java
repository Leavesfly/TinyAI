package io.leavesfly.tinyai.lora;

/**
 * LoRA配置类 - 管理LoRA微调的超参数
 * 
 * LoRA (Low-Rank Adaptation) 是一种高效的参数微调技术，
 * 该配置类管理所有相关的超参数设置。
 * 
 * @author leavesfly
 * @version 1.0
 */
public class LoraConfig {
    
    /**
     * 低秩矩阵的秩（rank）
     * 决定LoRA适配器的容量，通常设置为8、16、32、64等
     * 更高的秩提供更强的表达能力，但增加参数量
     */
    private final int rank;
    
    /**
     * 缩放参数alpha
     * 用于控制LoRA输出的幅度，scaling = alpha / rank
     * 通常设置为与rank相同的值，如alpha=16对应rank=16
     */
    private final double alpha;
    
    /**
     * dropout概率（可选）
     * 用于正则化，防止过拟合
     */
    private final double dropout;
    
    /**
     * 是否启用偏置项微调
     * 如果为true，同时微调LoRA参数和偏置项
     */
    private final boolean enableBias;
    
    /**
     * 目标模块类型
     * 指定哪些类型的层需要应用LoRA
     */
    private final String[] targetModules;
    
    /**
     * 构造函数 - 创建LoRA配置
     * 
     * @param rank LoRA矩阵的秩
     * @param alpha 缩放参数
     */
    public LoraConfig(int rank, double alpha) {
        this(rank, alpha, 0.0, false, new String[]{"linear"});
    }
    
    /**
     * 完整构造函数
     * 
     * @param rank LoRA矩阵的秩
     * @param alpha 缩放参数
     * @param dropout dropout概率
     * @param enableBias 是否启用偏置微调
     * @param targetModules 目标模块类型
     */
    public LoraConfig(int rank, double alpha, double dropout, boolean enableBias, String[] targetModules) {
        if (rank <= 0) {
            throw new IllegalArgumentException("Rank must be positive, got: " + rank);
        }
        if (alpha < 0) {
            throw new IllegalArgumentException("Alpha must be non-negative, got: " + alpha);
        }
        if (dropout < 0 || dropout >= 1) {
            throw new IllegalArgumentException("Dropout must be in [0, 1), got: " + dropout);
        }
        
        this.rank = rank;
        this.alpha = alpha;
        this.dropout = dropout;
        this.enableBias = enableBias;
        this.targetModules = targetModules != null ? targetModules.clone() : new String[]{"linear"};
    }
    
    /**
     * 创建常用的LoRA配置预设
     * 
     * @param rank 矩阵秩
     * @return LoRA配置
     */
    public static LoraConfig createDefault(int rank) {
        return new LoraConfig(rank, rank); // alpha = rank是常见设置
    }
    
    /**
     * 创建用于实验的低秩配置
     * 
     * @return 低秩LoRA配置
     */
    public static LoraConfig createLowRank() {
        return new LoraConfig(4, 8.0); // 非常小的rank用于快速实验
    }
    
    /**
     * 创建用于实际应用的中等秩配置
     * 
     * @return 中等秩LoRA配置
     */
    public static LoraConfig createMediumRank() {
        return new LoraConfig(16, 32.0); // 平衡性能和效率
    }
    
    /**
     * 创建用于复杂任务的高秩配置
     * 
     * @return 高秩LoRA配置
     */
    public static LoraConfig createHighRank() {
        return new LoraConfig(64, 128.0); // 更强的表达能力
    }
    
    /**
     * 获取矩阵秩
     * 
     * @return 矩阵秩
     */
    public int getRank() {
        return rank;
    }
    
    /**
     * 获取缩放参数
     * 
     * @return 缩放参数
     */
    public double getAlpha() {
        return alpha;
    }
    
    /**
     * 获取dropout概率
     * 
     * @return dropout概率
     */
    public double getDropout() {
        return dropout;
    }
    
    /**
     * 检查是否启用偏置微调
     * 
     * @return 是否启用偏置微调
     */
    public boolean isEnableBias() {
        return enableBias;
    }
    
    /**
     * 获取目标模块类型
     * 
     * @return 目标模块类型数组
     */
    public String[] getTargetModules() {
        return targetModules.clone();
    }
    
    /**
     * 计算缩放因子
     * 
     * @return 缩放因子 (alpha / rank)
     */
    public double getScaling() {
        return alpha / rank;
    }
    
    /**
     * 检查指定模块是否在目标列表中
     * 
     * @param moduleName 模块名称
     * @return 是否为目标模块
     */
    public boolean isTargetModule(String moduleName) {
        for (String target : targetModules) {
            if (target.equalsIgnoreCase(moduleName)) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * 计算相对于全参数微调的参数减少比例
     * 
     * @param originalInputDim 原始输入维度
     * @param originalOutputDim 原始输出维度
     * @return 参数减少比例
     */
    public double getParameterReduction(int originalInputDim, int originalOutputDim) {
        int originalParams = originalInputDim * originalOutputDim;
        int loraParams = rank * (originalInputDim + originalOutputDim);
        return 1.0 - (double) loraParams / originalParams;
    }
    
    /**
     * 验证配置的合理性
     * 
     * @param inputDim 输入维度
     * @param outputDim 输出维度
     * @throws IllegalArgumentException 当配置不合理时抛出
     */
    public void validate(int inputDim, int outputDim) {
        if (rank >= Math.min(inputDim, outputDim)) {
            throw new IllegalArgumentException(
                String.format("Rank %d should be smaller than min(input_dim=%d, output_dim=%d)", 
                            rank, inputDim, outputDim));
        }
    }
    
    /**
     * 创建配置的副本，允许修改某些参数
     * 
     * @param newRank 新的矩阵秩
     * @return 新的LoRA配置
     */
    public LoraConfig withRank(int newRank) {
        return new LoraConfig(newRank, alpha, dropout, enableBias, targetModules);
    }
    
    /**
     * 创建配置的副本，允许修改alpha参数
     * 
     * @param newAlpha 新的alpha值
     * @return 新的LoRA配置
     */
    public LoraConfig withAlpha(double newAlpha) {
        return new LoraConfig(rank, newAlpha, dropout, enableBias, targetModules);
    }
    
    @Override
    public String toString() {
        return String.format("LoraConfig{rank=%d, alpha=%.1f, scaling=%.4f, dropout=%.2f, enableBias=%s, targets=%s}", 
                            rank, alpha, getScaling(), dropout, enableBias, String.join(",", targetModules));
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        LoraConfig that = (LoraConfig) obj;
        return rank == that.rank &&
               Double.compare(that.alpha, alpha) == 0 &&
               Double.compare(that.dropout, dropout) == 0 &&
               enableBias == that.enableBias;
    }
    
    @Override
    public int hashCode() {
        int result = rank;
        result = 31 * result + Double.hashCode(alpha);
        result = 31 * result + Double.hashCode(dropout);
        result = 31 * result + Boolean.hashCode(enableBias);
        return result;
    }
}