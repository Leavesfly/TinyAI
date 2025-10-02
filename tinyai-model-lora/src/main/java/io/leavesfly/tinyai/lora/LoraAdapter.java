package io.leavesfly.tinyai.lora;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Parameter;

/**
 * LoRA适配器类 - 实现低秩矩阵分解的核心组件
 * 
 * LoRA (Low-Rank Adaptation) 是一种高效的参数微调技术，通过将原始权重矩阵W分解为：
 * W' = W + A * B
 * 其中：
 * - W 是冻结的预训练权重矩阵 (d x k)
 * - A 是可训练的下降矩阵 (d x r)  
 * - B 是可训练的上升矩阵 (r x k)
 * - r 是秩（rank），通常 r << min(d, k)
 * 
 * 这样可以大幅减少可训练参数数量，同时保持模型性能。
 * 
 * @author leavesfly
 * @version 1.0
 */
public class LoraAdapter {
    
    /**
     * LoRA配置参数
     */
    private final LoraConfig config;
    
    /**
     * 下降矩阵A：将输入从原始维度映射到低秩空间
     * 形状: (input_dim, rank)
     */
    private final Parameter matrixA;
    
    /**
     * 上升矩阵B：将低秩空间映射回输出维度
     * 形状: (rank, output_dim)
     */
    private final Parameter matrixB;
    
    /**
     * 缩放因子：控制LoRA输出的幅度
     * scaling = alpha / rank
     */
    private final double scaling;
    
    /**
     * 是否启用LoRA适配器
     */
    private boolean enabled = true;
    
    /**
     * 构造LoRA适配器
     * 
     * @param inputDim 输入维度
     * @param outputDim 输出维度  
     * @param config LoRA配置
     */
    public LoraAdapter(int inputDim, int outputDim, LoraConfig config) {
        this.config = config;
        this.scaling = (double) config.getAlpha() / config.getRank();
        
        // 初始化矩阵A：使用高斯分布
        NdArray initA = NdArray.likeRandomN(Shape.of(inputDim, config.getRank()))
                .divNum(Math.sqrt(inputDim));
        this.matrixA = new Parameter(initA);
        this.matrixA.setName("lora_A");
        
        // 初始化矩阵B：使用零初始化，确保训练开始时LoRA输出为0
        NdArray initB = NdArray.zeros(Shape.of(config.getRank(), outputDim));
        this.matrixB = new Parameter(initB);
        this.matrixB.setName("lora_B");
    }
    
    /**
     * LoRA前向传播
     * 计算 x * A * B * scaling
     * 
     * @param input 输入变量
     * @return LoRA输出
     */
    public Variable forward(Variable input) {
        if (!enabled) {
            // 如果LoRA被禁用，返回零张量
            Shape outputShape = Shape.of(input.getValue().getShape().getDimension(0), 
                                       matrixB.getValue().getShape().getDimension(1));
            return new Variable(NdArray.zeros(outputShape));
        }
        
        // 执行低秩矩阵乘法：input * A * B
        Variable hiddenOutput = input.matMul(matrixA);  // [batch, rank]
        Variable loraOutput = hiddenOutput.matMul(matrixB);  // [batch, output_dim]
        
        // 应用缩放因子
        if (scaling != 1.0) {
            Variable scalingVar = new Variable(NdArray.of(scaling));
            loraOutput = loraOutput.mul(scalingVar);
        }
        
        return loraOutput;
    }
    
    /**
     * 获取LoRA参数总数
     * 
     * @return 参数数量
     */
    public int getParameterCount() {
        return matrixA.getValue().getShape().size() + matrixB.getValue().getShape().size();
    }
    
    /**
     * 获取相对于全参数微调的参数减少比例
     * 
     * @param originalParamCount 原始参数数量
     * @return 参数减少比例
     */
    public double getParameterReduction(int originalParamCount) {
        return 1.0 - (double) getParameterCount() / originalParamCount;
    }
    
    /**
     * 启用LoRA适配器
     */
    public void enable() {
        this.enabled = true;
    }
    
    /**
     * 禁用LoRA适配器
     */
    public void disable() {
        this.enabled = false;
    }
    
    /**
     * 检查LoRA是否启用
     * 
     * @return 是否启用
     */
    public boolean isEnabled() {
        return enabled;
    }
    
    /**
     * 清除梯度
     */
    public void clearGrads() {
        matrixA.clearGrad();
        matrixB.clearGrad();
    }
    
    /**
     * 获取矩阵A
     * 
     * @return 矩阵A参数
     */
    public Parameter getMatrixA() {
        return matrixA;
    }
    
    /**
     * 获取矩阵B
     * 
     * @return 矩阵B参数
     */
    public Parameter getMatrixB() {
        return matrixB;
    }
    
    /**
     * 获取配置
     * 
     * @return LoRA配置
     */
    public LoraConfig getConfig() {
        return config;
    }
    
    /**
     * 获取缩放因子
     * 
     * @return 缩放因子
     */
    public double getScaling() {
        return scaling;
    }
    
    /**
     * 获取适配器信息字符串
     * 
     * @return 信息字符串
     */
    @Override
    public String toString() {
        return String.format("LoraAdapter{rank=%d, alpha=%.1f, scaling=%.4f, enabled=%s, params=%d}", 
                config.getRank(), config.getAlpha(), scaling, enabled, getParameterCount());
    }
}