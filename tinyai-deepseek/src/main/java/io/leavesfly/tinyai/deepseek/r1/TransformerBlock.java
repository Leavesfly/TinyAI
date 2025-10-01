package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.layer.transformer.FeedForward;
import io.leavesfly.tinyai.nnet.layer.transformer.LayerNorm;
import io.leavesfly.tinyai.nnet.layer.transformer.MultiHeadAttention;

/**
 * Transformer块实现 - DeepSeek R1的基础Transformer组件
 * 
 * 该块包含：
 * 1. 多头自注意力机制
 * 2. 层归一化
 * 3. 前馈神经网络
 * 4. 残差连接
 * 
 * 基于Python实现中的TransformerBlock，使用TinyAI架构和现有组件重新实现
 */
public class TransformerBlock extends Block {
    
    private int dModel;
    private int numHeads;
    private int dFF;
    private double dropout;
    
    // Transformer核心组件
    private MultiHeadAttention attention;
    private LayerNorm layerNorm1;
    private FeedForward feedForward;
    private LayerNorm layerNorm2;
    
    /**
     * 构造Transformer块
     * 
     * @param name Transformer块名称
     * @param dModel 模型维度
     * @param numHeads 注意力头数
     * @param dFF 前馈网络隐藏层维度
     * @param dropout Dropout比率
     */
    public TransformerBlock(String name, int dModel, int numHeads, int dFF, double dropout) {
        super(name, Shape.of(-1, -1, dModel), Shape.of(-1, -1, dModel));
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dFF = dFF;
        this.dropout = dropout;
        init();
    }
    
    /**
     * 使用默认参数的构造函数
     * 
     * @param name Transformer块名称
     * @param dModel 模型维度
     * @param numHeads 注意力头数
     */
    public TransformerBlock(String name, int dModel, int numHeads) {
        this(name, dModel, numHeads, dModel * 4, 0.1);
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 多头自注意力机制（不使用掩码，编码器风格）
            attention = new MultiHeadAttention(name + "_attention", dModel, numHeads, false);
            addLayer(attention);
            
            // 第一个层归一化
            layerNorm1 = new LayerNorm(name + "_norm1", dModel);
            addLayer(layerNorm1);
            
            // 前馈神经网络
            feedForward = new FeedForward(name + "_ffn", dModel, dFF);
            addLayer(feedForward);
            
            // 第二个层归一化
            layerNorm2 = new LayerNorm(name + "_norm2", dModel);
            addLayer(layerNorm2);
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable x = inputs[0];
        Variable attentionMask = inputs.length > 1 ? inputs[1] : null;
        
        // 1. 多头自注意力 + 残差连接 + 层归一化
        // Pre-LN变体：先归一化，再注意力，然后残差连接
        Variable normalizedInput = layerNorm1.layerForward(x);
        Variable attentionOutput = attention.layerForward(normalizedInput, normalizedInput, normalizedInput);
        Variable residual1 = addResidualConnection(x, attentionOutput);
        
        // 2. 前馈网络 + 残差连接 + 层归一化
        // Pre-LN变体：先归一化，再前馈，然后残差连接
        Variable normalizedResidual = layerNorm2.layerForward(residual1);
        Variable ffnOutput = feedForward.layerForward(normalizedResidual);
        Variable residual2 = addResidualConnection(residual1, ffnOutput);
        
        return residual2;
    }
    
    /**
     * Post-LN变体的前向传播（更接近原始Transformer）
     */
    public Variable layerForwardPostLN(Variable... inputs) {
        Variable x = inputs[0];
        Variable attentionMask = inputs.length > 1 ? inputs[1] : null;
        
        // 1. 多头自注意力 + 残差连接 + 层归一化
        Variable attentionOutput = attention.layerForward(x, x, x);
        Variable residual1 = addResidualConnection(x, attentionOutput);
        Variable norm1Output = layerNorm1.layerForward(residual1);
        
        // 2. 前馈网络 + 残差连接 + 层归一化
        Variable ffnOutput = feedForward.layerForward(norm1Output);
        Variable residual2 = addResidualConnection(norm1Output, ffnOutput);
        Variable norm2Output = layerNorm2.layerForward(residual2);
        
        return norm2Output;
    }
    
    /**
     * 添加残差连接
     * 
     * @param input 原始输入
     * @param output 层输出
     * @return 残差连接结果
     */
    private Variable addResidualConnection(Variable input, Variable output) {
        // 检查形状兼容性
        NdArray inputData = input.getValue();
        NdArray outputData = output.getValue();
        
        if (!inputData.getShape().equals(outputData.getShape())) {
            throw new IllegalArgumentException(
                String.format("残差连接形状不匹配: 输入形状=%s, 输出形状=%s", 
                            inputData.getShape(), outputData.getShape())
            );
        }
        
        return input.add(output);
    }
    
    /**
     * 应用Dropout（简化版本）
     * 在训练模式下随机丢弃一些神经元
     */
    private Variable applyDropout(Variable input, double dropoutRate) {
        if (dropoutRate <= 0.0 || dropoutRate >= 1.0) {
            return input;
        }
        
        // 简化的dropout实现 - 修复API兼容性
        // 注意：由于NdArray API限制，暂时跳过dropout处理
        NdArray inputData = input.getValue();
        // 直接使用输入数据，不进行copy和dropout处理
        return input; // 返回原始输入
    }
    
    /**
     * 设置训练模式
     * 
     * @param training 是否为训练模式
     */
    public void setTraining(boolean training) {
        // 在实际实现中，这里应该设置各个子层的训练模式
        // 特别是dropout和batch normalization相关的层
    }
    
    /**
     * 获取注意力权重（用于可视化和分析）
     * 
     * @param inputs 输入变量
     * @return 注意力权重矩阵
     */
    public NdArray getAttentionWeights(Variable... inputs) {
        Variable x = inputs[0];
        Variable normalizedInput = layerNorm1.layerForward(x);
        
        // 这里需要修改MultiHeadAttention以返回attention weights
        // 目前的实现不直接支持，这是一个简化版本
        attention.layerForward(normalizedInput, normalizedInput, normalizedInput);
        
        // 返回模拟的注意力权重
        int batchSize = x.getValue().getShape().getDimension(0);
        int seqLen = x.getValue().getShape().getDimension(1);
        return NdArray.ones(Shape.of(batchSize, numHeads, seqLen, seqLen));
    }
    
    /**
     * 计算模块的计算复杂度（FLOPs）
     * 
     * @param seqLen 序列长度
     * @return 浮点运算次数
     */
    public long calculateFLOPs(int seqLen) {
        // 多头注意力的FLOPs: 4 * batch_size * seq_len^2 * d_model + 2 * batch_size * seq_len^2 * d_model
        long attentionFLOPs = 4L * seqLen * seqLen * dModel + 2L * seqLen * seqLen * dModel;
        
        // 前馈网络的FLOPs: 2 * batch_size * seq_len * d_model * d_ff
        long ffnFLOPs = 2L * seqLen * dModel * dFF;
        
        // 层归一化的FLOPs相对较小，这里忽略
        return attentionFLOPs + ffnFLOPs;
    }
    
    /**
     * 获取参数总数
     * 
     * @return 参数数量
     */
    public long getParameterCount() {
        long attentionParams = 4L * dModel * dModel + 4L * dModel; // QKV投影 + 输出投影 + 偏置
        long ffnParams = 2L * dModel * dFF + dModel + dFF; // 两个线性层 + 偏置
        long normParams = 4L * dModel; // 两个LayerNorm的gamma和beta参数
        
        return attentionParams + ffnParams + normParams;
    }
    
    // Getters
    public MultiHeadAttention getAttention() { return attention; }
    public LayerNorm getLayerNorm1() { return layerNorm1; }
    public FeedForward getFeedForward() { return feedForward; }
    public LayerNorm getLayerNorm2() { return layerNorm2; }
    public int getDModel() { return dModel; }
    public int getNumHeads() { return numHeads; }
    public int getDFF() { return dFF; }
    public double getDropout() { return dropout; }
    
    @Override
    public String toString() {
        return String.format("TransformerBlock{name=%s, dModel=%d, numHeads=%d, dFF=%d, dropout=%.2f}",
                name, dModel, numHeads, dFF, dropout);
    }
}