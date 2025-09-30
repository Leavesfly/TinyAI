package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.math.Tanh;
import io.leavesfly.tinyai.func.matrix.MatMul;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.Parameter;
import io.leavesfly.tinyai.nnet.layer.transformer.LayerNorm;

/**
 * Transformer块实现
 * 
 * 包含多头自注意力机制和前馈神经网络，
 * 并使用残差连接和层归一化。
 * 
 * 结构：
 * 1. LayerNorm -> MultiHeadAttention -> 残差连接
 * 2. LayerNorm -> FeedForward -> 残差连接
 * 
 * @author leavesfly
 * @version 1.0
 */
public class TransformerBlock extends Layer {
    
    private int dModel;      // 模型维度
    private int numHeads;    // 注意力头数量
    private int dFF;         // 前馈网络隐藏维度
    private double dropout;  // Dropout比率
    
    // 组件
    private MultiHeadAttention attention;  // 多头注意力
    private LayerNorm norm1;              // 第一个层归一化
    private LayerNorm norm2;              // 第二个层归一化
    
    // 前馈网络参数
    private Parameter ffW1;    // 前馈网络第一层权重
    private Parameter ffB1;    // 前馈网络第一层偏置
    private Parameter ffW2;    // 前馈网络第二层权重
    private Parameter ffB2;    // 前馈网络第二层偏置
    
    /**
     * 构造Transformer块
     * 
     * @param name 层名称
     * @param dModel 模型维度
     * @param numHeads 注意力头数量
     * @param dFF 前馈网络隐藏维度
     * @param dropout Dropout比率
     */
    public TransformerBlock(String name, int dModel, int numHeads, int dFF, double dropout) {
        super(name, Shape.of(-1, -1, dModel), Shape.of(-1, -1, dModel));
        
        if (dModel <= 0 || numHeads <= 0 || dFF <= 0) {
            throw new IllegalArgumentException("dModel、numHeads和dFF必须大于0");
        }
        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException("dModel必须能被numHeads整除");
        }
        if (dropout < 0.0 || dropout > 1.0) {
            throw new IllegalArgumentException("dropout必须在0.0到1.0之间");
        }
        
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dFF = dFF;
        this.dropout = dropout;
        
        init();
    }
    
    /**
     * 默认构造函数
     */
    public TransformerBlock(String name, int dModel, int numHeads, int dFF) {
        this(name, dModel, numHeads, dFF, 0.1);
    }
    
    /**
     * 初始化权重矩阵
     */
    private NdArray initializeWeights(Shape shape) {
        NdArray weights = NdArray.of(shape);
        double fanIn = shape.getDimension(0);
        double fanOut = shape.getDimension(1);
        double limit = Math.sqrt(6.0 / (fanIn + fanOut));
        
        // 随机初始化
        for (int i = 0; i < weights.getShape().size(); i++) {
            double value = (Math.random() * 2.0 - 1.0) * limit;
            weights.set((float) value, i);
        }
        
        return weights;
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化多头注意力
            attention = new MultiHeadAttention(
                name + "_attention", dModel, numHeads, dropout
            );
            
            // 初始化层归一化
            norm1 = new LayerNorm(name + "_norm1", dModel);
            norm2 = new LayerNorm(name + "_norm2", dModel);
            
            // 初始化前馈网络权重
            ffW1 = new Parameter(initializeWeights(Shape.of(dModel, dFF)));
            ffB1 = new Parameter(NdArray.zeros(Shape.of(dFF)));
            ffW2 = new Parameter(initializeWeights(Shape.of(dFF, dModel)));
            ffB2 = new Parameter(NdArray.zeros(Shape.of(dModel)));
            
            params.put("ffW1", ffW1);
            params.put("ffB1", ffB1);
            params.put("ffW2", ffW2);
            params.put("ffB2", ffB2);
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];  // shape: (batch_size, seq_len, dModel)
        Variable mask = inputs.length > 1 ? inputs[1] : null;  // 可选的注意力掩码
        
        Variable x = input;
        
        // 1. 自注意力子层
        // LayerNorm -> MultiHeadAttention -> 残差连接
        Variable normed1 = norm1.layerForward(x);
        Variable attended = attention.forward(normed1, normed1, normed1, mask);
        x = addResidual(x, attended);
        
        // 2. 前馈神经网络子层
        // LayerNorm -> FeedForward -> 残差连接
        Variable normed2 = norm2.layerForward(x);
        Variable fedForward = feedForward(normed2);
        x = addResidual(x, fedForward);
        
        return x;
    }
    
    /**
     * 前馈神经网络
     * FFN(x) = max(0, xW1 + b1)W2 + b2
     * 
     * @param input 输入张量
     * @return 前馈网络输出
     */
    private Variable feedForward(Variable input) {
        // 第一层：线性变换 + ReLU激活
        Variable layer1 = linearTransform(input, ffW1, ffB1);
        layer1 = relu(layer1);
        
        // 应用Dropout
        if (dropout > 0.0) {
            layer1 = applyDropout(layer1);
        }
        
        // 第二层：线性变换
        Variable layer2 = linearTransform(layer1, ffW2, ffB2);
        
        // 应用Dropout
        if (dropout > 0.0) {
            layer2 = applyDropout(layer2);
        }
        
        return layer2;
    }
    
    /**
     * 线性变换：input @ weight + bias
     */
    private Variable linearTransform(Variable input, Parameter weight, Parameter bias) {
        MatMul matMul = new MatMul();
        NdArray result = matMul.forward(input.getValue(), weight.getValue());
        return addBias(new Variable(result), bias);
    }
    
    /**
     * ReLU激活函数
     */
    private Variable relu(Variable input) {
        NdArray inputArray = input.getValue();
        NdArray result = NdArray.of(inputArray.getShape());
        
        for (int i = 0; i < inputArray.getShape().size(); i++) {
            float value = inputArray.get(i);
            result.set(Math.max(0.0f, value), i);
        }
        
        return new Variable(result);
    }
    
    /**
     * GELU激活函数（可选的高级激活函数）
     * GELU(x) = x * Φ(x) = x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
     */
    private Variable gelu(Variable input) {
        NdArray inputArray = input.getValue();
        NdArray result = NdArray.of(inputArray.getShape());
        
        for (int i = 0; i < inputArray.getShape().size(); i++) {
            float x = inputArray.get(i);
            // 近似GELU计算
            double cdf = 0.5 * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * 
                               (x + 0.044715 * Math.pow(x, 3))));
            result.set((float)(x * cdf), i);
        }
        
        return new Variable(result);
    }
    
    /**
     * 添加偏置
     */
    private Variable addBias(Variable input, Parameter bias) {
        NdArray inputArray = input.getValue();
        NdArray biasArray = bias.getValue();
        NdArray result = NdArray.of(inputArray.getShape());
        
        int batchSize = inputArray.getShape().getDimension(0);
        int seqLen = inputArray.getShape().getDimension(1);
        int features = inputArray.getShape().getDimension(2);
        
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                for (int f = 0; f < features; f++) {
                    float value = inputArray.get(b, t, f) + biasArray.get(f);
                    result.set(value, b, t, f);
                }
            }
        }
        
        return new Variable(result);
    }
    
    /**
     * 残差连接
     */
    private Variable addResidual(Variable residual, Variable input) {
        NdArray residualArray = residual.getValue();
        NdArray inputArray = input.getValue();
        NdArray result = NdArray.of(residualArray.getShape());
        
        for (int i = 0; i < residualArray.getShape().size(); i++) {
            result.set(residualArray.get(i) + inputArray.get(i), i);
        }
        
        return new Variable(result);
    }
    
    /**
     * 应用Dropout（简化实现）
     */
    private Variable applyDropout(Variable input) {
        // 训练时随机将一些元素设为0
        // 这里简化为不应用dropout，实际应用中需要根据训练/评估模式决定
        return input;
    }
    
    /**
     * 获取注意力权重（用于可视化）
     */
    public Variable getAttentionWeights(Variable input, Variable mask) {
        Variable normed = norm1.layerForward(input);
        // 这里需要修改MultiHeadAttention类以返回注意力权重
        // 为简化实现，暂时返回null
        return null;
    }
    
    /**
     * 设置训练模式
     * 影响Dropout的行为
     */
    public void setTraining(boolean training) {
        // 在实际实现中，这里应该设置各个组件的训练状态
        // 影响Dropout和BatchNorm等层的行为
    }
    
    /**
     * 获取层的统计信息
     */
    public String getLayerStats() {
        return String.format(
            "TransformerBlock Stats:\n" +
            "  - dModel: %d\n" +
            "  - numHeads: %d\n" +
            "  - dFF: %d\n" +
            "  - dropout: %.3f\n" +
            "  - parameters: %d",
            dModel, numHeads, dFF, dropout, getTotalParameters()
        );
    }
    
    /**
     * 计算总参数数量
     */
    private int getTotalParameters() {
        int attentionParams = 4 * dModel * dModel + 4 * dModel; // Q,K,V,O权重和偏置
        int ffParams = dModel * dFF + dFF + dFF * dModel + dModel; // 前馈网络权重和偏置
        int normParams = 2 * 2 * dModel; // 两个LayerNorm，每个有缩放和偏移参数
        return attentionParams + ffParams + normParams;
    }
    
    // Getter方法
    public int getDModel() { return dModel; }
    public int getNumHeads() { return numHeads; }
    public int getDFF() { return dFF; }
    public double getDropout() { return dropout; }
    public MultiHeadAttention getAttention() { return attention; }
    public LayerNorm getNorm1() { return norm1; }
    public LayerNorm getNorm2() { return norm2; }
    
    @Override
    public String toString() {
        return String.format("TransformerBlock(dModel=%d, numHeads=%d, dFF=%d, dropout=%.2f)", 
                           dModel, numHeads, dFF, dropout);
    }
}