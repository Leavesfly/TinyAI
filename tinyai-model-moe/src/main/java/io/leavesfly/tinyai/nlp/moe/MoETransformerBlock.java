package io.leavesfly.tinyai.nlp.moe;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.layer.transformer.LayerNorm;
import io.leavesfly.tinyai.nnet.layer.transformer.MultiHeadAttention;

import java.util.ArrayList;
import java.util.List;

/**
 * MoE Transformer Block
 * <p>
 * 基于Mixture of Experts的Transformer块，将传统的FeedForward层替换为MoE层。
 * 这样可以大幅增加模型容量而不显著增加计算开销。
 * <p>
 * 结构：
 * Input 
 * → LayerNorm1 → Multi-Head Attention → Residual Connection
 * → LayerNorm2 → MoE Layer → Residual Connection 
 * → Output
 * <p>
 * 特点：
 * 1. 使用Pre-LayerNorm架构（与GPT-2相同）
 * 2. 用MoE层替换传统的FeedForward层
 * 3. 每个token可能激活不同的专家组合
 * 4. 支持稀疏计算和动态容量扩展
 * 5. 保持与标准Transformer的兼容性
 *
 * @author leavesfly
 * @version 1.0
 */
public class MoETransformerBlock extends Layer {
    
    private LayerNorm layerNorm1;           // 第一个层归一化
    private MultiHeadAttention attention;   // 多头自注意力
    private LayerNorm layerNorm2;           // 第二个层归一化
    private MoELayer moeLayer;              // MoE层（替换FeedForward）
    
    // 配置参数
    private int dModel;                     // 模型维度
    private int numHeads;                   // 注意力头数
    private int numExperts;                 // 专家数量
    private int dExpert;                    // 专家隐藏层维度
    private int topK;                       // Top-K专家选择
    private double dropoutRate;             // Dropout比率
    private boolean useNoise;               // 是否使用门控噪声
    private double noiseEpsilon;            // 噪声强度
    
    /**
     * 构造MoE Transformer Block
     * 
     * @param name 块名称
     * @param dModel 模型维度
     * @param numHeads 注意力头数
     * @param numExperts 专家数量
     * @param dExpert 专家隐藏层维度
     * @param topK Top-K专家选择
     * @param dropoutRate Dropout比率
     * @param useNoise 是否使用门控噪声
     * @param noiseEpsilon 噪声强度
     */
    public MoETransformerBlock(String name, int dModel, int numHeads, int numExperts,
                               int dExpert, int topK, double dropoutRate,
                               boolean useNoise, double noiseEpsilon) {
        super(name, Shape.of(-1, -1, dModel), Shape.of(-1, -1, dModel));
        
        if (dModel <= 0 || numHeads <= 0 || numExperts <= 0) {
            throw new IllegalArgumentException("dModel, numHeads和numExperts必须大于0");
        }
        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException("dModel必须能被numHeads整除");
        }
        if (topK <= 0 || topK > numExperts) {
            throw new IllegalArgumentException("topK必须在1到numExperts之间");
        }
        
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.numExperts = numExperts;
        this.dExpert = dExpert;
        this.topK = topK;
        this.dropoutRate = dropoutRate;
        this.useNoise = useNoise;
        this.noiseEpsilon = noiseEpsilon;
        
        init();
    }
    
    /**
     * 使用默认参数的构造函数
     */
    public MoETransformerBlock(String name, int dModel, int numHeads, int numExperts, int topK) {
        this(name, dModel, numHeads, numExperts, dModel * 4, topK, 0.1, true, 0.1);
    }
    
    /**
     * 简化的构造函数（默认8个专家，Top-2选择）
     */
    public MoETransformerBlock(String name, int dModel, int numHeads) {
        this(name, dModel, numHeads, 8, 2);
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化第一个层归一化
            layerNorm1 = new LayerNorm(name + "_ln1", dModel);
            
            // 初始化多头自注意力（使用掩码，用于解码器）
            attention = new MultiHeadAttention(name + "_attention", dModel, numHeads, true);
            
            // 初始化第二个层归一化
            layerNorm2 = new LayerNorm(name + "_ln2", dModel);
            
            // 初始化MoE层
            moeLayer = new MoELayer(
                name + "_moe", 
                numExperts, 
                dModel, 
                dExpert, 
                topK, 
                useNoise, 
                noiseEpsilon
            );
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable x = inputs[0];
        
        // 验证输入维度
        NdArray inputData = x.getValue();
        if (inputData.getShape().getDimension(2) != dModel) {
            throw new IllegalArgumentException(
                String.format("MoE Transformer Block输入维度不匹配。期望%d，实际%d", 
                             dModel, inputData.getShape().getDimension(2))
            );
        }
        
        // MoE-GPT使用Pre-LayerNorm架构
        
        // 1. Layer Norm + Multi-Head Attention + Residual Connection
        Variable norm1Output = layerNorm1.layerForward(x);
        Variable attentionOutput = attention.layerForward(norm1Output, norm1Output, norm1Output);
        Variable residual1 = addResidualConnection(x, attentionOutput);
        
        // 应用Dropout（如果需要）
        residual1 = applyDropout(residual1);
        
        // 2. Layer Norm + MoE Layer + Residual Connection
        Variable norm2Output = layerNorm2.layerForward(residual1);
        Variable moeOutput = moeLayer.layerForward(norm2Output);
        Variable residual2 = addResidualConnection(residual1, moeOutput);
        
        // 应用Dropout（如果需要）
        residual2 = applyDropout(residual2);
        
        return residual2;
    }
    
    /**
     * 添加残差连接
     */
    private Variable addResidualConnection(Variable input, Variable output) {
        return input.add(output);
    }
    
    /**
     * 应用Dropout（简化版本）
     * 实际实现中需要考虑训练/推理模式
     */
    private Variable applyDropout(Variable input) {
        if (dropoutRate > 0.0) {
            // 这里是简化的dropout实现
            // 实际应用中需要生成随机掩码并考虑训练/推理模式
            return input;
        }
        return input;
    }
    
    @Override
    public NdArray forward(NdArray... inputs) {
        return layerForward(new Variable(inputs[0])).getValue();
    }
    
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // MoE Transformer Block的反向传播需要依次通过各个子层
        List<NdArray> result = new ArrayList<>();
        result.add(yGrad);
        return result;
    }
    
    @Override
    public int requireInputNum() {
        return 1;
    }
    
    /**
     * 获取MoE层的负载均衡统计信息
     */
    public MoELayer.LoadBalancingStats getMoEStats() {
        return moeLayer.getLoadBalancingStats();
    }
    
    /**
     * 计算负载均衡损失
     * 
     * @return 负载均衡损失值
     */
    public float computeLoadBalancingLoss() {
        MoELayer.LoadBalancingStats stats = getMoEStats();
        
        // 如果没有处理任何token，返回0
        if (stats.totalTokens == 0) {
            return 0.0f;
        }
        
        // 使用负载不均衡系数作为损失
        // 负载不均衡系数越大，损失越大
        return (float) stats.loadImbalance;
    }
    
    /**
     * 重置MoE统计信息
     */
    public void resetMoEStats() {
        moeLayer.resetStats();
    }
    
    /**
     * 获取模型配置信息
     */
    public String getBlockConfig() {
        return String.format(
            "MoETransformerBlock Config:\n" +
            "  - Model Dim: %d\n" +
            "  - Num Heads: %d\n" +
            "  - Num Experts: %d\n" +
            "  - Expert Hidden Dim: %d\n" +
            "  - Top-K: %d\n" +
            "  - Dropout Rate: %.2f\n" +
            "  - Use Noise: %s\n" +
            "  - Noise Epsilon: %.3f",
            dModel, numHeads, numExperts, dExpert, topK, 
            dropoutRate, useNoise, noiseEpsilon
        );
    }
    
    /**
     * 计算MoE层的参数数量
     */
    public long getMoEParameterCount() {
        long totalParams = 0;
        
        // MoE层参数：专家网络 + 门控网络
        for (int i = 0; i < numExperts; i++) {
            // 每个专家：(dModel + 1) * dExpert + (dExpert + 1) * dModel
            totalParams += (dModel + 1) * dExpert + (dExpert + 1) * dModel;
        }
        
        // 门控网络：dModel * numExperts（无偏置）
        totalParams += dModel * numExperts;
        
        return totalParams;
    }
    
    /**
     * 计算总参数数量（包括注意力和LayerNorm）
     */
    public long getTotalParameterCount() {
        long totalParams = getMoEParameterCount();
        
        // 多头注意力参数（简化估算）
        // Q, K, V, Output线性层：4 * dModel * dModel
        totalParams += 4L * dModel * dModel;
        
        // LayerNorm参数：2 * 2 * dModel（两个LayerNorm，每个有gamma和beta）
        totalParams += 4L * dModel;
        
        return totalParams;
    }
    
    // Getter方法
    public LayerNorm getLayerNorm1() { return layerNorm1; }
    public MultiHeadAttention getAttention() { return attention; }
    public LayerNorm getLayerNorm2() { return layerNorm2; }
    public MoELayer getMoeLayer() { return moeLayer; }
    public int getDModel() { return dModel; }
    public int getNumHeads() { return numHeads; }
    public int getNumExperts() { return numExperts; }
    public int getDExpert() { return dExpert; }
    public int getTopK() { return topK; }
    public double getDropoutRate() { return dropoutRate; }
    public boolean isUseNoise() { return useNoise; }
    public double getNoiseEpsilon() { return noiseEpsilon; }
    
    @Override
    public String toString() {
        return String.format(
            "MoETransformerBlock(dModel=%d, numHeads=%d, numExperts=%d, topK=%d, params=%d)", 
            dModel, numHeads, numExperts, topK, getTotalParameterCount()
        );
    }
}