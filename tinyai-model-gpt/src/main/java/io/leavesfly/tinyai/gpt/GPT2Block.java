package io.leavesfly.tinyai.gpt;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.layer.transformer.FeedForward;
import io.leavesfly.tinyai.nnet.layer.transformer.LayerNorm;
import io.leavesfly.tinyai.nnet.layer.transformer.MultiHeadAttention;

import java.util.ArrayList;
import java.util.List;

/**
 * GPT-2 Transformer Block实现
 * 
 * GPT-2使用仅解码器的架构，每个block包含：
 * 1. 层归一化（前置）
 * 2. 带掩码的多头自注意力
 * 3. 残差连接
 * 4. 层归一化（前置）
 * 5. 前馈网络
 * 6. 残差连接
 * 
 * 注意：GPT-2使用Pre-LayerNorm结构（在子层之前应用层归一化）
 */
public class GPT2Block extends Layer {
    
    private LayerNorm layerNorm1;           // 第一个层归一化
    private MultiHeadAttention attention;    // 带掩码的多头自注意力
    private LayerNorm layerNorm2;           // 第二个层归一化
    private FeedForward feedForward;        // 前馈网络
    private double dropoutRate;
    
    /**
     * 构造GPT-2 Block
     * 
     * @param name 块名称
     * @param dModel 模型维度
     * @param numHeads 注意力头数
     * @param dFF 前馈网络隐藏维度
     * @param dropoutRate dropout比率
     */
    public GPT2Block(String name, int dModel, int numHeads, int dFF, double dropoutRate) {
        super(name, Shape.of(-1, -1, dModel), Shape.of(-1, -1, dModel));
        this.dropoutRate = dropoutRate;
        
        // 初始化各个子层
        this.layerNorm1 = new LayerNorm(name + "_ln1", dModel);
        this.attention = new MultiHeadAttention(name + "_attention", dModel, numHeads, true); // 使用掩码
        this.layerNorm2 = new LayerNorm(name + "_ln2", dModel);
        this.feedForward = new FeedForward(name + "_ffn", dModel, dFF);
        
        init();
    }
    
    /**
     * 使用默认参数的构造函数
     */
    public GPT2Block(String name, int dModel, int numHeads) {
        this(name, dModel, numHeads, dModel * 4, 0.1);
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 子层已经在构造函数中初始化了
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable x = inputs[0];
        
        // GPT-2 使用 Pre-LayerNorm 架构
        // 1. Layer Norm + Multi-Head Attention + Residual Connection
        Variable norm1Output = layerNorm1.layerForward(x);
        Variable attentionOutput = attention.layerForward(norm1Output, norm1Output, norm1Output);
        Variable residual1 = addResidualConnection(x, attentionOutput);
        
        // 2. Layer Norm + Feed Forward + Residual Connection
        Variable norm2Output = layerNorm2.layerForward(residual1);
        Variable ffnOutput = feedForward.layerForward(norm2Output);
        Variable residual2 = addResidualConnection(residual1, ffnOutput);
        
        return residual2;
    }
    
    /**
     * 添加残差连接
     * @param input 输入
     * @param output 子层输出
     * @return 残差连接结果
     */
    private Variable addResidualConnection(Variable input, Variable output) {
        return input.add(output);
    }
    
    /**
     * 应用Dropout（简化版本）
     * 在实际实现中，需要考虑训练/推理模式
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
        // GPT-2 Block的反向传播需要依次通过各个子层
        List<NdArray> result = new ArrayList<>();
        result.add(yGrad);
        return result;
    }
    
    @Override
    public int requireInputNum() {
        return 1;
    }
    
    /**
     * 获取第一个层归一化层
     */
    public LayerNorm getLayerNorm1() {
        return layerNorm1;
    }
    
    /**
     * 获取多头注意力层
     */
    public MultiHeadAttention getAttention() {
        return attention;
    }
    
    /**
     * 获取第二个层归一化层
     */
    public LayerNorm getLayerNorm2() {
        return layerNorm2;
    }
    
    /**
     * 获取前馈网络层
     */
    public FeedForward getFeedForward() {
        return feedForward;
    }
    
    /**
     * 获取dropout比率
     */
    public double getDropoutRate() {
        return dropoutRate;
    }
}