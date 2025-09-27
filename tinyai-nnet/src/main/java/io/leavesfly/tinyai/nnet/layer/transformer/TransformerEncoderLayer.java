package io.leavesfly.tinyai.nnet.layer.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;

import java.util.ArrayList;
import java.util.List;

/**
 * Transformer编码器层实现
 * 
 * 单个Transformer编码器层包含：
 * 1. 多头自注意力机制
 * 2. 残差连接和层归一化
 * 3. 前馈神经网络
 * 4. 残差连接和层归一化
 * 
 * EncoderLayer(x) = LayerNorm(x + FFN(LayerNorm(x + MultiHead(x))))
 */
public class TransformerEncoderLayer extends Layer {
    
    private MultiHeadAttention selfAttention;
    private LayerNorm layerNorm1;
    private FeedForward feedForward;
    private LayerNorm layerNorm2;
    private double dropoutRate;
    
    /**
     * 构造Transformer编码器层
     * 
     * @param name 层名称
     * @param dModel 模型维度
     * @param numHeads 注意力头数
     * @param dFF 前馈网络隐藏维度
     * @param dropoutRate dropout比率
     */
    public TransformerEncoderLayer(String name, int dModel, int numHeads, int dFF, double dropoutRate) {
        super(name, Shape.of(-1, -1, dModel), Shape.of(-1, -1, dModel));
        this.dropoutRate = dropoutRate;
        
        // 初始化各个子层
        this.selfAttention = new MultiHeadAttention(name + "_self_attention", dModel, numHeads, false);
        this.layerNorm1 = new LayerNorm(name + "_norm1", dModel);
        this.feedForward = new FeedForward(name + "_ffn", dModel, dFF);
        this.layerNorm2 = new LayerNorm(name + "_norm2", dModel);
        
        init();
    }
    
    /**
     * 使用默认参数的构造函数
     */
    public TransformerEncoderLayer(String name, int dModel, int numHeads) {
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
        
        // 1. 多头自注意力 + 残差连接 + 层归一化
        Variable attentionOutput = selfAttention.layerForward(x, x, x);
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
     */
    private Variable addResidualConnection(Variable input, Variable output) {
        // 残差连接：input + output
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
        // 编码器层的反向传播需要依次通过各个子层
        List<NdArray> result = new ArrayList<>();
        result.add(yGrad);
        return result;
    }
    
    @Override
    public int requireInputNum() {
        return 1;
    }
    
    /**
     * 获取自注意力层
     */
    public MultiHeadAttention getSelfAttention() {
        return selfAttention;
    }
    
    /**
     * 获取第一个层归一化层
     */
    public LayerNorm getLayerNorm1() {
        return layerNorm1;
    }
    
    /**
     * 获取前馈网络层
     */
    public FeedForward getFeedForward() {
        return feedForward;
    }
    
    /**
     * 获取第二个层归一化层
     */
    public LayerNorm getLayerNorm2() {
        return layerNorm2;
    }
}