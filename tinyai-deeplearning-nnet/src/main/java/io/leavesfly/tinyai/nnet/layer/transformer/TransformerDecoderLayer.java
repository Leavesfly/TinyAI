package io.leavesfly.tinyai.nnet.layer.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;

import java.util.ArrayList;
import java.util.List;

/**
 * Transformer解码器层实现
 * <p>
 * 单个Transformer解码器层包含：
 * 1. 带掩码的多头自注意力机制
 * 2. 残差连接和层归一化
 * 3. 多头交叉注意力机制（编码器-解码器注意力）
 * 4. 残差连接和层归一化
 * 5. 前馈神经网络
 * 6. 残差连接和层归一化
 * <p>
 * DecoderLayer(x, memory) = LayerNorm(x + FFN(LayerNorm(x + CrossAttention(x, memory, memory))))
 * 其中 x 先经过 MaskedSelfAttention
 */
public class TransformerDecoderLayer extends Layer {

    private MultiHeadAttention maskedSelfAttention;  // 带掩码的自注意力
    private LayerNorm layerNorm1;
    private MultiHeadAttention crossAttention;       // 交叉注意力（编码器-解码器）
    private LayerNorm layerNorm2;
    private FeedForward feedForward;
    private LayerNorm layerNorm3;
    private double dropoutRate;

    /**
     * 构造Transformer解码器层
     *
     * @param name        层名称
     * @param dModel      模型维度
     * @param numHeads    注意力头数
     * @param dFF         前馈网络隐藏维度
     * @param dropoutRate dropout比率
     */
    public TransformerDecoderLayer(String name, int dModel, int numHeads, int dFF, double dropoutRate) {
        super(name, Shape.of(-1, -1, dModel), Shape.of(-1, -1, dModel));
        this.dropoutRate = dropoutRate;

        // 初始化各个子层
        this.maskedSelfAttention = new MultiHeadAttention(name + "_masked_self_attention", dModel, numHeads, true);
        this.layerNorm1 = new LayerNorm(name + "_norm1", dModel);

        this.crossAttention = new MultiHeadAttention(name + "_cross_attention", dModel, numHeads, false);
        this.layerNorm2 = new LayerNorm(name + "_norm2", dModel);

        this.feedForward = new FeedForward(name + "_ffn", dModel, dFF);
        this.layerNorm3 = new LayerNorm(name + "_norm3", dModel);

        init();
    }

    /**
     * 使用默认参数的构造函数
     */
    public TransformerDecoderLayer(String name, int dModel, int numHeads) {
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
        Variable x = inputs[0];           // 解码器输入
        Variable memory = inputs[1];      // 编码器输出（用于交叉注意力）

        // 1. 带掩码的多头自注意力 + 残差连接 + 层归一化
        Variable selfAttentionOutput = maskedSelfAttention.layerForward(x, x, x);
        Variable residual1 = addResidualConnection(x, selfAttentionOutput);
        Variable norm1Output = layerNorm1.layerForward(residual1);

        // 2. 多头交叉注意力 + 残差连接 + 层归一化
        // Query来自解码器，Key和Value来自编码器
        Variable crossAttentionOutput = crossAttention.layerForward(norm1Output, memory, memory);
        Variable residual2 = addResidualConnection(norm1Output, crossAttentionOutput);
        Variable norm2Output = layerNorm2.layerForward(residual2);

        // 3. 前馈网络 + 残差连接 + 层归一化
        Variable ffnOutput = feedForward.layerForward(norm2Output);
        Variable residual3 = addResidualConnection(norm2Output, ffnOutput);
        Variable norm3Output = layerNorm3.layerForward(residual3);

        return norm3Output;
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
        Variable[] variables = new Variable[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            variables[i] = new Variable(inputs[i]);
        }
        return layerForward(variables).getValue();
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 解码器层的反向传播需要依次通过各个子层
        List<NdArray> result = new ArrayList<>();
        result.add(yGrad);
        result.add(yGrad); // 对编码器输出的梯度
        return result;
    }

    @Override
    public int requireInputNum() {
        return 2; // 解码器输入和编码器输出
    }

    /**
     * 获取带掩码的自注意力层
     */
    public MultiHeadAttention getMaskedSelfAttention() {
        return maskedSelfAttention;
    }

    /**
     * 获取第一个层归一化层
     */
    public LayerNorm getLayerNorm1() {
        return layerNorm1;
    }

    /**
     * 获取交叉注意力层
     */
    public MultiHeadAttention getCrossAttention() {
        return crossAttention;
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
     * 获取第三个层归一化层
     */
    public LayerNorm getLayerNorm3() {
        return layerNorm3;
    }
}