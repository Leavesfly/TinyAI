package io.leavesfly.tinyai.gpt1;

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
 * GPT-1 Transformer Block实现
 * <p>
 * GPT-1使用仅解码器的Transformer架构，每个block包含：
 * 1. 带掩码的多头自注意力机制
 * 2. 残差连接
 * 3. 层归一化
 * 4. 前馈网络
 * 5. 残差连接
 * 6. 层归一化
 * <p>
 * 注意：与GPT-2不同，GPT-1使用Post-LayerNorm结构（在子层之后应用层归一化）
 * 这与原始Transformer论文的架构一致
 *
 * @author 山泽
 * @version 1.0
 */
public class GPT1TransformerBlock extends Layer {

    /**
     * 多头自注意力层
     */
    private MultiHeadAttention attention;

    /**
     * 注意力层后的层归一化
     */
    private LayerNorm layerNorm1;

    /**
     * 前馈网络层
     */
    private FeedForward feedForward;

    /**
     * 前馈网络后的层归一化
     */
    private LayerNorm layerNorm2;

    /**
     * 配置信息
     */
    private GPT1Config config;

    /**
     * 构造GPT-1 Transformer Block
     *
     * @param name   块名称
     * @param config GPT-1配置
     */
    public GPT1TransformerBlock(String name, GPT1Config config) {
        super(name,
                Shape.of(-1, -1, config.getHiddenSize()),
                Shape.of(-1, -1, config.getHiddenSize()));

        this.config = config;
        init();
    }

    @Override
    public void init() {
        if (!alreadyInit) {
            // 1. 初始化多头自注意力层（带掩码）
            this.attention = new MultiHeadAttention(
                    name + "_attention",
                    config.getHiddenSize(),
                    config.getNumAttentionHeads(),
                    true  // 使用因果掩码（causal mask）
            );

            // 2. 初始化第一个层归一化（注意力后）
            this.layerNorm1 = new LayerNorm(
                    name + "_ln1",
                    config.getHiddenSize()
            );

            // 3. 初始化前馈网络
            this.feedForward = new FeedForward(
                    name + "_ffn",
                    config.getHiddenSize(),
                    config.getIntermediateSize()
            );

            // 4. 初始化第二个层归一化（前馈网络后）
            this.layerNorm2 = new LayerNorm(
                    name + "_ln2",
                    config.getHiddenSize()
            );

            alreadyInit = true;
        }
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable x = inputs[0];  // shape: (batchSize, sequenceLength, hiddenSize)

        // GPT-1 使用 Post-LayerNorm 架构（与原始Transformer一致）

        // 1. Multi-Head Self-Attention + Residual Connection + Layer Norm
        Variable attentionOutput = attention.layerForward(x, x, x);
        Variable residual1 = addResidualConnection(x, attentionOutput);
        Variable norm1Output = layerNorm1.layerForward(residual1);

        // 2. Feed Forward + Residual Connection + Layer Norm
        Variable ffnOutput = feedForward.layerForward(norm1Output);
        Variable residual2 = addResidualConnection(norm1Output, ffnOutput);
        Variable norm2Output = layerNorm2.layerForward(residual2);

        // 应用残差dropout（简化版本）
        norm2Output = applyResidualDropout(norm2Output);

        return norm2Output;
    }

    /**
     * 添加残差连接
     *
     * @param input  输入变量
     * @param output 子层输出变量
     * @return 残差连接后的结果
     */
    private Variable addResidualConnection(Variable input, Variable output) {
        return input.add(output);
    }

    /**
     * 应用残差连接的Dropout
     *
     * @param input 输入变量
     * @return 应用dropout后的变量
     */
    private Variable applyResidualDropout(Variable input) {
        // 在简化实现中，暂时返回原始输入
        // 实际训练时需要考虑训练/推理模式并实现真正的dropout
        if (config.getResidualDropoutProb() > 0.0) {
            // TODO: 实现真正的dropout
            return input;
        }
        return input;
    }

    /**
     * 应用注意力Dropout
     *
     * @param attentionOutput 注意力输出
     * @return 应用dropout后的输出
     */
    private Variable applyAttentionDropout(Variable attentionOutput) {
        // 在简化实现中，暂时返回原始输出
        if (config.getAttentionDropoutProb() > 0.0) {
            // TODO: 实现真正的dropout
            return attentionOutput;
        }
        return attentionOutput;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        return layerForward(new Variable(inputs[0])).getValue();
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 简化的反向传播实现
        // 实际实现需要通过各个子层的反向传播链
        List<NdArray> result = new ArrayList<>();
        result.add(yGrad);
        return result;
    }

    @Override
    public int requireInputNum() {
        return 1;
    }

    // ==================== Getter方法 ====================

    /**
     * 获取多头注意力层
     *
     * @return 多头注意力层
     */
    public MultiHeadAttention getAttention() {
        return attention;
    }

    /**
     * 获取第一个层归一化层（注意力后）
     *
     * @return 第一个层归一化层
     */
    public LayerNorm getLayerNorm1() {
        return layerNorm1;
    }

    /**
     * 获取前馈网络层
     *
     * @return 前馈网络层
     */
    public FeedForward getFeedForward() {
        return feedForward;
    }

    /**
     * 获取第二个层归一化层（前馈网络后）
     *
     * @return 第二个层归一化层
     */
    public LayerNorm getLayerNorm2() {
        return layerNorm2;
    }

    /**
     * 获取配置信息
     *
     * @return GPT-1配置
     */
    public GPT1Config getConfig() {
        return config;
    }

    /**
     * 获取隐藏层维度
     *
     * @return 隐藏层维度
     */
    public int getHiddenSize() {
        return config.getHiddenSize();
    }

    /**
     * 获取注意力头数
     *
     * @return 注意力头数
     */
    public int getNumAttentionHeads() {
        return config.getNumAttentionHeads();
    }

    /**
     * 获取前馈网络中间层维度
     *
     * @return 前馈网络中间层维度
     */
    public int getIntermediateSize() {
        return config.getIntermediateSize();
    }
}