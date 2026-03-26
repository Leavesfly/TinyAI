package io.leavesfly.tinyai.nnet.v2.layer.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.activation.ReLU;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.norm.LayerNorm;

/**
 * V2版本的TransformerDecoderLayer
 * <p>
 * Transformer解码器层，包含：
 * 1. 掩码多头自注意力子层（Masked Self-Attention）
 * 2. 编码器-解码器交叉注意力子层（Cross-Attention）
 * 3. 前馈神经网络子层
 * 4. 三个层归一化层
 * 5. 残差连接
 * <p>
 * 结构（Pre-LN）：
 * x -> LayerNorm -> Masked Self-Attention -> Add(x) ->
 * -> LayerNorm -> Cross-Attention(memory) -> Add ->
 * -> LayerNorm -> FFN -> Add -> output
 *
 * @author leavesfly
 * @version 2.0
 */
public class TransformerDecoderLayer extends Module {

    private final int dModel;
    private final int numHeads;
    private final int dFF;
    private final float dropout;
    private final boolean preLayerNorm;

    // 子模块
    private MultiHeadAttention selfAttention;      // 掩码自注意力
    private LayerNorm norm1;
    private MultiHeadAttention crossAttention;     // 编码器-解码器注意力
    private LayerNorm norm2;
    private Linear ffn1;
    private ReLU activation;
    private Linear ffn2;
    private LayerNorm norm3;

    /**
     * 构造函数
     *
     * @param name         层名称
     * @param dModel       模型维度
     * @param numHeads     注意力头数
     * @param dFF          前馈网络隐藏层维度
     * @param dropout      dropout比率
     * @param preLayerNorm 是否使用Pre-LayerNorm
     */
    public TransformerDecoderLayer(String name, int dModel, int numHeads, int dFF,
                                   float dropout, boolean preLayerNorm) {
        super(name);
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dFF = dFF;
        this.dropout = dropout;
        this.preLayerNorm = preLayerNorm;

        // 初始化子模块
        selfAttention = new MultiHeadAttention("self_attn", dModel, numHeads, dropout);
        norm1 = new LayerNorm("norm1", dModel);

        crossAttention = new MultiHeadAttention("cross_attn", dModel, numHeads, dropout);
        norm2 = new LayerNorm("norm2", dModel);

        // 前馈网络
        ffn1 = new Linear("ffn1", dModel, dFF, true);
        activation = new ReLU();
        ffn2 = new Linear("ffn2", dFF, dModel, true);
        norm3 = new LayerNorm("norm3", dModel);

        // 注册子模块
        registerModule("self_attn", selfAttention);
        registerModule("norm1", norm1);
        registerModule("cross_attn", crossAttention);
        registerModule("norm2", norm2);
        registerModule("ffn1", ffn1);
        registerModule("activation", activation);
        registerModule("ffn2", ffn2);
        registerModule("norm3", norm3);

        init();
    }

    /**
     * 构造函数（使用默认参数）
     *
     * @param name     层名称
     * @param dModel   模型维度
     * @param numHeads 注意力头数
     */
    public TransformerDecoderLayer(String name, int dModel, int numHeads) {
        this(name, dModel, numHeads, dModel * 4, 0.1f, true);
    }

    /**
     * 构造函数
     *
     * @param name     层名称
     * @param dModel   模型维度
     * @param numHeads 注意力头数
     * @param dFF      前馈网络隐藏层维度
     */
    public TransformerDecoderLayer(String name, int dModel, int numHeads, int dFF) {
        this(name, dModel, numHeads, dFF, 0.1f, true);
    }

    @Override
    public void resetParameters() {
        // 子模块的参数由其自己初始化
    }

    /**
     * 前向传播
     * <p>
     * 输入参数：
     * - inputs[0]: x - 解码器输入
     * - inputs[1]: memory - 编码器输出（可选）
     * - inputs[2]: tgtMask - 目标序列掩码（可选，因果掩码）
     * - inputs[3]: memoryMask - 编码器输出掩码（可选）
     * - inputs[4]: tgtKeyPaddingMask - 目标序列padding掩码（可选）
     * - inputs[5]: memoryKeyPaddingMask - 编码器输出padding掩码（可选）
     */
    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];
        Variable memory = inputs.length > 1 ? inputs[1] : null;
        Variable tgtMask = inputs.length > 2 ? inputs[2] : null;
        Variable memoryMask = inputs.length > 3 ? inputs[3] : null;
        Variable tgtKeyPaddingMask = inputs.length > 4 ? inputs[4] : null;
        Variable memoryKeyPaddingMask = inputs.length > 5 ? inputs[5] : null;

        if (preLayerNorm) {
            return forwardPreNorm(x, memory, tgtMask, memoryMask,
                    tgtKeyPaddingMask, memoryKeyPaddingMask);
        } else {
            return forwardPostNorm(x, memory, tgtMask, memoryMask,
                    tgtKeyPaddingMask, memoryKeyPaddingMask);
        }
    }

    /**
     * Pre-LayerNorm前向传播
     *
     * @param x                     解码器输入
     * @param memory                编码器输出
     * @param tgtMask               目标序列掩码（因果掩码）
     * @param memoryMask            编码器输出掩码
     * @param tgtKeyPaddingMask     目标序列padding掩码
     * @param memoryKeyPaddingMask  编码器输出padding掩码
     * @return 输出
     */
    private Variable forwardPreNorm(Variable x, Variable memory,
                                     Variable tgtMask, Variable memoryMask,
                                     Variable tgtKeyPaddingMask, Variable memoryKeyPaddingMask) {
        // 1. 掩码自注意力子层（Pre-LN）
        Variable normX = norm1.forward(x);
        Variable selfAttnOut = selfAttention.forward(normX, normX, normX,
                tgtMask, tgtKeyPaddingMask);
        Variable residual1 = x.add(selfAttnOut);

        // 2. 编码器-解码器交叉注意力子层（Pre-LN）
        if (memory != null) {
            Variable normResidual1 = norm2.forward(residual1);
            Variable crossAttnOut = crossAttention.forward(normResidual1, memory, memory,
                    memoryMask, memoryKeyPaddingMask);
            residual1 = residual1.add(crossAttnOut);
        }

        // 3. 前馈网络子层（Pre-LN）
        Variable normResidual2 = norm3.forward(residual1);
        Variable ffnOut = forwardFFN(normResidual2);
        return residual1.add(ffnOut);
    }

    /**
     * Post-LayerNorm前向传播
     *
     * @param x                     解码器输入
     * @param memory                编码器输出
     * @param tgtMask               目标序列掩码（因果掩码）
     * @param memoryMask            编码器输出掩码
     * @param tgtKeyPaddingMask     目标序列padding掩码
     * @param memoryKeyPaddingMask  编码器输出padding掩码
     * @return 输出
     */
    private Variable forwardPostNorm(Variable x, Variable memory,
                                      Variable tgtMask, Variable memoryMask,
                                      Variable tgtKeyPaddingMask, Variable memoryKeyPaddingMask) {
        // 1. 掩码自注意力子层（Post-LN）
        Variable selfAttnOut = selfAttention.forward(x, x, x,
                tgtMask, tgtKeyPaddingMask);
        Variable residual1 = x.add(selfAttnOut);
        Variable norm1Out = norm1.forward(residual1);

        // 2. 编码器-解码器交叉注意力子层（Post-LN）
        Variable current = norm1Out;
        if (memory != null) {
            Variable crossAttnOut = crossAttention.forward(current, memory, memory,
                    memoryMask, memoryKeyPaddingMask);
            Variable residual2 = current.add(crossAttnOut);
            current = norm2.forward(residual2);
        }

        // 3. 前馈网络子层（Post-LN）
        Variable ffnOut = forwardFFN(current);
        Variable residual3 = current.add(ffnOut);
        return norm3.forward(residual3);
    }

    /**
     * 前馈网络前向传播
     *
     * @param x 输入
     * @return 输出
     */
    private Variable forwardFFN(Variable x) {
        Variable h = ffn1.forward(x);
        h = activation.forward(h);
        Variable output = ffn2.forward(h);
        return output;
    }

    public int getDModel() {
        return dModel;
    }

    public int getNumHeads() {
        return numHeads;
    }

    public int getDFF() {
        return dFF;
    }

    public float getDropout() {
        return dropout;
    }

    public boolean isPreLayerNorm() {
        return preLayerNorm;
    }

    @Override
    public String toString() {
        return "TransformerDecoderBlock{" +
                "name='" + name + '\'' +
                ", dModel=" + dModel +
                ", numHeads=" + numHeads +
                ", dFF=" + dFF +
                ", dropout=" + dropout +
                ", preLayerNorm=" + preLayerNorm +
                '}';
    }
}
