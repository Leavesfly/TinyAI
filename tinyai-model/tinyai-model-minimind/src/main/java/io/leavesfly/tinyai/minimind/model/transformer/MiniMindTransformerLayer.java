package io.leavesfly.tinyai.minimind.model.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.attention.KVCache;
import io.leavesfly.tinyai.minimind.model.attention.MultiHeadAttention;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.activation.ReLU;
import io.leavesfly.tinyai.nnet.v2.layer.activation.SiLU;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.norm.LayerNorm;

/**
 * MiniMind Transformer 层
 * <p>
 * 架构: Pre-LN Transformer
 * <p>
 * 计算流程:
 * 1. x = input
 * 2. x_norm1 = LayerNorm(x)
 * 3. attn_out = MultiHeadAttention(x_norm1)
 * 4. x = x + attn_out (残差连接)
 * 5. x_norm2 = LayerNorm(x)
 * 6. ffn_out = FFN(x_norm2)  // SiLU 激活 + 两层线性变换
 * 7. x = x + ffn_out (残差连接)
 * 8. return x
 *
 * @author leavesfly
 * @version 1.0
 */
public class MiniMindTransformerLayer extends Module {

    /**
     * 注意力层前的 LayerNorm
     */
    private final LayerNorm attentionNorm;

    /**
     * 多头注意力层
     */
    private final MultiHeadAttention attention;

    /**
     * 前馈网络前的 LayerNorm
     */
    private final LayerNorm ffnNorm;

    /**
     * 前馈网络 - 第一层线性变换
     */
    private final Linear ffnLinear1;

    /**
     * 前馈网络 - 激活函数 (SiLU 或 ReLU)
     */
    private final Module ffnActivation;

    /**
     * 前馈网络 - 第二层线性变换
     */
    private final Linear ffnLinear2;

    /**
     * 注意力输出 Dropout
     */
    private final Dropout attentionDropout;

    /**
     * FFN 输出 Dropout
     */
    private final Dropout ffnDropout;

    /**
     * 是否使用 Pre-LayerNorm
     */
    private final boolean preLayerNorm;

    /**
     * 隐藏层维度
     */
    private final int hiddenSize;

    /**
     * 前馈网络隐藏层维度
     */
    private final int ffnHiddenSize;

    /**
     * 构造 Transformer 层
     *
     * @param name              层名称
     * @param hiddenSize        隐藏层维度
     * @param numHeads          注意力头数
     * @param ffnHiddenSize     前馈网络隐藏层维度
     * @param maxSeqLen         最大序列长度
     * @param dropoutRate       Dropout 比例
     * @param epsilon           LayerNorm 的 epsilon
     * @param activationFunction 激活函数类型 ("silu" 或 "relu")
     * @param preLayerNorm      是否使用 Pre-LayerNorm
     */
    public MiniMindTransformerLayer(String name, int hiddenSize, int numHeads, 
                                    int ffnHiddenSize, int maxSeqLen, 
                                    float dropoutRate, float epsilon,
                                    String activationFunction, boolean preLayerNorm) {
        super(name);

        this.hiddenSize = hiddenSize;
        this.ffnHiddenSize = ffnHiddenSize;
        this.preLayerNorm = preLayerNorm;

        // 1. 注意力层前的 LayerNorm
        this.attentionNorm = new LayerNorm("attention_norm", hiddenSize, epsilon);
        registerModule("attention_norm", attentionNorm);

        // 2. 多头注意力层
        this.attention = new MultiHeadAttention("attention", hiddenSize, numHeads, maxSeqLen, dropoutRate);
        registerModule("attention", attention);

        // 3. 前馈网络前的 LayerNorm
        this.ffnNorm = new LayerNorm("ffn_norm", hiddenSize, epsilon);
        registerModule("ffn_norm", ffnNorm);

        // 4. 前馈网络 - 两层 Linear + 激活函数
        this.ffnLinear1 = new Linear("ffn_linear1", hiddenSize, ffnHiddenSize, true);
        
        // 根据配置选择激活函数
        if ("relu".equalsIgnoreCase(activationFunction)) {
            this.ffnActivation = new ReLU("ffn_activation");
        } else {
            // 默认使用 SiLU
            this.ffnActivation = new SiLU("ffn_activation");
        }
        
        this.ffnLinear2 = new Linear("ffn_linear2", ffnHiddenSize, hiddenSize, true);

        // 5. Dropout 层
        this.attentionDropout = new Dropout("attention_dropout", dropoutRate);
        this.ffnDropout = new Dropout("ffn_dropout", dropoutRate);

        registerModule("ffn_linear1", ffnLinear1);
        registerModule("ffn_activation", ffnActivation);
        registerModule("ffn_linear2", ffnLinear2);
        registerModule("attention_dropout", attentionDropout);
        registerModule("ffn_dropout", ffnDropout);

        // 初始化参数
        init();
    }

    /**
     * 前向传播（不使用 KV-Cache）
     *
     * @param inputs 输入 Variable 数组
     * @return 输出 Variable
     */
    @Override
    public Variable forward(Variable... inputs) {
        return forwardWithCache(inputs[0], null, 0);
    }

    /**
     * 带 KV-Cache 的前向传播
     *
     * @param x        输入 Variable
     * @param kvCache  KV-Cache 对象（可为 null）
     * @param startPos 起始位置（用于 RoPE 和因果掩码）
     * @return 输出 Variable
     */
    public Variable forwardWithCache(Variable x, KVCache kvCache, int startPos) {
        Variable residual;

        // 1. 注意力子层
        if (preLayerNorm) {
            // Pre-LN: 先归一化，再残差
            Variable xNorm1 = attentionNorm.forward(x);
            Variable attnOut = attention.forwardWithCache(xNorm1, kvCache, startPos);
            attnOut = attentionDropout.forward(attnOut);
            x = x.add(attnOut);
        } else {
            // Post-LN: 先残差，再归一化
            residual = x;
            Variable attnOut = attention.forwardWithCache(x, kvCache, startPos);
            attnOut = attentionDropout.forward(attnOut);
            x = residual.add(attnOut);
            x = attentionNorm.forward(x);
        }

        // 2. 前馈网络子层
        if (preLayerNorm) {
            // Pre-LN: 先归一化，再残差
            Variable xNorm2 = ffnNorm.forward(x);
            Variable ffnOut = feedForward(xNorm2);
            ffnOut = ffnDropout.forward(ffnOut);
            x = x.add(ffnOut);
        } else {
            // Post-LN: 先残差，再归一化
            residual = x;
            Variable ffnOut = feedForward(x);
            ffnOut = ffnDropout.forward(ffnOut);
            x = residual.add(ffnOut);
            x = ffnNorm.forward(x);
        }

        return x;
    }

    /**
     * 前馈网络 (Feed-Forward Network)
     * <p>
     * 结构: Linear -> SiLU -> Linear
     * <p>
     * 数学表达:
     * FFN(x) = Linear2(SiLU(Linear1(x)))
     *
     * @param x 输入 Variable
     * @return 输出 Variable
     */
    private Variable feedForward(Variable x) {
        // 第一层线性变换: [batch, seq_len, hidden_size] -> [batch, seq_len, ffn_hidden_size]
        Variable h = ffnLinear1.forward(x);

        // SiLU 激活函数: SiLU(x) = x * sigmoid(x)
        h = ffnActivation.forward(h);

        // 第二层线性变换: [batch, seq_len, ffn_hidden_size] -> [batch, seq_len, hidden_size]
        Variable out = ffnLinear2.forward(h);

        return out;
    }

    /**
     * 设置训练模式
     *
     * @param training 是否为训练模式
     */
    public void setTraining(boolean training) {
        attention.setTraining(training);
        // Dropout 层会根据训练模式自动调整行为
    }

    /**
     * 获取多头注意力层
     */
    public MultiHeadAttention getAttention() {
        return attention;
    }

    /**
     * 获取隐藏层维度
     */
    public int getHiddenSize() {
        return hiddenSize;
    }

    /**
     * 获取前馈网络隐藏层维度
     */
    public int getFfnHiddenSize() {
        return ffnHiddenSize;
    }
}
