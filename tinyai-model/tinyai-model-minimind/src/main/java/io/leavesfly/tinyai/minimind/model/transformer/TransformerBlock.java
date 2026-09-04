package io.leavesfly.tinyai.minimind.model.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindConfig;
import io.leavesfly.tinyai.minimind.model.transformer.attention.KVCache;
import io.leavesfly.tinyai.minimind.model.transformer.attention.MultiHeadAttention;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.layer.norm.RMSNorm;

/**
 * MiniMind Transformer 层（对标 Python MiniMindBlock）
 * <p>
 * 架构: Pre-RMSNorm Transformer
 * <p>
 * 计算流程（对标 Python）:
 * 1. x = x + Attention(RMSNorm(x))
 * 2. x = x + SwiGLU_FFN(RMSNorm(x))
 *
 * @author TinyAI Team
 * @version 2.0
 */
public class TransformerBlock extends Module {

    /**
     * 注意力层前的 RMSNorm（对标 Python attention_norm）
     */
    private final RMSNorm attentionNorm;

    /**
     * 多头注意力层
     */
    private final MultiHeadAttention attention;

    /**
     * 前馈网络前的 RMSNorm（对标 Python ffn_norm）
     */
    private final RMSNorm ffnNorm;

    /**
     * SwiGLU 前馈网络（对标 Python FeedForward）
     */
    private final SwiGLUFeedForward feedForward;

    /**
     * 注意力输出 Dropout
     */
    private final Dropout attentionDropout;

    /**
     * FFN 输出 Dropout
     */
    private final Dropout ffnDropout;

    /**
     * 隐藏层维度
     */
    private final int hiddenSize;

    /**
     * 中间层维度
     */
    private final int intermediateSize;

    /**
     * 使用 MiniMindConfig 构造 Transformer 层（推荐）
     *
     * @param name   层名称
     * @param config 模型配置
     */
    public TransformerBlock(String name, MiniMindConfig config) {
        super(name);

        this.hiddenSize = config.getHiddenSize();
        this.intermediateSize = config.getIntermediateSize();

        // 1. RMSNorm（对标 Python self.attention_norm）
        this.attentionNorm = new RMSNorm("attention_norm", hiddenSize, config.getEpsilon());
        registerModule("attention_norm", attentionNorm);

        // 2. 多头注意力层
        this.attention = new MultiHeadAttention("attention", config);
        registerModule("attention", attention);

        // 3. RMSNorm（对标 Python self.ffn_norm）
        this.ffnNorm = new RMSNorm("ffn_norm", hiddenSize, config.getEpsilon());
        registerModule("ffn_norm", ffnNorm);

        // 4. SwiGLU 前馈网络（对标 Python FeedForward）
        this.feedForward = new SwiGLUFeedForward("ffn", hiddenSize, intermediateSize);
        registerModule("ffn", feedForward);

        // 5. Dropout 层
        this.attentionDropout = new Dropout("attention_dropout", config.getDropout());
        this.ffnDropout = new Dropout("ffn_dropout", config.getDropout());
        registerModule("attention_dropout", attentionDropout);
        registerModule("ffn_dropout", ffnDropout);

        // 初始化参数
        init();
    }

    /**
     * 兼容旧接口的构造函数
     */
    public TransformerBlock(String name, int hiddenSize, int numHeads,
                            int ffnHiddenSize, int maxSeqLen,
                            float dropoutRate, float epsilon,
                            String activationFunction, boolean preLayerNorm) {
        super(name);

        this.hiddenSize = hiddenSize;
        this.intermediateSize = ffnHiddenSize;

        // RMSNorm
        this.attentionNorm = new RMSNorm("attention_norm", hiddenSize, epsilon);
        registerModule("attention_norm", attentionNorm);

        // 多头注意力层（旧接口不支持 GQA，numKVHeads=numHeads）
        this.attention = new MultiHeadAttention("attention", hiddenSize, numHeads, maxSeqLen, dropoutRate);
        registerModule("attention", attention);

        // RMSNorm
        this.ffnNorm = new RMSNorm("ffn_norm", hiddenSize, epsilon);
        registerModule("ffn_norm", ffnNorm);

        // SwiGLU 前馈网络
        this.feedForward = new SwiGLUFeedForward("ffn", hiddenSize, ffnHiddenSize);
        registerModule("ffn", feedForward);

        // Dropout
        this.attentionDropout = new Dropout("attention_dropout", dropoutRate);
        this.ffnDropout = new Dropout("ffn_dropout", dropoutRate);
        registerModule("attention_dropout", attentionDropout);
        registerModule("ffn_dropout", ffnDropout);

        init();
    }

    /**
     * 前向传播（不使用 KV-Cache）
     */
    @Override
    public Variable forward(Variable... inputs) {
        return forwardWithCache(inputs[0], null, 0);
    }

    /**
     * 带 KV-Cache 的前向传播（对标 Python MiniMindBlock.forward）
     * <p>
     * Pre-RMSNorm 架构：
     * x = x + Attention(RMSNorm(x))
     * x = x + SwiGLU_FFN(RMSNorm(x))
     */
    public Variable forwardWithCache(Variable x, KVCache kvCache, int startPos) {
        // 1. 注意力子层: x = x + Attention(RMSNorm(x))
        Variable xNorm1 = attentionNorm.forward(x);
        Variable attnOut = attention.forwardWithCache(xNorm1, kvCache, startPos);
        attnOut = attentionDropout.forward(attnOut);
        x = x.add(attnOut);

        // 2. 前馈网络子层: x = x + SwiGLU(RMSNorm(x))
        Variable xNorm2 = ffnNorm.forward(x);
        Variable ffnOut = feedForward.forward(xNorm2);
        ffnOut = ffnDropout.forward(ffnOut);
        x = x.add(ffnOut);

        return x;
    }

    /**
     * 设置训练模式
     * <p>
     * 必须递归传播到全部子模块：forwardWithCache 中 attentionDropout 与 ffnDropout
     * 均会实际参与前向，若只切 attention，则 eval 模式下这两个 Dropout 仍然生效。
     */
    public void setTraining(boolean training) {
        // MultiHeadAttention 重写了 setTraining，需单独调用以同步其内部标记
        attention.setTraining(training);
        // 递归同步所有已注册子模块的 _training（含 attentionDropout / ffnDropout）
        train(training);
    }

    public MultiHeadAttention getAttention() {
        return attention;
    }

    public int getHiddenSize() {
        return hiddenSize;
    }

    public int getFfnHiddenSize() {
        return intermediateSize;
    }
}
