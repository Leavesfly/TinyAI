package io.leavesfly.tinyai.minimind.model.moe;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindConfig;
import io.leavesfly.tinyai.minimind.model.transformer.attention.KVCache;
import io.leavesfly.tinyai.minimind.model.transformer.attention.MultiHeadAttention;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.layer.norm.LayerNorm;

/**
 * MiniMind MoE Transformer 层
 * <p>
 * 集成了多头注意力和 MoE FFN 的 Transformer 层
 * 
 * 架构 (Pre-LayerNorm):
 * 1. x = x + MultiHeadAttention(LayerNorm(x))
 * 2. x = x + MoEBlock(LayerNorm(x))
 * 
 * 特点:
 * - 用 MoE 层替换标准 FFN
 * - 支持 KV-Cache 增量推理
 * - 计算负载均衡损失
 * - 专家使用统计
 * 
 * @author leavesfly
 * @version 1.0
 */
public class MoETransformerBlock extends Module {

    /**
     * 第一个归一化层（用于注意力）
     */
    private final LayerNorm norm1;

    /**
     * 多头自注意力层
     */
    private final MultiHeadAttention attention;

    /**
     * 第二个归一化层（用于 MoE）
     */
    private final LayerNorm norm2;

    /**
     * MoE 层
     */
    private final MoEBlock moeLayer;

    /**
     * 负载均衡损失计算器
     */
    private final LoadBalanceLoss loadBalanceLoss;

    /**
     * 模型配置
     */
    private final MiniMindConfig config;

    /**
     * 是否处于训练模式
     */
    private boolean training = true;

    /**
     * 构造 MoETransformerBlock
     *
     * @param name   层名称
     * @param config 模型配置
     */
    public MoETransformerBlock(String name, MiniMindConfig config) {
        super(name);
        this.config = config;

        int hiddenSize = config.getHiddenSize();
        int numHeads = config.getNumHeads();
        int maxSeqLen = config.getMaxSeqLen();
        float epsilon = config.getEpsilon();

        // 1. 第一个 LayerNorm
        this.norm1 = new LayerNorm(name + "_norm1", hiddenSize, epsilon);
        registerModule("norm1", norm1);

        // 2. 多头自注意力
        this.attention = new MultiHeadAttention(
            name + "_attn",
            hiddenSize,
            numHeads,
            maxSeqLen,
            0.0f  // dropout
        );
        registerModule("attention", attention);

        // 3. 第二个 LayerNorm
        this.norm2 = new LayerNorm(name + "_norm2", hiddenSize, epsilon);
        registerModule("norm2", norm2);

        // 4. MoE 层
        this.moeLayer = new MoEBlock(
            config.getHiddenSize(),
            config.getFfnHiddenSize(),
            config.getHiddenSize(),
            config.getNumExperts(),
            config.getNumExpertsPerToken(),
            config.getMoeNoiseFactor()
        );
        registerModule("moe", moeLayer);

        // 5. 负载均衡损失
        this.loadBalanceLoss = new LoadBalanceLoss(
            config.getMoeImportanceCoef(),
            config.getMoeLoadCoef()
        );
    }

    /**
     * 前向传播（不使用 KV-Cache）
     */
    @Override
    public Variable forward(Variable... inputs) {
        return forwardWithCache(inputs[0], null, 0).getOutput();
    }

    /**
     * 带 KV-Cache 的前向传播
     *
     * @param input    输入,形状 [batch_size, seq_len, hidden_size]
     * @param kvCache  KV-Cache（可为 null）
     * @param startPos 起始位置
     * @return 层输出结果
     */
    public LayerOutput forwardWithCache(Variable input, KVCache kvCache, int startPos) {
        // 1. 注意力部分: x = x + Attention(norm1(x))
        Variable norm1Output = norm1.forward(input);
        Variable attnOutput = (kvCache != null) 
            ? attention.forwardWithCache(norm1Output, kvCache, startPos)
            : attention.forward(norm1Output);
        Variable afterAttn = input.add(attnOutput);

        // 2. MoE 部分: x = x + MoE(norm2(x))
        Variable norm2Output = norm2.forward(afterAttn);
        
        // Router 只调用一次，结果同时用于专家路由和负载均衡损失计算
        ExpertRouter router = moeLayer.getRouter();
        ExpertRouter.RouterOutput routerOutput = router.forwardRouter(norm2Output);
        
        // 使用已计算好的 routerOutput 进行 MoE 前向传播（避免重复路由）
        Variable moeOutput = moeLayer.forwardWithRouterOutput(norm2Output, routerOutput);
        Variable output = afterAttn.add(moeOutput);

        // 3. 计算负载均衡损失（复用同一个 routerOutput）
        float balanceLoss = 0.0f;
        if (training && config.isMoeEnableLoadBalance()) {
            MoEBlock.LoadBalanceStats stats = moeLayer.getLoadBalanceStats(routerOutput);
            balanceLoss = loadBalanceLoss.computeLoss(stats, config.getNumExperts());
        }

        return new LayerOutput(output, balanceLoss);
    }

    /**
     * 设置训练模式
     */
    public void setTraining(boolean training) {
        this.training = training;
    }

    /**
     * 是否为训练模式
     */
    public boolean isTraining() {
        return training;
    }

    /**
     * 获取 MoE 层
     */
    public MoEBlock getMoELayer() {
        return moeLayer;
    }

    /**
     * 获取专家使用统计
     */
    public MoEBlock.ExpertUsageStats getUsageStats() {
        return moeLayer.getUsageStats();
    }

    /**
     * 重置统计信息
     */
    public void resetStats() {
        moeLayer.resetStats();
    }

    @Override
    public String toString() {
        return String.format("MoETransformerBlock(hidden=%d, experts=%d, topK=%d)",
            config.getHiddenSize(), config.getNumExperts(), config.getNumExpertsPerToken());
    }

    /**
     * 层输出结果（包含负载均衡损失）
     */
    public static class LayerOutput {
        private final Variable output;
        private final float balanceLoss;

        public LayerOutput(Variable output, float balanceLoss) {
            this.output = output;
            this.balanceLoss = balanceLoss;
        }

        public Variable getOutput() {
            return output;
        }

        public float getBalanceLoss() {
            return balanceLoss;
        }

        @Override
        public String toString() {
            return String.format("LayerOutput(shape=%s, balance_loss=%.6f)",
                output.getShape(), balanceLoss);
        }
    }
}
