package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.deepseek.base.DeepSeekBaseConfig;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.layer.norm.RMSNorm;

import java.util.HashMap;
import java.util.Map;

/**
 * DeepSeek-V3 Transformer块（Pre-RMSNorm + RoPE + MoE 架构）
 *
 * 对标 DeepSeek-V3 论文架构：
 * 1. 自注意力层（内置 RoPE 旋转位置编码）
 * 2. 混合专家层 (MoE) 替代传统 FFN
 *
 * 架构特点：
 * - Pre-RMSNorm 提升训练稳定性
 * - RoPE 旋转位置编码（在注意力层内部对 Q/K 应用）
 * - MoE 层实现参数高效（仅激活 Top-K 专家）
 *
 * @author leavesfly
 * @version 2.0
 */
public class DeepSeekV3TransformerBlock extends Module {

    private final DeepSeekBaseConfig config;

    // 注意力子层（内置 RoPE）
    private final DeepSeekV3Attention attention;
    private final RMSNorm layerNorm1;
    private final Dropout residDropout;

    // MoE 子层（替代传统 FFN）
    private final DeepSeekV3MoEBlock moeLayer;
    private final RMSNorm layerNorm2;

    // 因果掩码缓存（避免每次 forward 都重新生成）
    private final Map<Integer, Variable> causalMaskCache;

    /**
     * 构造函数
     *
     * @param name   模块名称
     * @param config V3 配置对象
     */
    public DeepSeekV3TransformerBlock(String name, DeepSeekBaseConfig config) {
        super(name);
        this.config = config;

        int dModel = config.getNEmbd();
        int numHeads = config.getNHead();
        int maxSeqLen = config.getNPositions();
        float dropoutRate = (float) config.getResidPdrop();
        float attnDropoutRate = (float) config.getAttnPdrop();

        // 初始化注意力子层（内置 RoPE，theta=10000）
        this.attention = new DeepSeekV3Attention(
                "attn", dModel, numHeads, maxSeqLen, attnDropoutRate, 10000.0f);
        this.layerNorm1 = new RMSNorm("ln1", dModel, (float) config.getLayerNormEpsilon());
        this.residDropout = new Dropout("resid_dropout", dropoutRate);

        // 初始化 MoE 子层
        this.moeLayer = new DeepSeekV3MoEBlock(name + "_moe", config);
        this.layerNorm2 = new RMSNorm("ln2", dModel, (float) config.getLayerNormEpsilon());

        // 注册所有子模块
        registerModule("attn", attention);
        registerModule("ln1", layerNorm1);
        registerModule("resid_dropout", residDropout);
        registerModule("moe", moeLayer);
        registerModule("ln2", layerNorm2);

        // 初始化因果掩码缓存
        this.causalMaskCache = new HashMap<>();
    }

    /**
     * 前向传播
     *
     * Pre-RMSNorm + RoPE + MoE 架构流程：
     * 1. 注意力分支: x -> RMSNorm -> Attention(含RoPE) -> Dropout -> Add(x)
     * 2. MoE 分支:   x -> RMSNorm -> MoE -> Add(x)
     *
     * @param inputs inputs[0] 为输入张量 [batch_size, seq_len, d_model]
     * @return 输出张量 [batch_size, seq_len, d_model]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("输入不能为空");
        }

        Variable x = inputs[0];
        int seqLen = x.getValue().getShape().getDimension(1);

        // 生成或获取缓存的因果掩码
        Variable causalMask = getCausalMask(seqLen);

        // ===== 注意力子层 (Pre-RMSNorm + RoPE) =====
        Variable normalized1 = layerNorm1.forward(x);
        Variable attnOutput = attention.forward(normalized1, causalMask);
        attnOutput = residDropout.forward(attnOutput);
        Variable residual1 = x.add(attnOutput);

        // ===== MoE 子层 (Pre-RMSNorm) =====
        Variable normalized2 = layerNorm2.forward(residual1);
        Variable moeOutput = moeLayer.forward(normalized2);
        Variable output = residual1.add(moeOutput);

        return output;
    }

    /**
     * 获取或生成因果掩码（带缓存）
     *
     * @param seqLen 序列长度
     * @return 因果掩码 [1, 1, seqLen, seqLen]
     */
    private Variable getCausalMask(int seqLen) {
        // 检查缓存
        if (causalMaskCache.containsKey(seqLen)) {
            return causalMaskCache.get(seqLen);
        }

        // 生成新的因果掩码
        Variable causalMask = DeepSeekV3Attention.generateCausalMask(seqLen);
        causalMaskCache.put(seqLen, causalMask);
        return causalMask;
    }

    /**
     * 带详细输出的前向传播（包含 MoE 损失）
     *
     * @param input 输入张量 [batch_size, seq_len, d_model]
     * @return 详细输出结果
     */
    public DetailedForwardResult forwardWithDetails(Variable input) {
        int seqLen = input.getValue().getShape().getDimension(1);

        // 生成或获取缓存的因果掩码
        Variable causalMask = getCausalMask(seqLen);

        // ===== 注意力子层 =====
        Variable normalized1 = layerNorm1.forward(input);
        Variable attnOutput = attention.forward(normalized1, causalMask);
        attnOutput = residDropout.forward(attnOutput);
        Variable residual1 = input.add(attnOutput);

        // ===== MoE 子层（获取详细结果） =====
        Variable normalized2 = layerNorm2.forward(residual1);
        DeepSeekV3MoEBlock.MoEOutput moeResult = moeLayer.computeMoE(normalized2);
        Variable output = residual1.add(moeResult.output);

        return new DetailedForwardResult(output, moeResult);
    }

    /**
     * 带任务类型的详细前向传播（包含 MoE 损失）
     *
     * @param input    输入张量 [batch_size, seq_len, d_model]
     * @param taskType 任务类型（当前保留接口，未来可用于任务感知路由）
     * @return 详细输出结果
     */
    public DetailedForwardResult forwardWithDetails(Variable input, io.leavesfly.tinyai.deepseek.base.TaskType taskType) {
        return forwardWithDetails(input);
    }

    /**
     * 获取配置对象
     */
    public DeepSeekBaseConfig getConfig() {
        return config;
    }

    /**
     * 获取 MoE 层
     */
    public DeepSeekV3MoEBlock getMoeLayer() {
        return moeLayer;
    }

    /**
     * 详细前向传播结果类
     */
    public static class DetailedForwardResult {
        /** Transformer 块的输出 */
        public final Variable output;
        /** MoE 层的详细结果 */
        public final DeepSeekV3MoEBlock.MoEOutput moeOutput;

        public DetailedForwardResult(Variable output, DeepSeekV3MoEBlock.MoEOutput moeOutput) {
            this.output = output;
            this.moeOutput = moeOutput;
        }

        /**
         * 获取负载均衡损失
         */
        public double getLoadBalanceLoss() {
            return moeOutput.loadBalanceLoss;
        }

        @Override
        public String toString() {
            return String.format(
                    "DetailedForwardResult{outputShape=%s, %s}",
                    output.getValue().getShape(),
                    moeOutput);
        }
    }
}