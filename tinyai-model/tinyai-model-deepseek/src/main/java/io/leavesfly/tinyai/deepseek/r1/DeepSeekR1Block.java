package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.deepseek.base.TaskType;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3TransformerBlock;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.norm.RMSNorm;

import java.util.ArrayList;
import java.util.List;

/**
 * DeepSeek-R1主体块（DeepSeekR1Block）
 *
 * 重要架构说明（根据论文 arXiv:2501.12948）：
 * DeepSeek-R1 与 DeepSeek-V3 使用完全相同的 MoE 基础架构（671B 参数），
 * 区别仅在于训练方式（R1 使用纯 RL 训练）。
 *
 * 架构组成（与 V3 完全一致）：
 * 1. Token嵌入层 - 将token ID转换为向量表示（RoPE 在注意力层提供位置信息）
 * 2. Transformer层堆叠（Pre-RMSNorm + RoPE + MoE）- 复用 V3 的 TransformerBlock
 * 3. 输出投影层 - 生成最终logits
 *
 * 与 V3 Block 的区别：
 * - R1 不包含 MTP（Multi-Token Prediction）头，MTP 是 V3 的训练辅助机制
 * - R1 的推理能力通过 RL 训练自然涌现，无需显式推理模块
 *
 * 数据流：
 * token_ids → embedding → MoE_transformer_layers → final_ln → output_projection
 *
 * @author leavesfly
 * @version 3.0
 */
public class DeepSeekR1Block extends Module {

    private final DeepSeekR1Config config;

    // 核心组件（与 V3 共享相同的 MoE 架构）
    private DeepSeekR1TokenEmbedding tokenEmbedding;
    private List<DeepSeekV3TransformerBlock> transformerBlocks;
    private RMSNorm finalLayerNorm;
    private Linear outputProjection;

    /**
     * 构造函数
     *
     * @param name   模块名称
     * @param config R1配置对象（继承自 DeepSeekBaseConfig，可直接传给 V3 的 TransformerBlock）
     */
    public DeepSeekR1Block(String name, DeepSeekR1Config config) {
        super(name);
        this.config = config;
        initializeComponents();
    }

    /**
     * 初始化所有组件
     *
     * R1Config 继承自 DeepSeekBaseConfig，而 DeepSeekV3TransformerBlock 已重构为
     * 接受 DeepSeekBaseConfig，因此可以直接传入 R1Config，无需 config 转换。
     */
    private void initializeComponents() {
        // 1. 初始化Token嵌入层
        tokenEmbedding = new DeepSeekR1TokenEmbedding(name + "_token_embedding", config);
        registerModule("token_embedding", tokenEmbedding);

        // 2. 初始化Transformer层堆叠（直接传入 R1Config，无需转换为 V3Config）
        transformerBlocks = new ArrayList<>();
        for (int i = 0; i < config.getNLayer(); i++) {
            DeepSeekV3TransformerBlock block = new DeepSeekV3TransformerBlock(
                    name + "_transformer_" + i, config);
            transformerBlocks.add(block);
            registerModule("transformer_" + i, block);
        }

        // 3. 初始化最终RMSNorm
        finalLayerNorm = new RMSNorm(
                name + "_final_ln",
                config.getNEmbd(),
                (float) config.getLayerNormEpsilon()
        );
        registerModule("final_ln", finalLayerNorm);

        // 4. 初始化输出投影层
        outputProjection = new Linear(
                name + "_output_proj",
                config.getNEmbd(),
                config.getVocabSize(),
                false
        );
        registerModule("output_proj", outputProjection);
    }

    /**
     * 前向传播
     *
     * @param inputs 输入变量，inputs[0]为token ID序列 [batch_size, seq_len]
     * @return logits输出 [batch_size, seq_len, vocab_size]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("输入不能为空");
        }

        Variable tokenIds = inputs[0];
        validateInput(tokenIds);

        // 1. Token嵌入
        Variable x = tokenEmbedding.forward(tokenIds);

        // 2. Transformer层堆叠
        for (DeepSeekV3TransformerBlock block : transformerBlocks) {
            x = block.forward(x);
        }

        // 3. 最终LayerNorm
        Variable normalized = finalLayerNorm.forward(x);

        // 4. 输出投影
        return outputProjection.forward(normalized);
    }

    /**
     * 带任务类型的前向传播
     *
     * @param tokenIds token ID序列 [batch_size, seq_len]
     * @param taskType 任务类型（用于任务感知路由）
     * @return logits输出 [batch_size, seq_len, vocab_size]
     */
    public Variable forward(Variable tokenIds, TaskType taskType) {
        validateInput(tokenIds);

        Variable x = tokenEmbedding.forward(tokenIds);

        for (DeepSeekV3TransformerBlock block : transformerBlocks) {
            if (taskType != null) {
                DeepSeekV3TransformerBlock.DetailedForwardResult detailedResult =
                        block.forwardWithDetails(x, taskType);
                x = detailedResult.output;
            } else {
                x = block.forward(x);
            }
        }

        Variable normalized = finalLayerNorm.forward(x);
        return outputProjection.forward(normalized);
    }

    /**
     * 带详细输出的前向传播（包含 MoE 损失）
     *
     * @param tokenIds token ID序列 [batch_size, seq_len]
     * @param taskType 任务类型（可选）
     * @return 详细输出结果
     */
    public DetailedForwardResult forwardWithDetails(Variable tokenIds, TaskType taskType) {
        validateInput(tokenIds);

        Variable x = tokenEmbedding.forward(tokenIds);

        double totalMoELoss = 0.0;
        for (DeepSeekV3TransformerBlock block : transformerBlocks) {
            DeepSeekV3TransformerBlock.DetailedForwardResult blockResult =
                    block.forwardWithDetails(x, taskType);
            x = blockResult.output;
            totalMoELoss += blockResult.getLoadBalanceLoss();
        }
        double avgMoELoss = totalMoELoss / transformerBlocks.size();

        Variable normalized = finalLayerNorm.forward(x);
        Variable logits = outputProjection.forward(normalized);

        return new DetailedForwardResult(logits, avgMoELoss);
    }

    /**
     * 验证输入的有效性
     */
    private void validateInput(Variable tokenIds) {
        NdArray data = tokenIds.getValue();
        if (data.getShape().getDimNum() != 2) {
            throw new IllegalArgumentException(
                    String.format("输入必须是2维张量 (batch_size, seq_len)，实际: %s",
                            data.getShape()));
        }

        int seqLen = data.getShape().getDimension(1);
        if (seqLen > config.getNPositions()) {
            throw new IllegalArgumentException(
                    String.format("序列长度(%d)超过最大位置数(%d)", seqLen, config.getNPositions()));
        }
    }

    /**
     * 估算参数数量
     */
    public long getParameterCount() {
        return config.estimateParameterCount();
    }

    /**
     * 打印架构信息
     */
    public void printArchitecture() {
        System.out.println("=".repeat(70));
        System.out.println("DeepSeek-R1 主体块架构（MoE，与 V3 共享）");
        System.out.println("=".repeat(70));
        System.out.printf("配置: %s\n", config);
        System.out.println("-".repeat(70));
        System.out.printf("Token嵌入层: %s（无位置嵌入，RoPE 在注意力层）\n",
                tokenEmbedding.getClass().getSimpleName());
        System.out.printf("Transformer块数量: %d (复用 V3 的 MoE TransformerBlock)\n",
                transformerBlocks.size());
        System.out.printf("专家数量: %d\n", config.getNumExperts());
        System.out.printf("Top-K选择: %d\n", config.getTopK());
        System.out.printf("架构模式: Pre-RMSNorm + RoPE + MoE\n");
        System.out.printf("训练方式: 纯RL（推理能力自然涌现）\n");
        System.out.printf("估算总参数: %s\n", formatParamCount(getParameterCount()));
        System.out.printf("激活参数: %s (%.1f%%)\n",
                formatParamCount(config.estimateActiveParameterCount()),
                config.getActivationRatio());
        System.out.println("=".repeat(70));
    }

    /**
     * 格式化参数数量
     */
    private String formatParamCount(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.2f B", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.2f M", count / 1_000_000.0);
        } else {
            return String.format("%,d", count);
        }
    }

    /**
     * 详细前向传播结果类
     */
    public static class DetailedForwardResult {
        /** 最终logits输出 */
        public final Variable logits;
        /** 平均 MoE 负载均衡损失 */
        public final double avgMoELoss;

        public DetailedForwardResult(Variable logits, double avgMoELoss) {
            this.logits = logits;
            this.avgMoELoss = avgMoELoss;
        }

        @Override
        public String toString() {
            return String.format(
                    "DetailedForwardResult{logitsShape=%s, avgMoELoss=%.6f}",
                    logits.getValue().getShape(), avgMoELoss);
        }
    }

    /**
     * 获取配置对象
     */
    public DeepSeekR1Config getConfig() {
        return config;
    }

    /**
     * 获取Transformer块列表
     */
    public List<DeepSeekV3TransformerBlock> getTransformerBlocks() {
        return transformerBlocks;
    }
}
