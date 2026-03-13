package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.deepseek.base.DeepSeekBaseConfig;
import io.leavesfly.tinyai.deepseek.base.DeepSeekModelBase;
import io.leavesfly.tinyai.func.Variable;

/**
 * DeepSeek-V3模型类
 *
 * DeepSeek-V3是一个基于混合专家模型(MoE)的大语言模型，
 * 通过任务感知路由实现高效的多任务处理和代码生成优化。
 *
 * 主要特性：
 * 1. 混合专家(MoE) - 8专家Top-2路由，参数激活率约25%
 * 2. 任务感知 - 支持推理、代码、数学、通用、多模态5种任务
 * 3. Pre-RMSNorm + RoPE 架构 - 提升训练稳定性
 * 4. MTP（Multi-Token Prediction）- V3 特有的训练辅助机制
 *
 * @author leavesfly
 * @version 2.0
 */
public class DeepSeekV3Model extends DeepSeekModelBase {

    private final DeepSeekV3Config config;
    private final DeepSeekV3Block v3Block;

    /**
     * 构造函数
     *
     * @param name   模型名称
     * @param config V3配置对象
     */
    public DeepSeekV3Model(String name, DeepSeekV3Config config) {
        super(name, new DeepSeekV3Block(name + "_main", config));
        this.config = config;
        this.v3Block = (DeepSeekV3Block) getModule();
        setDescription(buildDescription());
    }

    @Override
    public DeepSeekBaseConfig getBaseConfig() {
        return config;
    }

    /**
     * 构建模型描述信息
     */
    private String buildDescription() {
        return String.format(
                "DeepSeek-V3语言模型 | 参数量: %s | 激活参数: %s (%.1f%%) | 层数: %d | 维度: %d | " +
                        "专家数: %d | Top-K: %d | 架构: Pre-RMSNorm+RoPE+MoE",
                formatParamCount(config.estimateParameterCount()),
                formatParamCount(config.estimateActiveParameterCount()),
                config.getActivationRatio(),
                config.getNLayer(),
                config.getNEmbd(),
                config.getNumExperts(),
                config.getTopK()
        );
    }

    // ==================== 工厂方法 ====================

    /**
     * 创建标准DeepSeek-V3模型
     */
    public static DeepSeekV3Model createStandardModel(String name) {
        return new DeepSeekV3Model(name, DeepSeekV3Config.createStandardConfig());
    }

    /**
     * 创建微型DeepSeek-V3模型（用于快速测试）
     */
    public static DeepSeekV3Model createTinyModel(String name) {
        return new DeepSeekV3Model(name, DeepSeekV3Config.createTinyConfig());
    }

    /**
     * 创建小型DeepSeek-V3模型（用于学习和实验）
     */
    public static DeepSeekV3Model createSmallModel(String name) {
        return new DeepSeekV3Model(name, DeepSeekV3Config.createSmallConfig());
    }

    // ==================== V3 特有推理方法 ====================

    /**
     * 带详细信息的预测（包含 MoE 损失和隐藏状态）
     *
     * @param tokenIds token ID序列 [batch_size, seq_len]
     * @return 详细推理结果
     */
    public DeepSeekV3Block.DetailedForwardResult predictWithDetails(Variable tokenIds) {
        return v3Block.forwardWithDetails(tokenIds);
    }

    // ==================== 模型信息 ====================

    @Override
    public void printModelInfo() {
        System.out.println("=".repeat(80));
        System.out.println("DeepSeek-V3 模型详细信息");
        System.out.println("=".repeat(80));
        System.out.println("模型名称: " + getName());
        System.out.println("模型描述: " + buildDescription());
        System.out.println("-".repeat(80));
        System.out.println(config);
        System.out.println("-".repeat(80));
        if (v3Block != null) {
            v3Block.printArchitecture();
        }
        System.out.println("=".repeat(80));
    }

    /**
     * 获取配置摘要
     */
    public String getConfigSummary() {
        return String.format(
                "DeepSeek-V3配置摘要:\n" +
                        "  - 词汇表大小: %,d\n" +
                        "  - 嵌入维度: %d\n" +
                        "  - Transformer层数: %d\n" +
                        "  - 注意力头数: %d\n" +
                        "  - 专家数量: %d\n" +
                        "  - Top-K选择: %d\n" +
                        "  - 最大序列长度: %d\n" +
                        "  - 支持编程语言: %d种\n" +
                        "  - 架构: Pre-RMSNorm + RoPE + MoE\n" +
                        "  - 估算总参数: %s\n" +
                        "  - 激活参数: %s (%.1f%%)",
                config.getVocabSize(),
                config.getNEmbd(),
                config.getNLayer(),
                config.getNHead(),
                config.getNumExperts(),
                config.getTopK(),
                config.getNPositions(),
                config.getNumProgrammingLanguages(),
                formatParamCount(config.estimateParameterCount()),
                formatParamCount(config.estimateActiveParameterCount()),
                config.getActivationRatio()
        );
    }

    // ==================== Getter方法 ====================

    public DeepSeekV3Config getConfig() {
        return config;
    }

    public DeepSeekV3Block getV3Block() {
        return v3Block;
    }

    @Override
    public String toString() {
        return String.format(
                "DeepSeekV3Model{name='%s', params=%s, activeParams=%s, nLayer=%d, nEmbd=%d, experts=%d}",
                getName(),
                formatParamCount(config.estimateParameterCount()),
                formatParamCount(config.estimateActiveParameterCount()),
                config.getNLayer(),
                config.getNEmbd(),
                config.getNumExperts()
        );
    }
}
