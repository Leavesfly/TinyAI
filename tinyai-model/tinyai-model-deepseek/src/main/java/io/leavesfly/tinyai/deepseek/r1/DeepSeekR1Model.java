package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.deepseek.base.TaskType;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * DeepSeek-R1模型类
 * 
 * ⚠️ 重要架构说明（根据论文 arXiv:2501.12948）：
 * DeepSeek-R1 与 DeepSeek-V3 使用完全相同的 MoE 基础架构（671B 参数），
 * 唯一的区别在于训练方式：
 * 
 * - **V3**: 标准预训练 + 后训练（SFT）
 * - **R1**: V3-Base + 纯强化学习（RL）训练
 * 
 * R1 的推理能力（Chain-of-Thought）是通过 RL 训练自然涌现的，
 * 而不是通过显式的推理/反思模块实现。
 * 
 * 主要特性：
 * 1. 混合专家(MoE) - 8专家Top-2路由，参数激活率约25%
 * 2. 任务感知路由 - 支持推理、代码、数学、通用、多模态5种任务
 * 3. Pre-LayerNorm架构 - 提升训练稳定性
 * 4. RL训练 - 通过纯RL训练使推理能力自然涌现
 * 
 * @author leavesfly
 * @version 2.0
 */
public class DeepSeekR1Model extends Model {
    
    private final DeepSeekR1Config config;
    private final DeepSeekR1Block r1Block;
    
    /**
     * 构造函数
     * 
     * @param name 模型名称
     * @param config R1配置对象
     */
    public DeepSeekR1Model(String name, DeepSeekR1Config config) {
        super(name, new DeepSeekR1Block(name + "_main", config));
        this.config = config;
        this.r1Block = (DeepSeekR1Block) getModule();
        setDescription(buildDescription());
    }
    
    /**
     * 构建模型描述信息
     */
    private String buildDescription() {
        return String.format(
            "DeepSeek-R1语言模型 | 参数量: %s | 激活参数: %s (%.1f%%) | 层数: %d | 维度: %d | " +
            "专家数: %d | Top-K: %d | 架构: Pre-LayerNorm+MoE | 训练方式: 纯RL",
            formatParamCount(config.estimateParameterCount()),
            formatParamCount(config.estimateActiveParameterCount()),
            config.getActivationRatio(),
            config.getNLayer(),
            config.getNEmbd(),
            config.getNumExperts(),
            config.getTopK()
        );
    }
    
    /**
     * 格式化参数数量
     */
    private String formatParamCount(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.2fB", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.2fM", count / 1_000_000.0);
        } else {
            return String.format("%,d", count);
        }
    }
    
    // ==================== 工厂方法 ====================
    
    /**
     * 创建标准DeepSeek-R1模型
     */
    public static DeepSeekR1Model createStandardModel(String name) {
        return new DeepSeekR1Model(name, DeepSeekR1Config.createStandardConfig());
    }
    
    /**
     * 创建微型DeepSeek-R1模型（用于快速测试）
     */
    public static DeepSeekR1Model createTinyModel(String name) {
        return new DeepSeekR1Model(name, DeepSeekR1Config.createTinyConfig());
    }
    
    /**
     * 创建小型DeepSeek-R1模型（用于学习和实验）
     */
    public static DeepSeekR1Model createSmallModel(String name) {
        return new DeepSeekR1Model(name, DeepSeekR1Config.createSmallConfig());
    }
    
    // ==================== 推理方法 ====================
    
    /**
     * 标准预测方法
     * 
     * @param tokenIds token ID序列 [batch_size, seq_len]
     * @return logits输出 [batch_size, seq_len, vocab_size]
     */
    public Variable predict(Variable tokenIds) {
        return forward(tokenIds);
    }
    
    /**
     * 带任务类型的预测
     * 
     * TODO: 待 R1Block 重构为 MoE 架构后，启用任务感知路由
     * 
     * @param tokenIds token ID序列 [batch_size, seq_len]
     * @param taskType 任务类型（可选）
     * @return logits输出 [batch_size, seq_len, vocab_size]
     */
    public Variable predict(Variable tokenIds, TaskType taskType) {
        // 暂时使用标准前向传播，等 R1Block 重构后实现任务感知
        return forward(tokenIds);
    }
    
    /**
     * 推理任务（利用RL训练涌现的推理能力）
     * 
     * @param tokenIds token ID序列 [batch_size, seq_len]
     * @return 推理结果
     */
    public ReasoningResult performReasoning(Variable tokenIds) {
        Variable logits = predict(tokenIds, TaskType.REASONING);
        
        return new ReasoningResult(
            logits,
            0.0,  // TODO: 待 R1Block 重构后获取 MoE 损失
            TaskType.REASONING
        );
    }
    
    /**
     * 数学计算任务
     * 
     * @param tokenIds token ID序列 [batch_size, seq_len]
     * @return 数学计算结果
     */
    public ReasoningResult solveMath(Variable tokenIds) {
        Variable logits = predict(tokenIds, TaskType.MATH);
        
        return new ReasoningResult(
            logits,
            0.0,  // TODO: 待 R1Block 重构后获取 MoE 损失
            TaskType.MATH
        );
    }
    
    /**
     * 生成序列（贪婪解码）
     * 
     * @param promptIds 提示词token ID序列 [batch_size, prompt_len]
     * @param maxNewTokens 最大生成token数量
     * @return 生成的完整序列 [batch_size, prompt_len + maxNewTokens]
     */
    public NdArray generateSequence(NdArray promptIds, int maxNewTokens) {
        int batchSize = promptIds.getShape().getDimension(0);
        int promptLen = promptIds.getShape().getDimension(1);
        
        float[][] generatedSeq = new float[batchSize][promptLen + maxNewTokens];
        
        // 复制提示词
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < promptLen; t++) {
                generatedSeq[b][t] = promptIds.get(b, t);
            }
        }
        
        // 自回归生成
        for (int i = 0; i < maxNewTokens; i++) {
            int currentLen = promptLen + i;
            float[][] currentInput = new float[batchSize][currentLen];
            for (int b = 0; b < batchSize; b++) {
                System.arraycopy(generatedSeq[b], 0, currentInput[b], 0, currentLen);
            }
            
            // 预测下一个token
            Variable logits = predict(new Variable(NdArray.of(currentInput)));
            NdArray logitsArray = logits.getValue();
            
            // 贪婪选择（选择概率最大的token）
            for (int b = 0; b < batchSize; b++) {
                int nextToken = argmax(logitsArray, b, currentLen - 1);
                generatedSeq[b][currentLen] = nextToken;
            }
        }
        
        return NdArray.of(generatedSeq);
    }
    
    /**
     * 查找最大值的索引（argmax）
     */
    private int argmax(NdArray logits, int batchIdx, int seqIdx) {
        int vocabSize = logits.getShape().getDimension(2);
        int maxIdx = 0;
        float maxVal = logits.get(batchIdx, seqIdx, 0);
        
        for (int i = 1; i < vocabSize; i++) {
            float val = logits.get(batchIdx, seqIdx, i);
            if (val > maxVal) {
                maxVal = val;
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    // ==================== 模型信息 ====================
    
    /**
     * 打印模型详细信息
     */
    @Override
    public void printModelInfo() {
        System.out.println("=".repeat(80));
        System.out.println("DeepSeek-R1 模型详细信息");
        System.out.println("=".repeat(80));
        System.out.println("模型名称: " + getName());
        System.out.println("模型描述: " + buildDescription());
        System.out.println("-".repeat(80));
        System.out.println(config);
        System.out.println("-".repeat(80));
        if (r1Block != null) {
            r1Block.printArchitecture();
        }
        System.out.println("=".repeat(80));
    }
    
    /**
     * 获取配置摘要
     */
    public String getConfigSummary() {
        return String.format(
            "DeepSeek-R1配置摘要:\n" +
            "  - 词汇表大小: %,d\n" +
            "  - 嵌入维度: %d\n" +
            "  - Transformer层数: %d\n" +
            "  - 注意力头数: %d\n" +
            "  - 前馈网络维度: %d\n" +
            "  - 最大序列长度: %d\n" +
            "  - 专家数量: %d\n" +
            "  - Top-K选择: %d\n" +
            "  - 架构: Pre-LayerNorm + MoE\n" +
            "  - 训练方式: 纯RL\n" +
            "  - 估算总参数: %s\n" +
            "  - 激活参数: %s (%.1f%%)",
            config.getVocabSize(),
            config.getNEmbd(),
            config.getNLayer(),
            config.getNHead(),
            config.getNInner(),
            config.getNPositions(),
            config.getNumExperts(),
            config.getTopK(),
            formatParamCount(config.estimateParameterCount()),
            formatParamCount(config.estimateActiveParameterCount()),
            config.getActivationRatio()
        );
    }
    
    // ==================== Getter方法 ====================
    
    public DeepSeekR1Config getConfig() {
        return config;
    }
    
    public DeepSeekR1Block getR1Block() {
        return r1Block;
    }
    
    @Override
    public String toString() {
        return String.format(
            "DeepSeekR1Model{name='%s', params=%s, activeParams=%s, nLayer=%d, nEmbd=%d, experts=%d}",
            getName(), 
            formatParamCount(config.estimateParameterCount()),
            formatParamCount(config.estimateActiveParameterCount()),
            config.getNLayer(), 
            config.getNEmbd(),
            config.getNumExperts()
        );
    }
    
    // ==================== 内部类 ====================
    
    /**
     * 推理结果类（RL训练涌现的推理能力）
     */
    public static class ReasoningResult {
        /** 最终logits输出 */
        public final Variable logits;
        /** MoE负载均衡损失 */
        public final double moeLoss;
        /** 任务类型 */
        public final TaskType taskType;
        
        public ReasoningResult(Variable logits, double moeLoss, TaskType taskType) {
            this.logits = logits;
            this.moeLoss = moeLoss;
            this.taskType = taskType;
        }
        
        @Override
        public String toString() {
            return String.format(
                "ReasoningResult{\n" +
                "  任务类型: %s\n" +
                "  MoE损失: %.6f\n" +
                "  输出形状: %s\n" +
                "}",
                taskType.getDescription(),
                moeLoss,
                logits.getValue().getShape()
            );
        }
    }
}
