package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.deepseek.base.TaskType;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * DeepSeek-V3模型类
 * 
 * DeepSeek-V3是一个基于混合专家模型(MoE)的大语言模型，
 * 通过任务感知路由实现高效的多任务处理和代码生成优化。
 * 
 * 主要特性：
 * 1. 混合专家(MoE) - 8专家Top-2路由，参数激活率约25%
 * 2. 任务感知 - 支持推理、代码、数学、通用、多模态5种任务
 * 3. 代码优化 - 专门优化代码生成，支持10种编程语言
 * 4. Pre-LayerNorm架构 - 提升训练稳定性
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3Model extends Model {
    
    private final DeepSeekV3Config config;
    private final DeepSeekV3Block v3Block;
    
    /**
     * 构造函数
     * 
     * @param name 模型名称
     * @param config V3配置对象
     */
    public DeepSeekV3Model(String name, DeepSeekV3Config config) {
        super(name, new DeepSeekV3Block(name + "_main", config));
        this.config = config;
        this.v3Block = (DeepSeekV3Block) getModule();
        setDescription(buildDescription());
    }
    
    /**
     * 构建模型描述信息
     */
    private String buildDescription() {
        return String.format(
            "DeepSeek-V3语言模型 | 参数量: %s | 激活参数: %s (%.1f%%) | 层数: %d | 维度: %d | " +
            "专家数: %d | Top-K: %d | 架构: Pre-LayerNorm+MoE",
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
     * 带详细信息的预测
     * 
     * @param tokenIds token ID序列 [batch_size, seq_len]
     * @param taskType 任务类型（可选）
     * @return 详细推理结果
     */
    public DeepSeekV3Block.DetailedForwardResult predictWithDetails(Variable tokenIds, TaskType taskType) {
        return v3Block.forwardWithDetails(tokenIds, taskType);
    }
    
    /**
     * 代码生成任务（专门优化）
     * 
     * @param tokenIds token ID序列 [batch_size, seq_len]
     * @return 代码生成结果
     */
    public CodeGenerationResult generateCode(Variable tokenIds) {
        DeepSeekV3Block.DetailedForwardResult result = 
            v3Block.forwardWithDetails(tokenIds, TaskType.CODING);
        
        return new CodeGenerationResult(
            result.logits,
            result.taskType,
            result.avgMoELoss
        );
    }
    
    /**
     * 推理任务（MoE自然涌现推理能力）
     * 
     * @param tokenIds token ID序列 [batch_size, seq_len]
     * @return 推理结果
     */
    public ReasoningResult performReasoning(Variable tokenIds) {
        DeepSeekV3Block.DetailedForwardResult result = 
            v3Block.forwardWithDetails(tokenIds, TaskType.REASONING);
        
        return new ReasoningResult(
            result.logits,
            result.taskType,
            result.avgMoELoss
        );
    }
    
    /**
     * 数学计算任务（MoE自然涌现数学能力）
     * 
     * @param tokenIds token ID序列 [batch_size, seq_len]
     * @return 数学计算结果
     */
    public MathResult solveMath(Variable tokenIds) {
        DeepSeekV3Block.DetailedForwardResult result = 
            v3Block.forwardWithDetails(tokenIds, TaskType.MATH);
        
        return new MathResult(
            result.logits,
            result.taskType,
            result.avgMoELoss
        );
    }
    
    /**
     * 生成序列（贪婪解码）
     * 
     * @param promptIds 提示词token ID序列 [batch_size, prompt_len]
     * @param maxNewTokens 最大生成token数量
     * @param taskType 任务类型（可选）
     * @return 生成的完整序列 [batch_size, prompt_len + maxNewTokens]
     */
    public NdArray generateSequence(NdArray promptIds, int maxNewTokens, TaskType taskType) {
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
            
            // 预测下一个token（使用任务类型信息）
            Variable logits;
            if (taskType != null) {
                logits = predictWithDetails(new Variable(NdArray.of(currentInput)), taskType).logits;
            } else {
                logits = predict(new Variable(NdArray.of(currentInput)));
            }
            NdArray logitsArray = logits.getValue();
            
            // 贪婪选择
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
            "  - 支持任务类型: %d种\n" +
            "  - 支持编程语言: %d种\n" +
            "  - 架构: Pre-LayerNorm + MoE\n" +
            "  - 估算总参数: %s\n" +
            "  - 激活参数: %s (%.1f%%)",
            config.getVocabSize(),
            config.getNEmbd(),
            config.getNLayer(),
            config.getNHead(),
            config.getNumExperts(),
            config.getTopK(),
            config.getNPositions(),
            config.getNumTaskTypes(),
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
    
    // ==================== 内部结果类 ====================
    
    /**
     * 代码生成结果类
     */
    public static class CodeGenerationResult {
        public final Variable logits;
        public final TaskType taskType;
        public final double moeLoss;
        
        public CodeGenerationResult(Variable logits, TaskType taskType, double moeLoss) {
            this.logits = logits;
            this.taskType = taskType;
            this.moeLoss = moeLoss;
        }
        
        @Override
        public String toString() {
            return String.format(
                "CodeGenerationResult{taskType=%s, moeLoss=%.6f}",
                taskType, moeLoss
            );
        }
    }
    
    /**
     * 推理结果类
     */
    public static class ReasoningResult {
        public final Variable logits;
        public final TaskType taskType;
        public final double moeLoss;
        
        public ReasoningResult(Variable logits, TaskType taskType, double moeLoss) {
            this.logits = logits;
            this.taskType = taskType;
            this.moeLoss = moeLoss;
        }
        
        @Override
        public String toString() {
            return String.format(
                "ReasoningResult{taskType=%s, moeLoss=%.6f}",
                taskType != null ? taskType.getDescription() : "未知",
                moeLoss
            );
        }
    }
    
    /**
     * 数学计算结果（简化版）
     */
    public static class MathResult {
        public final Variable logits;
        public final TaskType taskType;
        public final double moeLoss;
            
        public MathResult(Variable logits, TaskType taskType, double moeLoss) {
            this.logits = logits;
            this.taskType = taskType;
            this.moeLoss = moeLoss;
        }
            
        @Override
        public String toString() {
            return String.format(
                "MathResult{taskType=%s, moeLoss=%.6f}",
                taskType, moeLoss
            );
        }
    }
}
