package io.leavesfly.tinyai.deepseek.v3.training;

import io.leavesfly.tinyai.deepseek.base.inference.DeepSeekBaseInference;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Model;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.ArrayList;
import java.util.List;

/**
 * DeepSeek-V3推理引擎
 * 
 * 支持多种生成策略和任务感知推理
 * 
 * 生成策略：
 * 1. Greedy贪婪解码 - 选择概率最高的token
 * 2. Temperature采样 - 控制生成随机性
 * 3. Top-K采样 - 从Top-K个候选中采样
 * 4. Top-P(Nucleus)采样 - 累积概率采样
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3Inference extends DeepSeekBaseInference {
    
    private final DeepSeekV3Model model;
    private final int maxSeqLen;

    /**
     * EOS token ID，生成时遇到此 token 立即停止。
     * DeepSeek 与 GPT 系列共享词表规范，EOS 对应 vocabSize - 1。
     */
    private final int eosTokenId;
    
    /**
     * 采样策略接口
     */
    private interface SamplingStrategy {
        int sampleToken(NdArray logits, int seqLen, int vocabSize);
    }
    
    /**
     * 构造函数
     */
    public DeepSeekV3Inference(DeepSeekV3Model model) {
        super();
        this.model = model;
        this.maxSeqLen = model.getConfig().getNPositions();
        this.eosTokenId = model.getConfig().getVocabSize() - 1;
    }
    
    // ==================== 模板方法 ====================
    
    /**
     * 使用指定采样策略的自回归生成模板方法
     * 
     * @param promptIds 提示词token序列
     * @param maxNewTokens 最大生成token数
     * @param strategy 采样策略
     * @return 生成结果
     */
    private GenerationResult generateWithStrategy(int[] promptIds, int maxNewTokens,
                                                  SamplingStrategy strategy) {
        List<Integer> generated = new ArrayList<>();
        for (int id : promptIds) {
            generated.add(id);
        }
        
        List<ReasoningStep> reasoningSteps = new ArrayList<>();
        
        for (int i = 0; i < maxNewTokens; i++) {
            if (generated.size() >= maxSeqLen) break;

            int[] currentSeq = toIntArray(generated);
            Variable inputVar = new Variable(createInputArray(currentSeq));
            
            var result = model.predictWithDetails(inputVar);
            NdArray logits = result.logits.getValue();
            
            int seqLen = currentSeq.length;
            int vocabSize = logits.getShape().getDimension(2);
            
            // 使用采样策略选择下一个token
            int nextToken = strategy.sampleToken(logits, seqLen, vocabSize);
            generated.add(nextToken);

            if (nextToken == eosTokenId) break;
            
            reasoningSteps.add(new ReasoningStep(
                i,
                0.0,  // confidence不再可用（MoE自然涌现）
                result.avgMoELoss
            ));
        }
        
        return new GenerationResult(toIntArray(generated), reasoningSteps);
    }
    
    // ==================== 贪婪解码 ====================
    
    /**
     * 贪婪解码生成
     * 
     * @param promptIds 提示词token序列 [1, prompt_len]
     * @param maxNewTokens 最大生成token数
     * @return 生成结果
     */
    public GenerationResult generateGreedy(int[] promptIds, int maxNewTokens) {
        return generateWithStrategy(promptIds, maxNewTokens, (logits, seqLen, vocabSize) -> {
            // 使用基类的argmax方法
            return argmax(logits, 0, seqLen - 1);
        });
    }
    
    // ==================== Temperature采样 ====================
    
    /**
     * Temperature采样生成
     * 
     * @param promptIds 提示词token序列
     * @param maxNewTokens 最大生成token数
     * @param temperature 温度参数（0.1-2.0）,越高越随机
     * @return 生成结果
     */
    public GenerationResult generateWithTemperature(int[] promptIds, int maxNewTokens,
                                                    float temperature) {
        return generateWithStrategy(promptIds, maxNewTokens, (logits, seqLen, vocabSize) -> {
            // 使用基类的applySoftmax方法
            float[] probs = applySoftmax(logits, 0, seqLen - 1, temperature);
            // 使用基类的sample方法
            return sample(probs);
        });
    }
    
    // ==================== Top-K采样 ====================
    
    /**
     * Top-K采样生成
     * 
     * @param promptIds 提示词token序列
     * @param maxNewTokens 最大生成token数
     * @param topK 保留前K个候选
     * @return 生成结果
     */
    public GenerationResult generateTopK(int[] promptIds, int maxNewTokens,
                                         int topK) {
        return generateWithStrategy(promptIds, maxNewTokens, (logits, seqLen, vocabSize) -> {
            // 使用基类的applySoftmax方法（temperature=1.0）
            float[] probs = applySoftmax(logits, 0, seqLen - 1, 1.0f);
            // 使用基类的applyTopK方法
            applyTopK(probs, topK);
            // 使用基类的sample方法
            return sample(probs);
        });
    }
    
    // ==================== Top-P (Nucleus)采样 ====================
    
    /**
     * Top-P(Nucleus)采样生成
     * 
     * @param promptIds 提示词token序列
     * @param maxNewTokens 最大生成token数
     * @param topP 累积概率阈值（0.9-0.95典型值）
     * @return 生成结果
     */
    public GenerationResult generateTopP(int[] promptIds, int maxNewTokens,
                                         float topP) {
        return generateWithStrategy(promptIds, maxNewTokens, (logits, seqLen, vocabSize) -> {
            // 使用基类的applySoftmax方法（temperature=1.0）
            float[] probs = applySoftmax(logits, 0, seqLen - 1, 1.0f);
            // 使用基类的applyTopP方法
            applyTopP(probs, topP);
            // 使用基类的sample方法
            return sample(probs);
        });
    }
    
    // ==================== 结果类 ====================
    
    /**
     * 推理步骤信息
     */
    public static class ReasoningStep {
        public final int tokenIndex;
        public final double confidence;
        public final double moeLoss;
        
        public ReasoningStep(int tokenIndex, double confidence, double moeLoss) {
            this.tokenIndex = tokenIndex;
            this.confidence = confidence;
            this.moeLoss = moeLoss;
        }
    }
    
    /**
     * 生成结果
     */
    public static class GenerationResult {
        public final int[] tokens;
        public final List<ReasoningStep> reasoningSteps;
        
        public GenerationResult(int[] tokens, List<ReasoningStep> reasoningSteps) {
            this.tokens = tokens;
            this.reasoningSteps = reasoningSteps;
        }
        
        /**
         * 打印推理追踪
         */
        public void printReasoningTrace() {
            System.out.println("\n推理追踪:");
            System.out.println("-".repeat(60));
            for (int i = 0; i < Math.min(5, reasoningSteps.size()); i++) {
                ReasoningStep step = reasoningSteps.get(i);
                System.out.printf("Step %d: 置信度=%.4f, MoE损失=%.6f%n",
                    step.tokenIndex, step.confidence, step.moeLoss);
            }
            if (reasoningSteps.size() > 5) {
                System.out.println("... (" + (reasoningSteps.size() - 5) + " more steps)");
            }
            System.out.println("-".repeat(60));
        }
    }
}