package io.leavesfly.tinyai.deepseek.v3.training;

import io.leavesfly.tinyai.deepseek.base.TaskType;
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
    
    /**
     * 构造函数
     */
    public DeepSeekV3Inference(DeepSeekV3Model model) {
        super();
        this.model = model;
    }
    
    // ==================== 贪婪解码 ====================
    
    /**
     * 贪婪解码生成
     * 
     * @param promptIds 提示词token序列 [1, prompt_len]
     * @param maxNewTokens 最大生成token数
     * @param taskType 任务类型
     * @return 生成结果
     */
    public GenerationResult generateGreedy(int[] promptIds, int maxNewTokens, 
                                           TaskType taskType) {
        List<Integer> generated = new ArrayList<>();
        for (int id : promptIds) {
            generated.add(id);
        }
        
        List<ReasoningStep> reasoningSteps = new ArrayList<>();
        
        for (int i = 0; i < maxNewTokens; i++) {
            int[] currentSeq = toIntArray(generated);
            Variable inputVar = new Variable(createInputArray(currentSeq));
            
            // 推理（带详细信息）
            var result = model.predictWithDetails(inputVar, taskType);
            NdArray logits = result.logits.getValue();
            
            // 选择最后一个位置的logits
            int seqLen = currentSeq.length;
            int nextToken = argmax(logits, 0, seqLen - 1);
            
            generated.add(nextToken);
            
            // 记录推理步骤
            reasoningSteps.add(new ReasoningStep(
                i,
                0.0,  // confidence不再可用（MoE自然涌现）
                result.avgMoELoss
            ));
        }
        
        return new GenerationResult(toIntArray(generated), reasoningSteps);
    }
    
    // ==================== Temperature采样 ====================
    
    /**
     * Temperature采样生成
     * 
     * @param promptIds 提示词token序列
     * @param maxNewTokens 最大生成token数
     * @param temperature 温度参数（0.1-2.0）,越高越随机
     * @param taskType 任务类型
     * @return 生成结果
     */
    public GenerationResult generateWithTemperature(int[] promptIds, int maxNewTokens,
                                                    float temperature, TaskType taskType) {
        List<Integer> generated = new ArrayList<>();
        for (int id : promptIds) {
            generated.add(id);
        }
        
        List<ReasoningStep> reasoningSteps = new ArrayList<>();
        
        for (int i = 0; i < maxNewTokens; i++) {
            int[] currentSeq = toIntArray(generated);
            Variable inputVar = new Variable(createInputArray(currentSeq));
            
            var result = model.predictWithDetails(inputVar, taskType);
            NdArray logits = result.logits.getValue();
            
            int seqLen = currentSeq.length;
            int vocabSize = logits.getShape().getDimension(2);
            
            // 应用temperature，跳过PAD token (id=0)
            float[] probs = new float[vocabSize];
            probs[0] = 0.0f;  // PAD token概率设为0
            float sum = 0.0f;
            for (int j = 1; j < vocabSize; j++) {
                float logit = logits.get(0, seqLen - 1, j);
                probs[j] = (float) Math.exp(logit / temperature);
                sum += probs[j];
            }
            
            // 归一化
            for (int j = 0; j < vocabSize; j++) {
                probs[j] /= sum;
            }
            
            // 采样
            int nextToken = sampleFromProbs(probs);
            generated.add(nextToken);
            
            reasoningSteps.add(new ReasoningStep(
                i,
                0.0,  // confidence不再可用（MoE自然涌现）
                result.avgMoELoss
            ));
        }
        
        return new GenerationResult(toIntArray(generated), reasoningSteps);
    }
    
    // ==================== Top-K采样 ====================
    
    /**
     * Top-K采样生成
     * 
     * @param promptIds 提示词token序列
     * @param maxNewTokens 最大生成token数
     * @param topK 保留前K个候选
     * @param taskType 任务类型
     * @return 生成结果
     */
    public GenerationResult generateTopK(int[] promptIds, int maxNewTokens,
                                         int topK, TaskType taskType) {
        List<Integer> generated = new ArrayList<>();
        for (int id : promptIds) {
            generated.add(id);
        }
        
        List<ReasoningStep> reasoningSteps = new ArrayList<>();
        
        for (int i = 0; i < maxNewTokens; i++) {
            int[] currentSeq = toIntArray(generated);
            Variable inputVar = new Variable(createInputArray(currentSeq));
            
            var result = model.predictWithDetails(inputVar, taskType);
            NdArray logits = result.logits.getValue();
            
            int seqLen = currentSeq.length;
            int vocabSize = logits.getShape().getDimension(2);
            
            // 获取logits，PAD token (id=0)设为负无穷大
            float[] logitArray = new float[vocabSize];
            logitArray[0] = Float.NEGATIVE_INFINITY;  // 排除PAD token
            for (int j = 1; j < vocabSize; j++) {
                logitArray[j] = logits.get(0, seqLen - 1, j);
            }
            
            // Top-K过滤
            int[] topKIndices = getTopKIndices(logitArray, topK);
            float[] topKProbs = new float[topK];
            float sum = 0.0f;
            for (int j = 0; j < topK; j++) {
                topKProbs[j] = (float) Math.exp(logitArray[topKIndices[j]]);
                sum += topKProbs[j];
            }
            
            // 归一化
            for (int j = 0; j < topK; j++) {
                topKProbs[j] /= sum;
            }
            
            // 采样
            int sampledIdx = sampleFromProbs(topKProbs);
            int nextToken = topKIndices[sampledIdx];
            generated.add(nextToken);
            
            reasoningSteps.add(new ReasoningStep(
                i,
                0.0,  // confidence不再可用（MoE自然涌现）
                result.avgMoELoss
            ));
        }
        
        return new GenerationResult(toIntArray(generated), reasoningSteps);
    }
    
    // ==================== Top-P (Nucleus)采样 ====================
    
    /**
     * Top-P(Nucleus)采样生成
     * 
     * @param promptIds 提示词token序列
     * @param maxNewTokens 最大生成token数
     * @param topP 累积概率阈值（0.9-0.95典型值）
     * @param taskType 任务类型
     * @return 生成结果
     */
    public GenerationResult generateTopP(int[] promptIds, int maxNewTokens,
                                         float topP, TaskType taskType) {
        List<Integer> generated = new ArrayList<>();
        for (int id : promptIds) {
            generated.add(id);
        }
        
        List<ReasoningStep> reasoningSteps = new ArrayList<>();
        
        for (int i = 0; i < maxNewTokens; i++) {
            int[] currentSeq = toIntArray(generated);
            Variable inputVar = new Variable(createInputArray(currentSeq));
            
            var result = model.predictWithDetails(inputVar, taskType);
            NdArray logits = result.logits.getValue();
            
            int seqLen = currentSeq.length;
            int vocabSize = logits.getShape().getDimension(2);
            
            // 获取并排序概率，跳过PAD token (id=0)
            float[] probs = new float[vocabSize];
            probs[0] = 0.0f;  // PAD token概率设为0
            float sum = 0.0f;
            for (int j = 1; j < vocabSize; j++) {
                float logit = logits.get(0, seqLen - 1, j);
                probs[j] = (float) Math.exp(logit);
                sum += probs[j];
            }
            
            // 归一化
            for (int j = 0; j < vocabSize; j++) {
                probs[j] /= sum;
            }
            
            // 排序并累积
            int[] sortedIndices = argsort(probs);
            float cumProb = 0.0f;
            List<Integer> nucleusIndices = new ArrayList<>();
            List<Float> nucleusProbs = new ArrayList<>();
            
            for (int j = sortedIndices.length - 1; j >= 0; j--) {
                int idx = sortedIndices[j];
                nucleusIndices.add(idx);
                nucleusProbs.add(probs[idx]);
                cumProb += probs[idx];
                if (cumProb >= topP) {
                    break;
                }
            }
            
            // 重新归一化并采样
            float[] nucleusProbArray = new float[nucleusProbs.size()];
            float nucleusSum = 0.0f;
            for (int j = 0; j < nucleusProbs.size(); j++) {
                nucleusProbArray[j] = nucleusProbs.get(j);
                nucleusSum += nucleusProbArray[j];
            }
            for (int j = 0; j < nucleusProbArray.length; j++) {
                nucleusProbArray[j] /= nucleusSum;
            }
            
            int sampledIdx = sampleFromProbs(nucleusProbArray);
            int nextToken = nucleusIndices.get(sampledIdx);
            generated.add(nextToken);
            
            reasoningSteps.add(new ReasoningStep(
                i,
                0.0,  // confidence不再可用（MoE自然涌现）
                result.avgMoELoss
            ));
        }
        
        return new GenerationResult(toIntArray(generated), reasoningSteps);
    }
    
    // ==================== 辅助方法（继承自DeepSeekBaseInference） ====================
    
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
