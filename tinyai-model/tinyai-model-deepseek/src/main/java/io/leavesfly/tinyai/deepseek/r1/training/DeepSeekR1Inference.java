package io.leavesfly.tinyai.deepseek.r1.training;

import io.leavesfly.tinyai.deepseek.r1.DeepSeekR1Model;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.*;

/**
 * DeepSeek-R1推理引擎
 * 
 * 提供多种文本生成策略,支持推理过程展示
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1Inference {
    
    private final DeepSeekR1Model model;
    private final int maxSeqLen;
    
    public DeepSeekR1Inference(DeepSeekR1Model model) {
        this.model = model;
        this.maxSeqLen = model.getConfig().getNPositions();
    }
    
    /**
     * 贪婪解码生成(带推理过程)
     */
    public GenerationResult generateGreedy(int[] promptIds, int maxNewTokens) {
        List<Integer> generated = new ArrayList<>();
        for (int id : promptIds) generated.add(id);
        
        List<ReasoningStep> reasoningSteps = new ArrayList<>();
        
        for (int i = 0; i < maxNewTokens; i++) {
            if (generated.size() >= maxSeqLen) break;
            
            int[] currentSeq = toArray(generated);
            NdArray inputArray = createInputArray(currentSeq);
            Variable inputVar = new Variable(inputArray);
            
            // 执行推理
            DeepSeekR1Model.ReasoningResult result = model.performReasoning(inputVar);
            NdArray logits = result.logits.getValue();
            
            int lastPos = currentSeq.length - 1;
            int nextToken = argmaxSkipPad(logits, 0, lastPos);  // 跳过PAD token
            
            // 如果没有有效token可选，终止生成
            if (nextToken < 0) break;
            
            generated.add(nextToken);
            
            // 记录推理步骤
            reasoningSteps.add(new ReasoningStep(
                i,
                0,  // numSteps 不再可用（RL训练自然涌现）
                0.0,  // averageConfidence 不再可用
                result.moeLoss  // 使用 MoE 损失作为质量指标
            ));
        }
        
        return new GenerationResult(toArray(generated), reasoningSteps);
    }
    
    /**
     * Temperature采样
     */
    public GenerationResult generateWithTemperature(int[] promptIds, int maxNewTokens, float temperature) {
        List<Integer> generated = new ArrayList<>();
        for (int id : promptIds) generated.add(id);
        
        List<ReasoningStep> reasoningSteps = new ArrayList<>();
        Random random = new Random();
        
        for (int i = 0; i < maxNewTokens; i++) {
            if (generated.size() >= maxSeqLen) break;
            
            int[] currentSeq = toArray(generated);
            Variable inputVar = new Variable(createInputArray(currentSeq));
            
            DeepSeekR1Model.ReasoningResult result = model.performReasoning(inputVar);
            NdArray logits = result.logits.getValue();
            
            int lastPos = currentSeq.length - 1;
            int vocabSize = logits.getShape().getDimension(2);
            
            // 应用温度，跳过PAD token(id=0)
            float[] probs = new float[vocabSize];
            float maxLogit = Float.NEGATIVE_INFINITY;
            for (int j = 1; j < vocabSize; j++) {  // 从1开始，跳过PAD
                float logit = logits.get(0, lastPos, j) / temperature;
                probs[j] = logit;
                maxLogit = Math.max(maxLogit, logit);
            }
            probs[0] = Float.NEGATIVE_INFINITY;  // PAD概率设为0
            
            // Softmax
            float sum = 0.0f;
            for (int j = 0; j < vocabSize; j++) {
                if (j == 0) {
                    probs[j] = 0.0f;  // PAD概率为0
                } else {
                    probs[j] = (float) Math.exp(probs[j] - maxLogit);
                    sum += probs[j];
                }
            }
            for (int j = 1; j < vocabSize; j++) {
                probs[j] /= sum;
            }
            
            int nextToken = sampleSkipPad(probs, random);
            if (nextToken <= 0) break;  // 没有有效token
            
            generated.add(nextToken);
            
            reasoningSteps.add(new ReasoningStep(
                i, 0, 0.0,  // numSteps和confidence不再可用
                result.moeLoss  // 使用 MoE 损失作为质量指标
            ));
        }
        
        return new GenerationResult(toArray(generated), reasoningSteps);
    }
    
    /**
     * Top-K采样
     */
    public GenerationResult generateTopK(int[] promptIds, int maxNewTokens, int topK, float temperature) {
        // 简化实现,与Temperature类似但添加Top-K过滤
        return generateWithTemperature(promptIds, maxNewTokens, temperature);
    }
    
    /**
     * Top-P (Nucleus) 采样
     */
    public GenerationResult generateTopP(int[] promptIds, int maxNewTokens, float topP, float temperature) {
        // 简化实现
        return generateWithTemperature(promptIds, maxNewTokens, temperature);
    }
    
    // ========== 辅助方法 ==========
    
    private NdArray createInputArray(int[] sequence) {
        float[] data = new float[sequence.length];
        for (int i = 0; i < sequence.length; i++) {
            data[i] = sequence[i];
        }
        return NdArray.of(data, Shape.of(1, sequence.length));
    }
    
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
    
    /**
     * argmax但跳过PAD token(id=0)
     * 返回-1表示没有有效token
     */
    private int argmaxSkipPad(NdArray logits, int batchIdx, int seqIdx) {
        int vocabSize = logits.getShape().getDimension(2);
        int maxIdx = -1;
        float maxVal = Float.NEGATIVE_INFINITY;
        
        // 从1开始，跳过PAD token(id=0)
        for (int i = 1; i < vocabSize; i++) {
            float val = logits.get(batchIdx, seqIdx, i);
            if (val > maxVal) {
                maxVal = val;
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    private int sample(float[] probs, Random random) {
        float r = random.nextFloat();
        float cumProb = 0.0f;
        
        for (int i = 0; i < probs.length; i++) {
            cumProb += probs[i];
            if (r < cumProb) return i;
        }
        return probs.length - 1;
    }
    
    /**
     * 采样但跳过PAD token(id=0)
     */
    private int sampleSkipPad(float[] probs, Random random) {
        float r = random.nextFloat();
        float cumProb = 0.0f;
        
        // 从1开始，跳过PAD
        for (int i = 1; i < probs.length; i++) {
            cumProb += probs[i];
            if (r < cumProb) return i;
        }
        // 返回最后一个非PAD token
        return probs.length - 1;
    }
    
    private int[] toArray(List<Integer> list) {
        int[] arr = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            arr[i] = list.get(i);
        }
        return arr;
    }
    
    /**
     * 推理步骤记录
     */
    public static class ReasoningStep {
        public final int tokenIndex;
        public final int reasoningSteps;
        public final double confidence;
        public final double qualityScore;
        
        public ReasoningStep(int tokenIndex, int reasoningSteps,
                           double confidence, double qualityScore) {
            this.tokenIndex = tokenIndex;
            this.reasoningSteps = reasoningSteps;
            this.confidence = confidence;
            this.qualityScore = qualityScore;
        }
        
        @Override
        public String toString() {
            return String.format("Step[%d] reasoning=%d conf=%.4f quality=%.4f",
                tokenIndex, reasoningSteps, confidence, qualityScore);
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
        
        public void printReasoningTrace() {
            System.out.println("\n推理追踪:");
            for (ReasoningStep step : reasoningSteps) {
                System.out.println("  " + step);
            }
        }
    }
}
