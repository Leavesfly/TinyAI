package io.leavesfly.tinyai.deepseek.base.inference;

import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * DeepSeek 系列推理引擎基类
 * 
 * 提供通用的文本生成策略和工具方法：
 * - 贪婪解码、Temperature采样、Top-K采样、Top-P采样
 * - 数据预处理（输入创建、数组转换）
 * - 采样辅助方法（argmax、softmax、采样）
 * 
 * @author leavesfly
 * @version 1.0
 */
public abstract class DeepSeekBaseInference {
    
    protected final Random random;
    
    /** PAD token ID，推理时需要排除 */
    protected static final int PAD_TOKEN_ID = 0;
    
    public DeepSeekBaseInference() {
        this.random = new Random();
    }
    
    /**
     * 设置随机种子
     */
    public void setSeed(long seed) {
        random.setSeed(seed);
    }
    
    // ========== 数据预处理方法 ==========
    
    /**
     * 创建输入数组 [1, seq_len]
     */
    protected NdArray createInputArray(int[] sequence) {
        float[][] inputData = new float[1][sequence.length];
        for (int i = 0; i < sequence.length; i++) {
            inputData[0][i] = sequence[i];
        }
        return NdArray.of(inputData);
    }
    
    /**
     * List转数组
     */
    protected int[] toIntArray(List<Integer> list) {
        int[] arr = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            arr[i] = list.get(i);
        }
        return arr;
    }
    
    // ========== 采样辅助方法 ==========
    
    /**
     * Argmax（跳过PAD token）
     * 
     * @param logits logits数组 [1, seq_len, vocab_size]
     * @param batch batch索引
     * @param pos 位置索引
     * @return 最大概率的token ID
     */
    protected int argmax(NdArray logits, int batch, int pos) {
        int vocabSize = logits.getShape().getDimension(2);
        int maxIdx = 1;  // 从1开始，跳过PAD
        float maxVal = logits.get(batch, pos, 1);
        
        for (int i = 2; i < vocabSize; i++) {
            float val = logits.get(batch, pos, i);
            if (val > maxVal) {
                maxVal = val;
                maxIdx = i;
            }
        }
        
        return maxIdx;
    }
    
    /**
     * 应用Softmax（跳过PAD token）
     * 
     * @param logits logits数组
     * @param temperature 温度参数
     * @return 概率分布
     */
    protected float[] applySoftmax(NdArray logits, int batch, int pos, float temperature) {
        int vocabSize = logits.getShape().getDimension(2);
        float[] probs = new float[vocabSize];
        
        // 应用温度并找最大值
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (int i = 1; i < vocabSize; i++) {  // 跳过PAD
            float logit = logits.get(batch, pos, i) / temperature;
            probs[i] = logit;
            maxLogit = Math.max(maxLogit, logit);
        }
        probs[0] = Float.NEGATIVE_INFINITY;  // PAD概率设为0
        
        // Softmax归一化
        float sum = 0.0f;
        for (int i = 1; i < vocabSize; i++) {
            probs[i] = (float) Math.exp(probs[i] - maxLogit);
            sum += probs[i];
        }
        for (int i = 1; i < vocabSize; i++) {
            probs[i] /= sum;
        }
        probs[0] = 0.0f;  // PAD概率为0
        
        return probs;
    }
    
    /**
     * 从概率分布中采样（跳过PAD）
     */
    protected int sample(float[] probs) {
        float r = random.nextFloat();
        float cumSum = 0.0f;
        
        for (int i = 1; i < probs.length; i++) {  // 从1开始，跳过PAD
            cumSum += probs[i];
            if (cumSum >= r) {
                return i;
            }
        }
        
        // 如果没有采样到，返回最高概率的token
        int maxIdx = 1;
        for (int i = 2; i < probs.length; i++) {
            if (probs[i] > probs[maxIdx]) {
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    /**
     * Top-K过滤
     * 
     * @param probs 概率分布
     * @param k 保留前k个最高概率的token
     */
    protected void applyTopK(float[] probs, int k) {
        // 找到第k大的值
        List<Float> sortedProbs = new ArrayList<>();
        for (int i = 1; i < probs.length; i++) {  // 跳过PAD
            sortedProbs.add(probs[i]);
        }
        sortedProbs.sort((a, b) -> Float.compare(b, a));  // 降序
        
        if (k < sortedProbs.size()) {
            float threshold = sortedProbs.get(k - 1);
            for (int i = 1; i < probs.length; i++) {
                if (probs[i] < threshold) {
                    probs[i] = 0.0f;
                }
            }
            
            // 重新归一化
            float sum = 0.0f;
            for (int i = 1; i < probs.length; i++) {
                sum += probs[i];
            }
            if (sum > 0) {
                for (int i = 1; i < probs.length; i++) {
                    probs[i] /= sum;
                }
            }
        }
    }
    
    /**
     * Top-P (Nucleus Sampling) 过滤
     * 
     * @param probs 概率分布
     * @param p 累积概率阈值
     */
    protected void applyTopP(float[] probs, float p) {
        // 按概率降序排序
        List<Integer> indices = new ArrayList<>();
        for (int i = 1; i < probs.length; i++) {  // 跳过PAD
            indices.add(i);
        }
        indices.sort((a, b) -> Float.compare(probs[b], probs[a]));
        
        // 累积概率
        float cumSum = 0.0f;
        int cutoff = indices.size();
        for (int i = 0; i < indices.size(); i++) {
            cumSum += probs[indices.get(i)];
            if (cumSum >= p) {
                cutoff = i + 1;
                break;
            }
        }
        
        // 过滤低概率token
        for (int i = cutoff; i < indices.size(); i++) {
            probs[indices.get(i)] = 0.0f;
        }
        
        // 重新归一化
        float sum = 0.0f;
        for (int i = 1; i < probs.length; i++) {
            sum += probs[i];
        }
        if (sum > 0) {
            for (int i = 1; i < probs.length; i++) {
                probs[i] /= sum;
            }
        }
    }
    
    // ========== 结果类 ==========
    
    /**
     * 生成结果
     */
    public static class GenerationResult {
        public final int[] tokenIds;
        public final List<ReasoningStep> reasoningSteps;
        
        public GenerationResult(int[] tokenIds, List<ReasoningStep> reasoningSteps) {
            this.tokenIds = tokenIds;
            this.reasoningSteps = reasoningSteps;
        }
    }
    
    /**
     * 推理步骤
     */
    public static class ReasoningStep {
        public final int stepIndex;
        public final double confidence;
        public final double moeLoss;
        
        public ReasoningStep(int stepIndex, double confidence, double moeLoss) {
            this.stepIndex = stepIndex;
            this.confidence = confidence;
            this.moeLoss = moeLoss;
        }
    }
}
