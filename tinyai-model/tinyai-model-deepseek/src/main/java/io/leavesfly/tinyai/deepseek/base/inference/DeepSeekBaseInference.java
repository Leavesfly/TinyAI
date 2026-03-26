package io.leavesfly.tinyai.deepseek.base.inference;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

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
     * 创建输入数组 [1, seq_len]，将int序列转换为float数组
     */
    protected NdArray createInputArray(int[] sequence) {
        float[] data = new float[sequence.length];
        for (int i = 0; i < sequence.length; i++) {
            data[i] = sequence[i];
        }
        return NdArray.of(data, Shape.of(1, sequence.length));
    }
    
    /**
     * List转int数组
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
     * @return 最大概率的token ID（跳过PAD_TOKEN_ID=0），词表仅1个token时返回0
     */
    protected int argmax(NdArray logits, int batch, int pos) {
        int vocabSize = logits.getShape().getDimension(2);
        if (vocabSize <= 1) {
            return 0;
        }
        
        int maxIdx = 1;
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
     * 优化：合并温度缩放、求max、exp计算为两次遍历（原三次），减少数组访问开销。
     * 
     * @param logits logits数组
     * @param batch batch索引
     * @param pos 位置索引
     * @param temperature 温度参数
     * @return 概率分布
     */
    protected float[] applySoftmax(NdArray logits, int batch, int pos, float temperature) {
        int vocabSize = logits.getShape().getDimension(2);
        float[] probs = new float[vocabSize];
        
        // 第一遍：应用温度并找最大值
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (int i = 1; i < vocabSize; i++) {
            float logit = logits.get(batch, pos, i) / temperature;
            probs[i] = logit;
            if (logit > maxLogit) {
                maxLogit = logit;
            }
        }
        
        // 第二遍：exp + 求和 + 归一化（合并为一次遍历求和，再一次归一化）
        float sum = 0.0f;
        for (int i = 1; i < vocabSize; i++) {
            float expVal = (float) Math.exp(probs[i] - maxLogit);
            probs[i] = expVal;
            sum += expVal;
        }
        if (sum > 0.0f) {
            float invSum = 1.0f / sum;
            for (int i = 1; i < vocabSize; i++) {
                probs[i] *= invSum;
            }
        }
        probs[0] = 0.0f;
        
        return probs;
    }
    
    /**
     * 从概率分布中采样（跳过PAD）
     * 
     * 当浮点精度导致累积概率未达到随机阈值时，回退到最高概率token。
     */
    protected int sample(float[] probs) {
        float r = random.nextFloat();
        float cumSum = 0.0f;
        int fallbackIdx = 1;
        float fallbackMax = probs.length > 1 ? probs[1] : 0.0f;
        
        for (int i = 1; i < probs.length; i++) {
            cumSum += probs[i];
            if (probs[i] > fallbackMax) {
                fallbackMax = probs[i];
                fallbackIdx = i;
            }
            if (cumSum >= r) {
                return i;
            }
        }
        
        // 浮点精度导致未命中，回退到最高概率token
        return fallbackIdx;
    }
    
    /**
     * 从概率分布中采样（通用版，不跳过任何token）
     */
    protected int sampleFromProbs(float[] probs) {
        float r = random.nextFloat();
        float cumProb = 0.0f;
        for (int i = 0; i < probs.length; i++) {
            cumProb += probs[i];
            if (r < cumProb) {
                return i;
            }
        }
        return probs.length - 1;
    }
    
    /**
     * 获取Top-K个最大值的索引
     * 
     * 使用最小堆实现，时间复杂度 O(n·log k)，优于原 O(n·k) 的选择排序。
     * 
     * @param values 值数组
     * @param k 保留前k个
     * @return Top-K索引数组（按值从大到小排列）
     */
    protected int[] getTopKIndices(float[] values, int k) {
        int effectiveK = Math.min(k, values.length);
        
        // 最小堆：堆顶是当前Top-K中最小的，方便淘汰
        java.util.PriorityQueue<int[]> minHeap = new java.util.PriorityQueue<>(
                effectiveK + 1, (a, b) -> Float.compare(values[a[0]], values[b[0]]));
        
        for (int i = 0; i < values.length; i++) {
            minHeap.offer(new int[]{i});
            if (minHeap.size() > effectiveK) {
                minHeap.poll();
            }
        }
        
        int[] result = new int[effectiveK];
        for (int i = effectiveK - 1; i >= 0; i--) {
            result[i] = minHeap.poll()[0];
        }
        return result;
    }
    
    /**
     * 按值升序排序，返回索引数组
     */
    protected int[] argsort(float[] array) {
        Integer[] indices = new Integer[array.length];
        for (int i = 0; i < array.length; i++) {
            indices[i] = i;
        }
        java.util.Arrays.sort(indices, (a, b) -> Float.compare(array[a], array[b]));
        int[] result = new int[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = indices[i];
        }
        return result;
    }
    
    /**
     * Top-K过滤
     * 
     * @param probs 概率分布
     * @param k 保留前k个最高概率的token
     */
    protected void applyTopK(float[] probs, int k) {
        int[] topIndices = getTopKIndices(probs, k + 1); // +1 因为包含PAD位置
        boolean[] keep = new boolean[probs.length];
        for (int idx : topIndices) {
            if (idx != PAD_TOKEN_ID) {
                keep[idx] = true;
            }
        }
        
        for (int i = 1; i < probs.length; i++) {
            if (!keep[i]) {
                probs[i] = 0.0f;
            }
        }
        
        renormalizeProbs(probs);
    }
    
    /**
     * Top-P (Nucleus Sampling) 过滤
     * 
     * @param probs 概率分布
     * @param p 累积概率阈值
     */
    protected void applyTopP(float[] probs, float p) {
        int[] sortedIndices = argsortDescending(probs);
        
        float cumSum = 0.0f;
        int cutoff = sortedIndices.length;
        for (int i = 0; i < sortedIndices.length; i++) {
            int idx = sortedIndices[i];
            if (idx == PAD_TOKEN_ID) {
                continue;
            }
            cumSum += probs[idx];
            if (cumSum >= p) {
                cutoff = i + 1;
                break;
            }
        }
        
        // 将cutoff之后的token概率置零
        for (int i = cutoff; i < sortedIndices.length; i++) {
            int idx = sortedIndices[i];
            if (idx != PAD_TOKEN_ID) {
                probs[idx] = 0.0f;
            }
        }
        
        renormalizeProbs(probs);
    }
    
    /**
     * 重新归一化概率分布（跳过PAD token）
     */
    private void renormalizeProbs(float[] probs) {
        float sum = 0.0f;
        for (int i = 1; i < probs.length; i++) {
            sum += probs[i];
        }
        if (sum > 0.0f) {
            float invSum = 1.0f / sum;
            for (int i = 1; i < probs.length; i++) {
                probs[i] *= invSum;
            }
        }
    }
    
    /**
     * 按值降序排序，返回索引数组
     */
    private int[] argsortDescending(float[] array) {
        Integer[] indices = new Integer[array.length];
        for (int i = 0; i < array.length; i++) {
            indices[i] = i;
        }
        java.util.Arrays.sort(indices, (a, b) -> Float.compare(array[b], array[a]));
        int[] result = new int[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = indices[i];
        }
        return result;
    }
}
