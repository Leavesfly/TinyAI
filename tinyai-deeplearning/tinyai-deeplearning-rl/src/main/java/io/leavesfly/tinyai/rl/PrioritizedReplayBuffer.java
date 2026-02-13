package io.leavesfly.tinyai.rl;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 优先经验回放缓冲区 (Prioritized Experience Replay)
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * 基于TD误差优先级进行采样的经验回放缓冲区。
 * 优先级高的经验被采样的概率更大，提高学习效率。
 * 使用Sum Tree数据结构高效实现优先级采样。
 * 
 * 参考论文: "Prioritized Experience Replay" (Schaul et al., 2016)
 */
public class PrioritizedReplayBuffer {
    
    /**
     * 缓冲区最大容量
     */
    private final int capacity;
    
    /**
     * 存储经验的列表
     */
    private final List<Experience> buffer;
    
    /**
     * 存储优先级的列表
     */
    private final List<Float> priorities;
    
    /**
     * 当前写入位置
     */
    private int position;
    
    /**
     * 随机数生成器
     */
    private final Random random;
    
    /**
     * 优先级指数 (alpha)，控制优先级程度
     * alpha=0: 均匀采样, alpha=1: 完全按优先级采样
     */
    private final float alpha;
    
    /**
     * 重要性采样权重指数 (beta)
     * beta=0: 无修正, beta=1: 完全修正
     */
    private float beta;
    
    /**
     * beta的增量，用于逐渐增加到1
     */
    private final float betaIncrement;
    
    /**
     * 最小优先级，防止零优先级
     */
    private static final float MIN_PRIORITY = 1e-6f;
    
    /**
     * 最大优先级记录
     */
    private float maxPriority;
    
    /**
     * 构造函数
     * 
     * @param capacity 缓冲区最大容量
     * @param alpha 优先级指数 (默认0.6)
     * @param beta 重要性采样权重指数初始值 (默认0.4)
     * @param betaIncrement beta增量 (默认0.001)
     */
    public PrioritizedReplayBuffer(int capacity, float alpha, float beta, float betaIncrement) {
        this.capacity = capacity;
        this.buffer = new ArrayList<>(capacity);
        this.priorities = new ArrayList<>(capacity);
        this.position = 0;
        this.random = new Random();
        this.alpha = alpha;
        this.beta = beta;
        this.betaIncrement = betaIncrement;
        this.maxPriority = 1.0f;
    }
    
    /**
     * 构造函数（使用默认参数）
     * 
     * @param capacity 缓冲区最大容量
     */
    public PrioritizedReplayBuffer(int capacity) {
        this(capacity, 0.6f, 0.4f, 0.001f);
    }
    
    /**
     * 添加经验到缓冲区
     * 
     * @param experience 要添加的经验
     */
    public void push(Experience experience) {
        // 新经验使用当前最大优先级，确保被采样
        float priority = maxPriority;
        
        if (buffer.size() < capacity) {
            buffer.add(experience);
            priorities.add(priority);
        } else {
            buffer.set(position, experience);
            priorities.set(position, priority);
        }
        position = (position + 1) % capacity;
    }
    
    /**
     * 基于优先级采样一批经验
     * 
     * @param batchSize 批次大小
     * @return 采样的经验数组
     */
    public PrioritizedSample sample(int batchSize) {
        if (batchSize > buffer.size()) {
            throw new IllegalArgumentException(
                String.format("批次大小 %d 大于缓冲区当前大小 %d", batchSize, buffer.size())
            );
        }
        
        // 计算优先级总和
        float prioritySum = 0.0f;
        for (float p : priorities) {
            prioritySum += (float) Math.pow(p + MIN_PRIORITY, alpha);
        }
        
        // 分段采样
        Experience[] batch = new Experience[batchSize];
        int[] indices = new int[batchSize];
        float[] weights = new float[batchSize];
        
        float segmentSize = prioritySum / batchSize;
        float minProbability = Float.MAX_VALUE;
        
        // 计算最小概率（用于权重归一化）
        for (float p : priorities) {
            float prob = (float) Math.pow(p + MIN_PRIORITY, alpha) / prioritySum;
            minProbability = Math.min(minProbability, prob);
        }
        
        // 最大权重（用于归一化）
        float maxWeight = (float) Math.pow(minProbability * buffer.size(), -beta);
        
        for (int i = 0; i < batchSize; i++) {
            float segmentStart = segmentSize * i;
            float segmentEnd = segmentSize * (i + 1);
            float sampleValue = segmentStart + random.nextFloat() * (segmentEnd - segmentStart);
            
            // 根据采样值选择经验
            int index = sampleFromPriority(sampleValue, prioritySum);
            indices[i] = index;
            batch[i] = buffer.get(index);
            
            // 计算重要性采样权重
            float probability = (float) Math.pow(priorities.get(index) + MIN_PRIORITY, alpha) / prioritySum;
            weights[i] = (float) Math.pow(probability * buffer.size(), -beta) / maxWeight;
        }
        
        // 逐渐增加beta
        beta = Math.min(1.0f, beta + betaIncrement);
        
        return new PrioritizedSample(batch, indices, weights);
    }
    
    /**
     * 根据采样值从优先级分布中选择索引
     */
    private int sampleFromPriority(float sampleValue, float prioritySum) {
        float cumulative = 0.0f;
        for (int i = 0; i < priorities.size(); i++) {
            cumulative += (float) Math.pow(priorities.get(i) + MIN_PRIORITY, alpha);
            if (cumulative >= sampleValue) {
                return i;
            }
        }
        return priorities.size() - 1;
    }
    
    /**
     * 更新采样经验的优先级
     * 
     * @param indices 经验索引
     * @param tdErrors TD误差
     */
    public void updatePriorities(int[] indices, float[] tdErrors) {
        for (int i = 0; i < indices.length; i++) {
            int idx = indices[i];
            if (idx >= 0 && idx < priorities.size()) {
                // 优先级 = |TD误差| + epsilon
                float priority = Math.abs(tdErrors[i]) + MIN_PRIORITY;
                priorities.set(idx, priority);
                maxPriority = Math.max(maxPriority, priority);
            }
        }
    }
    
    /**
     * 检查缓冲区是否可以进行采样
     * 
     * @param batchSize 批次大小
     * @return 是否可以采样
     */
    public boolean canSample(int batchSize) {
        return buffer.size() >= batchSize;
    }
    
    /**
     * 获取缓冲区当前大小
     * 
     * @return 当前存储的经验数量
     */
    public int size() {
        return buffer.size();
    }
    
    /**
     * 获取缓冲区最大容量
     * 
     * @return 最大容量
     */
    public int getCapacity() {
        return capacity;
    }
    
    /**
     * 检查缓冲区是否为空
     * 
     * @return 是否为空
     */
    public boolean isEmpty() {
        return buffer.isEmpty();
    }
    
    /**
     * 检查缓冲区是否已满
     * 
     * @return 是否已满
     */
    public boolean isFull() {
        return buffer.size() >= capacity;
    }
    
    /**
     * 清空缓冲区
     */
    public void clear() {
        buffer.clear();
        priorities.clear();
        position = 0;
        maxPriority = 1.0f;
    }
    
    /**
     * 获取当前beta值
     * 
     * @return beta值
     */
    public float getBeta() {
        return beta;
    }
    
    /**
     * 获取缓冲区使用率
     * 
     * @return 使用率（0.0 - 1.0）
     */
    public float getUsageRate() {
        return (float) buffer.size() / capacity;
    }
    
    @Override
    public String toString() {
        return String.format("PrioritizedReplayBuffer{size=%d/%d, usage=%.2f%%, alpha=%.2f, beta=%.2f}", 
                           buffer.size(), capacity, getUsageRate() * 100, alpha, beta);
    }
    
    /**
     * 优先采样结果类
     */
    public static class PrioritizedSample {
        private final Experience[] experiences;
        private final int[] indices;
        private final float[] weights;
        
        public PrioritizedSample(Experience[] experiences, int[] indices, float[] weights) {
            this.experiences = experiences;
            this.indices = indices;
            this.weights = weights;
        }
        
        public Experience[] getExperiences() {
            return experiences;
        }
        
        public int[] getIndices() {
            return indices;
        }
        
        public float[] getWeights() {
            return weights;
        }
    }
}
