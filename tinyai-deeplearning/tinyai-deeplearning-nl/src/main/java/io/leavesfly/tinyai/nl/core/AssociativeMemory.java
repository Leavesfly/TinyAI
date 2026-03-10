package io.leavesfly.tinyai.nl.core;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 关联记忆（AssociativeMemory）
 * 实现关联记忆模型，将输入映射到输出
 * 
 * <p>关联记忆是嵌入学习的核心概念之一。它通过键值对存储模式，
 * 使用注意力机制进行检索，并基于惊异度管理记忆优先级。</p>
 * 
 * @author TinyAI Team
 */
public class AssociativeMemory {
    
    /**
     * 键矩阵存储（存储所有的键）
     */
    private List<Variable> keys;
    
    /**
     * 值矩阵存储（对应的值）
     */
    private List<Variable> values;
    
    /**
     * 惊异度分数存储
     */
    private Map<Integer, Float> surpriseScores;
    
    /**
     * 记忆容量
     */
    private int memorySize;
    
    /**
     * 当前存储的记忆数量
     */
    private int currentSize;
    
    /**
     * 惊异度阈值，用于记忆修剪
     */
    private float surpriseThreshold;
    
    /**
     * 构造函数
     * 
     * @param memorySize 记忆容量
     * @param surpriseThreshold 惊异度阈值
     */
    public AssociativeMemory(int memorySize, float surpriseThreshold) {
        this.memorySize = Math.max(1, memorySize);
        this.surpriseThreshold = surpriseThreshold;
        this.keys = new ArrayList<>();
        this.values = new ArrayList<>();
        this.surpriseScores = new HashMap<>();
        this.currentSize = 0;
    }
    
    /**
     * 简化构造函数，使用默认阈值
     * 
     * @param memorySize 记忆容量
     */
    public AssociativeMemory(int memorySize) {
        this(memorySize, 0.5f);
    }
    
    /**
     * 存储关联记忆
     * 
     * @param key 键
     * @param value 值
     */
    public void store(Variable key, Variable value) {
        if (key == null || value == null) {
            return;
        }
        
        // 计算惊异度
        float surprise = computeSurprise(key);
        
        // 如果容量已满，需要考虑是否替换
        if (currentSize >= memorySize) {
            // 找到最低惊异度的记忆
            int lowestIndex = findLowestSurpriseIndex();
            float lowestSurprise = surpriseScores.getOrDefault(lowestIndex, 0.0f);
            
            // 如果新记忆的惊异度更高，替换旧记忆
            if (surprise > lowestSurprise) {
                keys.set(lowestIndex, key);
                values.set(lowestIndex, value);
                surpriseScores.put(lowestIndex, surprise);
            }
        } else {
            // 还有空间，直接添加
            keys.add(key);
            values.add(value);
            surpriseScores.put(currentSize, surprise);
            currentSize++;
        }
    }
    
    /**
     * 根据键检索值
     * 使用 softmax 加权的注意力机制进行检索
     * 
     * @param queryKey 查询键
     * @return 加权检索到的值
     */
    public Variable retrieve(Variable queryKey) {
        if (queryKey == null || currentSize == 0) {
            return null;
        }
        
        // 计算查询键与所有存储键的相似度
        float[] similarities = new float[currentSize];
        for (int i = 0; i < currentSize; i++) {
            similarities[i] = computeSimilarity(queryKey, keys.get(i));
        }
        
        // 对相似度应用 softmax 归一化得到注意力权重
        float[] weights = softmax(similarities);
        
        // 使用注意力权重对所有值进行加权求和
        Variable result = null;
        for (int i = 0; i < currentSize; i++) {
            Variable weightedValue = values.get(i).mul(new Variable(weights[i]));
            if (result == null) {
                result = weightedValue;
            } else {
                result = result.add(weightedValue);
            }
        }
        
        return result;
    }
    
    /**
     * 对浮点数组应用 softmax 归一化
     * 
     * @param scores 原始分数数组
     * @return 归一化后的概率分布
     */
    private float[] softmax(float[] scores) {
        float[] result = new float[scores.length];
        
        // 数值稳定性：减去最大值
        float maxScore = Float.NEGATIVE_INFINITY;
        for (float score : scores) {
            maxScore = Math.max(maxScore, score);
        }
        
        float sumExp = 0.0f;
        for (int i = 0; i < scores.length; i++) {
            result[i] = (float) Math.exp(scores[i] - maxScore);
            sumExp += result[i];
        }
        
        // 归一化
        if (sumExp > 0) {
            for (int i = 0; i < result.length; i++) {
                result[i] /= sumExp;
            }
        }
        
        return result;
    }
    
    /**
     * 计算输入的惊异度
     * 惊异度通过预测误差衡量
     * 
     * @param input 输入数据
     * @return 惊异度分数
     */
    public float computeSurprise(Variable input) {
        if (input == null || currentSize == 0) {
            return 1.0f; // 新数据，高惊异度
        }
        
        // 找到最相似的键
        float maxSimilarity = Float.NEGATIVE_INFINITY;
        
        for (int i = 0; i < currentSize; i++) {
            Variable key = keys.get(i);
            float similarity = computeSimilarity(input, key);
            maxSimilarity = Math.max(maxSimilarity, similarity);
        }
        
        // 惊异度 = 1 - 最大相似度
        // 相似度越高，惊异度越低
        return 1.0f - Math.max(0.0f, Math.min(1.0f, maxSimilarity));
    }
    
    /**
     * 根据惊异度修剪记忆
     * 移除低于阈值的记忆
     * 
     * @param threshold 阈值
     */
    public void prune(float threshold) {
        List<Variable> newKeys = new ArrayList<>();
        List<Variable> newValues = new ArrayList<>();
        Map<Integer, Float> newScores = new HashMap<>();
        
        int newIndex = 0;
        for (int i = 0; i < currentSize; i++) {
            float score = surpriseScores.getOrDefault(i, 0.0f);
            
            // 保留高于阈值的记忆
            if (score >= threshold) {
                newKeys.add(keys.get(i));
                newValues.add(values.get(i));
                newScores.put(newIndex, score);
                newIndex++;
            }
        }
        
        // 更新记忆
        this.keys = newKeys;
        this.values = newValues;
        this.surpriseScores = newScores;
        this.currentSize = newIndex;
    }
    
    /**
     * 计算两个变量之间的余弦相似度
     * 
     * @param v1 变量1
     * @param v2 变量2
     * @return 相似度分数 [-1, 1]
     */
    private float computeSimilarity(Variable v1, Variable v2) {
        if (v1 == null || v2 == null) {
            return 0.0f;
        }
        
        NdArray data1 = v1.getValue();
        NdArray data2 = v2.getValue();
        
        if (data1 == null || data2 == null) {
            return 0.0f;
        }
        
        try {
            // 将数组展平并计算余弦相似度
            // NdArray.flatten() 返回 Shape(1, N) 的 2D 数组
            NdArray flat1 = data1.flatten();
            NdArray flat2 = data2.flatten();
            
            int len1 = flat1.getShape().size();
            int len2 = flat2.getShape().size();
            int len = Math.min(len1, len2);
            
            // 判断 flatten 后的维度数来确定正确的索引方式
            int ndim1 = flat1.getShape().getDimNum();
            int ndim2 = flat2.getShape().getDimNum();
            
            float dotProduct = 0.0f;
            float norm1 = 0.0f;
            float norm2 = 0.0f;
            
            for (int i = 0; i < len; i++) {
                // 兼容 1D [N] 和 2D [1, N] 两种 flatten 结果
                int[] idx1 = (ndim1 == 1) ? new int[]{i} : new int[]{0, i};
                int[] idx2 = (ndim2 == 1) ? new int[]{i} : new int[]{0, i};
                
                float val1 = flat1.get(idx1);
                float val2 = flat2.get(idx2);
                dotProduct += val1 * val2;
                norm1 += val1 * val1;
                norm2 += val2 * val2;
            }
            
            // 余弦相似度
            if (norm1 > 0 && norm2 > 0) {
                return dotProduct / (float) (Math.sqrt(norm1) * Math.sqrt(norm2));
            }
            return 0.0f;
        } catch (Exception e) {
            return 0.0f;
        }
    }
    
    /**
     * 找到最低惊异度的记忆索引
     * 
     * @return 索引
     */
    private int findLowestSurpriseIndex() {
        int lowestIndex = 0;
        float lowestScore = Float.POSITIVE_INFINITY;
        
        for (Map.Entry<Integer, Float> entry : surpriseScores.entrySet()) {
            if (entry.getValue() < lowestScore) {
                lowestScore = entry.getValue();
                lowestIndex = entry.getKey();
            }
        }
        
        return lowestIndex;
    }
    
    /**
     * 清空所有记忆
     */
    public void clear() {
        keys.clear();
        values.clear();
        surpriseScores.clear();
        currentSize = 0;
    }
    
    // Getters
    
    public int getMemorySize() {
        return memorySize;
    }
    
    public int getCurrentSize() {
        return currentSize;
    }
    
    public float getSurpriseThreshold() {
        return surpriseThreshold;
    }
    
    public void setSurpriseThreshold(float surpriseThreshold) {
        this.surpriseThreshold = surpriseThreshold;
    }
    
    public List<Variable> getKeys() {
        return new ArrayList<>(keys);
    }
    
    public List<Variable> getValues() {
        return new ArrayList<>(values);
    }
}
