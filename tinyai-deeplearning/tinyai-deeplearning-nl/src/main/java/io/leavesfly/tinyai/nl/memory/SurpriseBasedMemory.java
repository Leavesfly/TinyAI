package io.leavesfly.tinyai.nl.memory;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nl.core.AssociativeMemory;

import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;

/**
 * 基于惊异度的记忆系统（SurpriseBasedMemory）
 * 使用惊异度驱动的记忆管理策略
 * 
 * <p>该系统根据输入数据的惊异度动态调整记忆存储策略：
 * - 高惊异度：优先存储，长期保留
 * - 低惊异度：选择性存储或丢弃
 * 这模拟了生物记忆系统对新奇事件的优先编码机制。</p>
 * 
 * @author TinyAI Team
 */
public class SurpriseBasedMemory {
    
    /**
     * 记忆条目类
     * 封装单个记忆及其元数据
     */
    private static class MemoryEntry implements Comparable<MemoryEntry> {
        Variable key;
        Variable value;
        float surpriseScore;
        long timestamp;
        int accessCount;
        
        MemoryEntry(Variable key, Variable value, float surpriseScore, long timestamp) {
            this.key = key;
            this.value = value;
            this.surpriseScore = surpriseScore;
            this.timestamp = timestamp;
            this.accessCount = 0;
        }
        
        @Override
        public int compareTo(MemoryEntry other) {
            // 按惊异度降序排列（高惊异度优先）
            return Float.compare(other.surpriseScore, this.surpriseScore);
        }
    }
    
    /**
     * 底层关联记忆（用于相似度计算和检索）
     */
    private AssociativeMemory memory;
    
    /**
     * 记忆条目的优先队列（按惊异度排序）
     */
    private PriorityQueue<MemoryEntry> priorityQueue;
    
    /**
     * 记忆条目列表（与底层 memory 保持同步）
     */
    private List<MemoryEntry> entryList;
    
    /**
     * 惊异度阈值
     * 只有超过阈值的记忆才会被存储
     */
    private float surpriseThreshold;
    
    /**
     * 最大记忆容量
     */
    private int maxCapacity;
    
    /**
     * 当前记忆索引
     */
    private int currentIndex;
    
    /**
     * 惊异度衰减率
     * 随着时间推移，惊异度会衰减
     */
    private float decayRate;
    
    /**
     * 是否启用惊异度衰减
     */
    private boolean enableDecay;
    
    /**
     * 构造函数
     * 
     * @param maxCapacity 最大容量
     * @param surpriseThreshold 惊异度阈值
     * @param decayRate 衰减率
     */
    public SurpriseBasedMemory(int maxCapacity, float surpriseThreshold, float decayRate) {
        this.maxCapacity = Math.max(1, maxCapacity);
        this.surpriseThreshold = Math.max(0.0f, Math.min(1.0f, surpriseThreshold));
        this.decayRate = Math.max(0.0f, Math.min(1.0f, decayRate));
        this.memory = new AssociativeMemory(maxCapacity, surpriseThreshold);
        this.priorityQueue = new PriorityQueue<>();
        this.entryList = new ArrayList<>();
        this.currentIndex = 0;
        this.enableDecay = true;
    }
    
    /**
     * 简化构造函数
     * 
     * @param maxCapacity 最大容量
     */
    public SurpriseBasedMemory(int maxCapacity) {
        this(maxCapacity, 0.3f, 0.001f);
    }
    
    /**
     * 存储记忆（自动计算惊异度）
     * 
     * @param key 记忆键
     * @param value 记忆值
     */
    public void store(Variable key, Variable value) {
        if (key == null || value == null) {
            return;
        }
        
        // 计算惊异度
        float surprise = memory.computeSurprise(key);
        
        // 只存储超过阈值的记忆
        if (surprise >= surpriseThreshold) {
            storeWithSurprise(key, value, surprise);
        }
    }
    
    /**
     * 存储记忆（指定惊异度）
     * 
     * @param key 记忆键
     * @param value 记忆值
     * @param surprise 惊异度分数
     */
    public void storeWithSurprise(Variable key, Variable value, float surprise) {
        if (key == null || value == null) {
            return;
        }
        
        long timestamp = System.currentTimeMillis();
        MemoryEntry entry = new MemoryEntry(key, value, surprise, timestamp);
        
        if (entryList.size() >= maxCapacity) {
            // 容量已满，找到最低惊异度的条目
            int lowestIdx = findLowestSurpriseIndex();
            if (lowestIdx >= 0 && surprise > entryList.get(lowestIdx).surpriseScore) {
                // 移除最低惊异度的条目
                MemoryEntry removed = entryList.remove(lowestIdx);
                priorityQueue.remove(removed);
                
                // 添加新条目
                entryList.add(entry);
                priorityQueue.offer(entry);
                
                // 重建底层 memory 以保持同步
                rebuildMemory();
            }
        } else {
            entryList.add(entry);
            priorityQueue.offer(entry);
            memory.store(key, value);
            currentIndex++;
        }
    }
    
    /**
     * 找到最低惊异度的条目索引
     */
    private int findLowestSurpriseIndex() {
        if (entryList.isEmpty()) {
            return -1;
        }
        int lowestIdx = 0;
        float lowestScore = entryList.get(0).surpriseScore;
        for (int i = 1; i < entryList.size(); i++) {
            if (entryList.get(i).surpriseScore < lowestScore) {
                lowestScore = entryList.get(i).surpriseScore;
                lowestIdx = i;
            }
        }
        return lowestIdx;
    }
    
    /**
     * 重建底层 AssociativeMemory 以保持与 entryList 同步
     */
    private void rebuildMemory() {
        memory.clear();
        for (MemoryEntry entry : entryList) {
            memory.store(entry.key, entry.value);
        }
    }
    
    /**
     * 检索记忆
     * 
     * @param queryKey 查询键
     * @return 检索到的记忆值
     */
    public Variable retrieve(Variable queryKey) {
        if (queryKey == null) {
            return null;
        }
        
        // 使用底层记忆检索
        Variable result = memory.retrieve(queryKey);
        
        // 更新访问计数（简化实现）
        // 实际应该找到对应的条目并更新
        
        return result;
    }
    
    /**
     * 计算输入的惊异度
     * 
     * @param input 输入数据
     * @return 惊异度分数
     */
    public float computeSurprise(Variable input) {
        return memory.computeSurprise(input);
    }
    
    /**
     * 应用惊异度衰减
     * 随着时间推移降低记忆的惊异度
     */
    public void applyDecay() {
        if (!enableDecay || entryList.isEmpty()) {
            return;
        }
        
        // 应用衰减并移除低于阈值的记忆
        PriorityQueue<MemoryEntry> newQueue = new PriorityQueue<>();
        List<MemoryEntry> newEntryList = new ArrayList<>();
        
        for (MemoryEntry entry : entryList) {
            entry.surpriseScore *= (1.0f - decayRate);
            if (entry.surpriseScore >= surpriseThreshold) {
                newQueue.offer(entry);
                newEntryList.add(entry);
            }
        }
        
        this.priorityQueue = newQueue;
        this.entryList = newEntryList;
        
        // 同步底层 memory
        rebuildMemory();
    }
    
    /**
     * 根据访问频率调整惊异度
     * 频繁访问的记忆惊异度会增加
     * 
     * @param boostFactor 增强因子
     */
    public void boostFrequentMemories(float boostFactor) {
        PriorityQueue<MemoryEntry> newQueue = new PriorityQueue<>();
        
        for (MemoryEntry entry : entryList) {
            float boost = 1.0f + (entry.accessCount * boostFactor);
            entry.surpriseScore = Math.min(1.0f, entry.surpriseScore * boost);
            newQueue.offer(entry);
        }
        
        this.priorityQueue = newQueue;
    }
    
    /**
     * 获取最惊异的记忆
     * 
     * @param topK 返回前K个
     * @return 记忆条目列表
     */
    public MemoryEntry[] getTopSurprisingMemories(int topK) {
        int count = Math.min(topK, priorityQueue.size());
        MemoryEntry[] result = new MemoryEntry[count];
        
        // 临时提取并保存
        MemoryEntry[] temp = new MemoryEntry[count];
        for (int i = 0; i < count; i++) {
            temp[i] = priorityQueue.poll();
            result[i] = temp[i];
        }
        
        // 恢复队列
        for (MemoryEntry entry : temp) {
            priorityQueue.offer(entry);
        }
        
        return result;
    }
    
    /**
     * 清空所有记忆
     */
    public void clear() {
        memory.clear();
        priorityQueue.clear();
        entryList.clear();
        currentIndex = 0;
    }
    
    /**
     * 获取当前记忆数量
     * 
     * @return 记忆数量
     */
    public int getSize() {
        return priorityQueue.size();
    }
    
    /**
     * 获取平均惊异度
     * 
     * @return 平均惊异度
     */
    public float getAverageSurprise() {
        if (priorityQueue.isEmpty()) {
            return 0.0f;
        }
        
        float total = 0.0f;
        for (MemoryEntry entry : priorityQueue) {
            total += entry.surpriseScore;
        }
        
        return total / priorityQueue.size();
    }
    
    /**
     * 获取统计信息
     * 
     * @return 统计信息字符串
     */
    public String getStatistics() {
        StringBuilder sb = new StringBuilder();
        sb.append("惊异度记忆系统统计:\n");
        sb.append(String.format("  总记忆数: %d/%d\n", getSize(), maxCapacity));
        sb.append(String.format("  平均惊异度: %.3f\n", getAverageSurprise()));
        sb.append(String.format("  惊异度阈值: %.3f\n", surpriseThreshold));
        sb.append(String.format("  衰减率: %.4f\n", decayRate));
        
        return sb.toString();
    }
    
    // Getters and Setters
    
    public float getSurpriseThreshold() {
        return surpriseThreshold;
    }
    
    public void setSurpriseThreshold(float surpriseThreshold) {
        this.surpriseThreshold = Math.max(0.0f, Math.min(1.0f, surpriseThreshold));
        memory.setSurpriseThreshold(surpriseThreshold);
    }
    
    public float getDecayRate() {
        return decayRate;
    }
    
    public void setDecayRate(float decayRate) {
        this.decayRate = Math.max(0.0f, Math.min(1.0f, decayRate));
    }
    
    public boolean isEnableDecay() {
        return enableDecay;
    }
    
    public void setEnableDecay(boolean enableDecay) {
        this.enableDecay = enableDecay;
    }
    
    public int getMaxCapacity() {
        return maxCapacity;
    }
    
    public AssociativeMemory getMemory() {
        return memory;
    }
}
