package io.leavesfly.tinyai.rl;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 经验回放缓冲区
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * ReplayBuffer类实现了经验回放机制，用于存储和采样历史经验数据。
 * 经验回放是深度强化学习中的重要技术，能够打破数据相关性，提高学习稳定性。
 * 支持固定大小的循环缓冲区，当容量满时会覆盖最旧的经验。
 */
public class ReplayBuffer {
    
    /**
     * 缓冲区最大容量
     */
    private final int capacity;
    
    /**
     * 存储经验的列表
     */
    private final List<Experience> buffer;
    
    /**
     * 当前写入位置
     */
    private int position;
    
    /**
     * 随机数生成器
     */
    private final Random random;
    
    /**
     * 构造函数
     * 
     * @param capacity 缓冲区最大容量
     */
    public ReplayBuffer(int capacity) {
        this.capacity = capacity;
        this.buffer = new ArrayList<>(capacity);
        this.position = 0;
        this.random = new Random();
    }
    
    /**
     * 添加经验到缓冲区
     * 
     * @param experience 要添加的经验
     */
    public void push(Experience experience) {
        if (buffer.size() < capacity) {
            // 缓冲区未满，直接添加
            buffer.add(experience);
        } else {
            // 缓冲区已满，覆盖旧经验
            buffer.set(position, experience);
        }
        position = (position + 1) % capacity;
    }
    
    /**
     * 随机采样一批经验
     * 
     * @param batchSize 批次大小
     * @return 采样的经验数组
     */
    public Experience[] sample(int batchSize) {
        if (batchSize > buffer.size()) {
            throw new IllegalArgumentException(
                String.format("批次大小 %d 大于缓冲区当前大小 %d", batchSize, buffer.size())
            );
        }
        
        Experience[] batch = new Experience[batchSize];
        for (int i = 0; i < batchSize; i++) {
            int randomIndex = random.nextInt(buffer.size());
            batch[i] = buffer.get(randomIndex);
        }
        
        return batch;
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
        position = 0;
    }
    
    /**
     * 获取缓冲区使用率
     * 
     * @return 使用率（0.0 - 1.0）
     */
    public float getUsageRate() {
        return (float) buffer.size() / capacity;
    }
    
    /**
     * 获取最近添加的经验
     * 
     * @param count 获取的数量
     * @return 最近的经验列表
     */
    public List<Experience> getRecent(int count) {
        if (count > buffer.size()) {
            count = buffer.size();
        }
        
        List<Experience> recent = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            int index = (position - 1 - i + capacity) % capacity;
            if (index < buffer.size()) {
                recent.add(buffer.get(index));
            }
        }
        
        return recent;
    }
    
    @Override
    public String toString() {
        return String.format("ReplayBuffer{size=%d/%d, usage=%.2f%%}", 
                           buffer.size(), capacity, getUsageRate() * 100);
    }
}