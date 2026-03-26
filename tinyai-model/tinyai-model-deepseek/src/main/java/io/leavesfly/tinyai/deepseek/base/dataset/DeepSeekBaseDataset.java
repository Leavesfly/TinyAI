package io.leavesfly.tinyai.deepseek.base.dataset;

import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * DeepSeek 系列数据集基类
 * 
 * 提供通用的数据加载、批次管理、序列处理功能。
 * R1 和 V3 的数据集都继承此基类，只需实现特定的批次数据结构。
 * 
 * @author leavesfly
 * @version 1.0
 */
public abstract class DeepSeekBaseDataset<T> {
    
    // ========== 通用字段 ==========
    protected final List<int[]> sequences;      // 完整序列
    protected final int maxSeqLength;           // 最大序列长度
    protected final int batchSize;              // 批次大小
    protected final boolean shuffle;            // 是否打乱数据
    
    protected int currentIndex;                 // 当前批次索引
    protected List<Integer> indices;            // 样本索引列表
    
    /**
     * 构造函数
     * 
     * @param sequences token序列列表
     * @param maxSeqLength 最大序列长度
     * @param batchSize 批次大小
     * @param shuffle 是否打乱数据
     */
    public DeepSeekBaseDataset(List<int[]> sequences, int maxSeqLength, 
                               int batchSize, boolean shuffle) {
        this.sequences = sequences;
        this.maxSeqLength = maxSeqLength;
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.currentIndex = 0;
        initIndices();
    }
    
    /**
     * 初始化索引
     */
    private void initIndices() {
        indices = new ArrayList<>();
        for (int i = 0; i < sequences.size(); i++) {
            indices.add(i);
        }
    }
    
    /**
     * 准备数据集（打乱或重置）
     * 
     * @param shouldShuffle 是否打乱
     */
    public void prepare(boolean shouldShuffle) {
        if (shouldShuffle && shuffle) {
            Collections.shuffle(indices, new Random());
        }
        currentIndex = 0;
    }
    
    /**
     * 是否还有下一批数据
     */
    public boolean hasNext() {
        return currentIndex < sequences.size();
    }
    
    /**
     * 获取下一批数据（核心方法，子类实现）
     * 
     * @return 批次数据
     */
    public abstract T nextBatch();
    
    /**
     * 创建输入和目标数据（通用逻辑）
     * 
     * 每个 sequence 的长度应为 maxSeqLength+1，其中：
     * - input: sequence[0 .. maxSeqLength-1]  (前 maxSeqLength 个token)
     * - target: sequence[1 .. maxSeqLength]   (后 maxSeqLength 个token，右移一位)
     * 
     * 如果 sequence 长度不足 maxSeqLength+1，则有效部分按上述规则填充，
     * 剩余位置补 0（padding）。
     * 
     * @param actualBatchSize 实际批次大小
     * @return [inputData, targetData]，每个维度为 [actualBatchSize][maxSeqLength]
     */
    protected float[][][] createInputTargetData(int actualBatchSize) {
        float[][] inputData = new float[actualBatchSize][maxSeqLength];
        float[][] targetData = new float[actualBatchSize][maxSeqLength];
        
        for (int i = 0; i < actualBatchSize; i++) {
            int dataIndex = indices.get(currentIndex + i);
            int[] sequence = sequences.get(dataIndex);
            
            // 有效长度：sequence 中可用于构建 input/target 对的 token 数
            // 序列至少需要2个token才能构成 input/target 对
            if (sequence.length < 2) {
                continue;
            }
            int validLen = Math.min(sequence.length - 1, maxSeqLength);
            
            // 输入: sequence[0 .. validLen-1]
            for (int j = 0; j < validLen; j++) {
                inputData[i][j] = sequence[j];
            }
            
            // 目标: sequence[1 .. validLen] (右移一位)
            for (int j = 0; j < validLen; j++) {
                targetData[i][j] = sequence[j + 1];
            }
            
            // 剩余位置已由数组初始化为 0（padding）
        }
        
        return new float[][][] { inputData, targetData };
    }
    
    /**
     * 获取当前批次的数据索引
     * 
     * @param actualBatchSize 实际批次大小
     * @return 数据索引列表
     */
    protected List<Integer> getCurrentBatchIndices(int actualBatchSize) {
        List<Integer> batchIndices = new ArrayList<>();
        for (int i = 0; i < actualBatchSize; i++) {
            batchIndices.add(indices.get(currentIndex + i));
        }
        return batchIndices;
    }
    
    /**
     * 推进当前索引
     * 
     * @param endIndex 结束索引
     */
    protected void advanceIndex(int endIndex) {
        currentIndex = endIndex;
    }
    
    /**
     * 计算实际批次大小
     * 
     * @return 实际批次大小
     */
    protected int calculateActualBatchSize() {
        int endIndex = Math.min(currentIndex + batchSize, sequences.size());
        return endIndex - currentIndex;
    }
    
    /**
     * 重置数据集
     */
    public void reset() {
        currentIndex = 0;
    }
    
    /**
     * 获取样本数量
     */
    public int getSampleCount() {
        return sequences.size();
    }
    
    /**
     * 获取批次数量
     */
    public int getBatchCount() {
        return (sequences.size() + batchSize - 1) / batchSize;
    }
    
    /**
     * 获取所有序列数据
     */
    public List<int[]> getSequences() {
        return sequences;
    }
    
    /**
     * 获取最大序列长度
     */
    public int getMaxSeqLength() {
        return maxSeqLength;
    }
    
    /**
     * 获取批次大小
     */
    public int getBatchSize() {
        return batchSize;
    }
    
    /**
     * 批次数据基类
     */
    public static class BaseBatch {
        protected final NdArray inputIds;
        protected final NdArray targetIds;
        
        public BaseBatch(NdArray inputIds, NdArray targetIds) {
            this.inputIds = inputIds;
            this.targetIds = targetIds;
        }
        
        public NdArray getInputIds() {
            return inputIds;
        }
        
        public NdArray getTargetIds() {
            return targetIds;
        }
    }
}
