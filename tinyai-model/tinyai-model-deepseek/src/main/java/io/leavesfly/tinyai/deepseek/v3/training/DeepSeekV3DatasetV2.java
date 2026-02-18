package io.leavesfly.tinyai.deepseek.v3.training;

import io.leavesfly.tinyai.deepseek.base.TaskType;
import io.leavesfly.tinyai.deepseek.base.dataset.DeepSeekBaseDataset;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.ArrayList;
import java.util.List;

/**
 * DeepSeek-V3数据集类（V2版本 - 基于共享基类）
 * 
 * 支持预训练、后训练两种模式的数据加载，
 * 特别支持任务类型标注，用于任务感知训练。
 * V3 特点：包含任务类型和代码语言标注。
 * 
 * @author leavesfly
 * @version 2.0
 */
public class DeepSeekV3DatasetV2 extends DeepSeekBaseDataset<DeepSeekV3DatasetV2.Batch> {
    
    // V3 特有字段
    private final List<TaskType> taskTypes;   // 任务类型（V3特有）
    private final List<String> codeLanguages; // 代码语言（代码任务专用）
    
    /**
     * 构造函数（预训练模式）
     */
    public DeepSeekV3DatasetV2(List<int[]> sequences, int maxSeqLength, 
                               int batchSize, boolean shuffle) {
        super(sequences, maxSeqLength, batchSize, shuffle);
        this.taskTypes = new ArrayList<>();
        this.codeLanguages = new ArrayList<>();
    }
    
    /**
     * 构造函数（任务感知模式）
     */
    public DeepSeekV3DatasetV2(List<int[]> sequences, List<TaskType> taskTypes,
                               int maxSeqLength, int batchSize, boolean shuffle) {
        super(sequences, maxSeqLength, batchSize, shuffle);
        this.taskTypes = taskTypes;
        this.codeLanguages = new ArrayList<>();
    }
    
    /**
     * 构造函数（代码任务专用）
     */
    public DeepSeekV3DatasetV2(List<int[]> sequences, List<TaskType> taskTypes,
                               List<String> codeLanguages, int maxSeqLength,
                               int batchSize, boolean shuffle) {
        super(sequences, maxSeqLength, batchSize, shuffle);
        this.taskTypes = taskTypes;
        this.codeLanguages = codeLanguages;
    }
    
    /**
     * 获取下一批数据（V3 特定实现）
     */
    @Override
    public Batch nextBatch() {
        int actualBatchSize = calculateActualBatchSize();
        int endIndex = currentIndex + actualBatchSize;
        
        // 创建输入和目标数据（复用基类方法）
        float[][][] data = createInputTargetData(actualBatchSize);
        float[][] inputData = data[0];
        float[][] targetData = data[1];
        
        // 准备任务类型和代码语言
        TaskType[] batchTaskTypes = new TaskType[actualBatchSize];
        String[] batchLanguages = new String[actualBatchSize];
        
        List<Integer> batchIndices = getCurrentBatchIndices(actualBatchSize);
        for (int i = 0; i < actualBatchSize; i++) {
            int dataIndex = batchIndices.get(i);
            
            // 任务类型
            if (!taskTypes.isEmpty() && dataIndex < taskTypes.size()) {
                batchTaskTypes[i] = taskTypes.get(dataIndex);
            } else {
                batchTaskTypes[i] = TaskType.GENERAL;
            }
            
            // 代码语言
            if (!codeLanguages.isEmpty() && dataIndex < codeLanguages.size()) {
                batchLanguages[i] = codeLanguages.get(dataIndex);
            }
        }
        
        // 推进索引
        advanceIndex(endIndex);
        
        NdArray inputIds = NdArray.of(inputData);
        NdArray targetIds = NdArray.of(targetData);
        
        return new Batch(inputIds, targetIds, batchTaskTypes, batchLanguages);
    }
    
    /**
     * V3 批次数据类
     */
    public static class Batch extends DeepSeekBaseDataset.BaseBatch {
        private final TaskType[] taskTypes;
        private final String[] codeLanguages;
        
        public Batch(NdArray inputIds, NdArray targetIds, 
                    TaskType[] taskTypes, String[] codeLanguages) {
            super(inputIds, targetIds);
            this.taskTypes = taskTypes;
            this.codeLanguages = codeLanguages;
        }
        
        public TaskType[] getTaskTypes() {
            return taskTypes;
        }
        
        public String[] getCodeLanguages() {
            return codeLanguages;
        }
    }
}
