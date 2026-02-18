package io.leavesfly.tinyai.deepseek.v3.training;

import io.leavesfly.tinyai.deepseek.base.TaskType;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * DeepSeek-V3数据集类
 * 
 * 支持预训练、后训练两种模式的数据加载,
 * 特别支持任务类型标注,用于任务感知训练
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3Dataset {
    
    private final List<int[]> sequences;      // 完整序列
    private final List<TaskType> taskTypes;   // 任务类型（V3特有）
    private final List<String> codeLanguages; // 代码语言（代码任务专用）
    private final int maxSeqLength;
    private final int batchSize;
    private final boolean shuffle;
    
    private int currentIndex;
    private List<Integer> indices;
    
    /**
     * 构造函数（预训练模式）
     * 
     * @param sequences token序列列表
     * @param maxSeqLength 最大序列长度
     * @param batchSize 批次大小
     * @param shuffle 是否打乱数据
     */
    public DeepSeekV3Dataset(List<int[]> sequences, int maxSeqLength, 
                             int batchSize, boolean shuffle) {
        this.sequences = sequences;
        this.taskTypes = new ArrayList<>();
        this.codeLanguages = new ArrayList<>();
        this.maxSeqLength = maxSeqLength;
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.currentIndex = 0;
        initIndices();
    }
    
    /**
     * 构造函数（任务感知模式）
     * 
     * @param sequences token序列列表
     * @param taskTypes 任务类型列表
     * @param maxSeqLength 最大序列长度
     * @param batchSize 批次大小
     * @param shuffle 是否打乱数据
     */
    public DeepSeekV3Dataset(List<int[]> sequences, List<TaskType> taskTypes,
                             int maxSeqLength, int batchSize, boolean shuffle) {
        this.sequences = sequences;
        this.taskTypes = taskTypes;
        this.codeLanguages = new ArrayList<>();
        this.maxSeqLength = maxSeqLength;
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.currentIndex = 0;
        initIndices();
    }
    
    /**
     * 构造函数（代码任务专用）
     * 
     * @param sequences token序列列表
     * @param taskTypes 任务类型列表
     * @param codeLanguages 代码语言列表
     * @param maxSeqLength 最大序列长度
     * @param batchSize 批次大小
     * @param shuffle 是否打乱数据
     */
    public DeepSeekV3Dataset(List<int[]> sequences, List<TaskType> taskTypes,
                             List<String> codeLanguages, int maxSeqLength,
                             int batchSize, boolean shuffle) {
        this.sequences = sequences;
        this.taskTypes = taskTypes;
        this.codeLanguages = codeLanguages;
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
     * 获取下一批数据
     * 
     * @return 批次数据
     */
    public Batch nextBatch() {
        int endIndex = Math.min(currentIndex + batchSize, sequences.size());
        int actualBatchSize = endIndex - currentIndex;
        
        // 准备输入和目标
        float[][] inputData = new float[actualBatchSize][maxSeqLength];
        float[][] targetData = new float[actualBatchSize][maxSeqLength];
        TaskType[] batchTaskTypes = new TaskType[actualBatchSize];
        String[] batchLanguages = new String[actualBatchSize];
        
        for (int i = 0; i < actualBatchSize; i++) {
            int dataIndex = indices.get(currentIndex + i);
            int[] sequence = sequences.get(dataIndex);
            
            // 填充或截断序列
            int seqLen = Math.min(sequence.length, maxSeqLength);
            
            // 输入：序列的前n-1个token
            for (int j = 0; j < seqLen - 1; j++) {
                inputData[i][j] = sequence[j];
            }
            
            // 目标：序列的后n-1个token（用于语言建模）
            for (int j = 1; j < seqLen; j++) {
                targetData[i][j - 1] = sequence[j];
            }
            
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
        
        currentIndex = endIndex;
        
        NdArray inputIds = NdArray.of(inputData);
        NdArray targetIds = NdArray.of(targetData);
        
        return new Batch(inputIds, targetIds, batchTaskTypes, batchLanguages);
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
     * 批次数据类
     */
    public static class Batch {
        private final NdArray inputIds;
        private final NdArray targetIds;
        private final TaskType[] taskTypes;
        private final String[] codeLanguages;
        
        public Batch(NdArray inputIds, NdArray targetIds, 
                    TaskType[] taskTypes, String[] codeLanguages) {
            this.inputIds = inputIds;
            this.targetIds = targetIds;
            this.taskTypes = taskTypes;
            this.codeLanguages = codeLanguages;
        }
        
        public NdArray getInputIds() {
            return inputIds;
        }
        
        public NdArray getTargetIds() {
            return targetIds;
        }
        
        public TaskType[] getTaskTypes() {
            return taskTypes;
        }
        
        public String[] getCodeLanguages() {
            return codeLanguages;
        }
        
        /**
         * 获取批次中主要的任务类型
         */
        public TaskType getMajorityTaskType() {
            if (taskTypes == null || taskTypes.length == 0) {
                return TaskType.GENERAL;
            }
            
            // 统计各任务类型出现次数
            int[] counts = new int[5];  // 5种任务类型
            for (TaskType type : taskTypes) {
                if (type != null) {
                    counts[type.getId()]++;
                }
            }
            
            // 找出最频繁的任务类型
            int maxCount = 0;
            int maxIdx = 0;
            for (int i = 0; i < counts.length; i++) {
                if (counts[i] > maxCount) {
                    maxCount = counts[i];
                    maxIdx = i;
                }
            }
            
            return TaskType.fromId(maxIdx);
        }
    }
    
    // ==================== 静态工厂方法 ====================
    
    /**
     * 创建虚拟预训练数据集（用于演示）
     */
    public static DeepSeekV3Dataset createDummyPretrainDataset(int numSamples, 
                                                                int seqLength, 
                                                                int batchSize) {
        List<int[]> sequences = new ArrayList<>();
        Random random = new Random(42);
        
        for (int i = 0; i < numSamples; i++) {
            int[] sequence = new int[seqLength];
            for (int j = 0; j < seqLength; j++) {
                sequence[j] = random.nextInt(1000);  // 假设词汇表大小1000
            }
            sequences.add(sequence);
        }
        
        return new DeepSeekV3Dataset(sequences, seqLength, batchSize, true);
    }
    
    /**
     * 创建虚拟后训练数据集（带任务类型）
     */
    public static DeepSeekV3Dataset createDummyPosttrainDataset(int numSamples,
                                                                 int seqLength,
                                                                 int batchSize) {
        List<int[]> sequences = new ArrayList<>();
        List<TaskType> taskTypes = new ArrayList<>();
        Random random = new Random(42);
        
        TaskType[] allTypes = TaskType.values();
        
        for (int i = 0; i < numSamples; i++) {
            int[] sequence = new int[seqLength];
            for (int j = 0; j < seqLength; j++) {
                sequence[j] = random.nextInt(1000);
            }
            sequences.add(sequence);
            
            // 随机分配任务类型
            TaskType taskType = allTypes[random.nextInt(allTypes.length)];
            taskTypes.add(taskType);
        }
        
        return new DeepSeekV3Dataset(sequences, taskTypes, seqLength, batchSize, true);
    }
    
    /**
     * 创建虚拟代码数据集
     */
    public static DeepSeekV3Dataset createDummyCodeDataset(int numSamples,
                                                            int seqLength,
                                                            int batchSize) {
        List<int[]> sequences = new ArrayList<>();
        List<TaskType> taskTypes = new ArrayList<>();
        List<String> languages = new ArrayList<>();
        Random random = new Random(42);
        
        String[] supportedLangs = {"Java", "Python", "JavaScript", "C++", "Go"};
        
        for (int i = 0; i < numSamples; i++) {
            int[] sequence = new int[seqLength];
            for (int j = 0; j < seqLength; j++) {
                sequence[j] = random.nextInt(1000);
            }
            sequences.add(sequence);
            taskTypes.add(TaskType.CODING);
            languages.add(supportedLangs[random.nextInt(supportedLangs.length)]);
        }
        
        return new DeepSeekV3Dataset(sequences, taskTypes, languages, seqLength, batchSize, true);
    }
    
    /**
     * 创建虚拟数据集（简化版）- 用于预训练和一般训练
     */
    public static DeepSeekV3Dataset createDummyDataset(int numSamples,
                                                        int seqLength,
                                                        int vocabSize,
                                                        int batchSize) {
        List<int[]> sequences = new ArrayList<>();
        List<TaskType> taskTypes = new ArrayList<>();
        Random random = new Random(42);
        
        TaskType[] allTypes = TaskType.values();
        
        for (int i = 0; i < numSamples; i++) {
            int[] sequence = new int[seqLength];
            for (int j = 0; j < seqLength; j++) {
                sequence[j] = random.nextInt(vocabSize);
            }
            sequences.add(sequence);
            taskTypes.add(allTypes[random.nextInt(allTypes.length)]);
        }
        
        return new DeepSeekV3Dataset(sequences, taskTypes, seqLength, batchSize, true);
    }
    
    /**
     * 创建代码数据集（带语言指定）
     */
    public static DeepSeekV3Dataset createCodeDataset(int numSamples,
                                                       int seqLength,
                                                       int vocabSize,
                                                       int batchSize,
                                                       String[] languages) {
        List<int[]> sequences = new ArrayList<>();
        List<TaskType> taskTypes = new ArrayList<>();
        List<String> langs = new ArrayList<>();
        Random random = new Random(42);
        
        for (int i = 0; i < numSamples; i++) {
            int[] sequence = new int[seqLength];
            for (int j = 0; j < seqLength; j++) {
                sequence[j] = random.nextInt(vocabSize);
            }
            sequences.add(sequence);
            taskTypes.add(TaskType.CODING);
            langs.add(languages[random.nextInt(languages.length)]);
        }
        
        return new DeepSeekV3Dataset(sequences, taskTypes, langs, seqLength, batchSize, true);
    }
}
