package io.leavesfly.tinyai.deepseek.v3.training;

import io.leavesfly.tinyai.deepseek.base.TaskType;
import io.leavesfly.tinyai.deepseek.base.dataset.DeepSeekBaseDataset;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * DeepSeek-V3数据集类
 * 
 * 继承 DeepSeekBaseDataset，复用通用的序列管理、批次迭代和 input/target 构建逻辑。
 * 在此基础上扩展任务类型标注和代码语言标注，用于任务感知训练。
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3Dataset extends DeepSeekBaseDataset<DeepSeekV3Dataset.Batch> {
    
    private final List<TaskType> taskTypes;   // 任务类型（V3特有）
    private final List<String> codeLanguages; // 代码语言（代码任务专用）
    private final List<float[]> lossMasks;    // 每个样本的 loss mask（后训练 answer-only loss 使用）
    private final List<Float> rewards;        // 奖励分数（RLHF 训练用）
    
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
        super(sequences, maxSeqLength, batchSize, shuffle);
        this.taskTypes = new ArrayList<>();
        this.codeLanguages = new ArrayList<>();
        this.lossMasks = new ArrayList<>();
        this.rewards = new ArrayList<>();
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
        super(sequences, maxSeqLength, batchSize, shuffle);
        this.taskTypes = taskTypes;
        this.codeLanguages = new ArrayList<>();
        this.lossMasks = new ArrayList<>();
        this.rewards = new ArrayList<>();
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
        super(sequences, maxSeqLength, batchSize, shuffle);
        this.taskTypes = taskTypes;
        this.codeLanguages = codeLanguages;
        this.lossMasks = new ArrayList<>();
        this.rewards = new ArrayList<>();
    }
    
    /**
     * 构造函数（后训练 ChatML 模式，带 loss mask）
     * 
     * @param sequences token序列列表
     * @param taskTypes 任务类型列表
     * @param lossMasks 每个样本的 loss mask（1.0=计算loss, 0.0=不计算）
     * @param maxSeqLength 最大序列长度
     * @param batchSize 批次大小
     * @param shuffle 是否打乱数据
     */
    public DeepSeekV3Dataset(List<int[]> sequences, List<TaskType> taskTypes,
                             List<float[]> lossMasks, int maxSeqLength,
                             int batchSize, boolean shuffle, boolean withLossMask) {
        super(sequences, maxSeqLength, batchSize, shuffle);
        this.taskTypes = taskTypes;
        this.codeLanguages = new ArrayList<>();
        this.lossMasks = lossMasks;
        this.rewards = new ArrayList<>();
    }
    
    /**
     * 构造函数（RLHF 模式，带 loss mask + 奖励分数）
     * 
     * 用于 RLHF 训练：结合 Answer-only Loss Mask 和奖励加权回归，
     * 只对 assistant 回复部分计算 loss，并按奖励分数加权梯度。
     * 
     * @param sequences token序列列表
     * @param taskTypes 任务类型列表
     * @param lossMasks 每个样本的 loss mask
     * @param rewards 每个样本的奖励分数
     * @param maxSeqLength 最大序列长度
     * @param batchSize 批次大小
     * @param shuffle 是否打乱数据
     */
    public DeepSeekV3Dataset(List<int[]> sequences, List<TaskType> taskTypes,
                             List<float[]> lossMasks, List<Float> rewards,
                             int maxSeqLength, int batchSize, boolean shuffle) {
        super(sequences, maxSeqLength, batchSize, shuffle);
        this.taskTypes = taskTypes;
        this.codeLanguages = new ArrayList<>();
        this.lossMasks = lossMasks;
        this.rewards = rewards;
    }
    
    /**
     * 获取下一批数据
     * 
     * 复用基类的 createInputTargetData 构建 input/target，
     * 并附加 V3 特有的任务类型和代码语言信息。
     * 
     * @return 批次数据
     */
    @Override
    public Batch nextBatch() {
        int actualBatchSize = calculateActualBatchSize();
        int endIndex = Math.min(currentIndex + batchSize, sequences.size());
        
        // 复用基类的 input/target 构建逻辑
        float[][][] inputTarget = createInputTargetData(actualBatchSize);
        
        // 构建 V3 特有的任务类型和代码语言
        TaskType[] batchTaskTypes = new TaskType[actualBatchSize];
        String[] batchLanguages = new String[actualBatchSize];
        float[] batchRewards = new float[actualBatchSize];
        List<Integer> batchIndices = getCurrentBatchIndices(actualBatchSize);
        
        // 构建 loss mask（如果有的话）
        boolean hasLossMaskData = !lossMasks.isEmpty();
        float[][] batchLossMaskData = hasLossMaskData ? new float[actualBatchSize][maxSeqLength] : null;
        
        for (int i = 0; i < actualBatchSize; i++) {
            int dataIndex = batchIndices.get(i);
            
            if (!taskTypes.isEmpty() && dataIndex < taskTypes.size()) {
                batchTaskTypes[i] = taskTypes.get(dataIndex);
            } else {
                batchTaskTypes[i] = TaskType.GENERAL;
            }
            
            if (!codeLanguages.isEmpty() && dataIndex < codeLanguages.size()) {
                batchLanguages[i] = codeLanguages.get(dataIndex);
            }
            
            if (!rewards.isEmpty() && dataIndex < rewards.size()) {
                batchRewards[i] = rewards.get(dataIndex);
            }
            
            // 构建该样本的 loss mask
            // loss mask 对应的是 target 位置（右移一位后的位置），
            // 因此 lossMask[j] 对应 target[j]，即原始序列中 sequence[j+1] 的 loss 权重
            if (hasLossMaskData && dataIndex < lossMasks.size()) {
                float[] sampleMask = lossMasks.get(dataIndex);
                int validLen = Math.min(sampleMask.length - 1, maxSeqLength);
                for (int j = 0; j < validLen; j++) {
                    batchLossMaskData[i][j] = sampleMask[j + 1];
                }
            }
        }
        
        advanceIndex(endIndex);
        
        NdArray inputIds = NdArray.of(inputTarget[0]);
        NdArray targetIds = NdArray.of(inputTarget[1]);
        NdArray batchLossMask = hasLossMaskData ? NdArray.of(batchLossMaskData) : null;
        
        return new Batch(inputIds, targetIds, batchTaskTypes, batchLanguages, batchLossMask, batchRewards);
    }
    
    /**
     * 批次数据类
     * 
     * 包含 lossMask 用于后训练的 answer-only loss masking：
     * - lossMask[i][j] = 1.0 表示该位置参与 loss 计算（assistant 回答部分）
     * - lossMask[i][j] = 0.0 表示该位置不参与 loss 计算（user 提问部分 / padding）
     * - lossMask 为 null 时表示所有位置都参与 loss 计算（预训练模式）
     */
    public static class Batch {
        private final NdArray inputIds;
        private final NdArray targetIds;
        private final TaskType[] taskTypes;
        private final String[] codeLanguages;
        private final NdArray lossMask;
        private final float[] rewards;
        
        public Batch(NdArray inputIds, NdArray targetIds, 
                    TaskType[] taskTypes, String[] codeLanguages) {
            this(inputIds, targetIds, taskTypes, codeLanguages, null, null);
        }
        
        public Batch(NdArray inputIds, NdArray targetIds, 
                    TaskType[] taskTypes, String[] codeLanguages, NdArray lossMask) {
            this(inputIds, targetIds, taskTypes, codeLanguages, lossMask, null);
        }
        
        public Batch(NdArray inputIds, NdArray targetIds, 
                    TaskType[] taskTypes, String[] codeLanguages, 
                    NdArray lossMask, float[] rewards) {
            this.inputIds = inputIds;
            this.targetIds = targetIds;
            this.taskTypes = taskTypes;
            this.codeLanguages = codeLanguages;
            this.lossMask = lossMask;
            this.rewards = rewards;
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
         * 获取奖励分数（RLHF 训练用）
         * 
         * @return 奖励分数数组，null 表示无奖励数据
         */
        public float[] getRewards() {
            return rewards;
        }
        
        /**
         * 获取 loss mask（后训练 answer-only loss 使用）
         * 
         * @return lossMask，null 表示全部位置参与 loss 计算
         */
        public NdArray getLossMask() {
            return lossMask;
        }
        
        /**
         * 是否有 loss mask（用于判断是否需要 answer-only loss）
         */
        public boolean hasLossMask() {
            return lossMask != null;
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
    
    /**
     * 判断数据集是否包含 Loss Mask
     */
    public boolean hasLossMasks() {
        return !lossMasks.isEmpty();
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
