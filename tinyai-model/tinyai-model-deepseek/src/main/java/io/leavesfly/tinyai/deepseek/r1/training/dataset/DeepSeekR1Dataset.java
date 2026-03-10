package io.leavesfly.tinyai.deepseek.r1.training.dataset;

import io.leavesfly.tinyai.deepseek.base.dataset.DeepSeekBaseDataset;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * DeepSeek-R1数据集类
 * 
 * 继承 DeepSeekBaseDataset，复用通用的序列管理、批次迭代和 input/target 构建逻辑。
 * 在此基础上扩展推理过程和奖励分数，支持 RLHF 训练模式。
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1Dataset extends DeepSeekBaseDataset<DeepSeekR1Dataset.Batch> {
    
    private final List<String> reasoning;     // 推理过程（RLHF用）
    private final List<Float> rewards;        // 奖励分数（RLHF用）
    
    /**
     * 构造函数（预训练模式）
     * 
     * @param sequences token序列列表
     * @param maxSeqLength 最大序列长度
     * @param batchSize 批次大小
     * @param shuffle 是否打乱数据
     */
    public DeepSeekR1Dataset(List<int[]> sequences, int maxSeqLength, 
                             int batchSize, boolean shuffle) {
        super(sequences, maxSeqLength, batchSize, shuffle);
        this.reasoning = new ArrayList<>();
        this.rewards = new ArrayList<>();
    }
    
    /**
     * 构造函数（RLHF模式）
     * 
     * @param sequences token序列列表
     * @param reasoning 推理过程文本列表
     * @param rewards 奖励分数列表
     * @param maxSeqLength 最大序列长度
     * @param batchSize 批次大小
     * @param shuffle 是否打乱数据
     */
    public DeepSeekR1Dataset(List<int[]> sequences, List<String> reasoning,
                             List<Float> rewards, int maxSeqLength,
                             int batchSize, boolean shuffle) {
        super(sequences, maxSeqLength, batchSize, shuffle);
        this.reasoning = reasoning;
        this.rewards = rewards;
    }
    
    /**
     * 获取下一批数据
     * 
     * 复用基类的 createInputTargetData 构建 input/target，
     * 并附加 R1 特有的推理过程和奖励分数信息。
     * 
     * @return 批次数据
     */
    @Override
    public Batch nextBatch() {
        int actualBatchSize = calculateActualBatchSize();
        int endIndex = Math.min(currentIndex + batchSize, sequences.size());
        
        // 复用基类的 input/target 构建逻辑
        float[][][] inputTarget = createInputTargetData(actualBatchSize);
        
        // 构建 R1 特有的推理过程和奖励分数
        String[] reasoningTexts = new String[actualBatchSize];
        float[] rewardScores = new float[actualBatchSize];
        List<Integer> batchIndices = getCurrentBatchIndices(actualBatchSize);
        
        for (int i = 0; i < actualBatchSize; i++) {
            int dataIndex = batchIndices.get(i);
            
            if (!reasoning.isEmpty() && dataIndex < reasoning.size()) {
                reasoningTexts[i] = reasoning.get(dataIndex);
            }
            if (!rewards.isEmpty() && dataIndex < rewards.size()) {
                rewardScores[i] = rewards.get(dataIndex);
            }
        }
        
        advanceIndex(endIndex);
        
        NdArray inputIds = NdArray.of(inputTarget[0]);
        NdArray targetIds = NdArray.of(inputTarget[1]);
        
        return new Batch(inputIds, targetIds, reasoningTexts, rewardScores);
    }
    
    /**
     * 批次数据类
     */
    public static class Batch {
        private final NdArray inputIds;
        private final NdArray targetIds;
        private final String[] reasoning;
        private final float[] rewards;
        
        public Batch(NdArray inputIds, NdArray targetIds, 
                    String[] reasoning, float[] rewards) {
            this.inputIds = inputIds;
            this.targetIds = targetIds;
            this.reasoning = reasoning;
            this.rewards = rewards;
        }
        
        public NdArray getInputIds() {
            return inputIds;
        }
        
        public NdArray getTargetIds() {
            return targetIds;
        }
        
        public String[] getReasoning() {
            return reasoning;
        }
        
        public float[] getRewards() {
            return rewards;
        }
        
        public int getBatchSize() {
            return inputIds.getShape().getDimension(0);
        }
    }
    
    /**
     * 创建示例数据集（用于测试）
     */
    public static DeepSeekR1Dataset createDummyDataset(int numSamples, int seqLength,
                                                        int vocabSize, int batchSize) {
        List<int[]> sequences = new ArrayList<>();
        Random random = new Random(42);
        
        for (int i = 0; i < numSamples; i++) {
            int[] seq = new int[seqLength];
            for (int j = 0; j < seqLength; j++) {
                seq[j] = random.nextInt(vocabSize);
            }
            sequences.add(seq);
        }
        
        return new DeepSeekR1Dataset(sequences, seqLength, batchSize, true);
    }
    
    /**
     * 创建RLHF示例数据集
     */
    public static DeepSeekR1Dataset createDummyRLHFDataset(int numSamples, int seqLength,
                                                            int vocabSize, int batchSize) {
        List<int[]> sequences = new ArrayList<>();
        List<String> reasoning = new ArrayList<>();
        List<Float> rewards = new ArrayList<>();
        Random random = new Random(42);
        
        for (int i = 0; i < numSamples; i++) {
            int[] seq = new int[seqLength];
            for (int j = 0; j < seqLength; j++) {
                seq[j] = random.nextInt(vocabSize);
            }
            sequences.add(seq);
            
            reasoning.add("推理步骤" + i);
            rewards.add(random.nextFloat());
        }
        
        return new DeepSeekR1Dataset(sequences, reasoning, rewards, 
                                     seqLength, batchSize, true);
    }
}
