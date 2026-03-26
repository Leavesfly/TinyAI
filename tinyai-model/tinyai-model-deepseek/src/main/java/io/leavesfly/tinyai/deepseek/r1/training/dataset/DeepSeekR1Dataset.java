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
    private final List<float[]> lossMasks;    // Loss Mask（SFT后训练用，标记哪些位置参与loss计算）
    
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
        this.lossMasks = new ArrayList<>();
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
        this.lossMasks = new ArrayList<>();
    }
    
    /**
     * 构造函数（SFT后训练模式，带 Loss Mask）
     * 
     * Loss Mask 用于实现 Answer-only Loss：只对 assistant 回复部分计算 loss，
     * user 指令部分不参与梯度更新，符合行业主流 SFT 训练标准。
     * 
     * @param sequences token序列列表
     * @param lossMasks 每个样本的 loss mask（1.0=参与loss, 0.0=忽略）
     * @param maxSeqLength 最大序列长度
     * @param batchSize 批次大小
     * @param shuffle 是否打乱数据
     */
    public DeepSeekR1Dataset(List<int[]> sequences, List<float[]> lossMasks,
                             int maxSeqLength, int batchSize, boolean shuffle) {
        super(sequences, maxSeqLength, batchSize, shuffle);
        this.reasoning = new ArrayList<>();
        this.rewards = new ArrayList<>();
        this.lossMasks = lossMasks;
    }
    
    /**
     * 构造函数（RLHF模式，带 Loss Mask + 奖励）
     * 
     * 行业标准 RLHF 训练：结合 Answer-only Loss Mask 和奖励加权回归，
     * 只对 assistant 回复部分计算 loss，并按奖励分数加权梯度。
     * 
     * @param sequences token序列列表
     * @param reasoning 推理过程文本列表
     * @param rewards 奖励分数列表
     * @param lossMasks 每个样本的 loss mask
     * @param maxSeqLength 最大序列长度
     * @param batchSize 批次大小
     * @param shuffle 是否打乱数据
     */
    public DeepSeekR1Dataset(List<int[]> sequences, List<String> reasoning,
                             List<Float> rewards, List<float[]> lossMasks,
                             int maxSeqLength, int batchSize, boolean shuffle) {
        super(sequences, maxSeqLength, batchSize, shuffle);
        this.reasoning = reasoning;
        this.rewards = rewards;
        this.lossMasks = lossMasks;
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
        
        // 构建 R1 特有的推理过程、奖励分数和 Loss Mask
        String[] reasoningTexts = new String[actualBatchSize];
        float[] rewardScores = new float[actualBatchSize];
        float[][] batchLossMasks = null;
        
        List<Integer> batchIndices = getCurrentBatchIndices(actualBatchSize);
        
        boolean hasLossMasks = !lossMasks.isEmpty();
        if (hasLossMasks) {
            batchLossMasks = new float[actualBatchSize][maxSeqLength];
        }
        
        for (int i = 0; i < actualBatchSize; i++) {
            int dataIndex = batchIndices.get(i);
            
            if (!reasoning.isEmpty() && dataIndex < reasoning.size()) {
                reasoningTexts[i] = reasoning.get(dataIndex);
            }
            if (!rewards.isEmpty() && dataIndex < rewards.size()) {
                rewardScores[i] = rewards.get(dataIndex);
            }
            if (hasLossMasks && dataIndex < lossMasks.size()) {
                float[] originalMask = lossMasks.get(dataIndex);
                // Loss Mask 需要右移一位对齐 target（target = input 右移一位）
                // target[j] = sequence[j+1]，对应的 mask 也应该是 mask[j+1]
                // 添加长度校验，避免数组越界
                if (originalMask.length >= 2) {
                    int validLen = Math.min(originalMask.length - 1, maxSeqLength);
                    for (int j = 0; j < validLen; j++) {
                        batchLossMasks[i][j] = originalMask[j + 1];
                    }
                }
                // 如果 originalMask.length < 2，batchLossMasks[i] 保持全零（默认值）
            }
        }
        
        advanceIndex(endIndex);
        
        NdArray inputIds = NdArray.of(inputTarget[0]);
        NdArray targetIds = NdArray.of(inputTarget[1]);
        NdArray lossMaskArray = hasLossMasks ? NdArray.of(batchLossMasks) : null;
        
        return new Batch(inputIds, targetIds, reasoningTexts, rewardScores, lossMaskArray);
    }
    
    /**
     * 判断数据集是否包含 Loss Mask
     */
    public boolean hasLossMasks() {
        return !lossMasks.isEmpty();
    }
    
    /**
     * 获取 Loss Mask 列表（供适配器使用）
     */
    public List<float[]> getLossMasks() {
        return lossMasks;
    }
    
    /**
     * 获取奖励分数列表（供适配器使用）
     */
    public List<Float> getRewardsList() {
        return rewards;
    }
    
    /**
     * 批次数据类
     */
    public static class Batch {
        private final NdArray inputIds;
        private final NdArray targetIds;
        private final String[] reasoning;
        private final float[] rewards;
        private final NdArray lossMask;
        
        public Batch(NdArray inputIds, NdArray targetIds, 
                    String[] reasoning, float[] rewards, NdArray lossMask) {
            this.inputIds = inputIds;
            this.targetIds = targetIds;
            this.reasoning = reasoning;
            this.rewards = rewards;
            this.lossMask = lossMask;
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
        
        /**
         * 获取 Loss Mask（可能为 null，表示不使用 mask）
         * 
         * 维度: [batchSize, seqLength]
         * 值: 1.0f 表示该位置参与 loss 计算，0.0f 表示忽略
         */
        public NdArray getLossMask() {
            return lossMask;
        }
        
        public boolean hasLossMask() {
            return lossMask != null;
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
