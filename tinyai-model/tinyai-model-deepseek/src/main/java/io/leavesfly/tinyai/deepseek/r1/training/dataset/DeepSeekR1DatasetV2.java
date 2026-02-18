package io.leavesfly.tinyai.deepseek.r1.training.dataset;

import io.leavesfly.tinyai.deepseek.base.dataset.DeepSeekBaseDataset;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.ArrayList;
import java.util.List;

/**
 * DeepSeek-R1数据集类（V2版本 - 基于共享基类）
 * 
 * 支持预训练、后训练和强化学习三种模式的数据加载。
 * R1 特点：包含推理过程和奖励分数（用于 RLHF）。
 * 
 * @author leavesfly
 * @version 2.0
 */
public class DeepSeekR1DatasetV2 extends DeepSeekBaseDataset<DeepSeekR1DatasetV2.Batch> {
    
    // R1 特有字段
    private final List<String> reasoning;     // 推理过程（RLHF用）
    private final List<Float> rewards;        // 奖励分数（RLHF用）
    
    /**
     * 构造函数（预训练模式）
     */
    public DeepSeekR1DatasetV2(List<int[]> sequences, int maxSeqLength, 
                               int batchSize, boolean shuffle) {
        super(sequences, maxSeqLength, batchSize, shuffle);
        this.reasoning = new ArrayList<>();
        this.rewards = new ArrayList<>();
    }
    
    /**
     * 构造函数（RLHF模式）
     */
    public DeepSeekR1DatasetV2(List<int[]> sequences, List<String> reasoning,
                               List<Float> rewards, int maxSeqLength,
                               int batchSize, boolean shuffle) {
        super(sequences, maxSeqLength, batchSize, shuffle);
        this.reasoning = reasoning;
        this.rewards = rewards;
    }
    
    /**
     * 获取下一批数据（R1 特定实现）
     */
    @Override
    public Batch nextBatch() {
        int actualBatchSize = calculateActualBatchSize();
        int endIndex = currentIndex + actualBatchSize;
        
        // 创建输入和目标数据（复用基类方法）
        float[][][] data = createInputTargetData(actualBatchSize);
        float[][] inputData = data[0];
        float[][] targetData = data[1];
        
        // 准备 RLHF 数据
        String[] reasoningTexts = new String[actualBatchSize];
        float[] rewardScores = new float[actualBatchSize];
        
        List<Integer> batchIndices = getCurrentBatchIndices(actualBatchSize);
        for (int i = 0; i < actualBatchSize; i++) {
            int dataIndex = batchIndices.get(i);
            
            // RLHF数据
            if (!reasoning.isEmpty() && dataIndex < reasoning.size()) {
                reasoningTexts[i] = reasoning.get(dataIndex);
            }
            if (!rewards.isEmpty() && dataIndex < rewards.size()) {
                rewardScores[i] = rewards.get(dataIndex);
            }
        }
        
        // 推进索引
        advanceIndex(endIndex);
        
        NdArray inputIds = NdArray.of(inputData);
        NdArray targetIds = NdArray.of(targetData);
        
        return new Batch(inputIds, targetIds, reasoningTexts, rewardScores);
    }
    
    /**
     * R1 批次数据类
     */
    public static class Batch extends DeepSeekBaseDataset.BaseBatch {
        private final String[] reasoning;
        private final float[] rewards;
        
        public Batch(NdArray inputIds, NdArray targetIds, 
                    String[] reasoning, float[] rewards) {
            super(inputIds, targetIds);
            this.reasoning = reasoning;
            this.rewards = rewards;
        }
        
        public String[] getReasoning() {
            return reasoning;
        }
        
        public float[] getRewards() {
            return rewards;
        }
    }
}
