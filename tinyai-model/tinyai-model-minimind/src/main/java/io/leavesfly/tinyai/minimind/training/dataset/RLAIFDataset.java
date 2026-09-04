package io.leavesfly.tinyai.minimind.training.dataset;

import io.leavesfly.tinyai.minimind.tokenizer.MiniMindTokenizer;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * RLAIF (Reinforcement Learning from AI Feedback) 数据集
 * <p>
 * 用于强化学习训练的数据集,支持:
 * 1. Prompt → 多个候选回答
 * 2. 奖励信号(可选,可用规则或奖励模型生成)
 * 3. 批处理和数据打乱
 * <p>
 * 数据格式:
 * - Prompt: 输入提示词
 * - Candidates: K个候选回答(模型生成或预先准备)
 * - Rewards: K个候选的奖励分数
 *
 * @author leavesfly
 * @since 2024
 */
public class RLAIFDataset {

    /**
     * RLAIF样本 - 一个prompt对应多个候选回答
     */
    public static class RLAIFSample {
        private final String prompt;
        private final List<String> candidates;
        private final float[] rewards;

        /**
         * 构造函数
         *
         * @param prompt     输入提示
         * @param candidates 候选回答列表
         * @param rewards    对应的奖励分数(可为null,训练时计算)
         */
        public RLAIFSample(String prompt, List<String> candidates, float[] rewards) {
            this.prompt = prompt;
            this.candidates = new ArrayList<>(candidates);
            this.rewards = rewards;
        }

        /**
         * 只有prompt的构造函数(奖励在训练时计算)
         */
        public RLAIFSample(String prompt, List<String> candidates) {
            this(prompt, candidates, null);
        }

        public String getPrompt() {
            return prompt;
        }

        public List<String> getCandidates() {
            return candidates;
        }

        public float[] getRewards() {
            return rewards;
        }

        public int getNumCandidates() {
            return candidates.size();
        }
    }

    /**
     * RLAIF批次数据
     */
    public static class Batch {
        private final NdArray[] candidateInputs;   // [K, batch_size, seq_len]
        private final NdArray[] candidateLabels;   // [K, batch_size, seq_len]
        private final float[][] rewards;           // [batch_size, K]
        private final int batchSize;
        private final int numCandidates;
        private final String[] prompts;            // 保留原始prompt用于奖励计算
        private final String[][] candidateTexts;   // [batch_size, K]
        /**
         * 逐样本标记"是否带有预设奖励"，旧构造路径下为 null
         */
        private final boolean[] presetRewardFlags;

        public Batch(NdArray[] candidateInputs, NdArray[] candidateLabels,
                     float[][] rewards, String[] prompts, String[][] candidateTexts) {
            this(candidateInputs, candidateLabels, rewards, prompts, candidateTexts, null);
        }

        public Batch(NdArray[] candidateInputs, NdArray[] candidateLabels,
                     float[][] rewards, String[] prompts, String[][] candidateTexts,
                     boolean[] presetRewardFlags) {
            this.candidateInputs = candidateInputs;
            this.candidateLabels = candidateLabels;
            this.rewards = rewards;
            this.prompts = prompts;
            this.candidateTexts = candidateTexts;
            this.presetRewardFlags = presetRewardFlags;
            this.batchSize = candidateInputs[0].getShape().getShapeDims()[0];
            this.numCandidates = candidateInputs.length;
        }

        public NdArray[] getCandidateInputs() {
            return candidateInputs;
        }

        public NdArray[] getCandidateLabels() {
            return candidateLabels;
        }

        public float[][] getRewards() {
            return rewards;
        }

        /**
         * 第 i 个样本是否带有预设奖励
         * <p>
         * 为什么需要这个标记：{@link #getRewards()} 对"没有预设奖励"的槽位填的是 0.0f，
         * 而 0.0f 同样可能是一个合法的预设奖励值。调用方若用 {@code rewards[i][k] != 0.0f}
         * 来判断，就会把"预设奖励恰好为 0"误判成"没有预设"，从而用规则奖励把它覆盖掉。
         * <p>
         * 旧构造路径（flags 为 null）退化为"该行存在非零奖励"的启发式判断，保持向后兼容。
         */
        public boolean hasPresetReward(int i) {
            if (presetRewardFlags != null) {
                return i < presetRewardFlags.length && presetRewardFlags[i];
            }
            if (i >= rewards.length) {
                return false;
            }
            for (float r : rewards[i]) {
                if (r != 0.0f) {
                    return true;
                }
            }
            return false;
        }

        public int getBatchSize() {
            return batchSize;
        }

        public int getNumCandidates() {
            return numCandidates;
        }

        public String[] getPrompts() {
            return prompts;
        }

        public String[][] getCandidateTexts() {
            return candidateTexts;
        }
    }

    private final MiniMindTokenizer tokenizer;
    private final int maxSeqLength;
    private final int batchSize;
    private final List<RLAIFSample> samples;

    private int currentIndex;
    private List<RLAIFSample> shuffledSamples;

    /**
     * 构造函数
     *
     * @param tokenizer    分词器
     * @param maxSeqLength 最大序列长度
     * @param batchSize    批次大小
     */
    public RLAIFDataset(MiniMindTokenizer tokenizer, int maxSeqLength, int batchSize) {
        this.tokenizer = tokenizer;
        this.maxSeqLength = maxSeqLength;
        this.batchSize = batchSize;
        this.samples = new ArrayList<>();
        this.currentIndex = 0;
    }

    /**
     * 添加样本
     */
    public void addSample(String prompt, List<String> candidates, float[] rewards) {
        samples.add(new RLAIFSample(prompt, candidates, rewards));
    }

    /**
     * 添加样本(无奖励)
     */
    public void addSample(String prompt, List<String> candidates) {
        samples.add(new RLAIFSample(prompt, candidates));
    }

    /**
     * 准备数据集
     *
     * @param shuffle 是否打乱
     */
    public void prepare(boolean shuffle) {
        shuffledSamples = new ArrayList<>(samples);
        if (shuffle) {
            Collections.shuffle(shuffledSamples);
        }
        currentIndex = 0;
    }

    /**
     * 是否还有下一批
     */
    public boolean hasNext() {
        return currentIndex < shuffledSamples.size();
    }

    /**
     * 获取下一批数据
     */
    public Batch nextBatch() {
        if (!hasNext()) {
            throw new IllegalStateException("No more batches available");
        }

        int endIndex = Math.min(currentIndex + batchSize, shuffledSamples.size());
        int actualBatchSize = endIndex - currentIndex;

        List<RLAIFSample> batchSamples = shuffledSamples.subList(currentIndex, endIndex);
        currentIndex = endIndex;

        // 确定最大候选数量(所有样本的最大值)
        int maxCandidates = batchSamples.stream()
                .mapToInt(RLAIFSample::getNumCandidates)
                .max()
                .orElse(1);

        // 准备batch数据
        NdArray[] candidateInputs = new NdArray[maxCandidates];
        NdArray[] candidateLabels = new NdArray[maxCandidates];
        float[][] batchRewards = new float[actualBatchSize][maxCandidates];
        String[] batchPrompts = new String[actualBatchSize];
        String[][] batchCandidateTexts = new String[actualBatchSize][maxCandidates];
        // 逐样本记录奖励是否来自数据集预设，供训练器区分"预设为 0"与"未预设"
        boolean[] presetRewardFlags = new boolean[actualBatchSize];

        // 为每个候选位置创建数组
        for (int k = 0; k < maxCandidates; k++) {
            float[][] inputs = new float[actualBatchSize][maxSeqLength];
            float[][] labels = new float[actualBatchSize][maxSeqLength];

            for (int i = 0; i < actualBatchSize; i++) {
                RLAIFSample sample = batchSamples.get(i);
                batchPrompts[i] = sample.getPrompt();

                // 如果候选数量不足,使用第一个候选填充
                int candidateIdx = Math.min(k, sample.getNumCandidates() - 1);
                String candidate = sample.getCandidates().get(candidateIdx);
                batchCandidateTexts[i][k] = candidate;

                // Tokenize: prompt + candidate
                String fullText = sample.getPrompt() + candidate;
                List<Integer> tokensList = tokenizer.encode(fullText);
                int[] tokens = tokensList.stream().mapToInt(Integer::intValue).toArray();

                // 准备input和label
                int copyLen = Math.min(tokens.length, maxSeqLength);
                for (int j = 0; j < copyLen - 1; j++) {
                    inputs[i][j] = tokens[j];
                    labels[i][j] = tokens[j + 1];
                }

                // Padding
                int padId = 0;  // 简化:使用0作为Pad ID
                for (int j = copyLen - 1; j < maxSeqLength; j++) {
                    inputs[i][j] = padId;
                    labels[i][j] = -100;  // 忽略padding的损失
                }

                // 奖励(如果有预设奖励)
                float[] presetRewards = sample.getRewards();
                if (presetRewards != null && presetRewards.length > 0) {
                    // 与候选一样做下标钳制，避免奖励数组比候选数组短时静默退化为 0
                    int rewardIdx = Math.min(candidateIdx, presetRewards.length - 1);
                    batchRewards[i][k] = presetRewards[rewardIdx];
                    presetRewardFlags[i] = true;
                } else {
                    batchRewards[i][k] = 0.0f;  // 默认奖励,训练时重新计算
                }
            }

            candidateInputs[k] = NdArray.of(inputs);
            candidateLabels[k] = NdArray.of(labels);
        }

        return new Batch(candidateInputs, candidateLabels, batchRewards, batchPrompts,
                batchCandidateTexts, presetRewardFlags);
    }

    /**
     * 重置迭代器
     */
    public void reset() {
        currentIndex = 0;
    }

    /**
     * 获取样本数量
     */
    public int getSampleCount() {
        return samples.size();
    }

    /**
     * 获取批次数量
     */
    public int getBatchCount() {
        return (int) Math.ceil((double) samples.size() / batchSize);
    }

    /**
     * 获取最大序列长度
     */
    public int getMaxSeqLength() {
        return maxSeqLength;
    }
}
