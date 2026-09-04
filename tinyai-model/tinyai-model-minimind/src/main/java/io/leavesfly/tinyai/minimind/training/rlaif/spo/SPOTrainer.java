package io.leavesfly.tinyai.minimind.training.rlaif.spo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindBlock;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.training.dataset.RLAIFDataset;
import io.leavesfly.tinyai.minimind.training.rlaif.BaseRLTrainer;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * SPO (Simplified Policy Optimization) 训练器
 * <p>
 * 简化的策略优化算法,无需Critic网络。核心思想:
 * 1. 生成K个候选回答
 * 2. 计算奖励R(y)
 * 3. 计算优势A(y) = R(y) - mean(R)
 * 4. 策略梯度优化（带基线的 REINFORCE）
 * <p>
 * 内存与正确性策略: 每个候选单独前向 + 单独 backward（梯度在参数上自然累积），
 * 全部候选完成后再统一裁剪与更新。这样避免同时持有 K 份 [batch, seq, vocab] 的计算图，
 * 也让 K 个候选的梯度按 1/K 等权合并。
 * <p>
 * 训练循环、日志、检查点、NaN 保护、结束后复位 eval 模式均复用 {@link BaseRLTrainer}
 * 的基类实现，本类不再重复声明 maxEpochs/currentStep/lossHistory 等状态字段
 * （重复声明会遮蔽基类字段，导致基类的检查点保存与日志读到永远为 0/空的另一份状态）。
 *
 * @author leavesfly
 * @since 2024
 */
public class SPOTrainer extends BaseRLTrainer {

    private final RLAIFDataset dataset;
    private final SPOConfig config;
    private final SPOLoss spoLoss;
    private final Adam optimizer;

    private final List<Float> rewardHistory;

    /**
     * 构造函数
     *
     * @param model   策略网络
     * @param dataset RLAIF 数据集
     * @param config  SPO 配置
     */
    public SPOTrainer(MiniMindModel model, RLAIFDataset dataset, SPOConfig config) {
        super(model);
        this.dataset = dataset;
        this.config = config;
        this.spoLoss = new SPOLoss(config);
        this.optimizer = new Adam(model, config.getLearningRate(), 0.9f, 0.999f, 1e-8f);

        // 训练配置写入基类字段，保证基类的训练循环/日志/检查点逻辑读到同一份状态
        this.maxEpochs = 1;
        this.logInterval = 10;
        this.saveInterval = 500;
        this.maxGradNorm = config.getMaxGradNorm();
        this.checkpointDir = "./checkpoints/minimind/spo";

        this.rewardHistory = new ArrayList<>();
    }

    /**
     * 配置训练
     */
    public SPOTrainer configure(int maxEpochs, int logInterval) {
        this.maxEpochs = maxEpochs;
        this.logInterval = logInterval;
        return this;
    }

    // ==================== 核心训练逻辑 ====================

    /**
     * 训练一步
     * <p>
     * 计算图链路（逐候选）:
     * {@code forwardWithAux(input) → spoLoss.computeCandidateLoss(logits, labels, A_k) → 标量损失
     * → loss.backward() → 梯度回传到模型参数}
     */
    @Override
    protected float trainStep(Object batch) {
        RLAIFDataset.Batch rlBatch = (RLAIFDataset.Batch) batch;
        model.setTraining(true);

        int numCandidates = rlBatch.getNumCandidates();
        int batchSize = rlBatch.getBatchSize();
        NdArray[] candidateInputs = rlBatch.getCandidateInputs();
        NdArray[] candidateLabels = rlBatch.getCandidateLabels();

        // 1. 计算奖励与优势（纯数值，不参与梯度）
        float[][] rewards = computeRewards(rlBatch);
        float[][] advantages = spoLoss.computeNormalizedAdvantages(rewards);
        rewardHistory.add(averageReward(rewards));

        model.clearGrads();

        float entropyCoef = config.getEntropyCoef();
        float lossSum = 0.0f;
        int validCandidates = 0;

        // 2. 逐候选前向 + 反向（梯度累积）
        for (int k = 0; k < numCandidates; k++) {
            Variable inputVar = new Variable(candidateInputs[k]);
            Variable labelVar = new Variable(candidateLabels[k]);
            labelVar.setRequireGrad(false);

            MiniMindBlock.MoEOutput output = forwardWithAux(inputVar);
            Variable logits = output.getOutput();

            Variable policyLoss = spoLoss.computeCandidateLoss(
                    logits, labelVar, column(advantages, k, batchSize));

            // 熵正则化：标量平均熵，避免损失退化为非标量张量
            Variable entropy = spoLoss.computeEntropy(logits);
            Variable loss = policyLoss.sub(entropy.mul(constant(entropyCoef)));

            // 并入 MoE 辅助损失，并按候选数等权平均
            loss = withMoeAuxLoss(loss, output).mul(constant(1.0f / numCandidates));

            float lossValue = loss.getValue().getNumber().floatValue();
            if (!Float.isFinite(lossValue)) {
                System.err.printf("警告: SPO 候选 %d 损失异常(%s)，已跳过%n",
                        k, Float.isNaN(lossValue) ? "NaN" : "Inf");
                loss.unChainBackward();
                continue;
            }

            loss.backward();
            loss.unChainBackward();

            lossSum += lossValue;
            validCandidates++;
        }

        // 3. 全部候选累积完成后统一裁剪与更新
        clipGradients(model, config.getMaxGradNorm());
        optimizer.update();
        model.clearGrads();

        return validCandidates > 0 ? lossSum / validCandidates : Float.NaN;
    }

    /**
     * 计算奖励
     * <p>
     * 优先使用数据集预设的奖励；仅当该样本确实没有预设奖励时，才回退到规则奖励。
     * 判定依据是 {@link RLAIFDataset.Batch#hasPresetReward(int)} 而不是
     * {@code rewards[i][k] != 0.0f}——后者会把"预设奖励恰好为 0"误判成"没有预设"。
     */
    private float[][] computeRewards(RLAIFDataset.Batch batch) {
        int batchSize = batch.getBatchSize();
        int numCandidates = batch.getNumCandidates();
        String[][] candidateTexts = batch.getCandidateTexts();
        float[][] presetRewards = batch.getRewards();

        float[][] rewards = new float[batchSize][numCandidates];

        for (int i = 0; i < batchSize; i++) {
            boolean preset = batch.hasPresetReward(i);
            for (int k = 0; k < numCandidates; k++) {
                if (preset) {
                    rewards[i][k] = (k < presetRewards[i].length) ? presetRewards[i][k] : 0.0f;
                } else {
                    rewards[i][k] = ruleBasedReward(candidateTexts[i][k]);
                }
            }
        }

        return rewards;
    }

    /**
     * 规则奖励（无奖励模型时的兜底）
     * <p>
     * 由两项组成：
     * 1. 长度奖励：过短（&lt;=10 字符，基本没有信息量）或过长（&gt;=200 字符，通常是复读）不给分
     * 2. 重复惩罚：按字符 4-gram 的重复率连续扣分，而不是"命中即扣固定分"
     */
    private static float ruleBasedReward(String text) {
        if (text == null || text.isEmpty()) {
            return 0.0f;
        }

        float reward = 0.0f;
        int length = text.length();

        // 长度奖励（适中长度）
        if (length > 10 && length < 200) {
            reward += 0.5f;
        }

        // 重复惩罚：4-gram 重复率越高扣得越多，最多扣 0.5
        float repetitionRatio = repetitionRatio(text, 4);
        reward -= 0.5f * repetitionRatio;

        return reward;
    }

    /**
     * 计算字符 n-gram 的重复率
     * <p>
     * 为什么不用 {@code text.contains(text.substring(0, n))}：任何字符串都必然包含自己的前缀，
     * 该判断恒为 true，等价于给所有候选都扣同一份惩罚——奖励失去区分度，优势全为 0，
     * 策略梯度不会有任何有效更新。
     *
     * @param text  待检测文本
     * @param nGram n-gram 长度
     * @return 重复出现的 n-gram 占全部 n-gram 的比例，范围 [0, 1)
     */
    private static float repetitionRatio(String text, int nGram) {
        if (text.length() < nGram * 2) {
            return 0.0f;
        }

        Set<String> seen = new HashSet<>();
        int repeats = 0;
        int total = 0;
        for (int i = 0; i + nGram <= text.length(); i++) {
            total++;
            if (!seen.add(text.substring(i, i + nGram))) {
                repeats++;
            }
        }
        return total > 0 ? (float) repeats / total : 0.0f;
    }

    // ==================== 工具方法 ====================

    /**
     * 取出 [batchSize][numCandidates] 矩阵的第 k 列
     */
    private static float[] column(float[][] matrix, int k, int batchSize) {
        float[] column = new float[batchSize];
        for (int i = 0; i < batchSize; i++) {
            column[i] = (i < matrix.length && k < matrix[i].length) ? matrix[i][k] : 0.0f;
        }
        return column;
    }

    /**
     * 计算平均奖励
     */
    private static float averageReward(float[][] rewards) {
        float sum = 0.0f;
        int count = 0;
        for (float[] row : rewards) {
            for (float r : row) {
                sum += r;
                count++;
            }
        }
        return count > 0 ? sum / count : 0.0f;
    }

    public List<Float> getRewardHistory() {
        return new ArrayList<>(rewardHistory);
    }

    public SPOConfig getConfig() {
        return config;
    }

    // ==================== BaseTrainer 抽象方法实现 ====================

    @Override
    protected String getTrainerName() {
        return "SPO";
    }

    @Override
    protected void printTrainingInfo() {
        System.out.println("=".repeat(70));
        System.out.println("开始SPO训练");
        System.out.println("配置: " + config);
        System.out.println("样本数: " + dataset.getSampleCount());
        System.out.println("最大轮次: " + maxEpochs);
        System.out.println("=".repeat(70));
    }

    @Override
    protected void printTrainingLog() {
        double avgLoss = lossHistory.stream()
                .skip(Math.max(0, lossHistory.size() - logInterval))
                .mapToDouble(Float::doubleValue)
                .average()
                .orElse(0.0);

        double avgReward = rewardHistory.stream()
                .skip(Math.max(0, rewardHistory.size() - logInterval))
                .mapToDouble(Float::doubleValue)
                .average()
                .orElse(0.0);

        System.out.printf("Epoch %d/%d | Step %d | Loss: %.4f | Reward: %.4f%n",
                currentEpoch + 1, maxEpochs, currentStep, avgLoss, avgReward);
    }

    @Override
    protected void prepareDataset() {
        dataset.prepare(true);
    }

    @Override
    protected boolean hasNextBatch() {
        return dataset.hasNext();
    }

    @Override
    protected Object getNextBatch() {
        return dataset.nextBatch();
    }

    @Override
    protected void resetDataset() {
        dataset.reset();
    }

    @Override
    protected String getCheckpointPrefix() {
        return "spo";
    }
}
