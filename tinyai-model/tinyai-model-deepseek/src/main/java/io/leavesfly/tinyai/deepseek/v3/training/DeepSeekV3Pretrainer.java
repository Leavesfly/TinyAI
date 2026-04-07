package io.leavesfly.tinyai.deepseek.v3.training;

import io.leavesfly.tinyai.deepseek.base.DeepSeekTrainerBase;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Block;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Config;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3MTPHead;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Model;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.ArrayList;
import java.util.List;

/**
 * DeepSeek-V3预训练器
 * <p>
 * 实现因果语言建模(Causal Language Modeling)预训练,
 * 特别优化MoE负载均衡和任务感知能力
 * <p>
 * 关键特性：
 * 1. MoE负载均衡损失 - 确保专家均匀使用
 * 2. 任务感知训练 - 提升任务路由准确性
 * 3. Warmup + Cosine衰减学习率
 * 4. 梯度裁剪防止爆炸
 *
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3Pretrainer extends DeepSeekTrainerBase {

    private final DeepSeekV3Model model;
    private final DeepSeekV3Config config;
    private final DeepSeekV3Dataset dataset;
    private final SoftmaxCrossEntropy lossFunction;
    private final Adam optimizer;

    // 训练超参数
    private float initialLearningRate;
    private float minLearningRate;
    private int warmupSteps;
    private float moeLoadBalanceWeight;  // MoE负载均衡权重(V3特有)
    private int saveInterval;

    // 训练状态
    private float currentLearningRate;
    private List<Float> moeLossHistory;      // MoE损失历史(V3特有)
    private List<Float> confidenceHistory;

    /**
     * 构造函数
     */
    public DeepSeekV3Pretrainer(DeepSeekV3Model model, DeepSeekV3Dataset dataset) {
        super(model, 10, 1.0f, 100, "./checkpoints/deepseek_v3/pretrain");
        this.model = model;
        this.config = model.getConfig();
        this.dataset = dataset;
        this.lossFunction = new SoftmaxCrossEntropy();

        // 默认超参数
        this.initialLearningRate = 2.5e-4f;
        this.minLearningRate = 1e-5f;
        this.warmupSteps = 2000;
        this.moeLoadBalanceWeight = (float) config.getLoadBalanceLossWeight();
        this.saveInterval = 5000;

        // 创建优化器
        this.optimizer = new Adam(model, initialLearningRate, 0.9f, 0.999f, 1e-8f);

        // 初始化状态
        this.currentLearningRate = 0.0f;
        this.moeLossHistory = new ArrayList<>();
        this.confidenceHistory = new ArrayList<>();
    }

    /**
     * 配置训练参数
     */
    public DeepSeekV3Pretrainer configure(int maxEpochs, float learningRate,
                                          int warmupSteps, float maxGradNorm) {
        this.maxEpochs = maxEpochs;
        this.initialLearningRate = learningRate;
        this.warmupSteps = warmupSteps;
        this.maxGradNorm = maxGradNorm;
        return this;
    }

    /**
     * 配置MoE参数
     */
    public DeepSeekV3Pretrainer configureMoE(float moeLoadBalanceWeight) {
        this.moeLoadBalanceWeight = moeLoadBalanceWeight;
        return this;
    }

    /**
     * 设置检查点配置
     */
    public DeepSeekV3Pretrainer setCheckpoint(String checkpointDir, int saveInterval) {
        this.checkpointDir = checkpointDir;
        this.saveInterval = saveInterval;
        return this;
    }

    /**
     * 开始训练
     */
    @Override
    public void train() {
        System.out.println("=".repeat(80));
        System.out.println("DeepSeek-V3 预训练（含MoE负载均衡）");
        System.out.println("=".repeat(80));
        System.out.println("模型参数:");
        System.out.println("  - 嵌入维度: " + config.getNEmbd());
        System.out.println("  - Transformer层数: " + config.getNLayer());
        System.out.println("  - 注意力头数: " + config.getNHead());
        System.out.println("  - 专家数量: " + config.getNumExperts());
        System.out.println("  - Top-K选择: " + config.getTopK());
        System.out.println("  - 总参数量: " + formatParamCount(config.estimateParameterCount()));
        System.out.println("  - 激活参数: " + formatParamCount(config.estimateActiveParameterCount()) +
                " (" + String.format("%.1f%%", config.getActivationRatio()) + ")");
        System.out.println("训练配置:");
        System.out.println("  - 训练样本: " + dataset.getSampleCount());
        System.out.println("  - 批次数量: " + dataset.getBatchCount());
        System.out.println("  - 最大轮次: " + maxEpochs);
        System.out.println("  - 初始学习率: " + initialLearningRate);
        System.out.println("  - Warmup步数: " + warmupSteps);
        System.out.println("  - MoE负载均衡权重: " + moeLoadBalanceWeight);
        System.out.println("=".repeat(80));

        // 创建检查点目录
        createCheckpointDir();

        // 训练循环
        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();
        }

        // 保存最终模型
        saveCheckpoint("final");

        System.out.println("\n训练完成!");
        System.out.println("最终语言模型损失: " + getAverage(lossHistory, 100));
        System.out.println("平均推理置信度: " + getAverage(confidenceHistory, 100));
    }

    /**
     * 训练一个epoch
     */
    private void trainOneEpoch() {
        dataset.prepare(true);
        EpochStats stats = new EpochStats();

        while (dataset.hasNext()) {
            processBatch(dataset.nextBatch(), stats);
        }

        System.out.printf("Epoch %d 完成 | 平均LM损失: %.4f | 平均置信度: %.4f | 耗时: %d ms%n",
                currentEpoch + 1, stats.avgLoss(), stats.avgConfidence(), stats.elapsedMs());
        dataset.reset();
    }

    /**
     * 处理单个batch：训练一步 + 记录历史 + 定期日志/保存
     */
    private void processBatch(DeepSeekV3Dataset.Batch batch, EpochStats stats) {
        StepResult stepResult = trainStep(batch);
        stats.accumulate(stepResult);
        globalStep++;

        lossHistory.add(stepResult.languageModelLoss);
        moeLossHistory.add(stepResult.moeLoss);
        confidenceHistory.add(stepResult.confidence);

        if (globalStep % logInterval == 0) {
            logStepProgress();
        }
        if (globalStep % saveInterval == 0) {
            saveCheckpoint("step_" + globalStep);
        }
    }

    /**
     * 打印当前步骤训练进度
     */
    private void logStepProgress() {
        float avgLoss = getAverage(lossHistory, logInterval);
        float avgConf = getAverage(confidenceHistory, logInterval);
        System.out.printf("Epoch %d/%d | Step %d | LM Loss: %.4f | Confidence: %.4f | LR: %.6f%n",
                currentEpoch + 1, maxEpochs, globalStep, avgLoss, avgConf, currentLearningRate);
    }

    /**
     * 训练单步（含MoE负载均衡 + Multi-Token Prediction）
     */
    private StepResult trainStep(DeepSeekV3Dataset.Batch batch) {
        updateLearningRate();

        NdArray inputIds = batch.getInputIds();
        NdArray targetIds = batch.getTargetIds();

        // 前向传播，result 中同时携带各层 MoE 负载均衡损失和隐藏状态
        DeepSeekV3Block.DetailedForwardResult result = model.predictWithDetails(new Variable(inputIds));

        // 计算语言模型损失
        Variable[] shaped = reshapeForLoss(inputIds, targetIds, result.logits);
        Variable lmLoss = lossFunction.loss(shaped[1], shaped[0]);
        float lmLossValue = lmLoss.getValue().getNumber().floatValue();
        float moeLossValue = (float) result.avgMoELoss;

        // 计算置信度：对 logits 做 softmax，取每个位置最大概率的平均值
        float confidenceValue = computeConfidence(result.logits);

        // 构建总损失：LM损失 + MoE负载均衡损失
        Variable totalLoss = buildTotalLoss(lmLoss, moeLossValue);

        // Multi-Token Prediction 损失（DeepSeek-V3 论文核心创新）
        DeepSeekV3MTPHead mtpHead = model.getV3Block().getMtpHead();
        if (mtpHead != null && result.hiddenStates != null) {
            Variable mtpLoss = mtpHead.computeMTPLoss(result.hiddenStates, new Variable(inputIds), targetIds, lossFunction);
            totalLoss = totalLoss.add(mtpLoss);
        }

        // 反向传播
        model.clearGrads();
        totalLoss.backward();
        clipGradients();
        optimizer.update();
        totalLoss.unChainBackward();

        return new StepResult(lmLossValue, moeLossValue, confidenceValue);
    }

    /**
     * 打印批数据详情(用于调试数据结构)
     */
    private void printBatchDetails(DeepSeekV3Dataset.Batch batch, NdArray inputIds, NdArray targetIds) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🔍 批数据详情检查 (Step " + globalStep + ")");
        System.out.println("=".repeat(80));

        // 1. 批次基本信息
        System.out.println("[批次信息]");

        // 2. 输入数据（使用NdArray的toString按形状打印）
        System.out.println("\n[输入数据 - 按形状打印]");
        System.out.println("  - 词汇表大小: " + config.getVocabSize());
        System.out.println(inputIds.toString());

        // 检查是否有超出词汇表的token ID
        float[] inputData = inputIds.getArray();
        boolean hasInvalidTokens = false;
        for (float val : inputData) {
            if (val >= config.getVocabSize() || val < 0) {
                hasInvalidTokens = true;
                break;
            }
        }
        if (hasInvalidTokens) {
            System.out.println("  ⚠️ 警告: 发现超出词汇表范围的token ID!");
        } else {
            System.out.println("  ✓ 所有token ID均在有效范围内");
        }

        // 3. 目标数据（使用NdArray的toString按形状打印）
        System.out.println("\n[目标数据 - 按形状打印]");
        System.out.println(targetIds.toString());

        // 检查目标是否有超出词汇表的token ID
        float[] targetData = targetIds.getArray();
        boolean hasInvalidTargets = false;
        for (float val : targetData) {
            if (val >= config.getVocabSize() || val < 0) {
                hasInvalidTargets = true;
                break;
            }
        }
        if (hasInvalidTargets) {
            System.out.println("  ⚠️ 警告: 发现超出词汇表范围的目标token ID!");
        } else {
            System.out.println("  ✓ 所有目标token ID均在有效范围内");
        }

        // 4. 数据对齐检查（目标应该是输入左移1位）
        System.out.println("\n[数据对齐检查]");
        Shape inputShape = inputIds.getShape();
        int seqLen = inputShape.getDimension(1);
        boolean isAligned = true;
        for (int i = 0; i < Math.min(5, seqLen - 1); i++) {
            if (Math.abs(inputData[i + 1] - targetData[i]) > 0.001) {
                isAligned = false;
                break;
            }
        }
        if (isAligned && seqLen > 1) {
            System.out.println("  ✓ 目标序列 = 输入序列左移1位 (符合预期)");
        } else {
            System.out.println("  ⚠️ 注意: 目标和输入可能未按预期对齐");
        }

        // 5. 填充值分析（检查0的分布情况）
        System.out.println("\n[填充值分析]");
        int batchSize = inputShape.getDimension(0);
        int[] paddingCounts = new int[batchSize];
        int[] validTokenCounts = new int[batchSize];

        for (int i = 0; i < batchSize; i++) {
            int validCount = 0;
            int paddingCount = 0;
            for (int j = 0; j < seqLen; j++) {
                int idx = i * seqLen + j;
                if (Math.abs(inputData[idx]) < 0.001) {  // 假设0是填充值
                    paddingCount++;
                } else {
                    validCount++;
                }
            }
            paddingCounts[i] = paddingCount;
            validTokenCounts[i] = validCount;
        }

        System.out.println("  - 各样本有效token数量:");
        for (int i = 0; i < batchSize; i++) {
            float ratio = (validTokenCounts[i] * 100.0f) / seqLen;
            System.out.printf("    样本%d: %d个有效token, %d个填充 (有效率: %.1f%%)%n",
                    i + 1, validTokenCounts[i], paddingCounts[i], ratio);
        }

        // 检查是否有token ID为0但不是填充的情况
        int zeroCount = 0;
        for (float val : inputData) {
            if (Math.abs(val) < 0.001) zeroCount++;
        }
        float zeroProportion = (zeroCount * 100.0f) / inputData.length;
        System.out.printf("  - 整批数据中0的占比: %.1f%% (%d/%d)%n",
                zeroProportion, zeroCount, inputData.length);

        if (zeroProportion > 50) {
            System.out.println("  ⚠️ 警告: 填充值占比过高(>50%)，可能影响训练效果!");
        }

        System.out.println("=".repeat(80) + "\n");
    }

    /**
     * 将 logits 和 targets reshape 为 2D（供损失函数使用）
     * 返回 [logits2D, targets2D]
     */
    private Variable[] reshapeForLoss(NdArray inputIds, NdArray targetIds, Variable logits) {
        int batchSize = inputIds.getShape().getDimension(0);
        int seqLen = inputIds.getShape().getDimension(1);
        int vocabSize = model.getConfig().getVocabSize();
        Variable logits2D = logits.reshape(Shape.of(batchSize * seqLen, vocabSize));
        Variable targets2D = new Variable(targetIds).reshape(Shape.of(batchSize * seqLen, 1));
        return new Variable[]{logits2D, targets2D};
    }

    /**
     * 融合语言模型损失与 MoE 负载均衡损失
     */
    private Variable buildTotalLoss(Variable lmLoss, float moeLoss) {
        if (moeLoadBalanceWeight <= 0) return lmLoss;
        Variable moeLossVar = new Variable(NdArray.of(new float[]{moeLoss * moeLoadBalanceWeight}));
        return lmLoss.add(moeLossVar);
    }

    /**
     * 更新学习率(warmup + cosine衰减)
     */
    private void updateLearningRate() {
        if (globalStep < warmupSteps) {
            // 线性warmup
            currentLearningRate = initialLearningRate * ((float) globalStep / warmupSteps);
        } else {
            // 余弦退火
            int totalSteps = maxEpochs * dataset.getBatchCount();
            int decaySteps = totalSteps - warmupSteps;
            int currentDecayStep = globalStep - warmupSteps;

            double cosineDecay = 0.5 * (1 + Math.cos(Math.PI * currentDecayStep / decaySteps));
            float decayedLR = (initialLearningRate - minLearningRate) * (float) cosineDecay + minLearningRate;
            currentLearningRate = Math.max(decayedLR, minLearningRate);
        }

        optimizer.setLearningRate(currentLearningRate);
    }

    /**
     * 格式化参数数量
     */
    private String formatParamCount(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.2fB", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.2fM", count / 1_000_000.0);
        } else {
            return String.format("%,d", count);
        }
    }

    @Override
    protected String getTrainerName() {
        return "DeepSeek-V3 Pretrain";
    }

    @Override
    protected String getCheckpointPrefix() {
        return "deepseek_v3_pretrain";
    }

    /**
     * 计算基于 softmax 最大概率的置信度
     * <p>
     * 对 logits 沿词汇维度做 softmax 得到概率分布，
     * 取每个 token 位置上最大概率值，再对所有位置求平均。
     * 值域 [1/vocabSize, 1.0]，越接近 1 表示模型越"确信"。
     *
     * @param logits 模型输出 [batch_size, seq_len, vocab_size]
     * @return 平均最大概率置信度
     */
    private float computeConfidence(Variable logits) {
        NdArray probabilities = logits.getValue().softMax();
        float[] probData = probabilities.getArray();
        int[] dims = probabilities.getShape().getShapeDims();

        int batchSize = dims[0];
        int seqLen = dims[1];
        int vocabSize = dims[2];

        double sumMaxProb = 0.0;
        int totalPositions = batchSize * seqLen;

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int offset = (b * seqLen + s) * vocabSize;
                float maxProb = Float.NEGATIVE_INFINITY;
                for (int v = 0; v < vocabSize; v++) {
                    float prob = probData[offset + v];
                    if (prob > maxProb) {
                        maxProb = prob;
                    }
                }
                sumMaxProb += maxProb;
            }
        }

        return (float) (sumMaxProb / totalPositions);
    }

    /**
     * 单步训练结果
     */
    private static class StepResult {
        final float languageModelLoss;  // 语言模型（下一词预测）损失
        final float moeLoss;            // MoE 负载均衡损失
        final float confidence;         // 基于 softmax 最大概率的推理置信度

        StepResult(float languageModelLoss, float moeLoss, float confidence) {
            this.languageModelLoss = languageModelLoss;
            this.moeLoss = moeLoss;
            this.confidence = confidence;
        }
    }

    /**
     * epoch 内累计统计，避免在 trainOneEpoch 中散落多个计数变量
     */
    private static class EpochStats {
        double totalLoss = 0.0;
        double totalMoeLoss = 0.0;
        double totalConfidence = 0.0;
        int batchCount = 0;
        final long startTime = System.currentTimeMillis();

        void accumulate(StepResult result) {
            totalLoss += result.languageModelLoss;
            totalMoeLoss += result.moeLoss;
            totalConfidence += result.confidence;
            batchCount++;
        }

        double avgLoss() {
            return batchCount > 0 ? totalLoss / batchCount : 0.0;
        }

        double avgMoeLoss() {
            return batchCount > 0 ? totalMoeLoss / batchCount : 0.0;
        }

        double avgConfidence() {
            return batchCount > 0 ? totalConfidence / batchCount : 0.0;
        }

        long elapsedMs() {
            return System.currentTimeMillis() - startTime;
        }
    }

    /**
     * 训练统计信息
     */
    public static class TrainingStats {
        public final int totalSteps;
        public final double avgLoss;
        public final double avgMoeLoss;
        public final double avgConfidence;

        public TrainingStats(int totalSteps, double avgLoss,
                             double avgMoeLoss, double avgConfidence) {
            this.totalSteps = totalSteps;
            this.avgLoss = avgLoss;
            this.avgMoeLoss = avgMoeLoss;
            this.avgConfidence = avgConfidence;
        }

        @Override
        public String toString() {
            return String.format("TrainingStats[steps=%d, loss=%.4f, moeLoss=%.6f, conf=%.4f]",
                    totalSteps, avgLoss, avgMoeLoss, avgConfidence);
        }
    }

    /**
     * 获取训练统计信息
     */
    public TrainingStats getStats() {
        double avgLoss = lossHistory.stream().mapToDouble(f -> f).average().orElse(0.0);
        double avgMoeLoss = moeLossHistory.stream().mapToDouble(f -> f).average().orElse(0.0);
        double avgConf = confidenceHistory.stream().mapToDouble(f -> f).average().orElse(0.0);
        return new TrainingStats(globalStep, avgLoss, avgMoeLoss, avgConf);
    }
}
