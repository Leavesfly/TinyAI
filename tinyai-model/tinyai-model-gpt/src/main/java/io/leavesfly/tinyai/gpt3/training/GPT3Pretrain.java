package io.leavesfly.tinyai.gpt3.training;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.gpt3.GPT3Config;
import io.leavesfly.tinyai.gpt3.GPT3Model;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.SGD;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * GPT-3预训练器
 *
 * 实现因果语言建模（Causal Language Modeling）预训练，对应GPT-3论文训练策略：
 * - 线性Warmup + 余弦学习率退火（Cosine Annealing）
 * - 梯度裁剪（Gradient Clipping），防止梯度爆炸
 * - 检查点按Epoch间隔保存
 * - 梯度检查点（由GPT3Config控制，在GPT3MainBlock中实现）
 *
 * GPT-3论文中的学习率：
 * - 小型(125M)：6e-4
 * - 中型(350M)：3e-4
 * - 大型(1.3B)：2e-4
 *
 * @author TinyAI
 * @since 2024
 */
public class GPT3Pretrain {

    private final GPT3Model model;
    private final GPT3Config config;
    private final GPT3Dataset dataset;
    private final SoftmaxCrossEntropy lossFunction;
    private final SGD optimizer;

    // 训练超参数
    private int maxEpochs;
    private float initialLearningRate;
    private float minLearningRate;
    private int warmupSteps;
    private float maxGradNorm;
    private int logInterval;
    private int saveInterval;
    private String checkpointDir;

    // 训练状态
    private int currentEpoch;
    private int globalStep;
    private float currentLearningRate;
    private List<Float> lossHistory;

    /**
     * 构造函数
     *
     * @param model   GPT-3模型
     * @param dataset 训练数据集
     */
    public GPT3Pretrain(GPT3Model model, GPT3Dataset dataset) {
        this.model = model;
        this.config = model.getConfig();
        this.dataset = dataset;
        this.lossFunction = new SoftmaxCrossEntropy();

        // 默认超参数（遵循GPT-3论文，按小型模型设置）
        this.maxEpochs = 10;
        this.initialLearningRate = 6e-4f;
        this.minLearningRate = 1e-5f;
        this.warmupSteps = 2000;
        this.maxGradNorm = 1.0f;
        this.logInterval = 100;
        this.saveInterval = 1;
        this.checkpointDir = "./checkpoints/gpt3_pretrain";

        this.optimizer = new SGD(model, initialLearningRate);

        this.currentEpoch = 0;
        this.globalStep = 0;
        this.currentLearningRate = 0.0f;
        this.lossHistory = new ArrayList<>();
    }

    /**
     * 配置训练参数
     *
     * @param maxEpochs    最大训练轮次
     * @param learningRate 初始学习率
     * @param warmupSteps  warmup步数
     * @param maxGradNorm  梯度裁剪阈值
     * @return this
     */
    public GPT3Pretrain configure(int maxEpochs, float learningRate,
                                   int warmupSteps, float maxGradNorm) {
        this.maxEpochs = maxEpochs;
        this.initialLearningRate = learningRate;
        this.warmupSteps = warmupSteps;
        this.maxGradNorm = maxGradNorm;
        this.optimizer.setLearningRate(learningRate);
        return this;
    }

    /**
     * 设置检查点配置
     *
     * @param checkpointDir 检查点目录
     * @param saveInterval  保存间隔（Epoch数）
     * @return this
     */
    public GPT3Pretrain setCheckpoint(String checkpointDir, int saveInterval) {
        this.checkpointDir = checkpointDir;
        this.saveInterval = saveInterval;
        return this;
    }

    /**
     * 开始预训练
     */
    public void train() {
        dataset.prepare(true);

        System.out.println("=".repeat(60));
        System.out.println("GPT-3 预训练");
        System.out.println("=".repeat(60));
        System.out.println("模型参数:");
        System.out.println("  - 隐藏维度: " + config.getNEmbd());
        System.out.println("  - 层数: " + config.getNLayer());
        System.out.println("  - 注意力头: " + config.getNHead());
        System.out.println("  - 序列长度: " + config.getNPositions());
        System.out.println("  - 并行注意力: " + config.isParallelAttention());
        System.out.println("  - RoPE: " + config.isUseRotaryEmbedding());
        System.out.println("  - 稀疏注意力: " + config.isSparseAttention());
        System.out.println("  - 梯度检查点: " + config.isGradientCheckpointing());
        System.out.println("训练配置:");
        System.out.println("  - 训练样本: " + dataset.getSampleCount());
        System.out.println("  - 批次数量: " + dataset.getBatchCount());
        System.out.println("  - 最大轮次: " + maxEpochs);
        System.out.println("  - 初始学习率: " + initialLearningRate);
        System.out.println("  - Warmup步数: " + warmupSteps);
        System.out.println("=".repeat(60));

        createCheckpointDir();

        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();

            if ((currentEpoch + 1) % saveInterval == 0) {
                saveCheckpoint("epoch_" + (currentEpoch + 1));
            }
        }

        saveCheckpoint("final");

        System.out.println("\n预训练完成!");
        System.out.println("最终损失: " + getAverageLoss(100));
    }

    /**
     * 训练一个epoch
     */
    private void trainOneEpoch() {
        dataset.prepare(true);

        double epochLoss = 0.0;
        int batchCount = 0;
        long epochStartTime = System.currentTimeMillis();

        while (dataset.hasNext()) {
            GPT3Dataset.Batch batch = dataset.nextBatch();

            float stepLoss = trainStep(batch);

            epochLoss += stepLoss;
            batchCount++;
            globalStep++;

            lossHistory.add(stepLoss);

            if (globalStep % logInterval == 0) {
                float avgLoss = getAverageLoss(logInterval);
                System.out.printf("Epoch %d/%d | Step %d | Loss: %.4f | LR: %.6f%n",
                        currentEpoch + 1, maxEpochs, globalStep, avgLoss, currentLearningRate);
            }
        }

        long elapsed = System.currentTimeMillis() - epochStartTime;
        double avgEpochLoss = batchCount > 0 ? epochLoss / batchCount : 0.0;

        System.out.printf("Epoch %d 完成 | 平均损失: %.4f | 耗时: %d ms%n",
                currentEpoch + 1, avgEpochLoss, elapsed);

        dataset.reset();
    }

    /**
     * 训练单步
     *
     * @param batch 批次数据
     * @return 损失值
     */
    private float trainStep(GPT3Dataset.Batch batch) {
        updateLearningRate();

        NdArray inputIds  = batch.getInputIds();
        NdArray targetIds = batch.getTargetIds();

        Variable inputVar = new Variable(inputIds);

        // 前向传播：logits shape = (batch, seq_len, vocab_size)
        Variable logits = model.predict(inputVar);

        NdArray logitsArray = logits.getValue();
        int batchSize  = logitsArray.getShape().getDimension(0);
        int seqLen     = logitsArray.getShape().getDimension(1);
        int vocabSize  = logitsArray.getShape().getDimension(2);

        // 重塑为2D计算交叉熵损失
        Variable reshapedLogits  = logits.reshape(Shape.of(batchSize * seqLen, vocabSize));
        Variable reshapedTargets = new Variable(targetIds.reshape(Shape.of(batchSize * seqLen, 1)));

        Variable loss = lossFunction.loss(reshapedTargets, reshapedLogits);
        float lossValue = loss.getValue().getNumber().floatValue();

        model.clearGrads();
        loss.backward();
        clipGradients();
        optimizer.update();
        loss.unChainBackward();

        return lossValue;
    }

    /**
     * 更新学习率（线性Warmup + 余弦退火）
     */
    private void updateLearningRate() {
        if (globalStep < warmupSteps) {
            currentLearningRate = initialLearningRate * ((float) globalStep / Math.max(1, warmupSteps));
        } else {
            int totalSteps     = maxEpochs * dataset.getBatchCount();
            int decaySteps     = Math.max(1, totalSteps - warmupSteps);
            int currentDecay   = globalStep - warmupSteps;

            double cosineDecay = 0.5 * (1 + Math.cos(Math.PI * currentDecay / decaySteps));
            float decayedLR    = (initialLearningRate - minLearningRate) * (float) cosineDecay + minLearningRate;
            currentLearningRate = Math.max(decayedLR, minLearningRate);
        }
        optimizer.setLearningRate(currentLearningRate);
    }

    /**
     * 梯度裁剪（全局L2范数裁剪）
     */
    private void clipGradients() {
        double totalNorm = 0.0;

        Map<String, Parameter> params = model.getAllParams();
        for (Parameter param : params.values()) {
            if (param.getGrad() != null) {
                NdArray grad = param.getGrad();
                totalNorm += grad.mul(grad).sum().getNumber().doubleValue();
            }
        }

        totalNorm = Math.sqrt(totalNorm);

        if (totalNorm > maxGradNorm) {
            float scale = (float) (maxGradNorm / totalNorm);
            for (Parameter param : params.values()) {
                if (param.getGrad() != null) {
                    param.setGrad(param.getGrad().mulNum(scale));
                }
            }
        }
    }

    /**
     * 保存检查点
     */
    private void saveCheckpoint(String suffix) {
        try {
            String filename = String.format("gpt3_pretrain_%s.model", suffix);
            String filepath = checkpointDir + File.separator + filename;
            model.saveModel(filepath);
            System.out.println("检查点已保存: " + filepath);
        } catch (Exception e) {
            System.err.println("保存检查点失败: " + e.getMessage());
        }
    }

    /**
     * 创建检查点目录
     */
    private void createCheckpointDir() {
        try {
            Files.createDirectories(Paths.get(checkpointDir));
        } catch (IOException e) {
            System.err.println("创建检查点目录失败: " + e.getMessage());
        }
    }

    /**
     * 获取最近N步的平均损失
     */
    private float getAverageLoss(int n) {
        if (lossHistory.isEmpty()) return 0.0f;
        int start = Math.max(0, lossHistory.size() - n);
        float sum = 0.0f;
        for (int i = start; i < lossHistory.size(); i++) {
            sum += lossHistory.get(i);
        }
        return sum / (lossHistory.size() - start);
    }

    /**
     * 获取训练统计信息
     */
    public TrainingStats getStats() {
        return new TrainingStats(
                currentEpoch,
                globalStep,
                currentLearningRate,
                lossHistory.isEmpty() ? 0.0f : lossHistory.get(lossHistory.size() - 1),
                getAverageLoss(100)
        );
    }

    /**
     * 训练统计信息
     */
    public static class TrainingStats {
        public final int epoch;
        public final int step;
        public final float learningRate;
        public final float currentLoss;
        public final float avgLoss;

        public TrainingStats(int epoch, int step, float learningRate,
                             float currentLoss, float avgLoss) {
            this.epoch = epoch;
            this.step = step;
            this.learningRate = learningRate;
            this.currentLoss = currentLoss;
            this.avgLoss = avgLoss;
        }

        @Override
        public String toString() {
            return String.format(
                    "TrainingStats{epoch=%d, step=%d, lr=%.6f, loss=%.4f, avgLoss=%.4f}",
                    epoch, step, learningRate, currentLoss, avgLoss);
        }
    }
}
