package io.leavesfly.tinyai.gpt3.training;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.gpt3.GPT3Config;
import io.leavesfly.tinyai.gpt3.GPT3Model;
import io.leavesfly.tinyai.ml.loss.Loss;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.Adam;
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
 * GPT-3微调训练器（Posttrain/SFT Finetune）
 *
 * 在预训练模型基础上进行任务特定的有监督微调（Supervised Fine-Tuning），
 * 对应GPT-3论文中的few-shot/fine-tune策略：
 * - 较小的学习率（相比预训练缩小10倍）
 * - 支持Masked Loss（只在Response部分计算loss，Instruction部分不参与）
 * - 早停机制（Early Stopping）防止过拟合
 * - 验证集评估 + 最佳模型保存
 *
 * @author TinyAI
 * @since 2024
 */
public class GPT3Finetune {

    private final GPT3Model model;
    private final GPT3Config config;
    private final GPT3Dataset trainDataset;
    private final GPT3Dataset valDataset;
    private final SoftmaxCrossEntropy lossFunction;
    private final SoftmaxCrossEntropy perTokenLossFunction;
    private final Adam optimizer;

    // 微调超参数
    private int maxEpochs;
    private float learningRate;
    private float maxGradNorm;
    private int logInterval;
    private int evalInterval;
    private int patience;
    private String checkpointDir;

    // 学习率调度参数（Warmup支持）
    private int warmupSteps;
    private float initialLearningRate;

    // 训练状态
    private int currentEpoch;
    private int globalStep;
    private List<Float> trainLossHistory;
    private List<Float> valLossHistory;
    private float bestValLoss;
    private int stepsWithoutImprovement;

    /**
     * 构造函数
     *
     * @param model        预训练的GPT-3模型
     * @param trainDataset 训练数据集
     * @param valDataset   验证数据集
     */
    public GPT3Finetune(GPT3Model model, GPT3Dataset trainDataset, GPT3Dataset valDataset) {
        this.model = model;
        this.config = model.getConfig();
        this.trainDataset = trainDataset;
        this.valDataset = valDataset;
        this.lossFunction = new SoftmaxCrossEntropy();
        this.perTokenLossFunction = new SoftmaxCrossEntropy(Loss.Reduction.NONE);

        // 微调默认超参数（学习率比预训练小，但使用Adam优化器加速收敛）
        this.maxEpochs = 5;
        this.learningRate = 1e-4f;  // 提高学习率以加速收敛
        this.initialLearningRate = this.learningRate;
        this.warmupSteps = 50;      // 50步线性warmup稳定训练初期
        this.maxGradNorm = 1.0f;
        this.logInterval = 50;
        this.evalInterval = 100;
        this.patience = 3;
        this.checkpointDir = "./checkpoints/gpt3/finetune";

        // 使用Adam优化器（比SGD收敛更快更稳定）
        this.optimizer = new Adam(model, learningRate, 0.9f, 0.999f, 1e-8f);

        this.currentEpoch = 0;
        this.globalStep = 0;
        this.trainLossHistory = new ArrayList<>();
        this.valLossHistory = new ArrayList<>();
        this.bestValLoss = Float.MAX_VALUE;
        this.stepsWithoutImprovement = 0;
    }

    /**
     * 配置微调参数
     *
     * @param maxEpochs    最大训练轮次
     * @param learningRate 学习率
     * @param patience     早停耐心值
     * @return this
     */
    public GPT3Finetune configure(int maxEpochs, float learningRate, int patience) {
        this.maxEpochs = maxEpochs;
        this.learningRate = learningRate;
        this.initialLearningRate = learningRate;
        this.patience = patience;
        this.optimizer.setLearningRate(learningRate);
        return this;
    }

    /**
     * 设置Warmup步数
     *
     * @param warmupSteps warmup预热步数（默认50步）
     * @return this
     */
    public GPT3Finetune setWarmupSteps(int warmupSteps) {
        this.warmupSteps = warmupSteps;
        return this;
    }

    /**
     * 设置检查点配置
     *
     * @param checkpointDir 检查点目录
     * @param evalInterval  验证评估间隔（step数）
     * @return this
     */
    public GPT3Finetune setCheckpoint(String checkpointDir, int evalInterval) {
        this.checkpointDir = checkpointDir;
        this.evalInterval = evalInterval;
        return this;
    }

    /**
     * 开始微调训练
     */
    public void train() {
        System.out.println("=".repeat(60));
        System.out.println("GPT-3 微调训练 (Finetune/Posttrain)");
        System.out.println("=".repeat(60));
        System.out.println("模型配置:");
        System.out.println("  - 模型: " + model.getName());
        System.out.println("  - 参数量: " + model.getAllParams().size());
        System.out.println("微调配置:");
        System.out.println("  - 训练样本: " + trainDataset.getSampleCount());
        System.out.println("  - 验证样本: " + valDataset.getSampleCount());
        System.out.println("  - 最大轮次: " + maxEpochs);
        System.out.println("  - 学习率: " + learningRate);
        System.out.println("  - Warmup步数: " + warmupSteps);
        System.out.println("  - 优化器: Adam");
        System.out.println("  - 早停耐心: " + patience);
        System.out.println("=".repeat(60));

        createCheckpointDir();

        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();

            // 每个epoch结束后进行验证
            float valLoss = evaluate();
            valLossHistory.add(valLoss);

            System.out.printf("Epoch %d 验证损失: %.4f%n", currentEpoch + 1, valLoss);

            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                stepsWithoutImprovement = 0;
                saveCheckpoint("best");
                System.out.println("✓ 保存最佳模型 (val_loss: " + String.format("%.4f", bestValLoss) + ")");
            } else {
                stepsWithoutImprovement++;
                System.out.println("连续 " + stepsWithoutImprovement + " 个epoch未改善");

                if (stepsWithoutImprovement >= patience) {
                    System.out.println("触发早停机制，训练结束");
                    break;
                }
            }
        }

        System.out.println("\n微调完成!");
        System.out.println("最佳验证损失: " + bestValLoss);
    }

    /**
     * 训练一个epoch
     */
    private void trainOneEpoch() {
        trainDataset.prepare(true);

        double epochLoss = 0.0;
        int batchCount = 0;
        long epochStartTime = System.currentTimeMillis();

        while (trainDataset.hasNext()) {
            GPT3Dataset.Batch batch = trainDataset.nextBatch();

            float stepLoss = trainStep(batch);

            epochLoss += stepLoss;
            batchCount++;
            globalStep++;

            trainLossHistory.add(stepLoss);

            if (globalStep % logInterval == 0) {
                float avgLoss = getAverageLoss(trainLossHistory, logInterval);
                System.out.printf("Epoch %d/%d | Step %d | Train Loss: %.4f%n",
                        currentEpoch + 1, maxEpochs, globalStep, avgLoss);
            }

            if (globalStep % evalInterval == 0) {
                float valLoss = evaluate();
                System.out.printf("  Validation Loss: %.4f%n", valLoss);
            }
        }

        long elapsed = System.currentTimeMillis() - epochStartTime;
        double avgEpochLoss = batchCount > 0 ? epochLoss / batchCount : 0.0;

        System.out.printf("Epoch %d 完成 | 训练损失: %.4f | 耗时: %d ms%n",
                currentEpoch + 1, avgEpochLoss, elapsed);

        trainDataset.reset();
    }

    /**
     * 训练单步
     *
     * 支持Masked Loss：当batch包含lossMask时，只在mask=1的位置计算loss（Response部分），
     * mask=0的位置（Instruction和padding）不参与loss计算和梯度更新。
     *
     * @param batch 批次数据
     * @return 损失值
     */
    private float trainStep(GPT3Dataset.Batch batch) {
        // 更新学习率（支持Warmup）
        updateLearningRate();

        NdArray inputIds  = batch.getInputIds();
        NdArray targetIds = batch.getTargetIds();
        NdArray lossMask  = batch.getLossMask();

        Variable inputVar = new Variable(inputIds);
        Variable logits   = model.predict(inputVar);

        int[] shape    = logits.getValue().getShape().getShapeDims();
        int batchSize  = shape[0];
        int seqLen     = shape[1];
        int vocabSize  = shape[2];

        Variable logits2D  = logits.reshape(Shape.of(batchSize * seqLen, vocabSize));
        Variable targetVar = new Variable(targetIds.reshape(Shape.of(batchSize * seqLen, 1)));

        Variable loss = buildLoss(logits2D, targetVar, lossMask, batchSize, seqLen);

        float lossValue = loss.getValue().getNumber().floatValue();

        model.clearGrads();
        loss.backward();
        clipGradients();
        optimizer.update();
        loss.unChainBackward();

        return lossValue;
    }

    /**
     * 在验证集上评估
     *
     * @return 验证损失
     */
    private float evaluate() {
        valDataset.prepare(false);

        double totalLoss = 0.0;
        int batchCount = 0;

        while (valDataset.hasNext()) {
            GPT3Dataset.Batch batch = valDataset.nextBatch();

            NdArray inputIds  = batch.getInputIds();
            NdArray targetIds = batch.getTargetIds();
            NdArray lossMask  = batch.getLossMask();

            Variable inputVar = new Variable(inputIds);
            Variable logits   = model.predict(inputVar);

            int[] shape    = logits.getValue().getShape().getShapeDims();
            int batchSize  = shape[0];
            int seqLen     = shape[1];
            int vocabSize  = shape[2];

            Variable logits2D  = logits.reshape(Shape.of(batchSize * seqLen, vocabSize));
            Variable targetVar = new Variable(targetIds.reshape(Shape.of(batchSize * seqLen, 1)));

            Variable loss = buildLoss(logits2D, targetVar, lossMask, batchSize, seqLen);
            totalLoss += loss.getValue().getNumber().floatValue();
            batchCount++;
        }

        valDataset.reset();
        return batchCount > 0 ? (float) (totalLoss / batchCount) : 0.0f;
    }

    /**
     * 构建损失：有lossMask时做masked loss，否则全序列loss
     */
    private Variable buildLoss(Variable logits2D, Variable targetVar,
                                NdArray lossMask, int batchSize, int seqLen) {
        if (lossMask != null) {
            Variable perTokenLoss = perTokenLossFunction.loss(targetVar, logits2D);
            NdArray mask1D = lossMask.reshape(Shape.of(batchSize * seqLen, 1));
            Variable maskVar = new Variable(mask1D);
            Variable maskedLoss = perTokenLoss.mul(maskVar);

            float validTokenCount = mask1D.sum().getNumber().floatValue();
            if (validTokenCount < 1.0f) validTokenCount = 1.0f;

            return maskedLoss.sum().mul(new Variable(NdArray.of(1.0f / validTokenCount)));
        } else {
            return lossFunction.loss(targetVar, logits2D);
        }
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
     * 更新学习率（支持线性Warmup）
     *
     * 在训练初期，学习率从0线性增加到目标学习率，
     * 避免初期梯度不稳定导致的震荡。
     */
    private void updateLearningRate() {
        float currentLr;
        if (globalStep < warmupSteps) {
            // 线性warmup：从很小的值线性增加到目标学习率
            currentLr = initialLearningRate * ((float) (globalStep + 1) / warmupSteps);
        } else {
            // warmup结束后保持目标学习率
            currentLr = initialLearningRate;
        }
        this.learningRate = currentLr;
        optimizer.setLearningRate(currentLr);
    }

    /**
     * 保存检查点
     */
    private void saveCheckpoint(String suffix) {
        try {
            String filename = String.format("gpt3_finetune_%s.model", suffix);
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
    private float getAverageLoss(List<Float> history, int n) {
        if (history.isEmpty()) return 0.0f;
        int start = Math.max(0, history.size() - n);
        float sum = 0.0f;
        for (int i = start; i < history.size(); i++) {
            sum += history.get(i);
        }
        return sum / (history.size() - start);
    }

    /**
     * 获取训练统计信息
     */
    public FinetuneStats getStats() {
        return new FinetuneStats(
                currentEpoch,
                globalStep,
                trainLossHistory.isEmpty() ? 0.0f : trainLossHistory.get(trainLossHistory.size() - 1),
                valLossHistory.isEmpty() ? 0.0f : valLossHistory.get(valLossHistory.size() - 1),
                bestValLoss,
                stepsWithoutImprovement
        );
    }

    /**
     * 微调统计信息
     */
    public static class FinetuneStats {
        public final int epoch;
        public final int step;
        public final float trainLoss;
        public final float valLoss;
        public final float bestValLoss;
        public final int patienceCount;

        public FinetuneStats(int epoch, int step, float trainLoss,
                             float valLoss, float bestValLoss, int patienceCount) {
            this.epoch = epoch;
            this.step = step;
            this.trainLoss = trainLoss;
            this.valLoss = valLoss;
            this.bestValLoss = bestValLoss;
            this.patienceCount = patienceCount;
        }

        @Override
        public String toString() {
            return String.format(
                    "FinetuneStats{epoch=%d, step=%d, trainLoss=%.4f, valLoss=%.4f, bestValLoss=%.4f, patience=%d}",
                    epoch, step, trainLoss, valLoss, bestValLoss, patienceCount);
        }
    }
}
