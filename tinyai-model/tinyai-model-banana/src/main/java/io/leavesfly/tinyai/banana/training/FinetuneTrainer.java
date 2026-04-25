package io.leavesfly.tinyai.banana.training;

import io.leavesfly.tinyai.banana.block.BananaBlock;
import io.leavesfly.tinyai.banana.config.BananaConfig;
import io.leavesfly.tinyai.banana.model.BananaModel;
import io.leavesfly.tinyai.banana.training.dataset.BananaDataset;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.util.Config;

import java.io.File;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Banana 多模态微调器。
 *
 * <p>在预训练模型基础上进行任务特定的微调训练，默认采用与 {@link PretrainTrainer} 相同的
 * <b>CLIP 风格对称 InfoNCE 对比损失</b>，确保跨模态对齐特性在微调阶段得以保留。</p>
 *
 * <p>与预训练的主要区别：</p>
 * <ul>
 *   <li>更小的学习率（通常小一个数量级）；</li>
 *   <li>验证集评估（每个 epoch 评估一次）；</li>
 *   <li>早停机制：验证损失未改进超过 {@code patience} 轮自动停止；</li>
 *   <li>最佳模型保存：仅当验证损失创新低时覆盖保存。</li>
 * </ul>
 *
 * <p>历史实现曾错误地把 {@code forwardMultiModal} 返回的 vocab logits 当作 embedding
 * 做 pool+MSE，现已修正为直接使用 {@code encodeText + encodeImage} 得到的 hidden 特征做对齐。</p>
 *
 * @author TinyAI
 * @since 2024
 */
public class FinetuneTrainer {

    private final BananaModel model;
    private final BananaBlock bananaBlock;
    private final BananaConfig config;
    private final BananaDataset trainDataset;
    private final BananaDataset valDataset;
    private final Adam optimizer;

    // 微调超参数(与预训练不同)
    private int maxEpochs;
    private float learningRate;           // 微调学习率通常更小
    private float maxGradNorm;
    private int logInterval;
    private int patience;                 // 早停耐心值
    private String checkpointDir;

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
     * @param model        预训练的Banana模型
     * @param trainDataset 训练数据集
     * @param valDataset   验证数据集
     */
    public FinetuneTrainer(BananaModel model, BananaDataset trainDataset, BananaDataset valDataset) {
        if (model == null) {
            throw new IllegalArgumentException("model 不能为 null");
        }
        if (trainDataset == null) {
            throw new IllegalArgumentException("trainDataset 不能为 null");
        }
        if (valDataset == null) {
            throw new IllegalArgumentException("valDataset 不能为 null");
        }
        this.model = model;
        this.bananaBlock = model.getBananaBlock();
        this.config = model.getConfig();
        this.trainDataset = trainDataset;
        this.valDataset = valDataset;

        // 默认配置(微调学习率更小)
        this.maxEpochs = 5;
        this.learningRate = 1e-5f;     // 比预训练小10倍
        this.maxGradNorm = 1.0f;
        this.logInterval = 50;
        this.patience = 3;
        this.checkpointDir = "./checkpoints/banana/finetune";

        // 创建优化器
        this.optimizer = new Adam(model, learningRate, 0.9f, 0.999f, 1e-8f);

        // 初始化状态
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
    public FinetuneTrainer configure(int maxEpochs, float learningRate, int patience) {
        this.maxEpochs = maxEpochs;
        this.learningRate = learningRate;
        this.patience = patience;
        this.optimizer.setLearningRate(learningRate);
        return this;
    }

    /**
     * 设置检查点配置
     *
     * @param checkpointDir 检查点目录
     * @return this
     */
    public FinetuneTrainer setCheckpoint(String checkpointDir) {
        this.checkpointDir = checkpointDir;
        return this;
    }

    /**
     * 开始微调
     */
    public void train() {
        System.out.println("=".repeat(60));
        System.out.println("开始Banana多模态微调");
        System.out.println("=".repeat(60));
        System.out.println("模型配置: Banana " + config.getClass().getSimpleName());
        long totalParams = calculateTotalParams();
        System.out.println("总参数量: " + totalParams + " (" + String.format("%.2fM", totalParams / 1_000_000.0) + ")");
        System.out.println("训练样本数: " + trainDataset.getSampleCount());
        System.out.println("验证样本数: " + valDataset.getSampleCount());
        System.out.println("最大轮次: " + maxEpochs);
        System.out.println("微调学习率: " + learningRate);
        System.out.println("早停耐心值: " + patience);
        System.out.println("=".repeat(60));

        // 创建检查点目录
        createCheckpointDir();

        // 微调循环
        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();

            // 验证
            float valLoss = evaluate();
            valLossHistory.add(valLoss);

            System.out.println(String.format(
                    "Epoch %d 验证损失: %.4f | 最佳验证损失: %.4f",
                    currentEpoch + 1, valLoss, bestValLoss
            ));

            // 检查是否改进
            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                stepsWithoutImprovement = 0;
                saveBestModel();
            } else {
                stepsWithoutImprovement++;
                System.out.println("未改进轮数: " + stepsWithoutImprovement + "/" + patience);
            }

            // 早停检查
            if (stepsWithoutImprovement >= patience) {
                System.out.println("触发早停,微调结束");
                break;
            }
        }

        System.out.println("微调完成!");
        System.out.println("最佳验证损失: " + bestValLoss);
    }

    /**
     * 训练一个epoch
     */
    private void trainOneEpoch() {
        trainDataset.prepare(true);  // 打乱数据

        // 设置训练模式
        boolean prevTrain = io.leavesfly.tinyai.util.Config.train;
        io.leavesfly.tinyai.util.Config.train = true;

        try {
            double epochLoss = 0.0;
            int batchCount = 0;

            long epochStartTime = System.currentTimeMillis();

            while (trainDataset.hasNextBatch()) {
                BananaDataset.Batch batch = trainDataset.getNextBatch();

                // 训练一步
                float stepLoss = trainStep(batch);

                epochLoss += stepLoss;
                batchCount++;
                globalStep++;

                // 记录损失
                trainLossHistory.add(stepLoss);

                // 打印日志
                if (globalStep % logInterval == 0) {
                    double avgLoss = trainLossHistory.stream()
                            .skip(Math.max(0, trainLossHistory.size() - logInterval))
                            .mapToDouble(Float::doubleValue)
                            .average()
                            .orElse(0.0);

                    System.out.printf("Epoch %d/%d | Step %d | Train Loss: %.4f%n",
                            currentEpoch + 1, maxEpochs, globalStep, avgLoss);
                }
            }

            long epochEndTime = System.currentTimeMillis();
            double avgEpochLoss = batchCount > 0 ? epochLoss / batchCount : 0.0;

            System.out.println(String.format(
                    "Epoch %d 训练完成 | 平均损失: %.4f | 耗时: %d ms",
                    currentEpoch + 1, avgEpochLoss, epochEndTime - epochStartTime
            ));
        } finally {
            Config.train = prevTrain;
        }

        trainDataset.reset();
    }

    /**
     * 训练一步。
     *
     * <p>采用 CLIP 风格对称 InfoNCE 对比损失：</p>
     * <ol>
     *   <li>{@code encodeText} + {@code encodeImage} 得到序列级 hidden 特征；</li>
     *   <li>可选通过 fusion 层双向增强（若启用）；</li>
     *   <li>序列维度均值池化 → L2 归一化 → 对比损失。</li>
     * </ol>
     *
     * @param batch 批次数据
     * @return 损失值
     */
    private float trainStep(BananaDataset.Batch batch) {
        Variable textVar = new Variable(batch.getTextInput());
        Variable imageVar = new Variable(batch.getImageInput());

        // 编码 + 可选双向融合 + 池化 + L2 归一化（与 PretrainTrainer 共用口径）
        Variable[] pair = BaseBananaTrainer.encodeAndPoolPair(model, bananaBlock, textVar, imageVar);
        Variable textEmb = pair[0];
        Variable imageEmb = pair[1];

        // CLIP 对比损失
        Variable loss = BaseBananaTrainer.computeContrastiveLoss(
                textEmb, imageEmb, BaseBananaTrainer.DEFAULT_LOGIT_SCALE);

        float lossValue = loss.getValue().getNumber().floatValue();

        // 梯度更新
        model.clearGrads();
        loss.backward();
        clipGradients();
        optimizer.update();
        loss.unChainBackward();

        return lossValue;
    }

    /**
     * 评估验证集。
     *
     * <p>使用与 {@link #trainStep(BananaDataset.Batch)} 相同的损失口径（对比损失），
     * 只是关闭训练模式，不更新参数。</p>
     *
     * @return 平均验证损失
     */
    private float evaluate() {
        valDataset.prepare(false);

        boolean prevTrain = io.leavesfly.tinyai.util.Config.train;
        io.leavesfly.tinyai.util.Config.train = false;

        try {
            double totalLoss = 0.0;
            int batchCount = 0;

            while (valDataset.hasNextBatch()) {
                BananaDataset.Batch batch = valDataset.getNextBatch();

                Variable textVar = new Variable(batch.getTextInput());
                Variable imageVar = new Variable(batch.getImageInput());

                // 验证口径与 trainStep 完全一致，避免 train/eval 评估偏差
                Variable[] pair = BaseBananaTrainer.encodeAndPoolPair(
                        model, bananaBlock, textVar, imageVar);
                Variable textEmb = pair[0];
                Variable imageEmb = pair[1];

                Variable loss = BaseBananaTrainer.computeContrastiveLoss(
                        textEmb, imageEmb, BaseBananaTrainer.DEFAULT_LOGIT_SCALE);

                totalLoss += loss.getValue().getNumber().floatValue();
                batchCount++;

                // 验证时不参与反向传播，断开计算图避免内存累积
                loss.unChainBackward();
            }

            return batchCount > 0 ? (float) (totalLoss / batchCount) : 0.0f;
        } finally {
            io.leavesfly.tinyai.util.Config.train = prevTrain;
        }
    }

    /**
     * 梯度裁剪
     */
    private void clipGradients() {
        BaseBananaTrainer.clipGradients(model, maxGradNorm);
    }

    /**
     * 保存最佳模型
     */
    private void saveBestModel() {
        String filename = "best_model.model";
        String filepath = Paths.get(checkpointDir, filename).toString();

        try {
            model.save(new File(filepath));
            System.out.println("最佳模型已保存: " + filepath);
        } catch (Exception e) {
            System.err.println("保存最佳模型失败: " + e.getMessage());
        }
    }

    /**
     * 创建检查点目录
     */
    private void createCheckpointDir() {
        BaseBananaTrainer.createCheckpointDir(checkpointDir);
    }

    /**
     * 计算总参数量
     */
    private long calculateTotalParams() {
        return BaseBananaTrainer.calculateTotalParams(model);
    }

    /**
     * 获取训练损失历史
     */
    public List<Float> getTrainLossHistory() {
        return new ArrayList<>(trainLossHistory);
    }

    /**
     * 获取验证损失历史
     */
    public List<Float> getValLossHistory() {
        return new ArrayList<>(valLossHistory);
    }

    /**
     * 获取最佳验证损失
     */
    public float getBestValLoss() {
        return bestValLoss;
    }
}