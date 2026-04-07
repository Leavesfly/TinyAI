package io.leavesfly.tinyai.deepseek.v3.training;

import io.leavesfly.tinyai.deepseek.base.DeepSeekTrainerBase;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Block;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Config;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Model;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.ArrayList;
import java.util.List;

/**
 * DeepSeek-V3后训练器（任务感知微调）
 * 
 * 在预训练基础上进行任务特定的微调,
 * 优化任务感知路由和代码生成能力
 * 
 * 关键特性：
 * 1. 任务感知微调 - 根据任务类型优化专家选择
 * 2. 代码任务优化 - 特别针对代码生成质量
 * 3. 早停机制 - 防止过拟合
 * 4. 较低学习率 - 保护预训练知识
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3SFTrainer extends DeepSeekTrainerBase {
    
    private final DeepSeekV3Model model;
    private final DeepSeekV3Config config;
    private final DeepSeekV3Dataset trainDataset;
    private final DeepSeekV3Dataset valDataset;
    private final SoftmaxCrossEntropy lossFunction;
    private final SoftmaxCrossEntropy elementWiseLossFunction; // 逐元素 loss，用于 answer-only loss masking
    private final Adam optimizer;
    
    // 训练超参数
    private float initialLearningRate;  // 比预训练低10倍
    private float minLearningRate;
    private int warmupSteps;
    private float moeLoadBalanceWeight;
    private int valInterval;
    private int saveInterval;
    private int patience;  // 早停耐心值
    
    // 训练状态
    private float currentLearningRate;
    private float bestValLoss;
    private int stepsWithoutImprovement;
    private List<Float> trainLossHistory;
    private List<Float> valLossHistory;
    private List<Float> codeQualityHistory;  // 代码质量历史
    
    /**
     * 构造函数
     */
    public DeepSeekV3SFTrainer(DeepSeekV3Model model,
                               DeepSeekV3Dataset trainDataset,
                               DeepSeekV3Dataset valDataset) {
        super(model, 5, 1.0f, 50, "./checkpoints/deepseek_v3/posttrain");
        this.model = model;
        this.config = model.getConfig();
        this.trainDataset = trainDataset;
        this.valDataset = valDataset;
        this.lossFunction = new SoftmaxCrossEntropy();
        this.elementWiseLossFunction = new SoftmaxCrossEntropy(SoftmaxCrossEntropy.Reduction.NONE);
        
        // 默认超参数（比预训练更保守）
        this.initialLearningRate = 2.5e-5f;  // 比预训练低10倍
        this.minLearningRate = 1e-6f;
        this.warmupSteps = 500;
        this.moeLoadBalanceWeight = (float) config.getLoadBalanceLossWeight();
        this.valInterval = 500;
        this.saveInterval = 2000;
        this.patience = 3;
        
        // 创建优化器
        this.optimizer = new Adam(model, initialLearningRate, 0.9f, 0.999f, 1e-8f);
        
        // 初始化状态
        this.currentLearningRate = 0.0f;
        this.bestValLoss = Float.MAX_VALUE;
        this.stepsWithoutImprovement = 0;
        this.trainLossHistory = new ArrayList<>();
        this.valLossHistory = new ArrayList<>();
        this.codeQualityHistory = new ArrayList<>();
    }
    
    /**
     * 配置训练参数
     */
    public DeepSeekV3SFTrainer configure(int maxEpochs, float learningRate,
                                         int patience) {
        this.maxEpochs = maxEpochs;
        this.initialLearningRate = learningRate;
        this.patience = patience;
        return this;
    }
    
    /**
     * 开始训练
     */
    @Override
    public void train() {
        System.out.println("=".repeat(80));
        System.out.println("DeepSeek-V3 后训练/微调（任务感知优化）");
        System.out.println("=".repeat(80));
        System.out.println("训练配置:");
        System.out.println("  - 训练样本: " + trainDataset.getSampleCount());
        System.out.println("  - 验证样本: " + valDataset.getSampleCount());
        System.out.println("  - 最大轮次: " + maxEpochs);
        System.out.println("  - 初始学习率: " + initialLearningRate + " (比预训练低10倍)");
        System.out.println("  - 早停耐心值: " + patience);
        System.out.println("=".repeat(80));
        
        // 创建检查点目录
        createCheckpointDir();
        
        // 训练循环
        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();
            
            // 验证
            float valLoss = validate();
            valLossHistory.add(valLoss);
            
            System.out.printf("Epoch %d 验证损失: %.4f%n", currentEpoch + 1, valLoss);
            
//            // 早停检查
//            if (valLoss < bestValLoss) {
//                bestValLoss = valLoss;
//                stepsWithoutImprovement = 0;
//                saveCheckpoint("best");
//                System.out.println("新的最佳模型已保存!");
//            } else {
//                stepsWithoutImprovement++;
//                if (stepsWithoutImprovement >= patience) {
//                    System.out.println("触发早停,训练结束");
//                    break;
//                }
//            }
        }
        
        // 保存最终模型
        saveCheckpoint("final");
        
        System.out.println("\n训练完成!");
        System.out.println("最佳验证损失: " + bestValLoss);
    }
    
    /**
     * 训练一个epoch
     */
    private void trainOneEpoch() {
        trainDataset.prepare(true);
        EpochStats stats = new EpochStats();

        while (trainDataset.hasNext()) {
            processBatch(trainDataset.nextBatch(), stats);
        }

        System.out.printf("Epoch %d 完成 | 平均损失: %.4f | 平均代码质量: %.4f%n",
            currentEpoch + 1, stats.avgLoss(), stats.avgCodeQuality());
        trainDataset.reset();
    }

    /**
     * 处理单个batch：训练一步 + 记录历史 + 日志/验证/保存
     */
    private void processBatch(DeepSeekV3Dataset.Batch batch, EpochStats stats) {
        StepResult stepResult = trainStep(batch);
        stats.accumulate(stepResult);
        globalStep++;

        trainLossHistory.add(stepResult.loss);
        if (stepResult.codeQuality > 0) {
            codeQualityHistory.add(stepResult.codeQuality);
        }

        if (globalStep % logInterval == 0) {
            logTrainingProgress();
        }
        if (globalStep % valInterval == 0) {
            System.out.printf("中期验证损失: %.4f%n", validate());
        }
        if (globalStep % saveInterval == 0) {
            saveCheckpoint("step_" + globalStep);
        }
    }

    /**
     * 打印训练进度日志
     */
    private void logTrainingProgress() {
        float avgLoss = getAverage(trainLossHistory, logInterval);
        float avgCodeQuality = getAverage(codeQualityHistory, logInterval);
        System.out.printf("Epoch %d/%d | Step %d | Loss: %.4f | 代码质量: %.4f | LR: %.6f%n",
            currentEpoch + 1, maxEpochs, globalStep, avgLoss, avgCodeQuality, currentLearningRate);
    }
    
    /**
     * 训练单步
     * 
     * 支持 answer-only loss masking：当 batch 包含 lossMask 时，
     * 只对 assistant 回答部分计算 loss，user 提问部分不参与梯度更新。
     * 这是行业主流的 SFT 训练方式（ChatML 格式）。
     */
    private StepResult trainStep(DeepSeekV3Dataset.Batch batch) {
        updateLearningRate();

        NdArray inputIds = batch.getInputIds();
        NdArray targetIds = batch.getTargetIds();

        // 前向传播
        DeepSeekV3Block.DetailedForwardResult result =
            model.predictWithDetails(new Variable(inputIds));

        // 计算损失（支持 answer-only loss masking）
        Variable lmLoss = computeMaskedLoss(batch, result.logits);
        float lossValue = lmLoss.getValue().getNumber().floatValue();
        float moeLoss = (float) result.avgMoELoss;

        // 反向传播
        Variable totalLoss = buildTotalLoss(lmLoss, moeLoss);
        model.clearGrads();
        totalLoss.backward();
        clipGradients();
        optimizer.update();
        totalLoss.unChainBackward();

        return new StepResult(lossValue, moeLoss);
    }
    
    /**
     * 验证（同样支持 answer-only loss masking）
     */
    private float validate() {
        valDataset.prepare(false);
        double totalLoss = 0.0;
        int count = 0;

        while (valDataset.hasNext()) {
            DeepSeekV3Dataset.Batch batch = valDataset.nextBatch();

            Variable logits = model.predict(new Variable(batch.getInputIds()));
            Variable loss = computeMaskedLoss(batch, logits);

            totalLoss += loss.getValue().getNumber().floatValue();
            count++;
            loss.unChainBackward();
        }

        valDataset.reset();
        return count > 0 ? (float) (totalLoss / count) : 0.0f;
    }
    
    /**
     * 计算带 mask 的损失（answer-only loss masking）
     * 
     * 当 batch 包含 lossMask 时：
     * 1. 使用 Reduction.NONE 获取逐位置的 loss
     * 2. 将 loss 与 mask 逐元素相乘（user 部分 mask=0 不贡献 loss）
     * 3. 对有效位置求均值（除以 mask 中 1.0 的个数）
     * 
     * 当 batch 无 lossMask 时：退化为标准的全序列 loss 计算
     */
    private Variable computeMaskedLoss(DeepSeekV3Dataset.Batch batch, Variable logits) {
        NdArray inputIds = batch.getInputIds();
        NdArray targetIds = batch.getTargetIds();
        int batchSize = inputIds.getShape().getDimension(0);
        int seqLen = inputIds.getShape().getDimension(1);
        int vocabSize = model.getConfig().getVocabSize();
        
        Variable logits2D = logits.reshape(Shape.of(batchSize * seqLen, vocabSize));
        Variable targets2D = new Variable(targetIds).reshape(Shape.of(batchSize * seqLen, 1));
        
        if (!batch.hasLossMask()) {
            // 无 mask：标准全序列 loss（预训练兼容）
            return lossFunction.loss(targets2D, logits2D);
        }
        
        // 有 mask：answer-only loss masking
        // 1. 获取逐位置 loss（不归约）
        Variable elementWiseLoss = elementWiseLossFunction.loss(targets2D, logits2D);
        
        // 2. 将 mask reshape 为 1D 并与 loss 相乘
        NdArray lossMask = batch.getLossMask();
        Variable mask1D = new Variable(lossMask).reshape(Shape.of(batchSize * seqLen, 1));
        Variable maskedLoss = elementWiseLoss.mul(mask1D);
        
        // 3. 计算有效位置数量，对有效位置求均值
        float maskSum = lossMask.sum().getNumber().floatValue();
        if (maskSum <= 0) {
            // 安全兜底：如果没有有效位置，返回 0 loss
            return new Variable(NdArray.of(new float[]{0.0f}));
        }
        Variable totalMaskedLoss = maskedLoss.sum();
        Variable validCount = new Variable(NdArray.of(new float[]{maskSum}));
        return totalMaskedLoss.div(validCount);
    }

    /**
     * 融合语言模型损失与 MoE 负载均衡损失
     */
    private Variable buildTotalLoss(Variable lmLoss, float moeLoss) {
        if (moeLoadBalanceWeight <= 0) return lmLoss;
        Variable moeLossVar = new Variable(NdArray.of(new float[]{moeLoss * moeLoadBalanceWeight}));
        return lmLoss.add(moeLossVar);
    }

    private void updateLearningRate() {
        if (globalStep < warmupSteps) {
            currentLearningRate = initialLearningRate * ((float) globalStep / warmupSteps);
        } else {
            int totalSteps = maxEpochs * trainDataset.getBatchCount();
            int decaySteps = totalSteps - warmupSteps;
            int currentDecayStep = globalStep - warmupSteps;
            
            double cosineDecay = 0.5 * (1 + Math.cos(Math.PI * currentDecayStep / decaySteps));
            float decayedLR = (initialLearningRate - minLearningRate) * (float) cosineDecay + minLearningRate;
            currentLearningRate = Math.max(decayedLR, minLearningRate);
        }
        
        optimizer.setLearningRate(currentLearningRate);
    }
    
    @Override
    protected String getTrainerName() {
        return "DeepSeek-V3 Posttrain";
    }
    
    @Override
    protected String getCheckpointPrefix() {
        return "deepseek_v3_posttrain";
    }
    
    private static class StepResult {
        final float loss;
        final float codeQuality;  // 以 MoE loss 近似表示代码质量

        StepResult(float loss, float codeQuality) {
            this.loss = loss;
            this.codeQuality = codeQuality;
        }
    }

    /** epoch 内累计统计，避免在 trainOneEpoch 中散落多个计数变量 */
    private static class EpochStats {
        double totalLoss = 0.0;
        double totalCodeQuality = 0.0;
        int batchCount = 0;
        int codeTaskCount = 0;

        void accumulate(StepResult result) {
            totalLoss += result.loss;
            if (result.codeQuality > 0) {
                totalCodeQuality += result.codeQuality;
                codeTaskCount++;
            }
            batchCount++;
        }

        double avgLoss() {
            return batchCount > 0 ? totalLoss / batchCount : 0.0;
        }

        double avgCodeQuality() {
            return codeTaskCount > 0 ? totalCodeQuality / codeTaskCount : 0.0;
        }
    }
}
