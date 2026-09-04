package io.leavesfly.tinyai.minimind.training;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindBlock;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.training.dataset.SFTDataset;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.io.File;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * SFT(Supervised Fine-Tuning)训练器
 * 
 * 实现指令微调,仅计算模型输出部分的损失
 * 
 * @author leavesfly
 * @since 2024
 */
public class SFTTrainer extends BaseTrainer {
    
    /**
     * 标签忽略位标记（对齐 SoftmaxCE 的 ignore_index 约定）
     */
    private static final int IGNORE_INDEX = -100;
    
    private final SFTDataset dataset;
    private final SoftmaxCrossEntropy lossFunction;
    private final Adam optimizer;
    
    private float learningRate;
    private int warmupSteps = 0;          // 学习率预热步数（默认不预热）
    private int accumulationSteps = 1;    // 梯度累积步数
    private float currentLearningRate;
    private int accumulationCounter = 0;
    
    /**
     * 构造函数
     */
    public SFTTrainer(MiniMindModel model, SFTDataset dataset) {
        super(model);
        this.dataset = dataset;
        this.lossFunction = new SoftmaxCrossEntropy();
        
        // 默认配置(较小学习率,避免灾难性遗忘)
        this.maxEpochs = 3;
        this.learningRate = 5e-5f;
        this.maxGradNorm = 1.0f;
        this.logInterval = 50;
        this.saveInterval = 500;
        this.checkpointDir = "./checkpoints/minimind/sft";
        
        // 创建优化器
        this.optimizer = new Adam(model, learningRate, 0.9f, 0.999f, 1e-8f);
    }
    
    /**
     * 配置训练参数
     */
    public SFTTrainer configure(int maxEpochs, float learningRate, float maxGradNorm) {
        this.maxEpochs = maxEpochs;
        this.learningRate = learningRate;
        this.currentLearningRate = learningRate;
        this.maxGradNorm = maxGradNorm;
        
        optimizer.setLearningRate(learningRate);
        return this;
    }

    /**
     * 设置梯度累积步数
     */
    public SFTTrainer setAccumulationSteps(int accumulationSteps) {
        this.accumulationSteps = Math.max(1, accumulationSteps);
        return this;
    }
    
    /**
     * 设置学习率预热步数
     */
    public SFTTrainer setWarmupSteps(int warmupSteps) {
        this.warmupSteps = Math.max(0, warmupSteps);
        return this;
    }
    
    // ==================== 实现抽象方法 ====================
    
    @Override
    protected float trainStep(Object batch) {
        // 更新学习率（对标 Python get_lr: cosine with 10% floor）
        updateLearningRate();
        
        SFTDataset.Batch sftBatch = (SFTDataset.Batch) batch;
        
        NdArray inputArray = sftBatch.getInput();
        // 关键：应用数据集提供的 lossMask，将 prompt / padding 位置的标签置为 -100。
        // SoftmaxCE 对负数标签会前向跳过、反向置零，因此损失仅由 assistant 回复部分贡献。
        // 若直接用原始 labels，prompt 会被当成学习目标（指令微调退化为续写整段对话），
        // 且 padding 位的标签是 pad token id(0)，会把模型训成去预测 PAD。
        NdArray labelArray = applyIgnoreIndex(
            sftBatch.getLabels(), sftBatch.getLossMask(), IGNORE_INDEX);
        
        Variable input = new Variable(inputArray);
        Variable labels = new Variable(labelArray);
        labels.setRequireGrad(false);
        
        // 前向传播（同时取出 MoE 负载均衡损失；Dense 模式下为 0 常量）
        MiniMindBlock.MoEOutput output = forwardWithAux(input);
        Variable logits = output.getOutput();
        
        int[] logitsShape = logits.getValue().getShape().getShapeDims();
        int totalTokens = logitsShape[0] * logitsShape[1];
        int vocabSize = logitsShape[2];
        
        Variable logitsReshaped = logits.reshape(Shape.of(totalTokens, vocabSize));
        Variable labelsReshaped = labels.reshape(Shape.of(totalTokens, 1));
        
        Variable ceLoss = lossFunction.loss(labelsReshaped, logitsReshaped);
        Variable loss = withMoeAuxLoss(ceLoss, output);
        float lossValue = loss.getValue().getNumber().floatValue();
        
        // 损失异常时跳过本 batch：不反向、不推进累积计数，并释放计算图避免泄漏
        if (Float.isNaN(lossValue) || Float.isInfinite(lossValue)) {
            System.err.println("警告: 损失值异常 (" + lossValue + "), 跳过此batch");
            loss.unChainBackward();
            model.clearGrads();
            accumulationCounter = 0;
            return Float.NaN;
        }
        
        // 梯度累积
        if (accumulationSteps > 1) {
            loss = loss.mul(constant(1.0f / accumulationSteps));
        }
        
        loss.backward();
        
        accumulationCounter++;
        
        if (accumulationCounter % accumulationSteps == 0) {
            clipGradients();
            optimizer.update();
            model.clearGrads();
            accumulationCounter = 0;
        }
        
        loss.unChainBackward();
        
        return lossValue;
    }
    
    /**
     * 更新学习率（对标 Python get_lr: cosine with 10% floor）
     */
    private void updateLearningRate() {
        int totalSteps = maxEpochs * dataset.getBatchCount();
        // warmup 步数不得超过总步数，否则 LR 全程停在升温段、永远到不了峰值
        int effectiveWarmup = Math.min(warmupSteps, totalSteps);
        currentLearningRate = computeScheduledLearningRate(
            learningRate, currentStep, totalSteps, effectiveWarmup);
        optimizer.setLearningRate(currentLearningRate);
    }
    
    @Override
    protected String getTrainerName() {
        return "SFT微调";
    }
    
    @Override
    protected void printTrainingInfo() {
        System.out.println("=".repeat(60));
        System.out.println("开始SFT微调");
        System.out.println("=".repeat(60));
        System.out.println("训练样本数: " + dataset.getSampleCount());
        System.out.println("批次数量: " + dataset.getBatchCount());
        System.out.println("最大轮次: " + maxEpochs);
        System.out.println("学习率: " + learningRate);
        System.out.println("=".repeat(60));
    }
    
    @Override
    protected void prepareDataset() {
        dataset.prepare(true);
    }
    
    @Override
    protected boolean hasNextBatch() {
        return dataset.hasNextBatch();
    }
    
    @Override
    protected Object getNextBatch() {
        return dataset.getNextBatch();
    }
    
    @Override
    protected void resetDataset() {
        dataset.reset();
    }
    
    @Override
    protected String getCheckpointPrefix() {
        return "sft_checkpoint";
    }
}