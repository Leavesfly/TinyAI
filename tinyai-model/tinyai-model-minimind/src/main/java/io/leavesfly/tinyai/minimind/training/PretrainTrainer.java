package io.leavesfly.tinyai.minimind.training;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindBlock;
import io.leavesfly.tinyai.minimind.model.MiniMindConfig;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.training.dataset.PretrainDataset;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * MiniMind预训练Trainer
 * 
 * 实现因果语言建模(Causal Language Modeling)预训练
 * 支持学习率调度、梯度裁剪、检查点保存等功能
 * 
 * @author leavesfly
 * @since 2024
 */
public class PretrainTrainer extends BaseTrainer {
    
    private final MiniMindConfig config;
    private final PretrainDataset dataset;
    private final SoftmaxCrossEntropy lossFunction;
    private final Adam optimizer;
    
    // 训练配置
    private float initialLearningRate;
    private int warmupSteps;     // 学习率预热步数
    private int accumulationSteps = 1;  // 梯度累积步数（对标 Python accumulation_steps）
    
    // 训练状态
    private float currentLearningRate;
    private int accumulationCounter = 0;  // 梯度累积计数器
    
    /**
     * 构造函数
     * 
     * @param model 模型
     * @param dataset 预训练数据集
     */
    public PretrainTrainer(MiniMindModel model, PretrainDataset dataset) {
        super(model);
        this.config = model.getConfig();
        this.dataset = dataset;
        this.lossFunction = new SoftmaxCrossEntropy();
        
        // 默认配置
        this.maxEpochs = 10;
        this.initialLearningRate = 1e-4f;
        this.maxGradNorm = 1.0f;
        this.warmupSteps = 1000;
        this.logInterval = 100;
        this.saveInterval = 1000;
        this.checkpointDir = "./checkpoints";
        
        // 创建优化器(Adam，本框架未实现权重衰减)
        this.optimizer = new Adam(model, initialLearningRate, 0.9f, 0.999f, 1e-8f);
        
        // 初始化状态
        this.currentLearningRate = 0.0f;
    }
    
    /**
     * 配置训练参数
     */
    public PretrainTrainer configure(int maxEpochs, float learningRate, 
                                      int warmupSteps, float maxGradNorm) {
        this.maxEpochs = maxEpochs;
        this.initialLearningRate = learningRate;
        this.warmupSteps = warmupSteps;
        this.maxGradNorm = maxGradNorm;
        return this;
    }

    /**
     * 设置梯度累积步数（对标 Python accumulation_steps）
     */
    public PretrainTrainer setAccumulationSteps(int accumulationSteps) {
        this.accumulationSteps = Math.max(1, accumulationSteps);
        return this;
    }
    
    // ==================== 实现抽象方法 ====================
    
    @Override
    protected float trainStep(Object batch) {
        // 更新学习率
        updateLearningRate();
        
        PretrainDataset.Batch pretrainBatch = (PretrainDataset.Batch) batch;
        
        NdArray inputArray = pretrainBatch.getInput();
        NdArray targetArray = pretrainBatch.getTarget();
        
        Variable input = new Variable(inputArray);
        Variable target = new Variable(targetArray);
        
        // 前向传播（同时取出 MoE 负载均衡损失；Dense 模式下为 0 常量）
        MiniMindBlock.MoEOutput output = forwardWithAux(input);
        Variable logits = output.getOutput();
        
        int[] logitsShape = logits.getValue().getShape().getShapeDims();
        int totalTokens = logitsShape[0] * logitsShape[1];
        int vocabSize = logitsShape[2];
        
        Variable logitsReshaped = logits.reshape(Shape.of(totalTokens, vocabSize));
        Variable targetReshaped = target.reshape(Shape.of(totalTokens, 1));
        
        // 计算损失：语言建模 CE + MoE 负载均衡辅助损失
        Variable ceLoss = lossFunction.loss(targetReshaped, logitsReshaped);
        Variable loss = withMoeAuxLoss(ceLoss, output);
        float lossValue = loss.getValue().getNumber().floatValue();
        
        // 损失异常时跳过本 batch：不反向、不推进累积计数，并释放计算图
        // （若继续 backward，NaN 会污染已累积的梯度，使后续所有参数变成 NaN）
        if (Float.isNaN(lossValue) || Float.isInfinite(lossValue)) {
            System.err.println("警告: 损失值异常 (" + lossValue + "), 跳过此batch");
            loss.unChainBackward();
            model.clearGrads();
            accumulationCounter = 0;
            return Float.NaN;
        }
        
        // 梯度累积：将损失除以累积步数（对标 Python: loss = loss / accumulation_steps）
        if (accumulationSteps > 1) {
            loss = loss.mul(constant(1.0f / accumulationSteps));
        }
        
        // 反向传播（累积梯度）
        loss.backward();
        
        accumulationCounter++;
        
        // 每 accumulation_steps 步更新一次参数
        if (accumulationCounter % accumulationSteps == 0) {
            // 梯度裁剪
            clipGradients();
            
            // 更新参数
            optimizer.update();
            
            // 清除梯度
            model.clearGrads();
            
            accumulationCounter = 0;
        }
        
        // 断开计算图
        loss.unChainBackward();
        
        return lossValue;
    }
    
    @Override
    protected String getTrainerName() {
        return "预训练";
    }
    
    @Override
    protected void printTrainingInfo() {
        System.out.println("=".repeat(60));
        System.out.println("开始预训练");
        System.out.println("=".repeat(60));
        System.out.println("模型配置: " + config.getModelSize());
        long totalParams = 0;
        for (var param : model.getAllParams().values()) {
            int[] dims = param.getValue().getShape().getShapeDims();
            long size = 1;
            for (int d : dims) size *= d;
            totalParams += size;
        }
        System.out.println("总参数量: " + totalParams + " (" + String.format("%.2fM", totalParams / 1_000_000.0) + ")");
        System.out.println("训练样本数: " + dataset.getSampleCount());
        System.out.println("批次数量: " + dataset.getBatchCount());
        System.out.println("最大轮次: " + maxEpochs);
        System.out.println("初始学习率: " + initialLearningRate);
        System.out.println("=".repeat(60));
    }
    
    @Override
    protected void prepareDataset() {
        dataset.prepare(true);  // 打乱数据
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
        return "checkpoint";
    }
    
    @Override
    protected void printTrainingLog() {
        double avgLoss = lossHistory.stream()
            .skip(Math.max(0, lossHistory.size() - logInterval))
            .mapToDouble(Float::doubleValue)
            .average()
            .orElse(0.0);
        
        System.out.printf("Epoch %d/%d | Step %d | Loss: %.4f | LR: %.6f%n",
            currentEpoch + 1, maxEpochs, currentStep, avgLoss, currentLearningRate);
    }
    
    // ==================== PretrainTrainer 特有方法 ====================
    
    /**
     * 更新学习率（对标 Python get_lr）
     * <p>
     * 预热段：线性升温，首步为 {@code lr/warmupSteps} 而非 0（避开一次空更新）；
     * 退火段：{@code lr * (0.1 + 0.45 * (1 + cos(π * step / total_steps)))}，即余弦退火到 10% floor。
     */
    private void updateLearningRate() {
        int totalSteps = maxEpochs * dataset.getBatchCount();
        // warmup 步数不得超过总步数，否则 LR 全程停在升温段、永远到不了峰值
        int effectiveWarmup = Math.min(warmupSteps, totalSteps);
        currentLearningRate = computeScheduledLearningRate(
            initialLearningRate, currentStep, totalSteps, effectiveWarmup);
        optimizer.setLearningRate(currentLearningRate);
    }
}