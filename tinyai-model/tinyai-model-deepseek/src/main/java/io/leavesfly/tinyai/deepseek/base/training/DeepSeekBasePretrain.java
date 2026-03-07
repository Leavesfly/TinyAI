package io.leavesfly.tinyai.deepseek.base.training;

import io.leavesfly.tinyai.deepseek.base.utils.CheckpointManager;
import io.leavesfly.tinyai.deepseek.base.utils.TrainingMonitor;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ml.optimize.Optimizer;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * DeepSeek 共享预训练器基类
 * 
 * 提供 R1 和 V3 预训练的通用逻辑：
 * 1. 训练循环（Epoch → Batch → Step）
 * 2. 学习率调度（Warmup + Cosine Decay）
 * 3. 梯度裁剪
 * 4. 检查点保存
 * 5. 训练监控
 * 
 * 子类只需实现：
 * - computeLoss(): 计算模型特定的损失（如 MoE 损失）
 * - createOptimizer(): 创建优化器（SGD/Adam）
 * 
 * @author leavesfly
 * @version 1.0
 */
public abstract class DeepSeekBasePretrain<M extends Model> {
    
    protected final M model;
    protected final SoftmaxCrossEntropy lossFunction;
    protected Optimizer optimizer;
    
    // 训练超参数（protected 供子类访问）
    protected int maxEpochs;
    protected float initialLearningRate;
    protected float minLearningRate;
    protected int warmupSteps;
    protected float maxGradNorm;
    protected int logInterval;
    protected int saveInterval;
    protected String checkpointDir;
    
    // 训练状态
    protected int currentEpoch;
    protected int globalStep;
    protected float currentLearningRate;
    protected List<Float> lossHistory;
    
    /**
     * 构造函数
     */
    public DeepSeekBasePretrain(M model) {
        this.model = model;
        this.lossFunction = new SoftmaxCrossEntropy();
        
        // 默认超参数
        this.maxEpochs = 10;
        this.initialLearningRate = 2.5e-4f;
        this.minLearningRate = 1e-5f;
        this.warmupSteps = 2000;
        this.maxGradNorm = 1.0f;
        this.logInterval = 100;
        this.saveInterval = 5000;
        this.checkpointDir = "./checkpoints/deepseek_pretrain";
        
        // 初始化状态
        this.currentEpoch = 0;
        this.globalStep = 0;
        this.currentLearningRate = 0.0f;
        this.lossHistory = new ArrayList<>();
        
        // 创建优化器（由子类实现）
        this.optimizer = createOptimizer();
    }
    
    /**
     * 创建优化器（子类实现：R1 使用 SGD，V3 使用 Adam）
     */
    protected abstract Optimizer createOptimizer();
    
    /**
     * 计算损失（子类实现，可添加 MoE 损失等）
     * 
     * @param inputIds 输入序列 [batch_size, seq_len]
     * @param targetIds 目标序列 [batch_size, seq_len]
     * @return 损失值
     */
    protected abstract Variable computeLoss(NdArray inputIds, NdArray targetIds);
    
    /**
     * 配置训练参数
     */
    public DeepSeekBasePretrain<M> configure(int maxEpochs, float learningRate,
                                             int warmupSteps, float maxGradNorm) {
        this.maxEpochs = maxEpochs;
        this.initialLearningRate = learningRate;
        this.warmupSteps = warmupSteps;
        this.maxGradNorm = maxGradNorm;
        optimizer.setLearningRate(learningRate);
        return this;
    }
    
    /**
     * 设置检查点目录
     */
    public DeepSeekBasePretrain<M> setCheckpoint(String dir, int interval) {
        this.checkpointDir = dir;
        this.saveInterval = interval;
        return this;
    }
    
    /**
     * 主训练流程
     */
    public void train() {
        System.out.println("=".repeat(70));
        System.out.println("DeepSeek 预训练");
        System.out.println("=".repeat(70));
        System.out.println("模型: " + model.getName());
        System.out.println("最大轮次: " + maxEpochs);
        System.out.println("初始学习率: " + initialLearningRate);
        System.out.println("Warmup步数: " + warmupSteps);
        System.out.println("=".repeat(70));
        
        createCheckpointDir();
        
        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();
        }
        
        saveCheckpoint("final");
        System.out.println("\n预训练完成!");
    }
    
    /**
     * 训练一个Epoch
     */
    protected void trainOneEpoch() {
        prepareDataset(true);
        
        double epochLoss = 0.0;
        int count = 0;
        
        while (hasNextBatch()) {
            Batch batch = nextBatch();
            
            // 训练一个batch
            float stepLoss = trainOneBatch(batch.getInputIds(), batch.getTargetIds());
            
            epochLoss += stepLoss;
            count++;
            globalStep++;
            
            // 日志输出
            if (globalStep % logInterval == 0) {
                System.out.printf("Epoch %d | Step %d | Loss: %.4f | LR: %.6f%n",
                    currentEpoch + 1, globalStep, stepLoss, currentLearningRate);
            }
            
            // 保存检查点
            if (globalStep % saveInterval == 0) {
                saveCheckpoint(String.format("step_%d", globalStep));
            }
        }
        
        float avgLoss = (float) (epochLoss / count);
        lossHistory.add(avgLoss);
        
        System.out.printf("\nEpoch %d 完成 | 平均Loss: %.4f%n", currentEpoch + 1, avgLoss);
    }
    
    /**
     * 训练一个Batch
     */
    protected float trainOneBatch(NdArray inputIds, NdArray targetIds) {
        // 更新学习率
        updateLearningRate();
        
        // 前向传播 + 计算损失
        Variable loss = computeLoss(inputIds, targetIds);
        
        float lossValue = loss.getValue().getNumber().floatValue();
        
        // 反向传播
        model.clearGrads();
        loss.backward();
        
        // 梯度裁剪
        clipGradients();
        
        // 更新参数
        optimizer.update();
        
        // 断开计算图
        loss.unChainBackward();
        
        return lossValue;
    }
    
    /**
     * 更新学习率（Warmup + Cosine Decay）
     */
    protected void updateLearningRate() {
        if (globalStep < warmupSteps) {
            // Warmup阶段：线性增长
            currentLearningRate = initialLearningRate * ((float) globalStep / warmupSteps);
        } else {
            // Cosine衰减
            float progress = (float) (globalStep - warmupSteps) / 
                           (maxEpochs * getStepsPerEpoch() - warmupSteps);
            currentLearningRate = minLearningRate + 
                (initialLearningRate - minLearningRate) * 
                (float) (0.5 * (1 + Math.cos(Math.PI * progress)));
        }
        
        optimizer.setLearningRate(currentLearningRate);
    }
    
    /**
     * 梯度裁剪
     */
    protected void clipGradients() {
        double totalNorm = 0.0;
        
        Map<String, Parameter> params = model.getModule().namedParameters("", true);
        for (Parameter param : params.values()) {
            if (param.grad() != null) {
                NdArray grad = param.grad();
                double paramNorm = Math.sqrt(grad.mul(grad).sum().getNumber().doubleValue());
                totalNorm += paramNorm * paramNorm;
            }
        }
        
        totalNorm = Math.sqrt(totalNorm);
        
        if (totalNorm > maxGradNorm) {
            float scale = (float) (maxGradNorm / (totalNorm + 1e-6));
            for (Parameter param : params.values()) {
                if (param.grad() != null) {
                    // 统一使用 mulNum 方式，与其他 Trainer 保持一致
                    param.setGrad(param.grad().mulNum(scale));
                }
            }
        }
    }
    
    /**
     * 保存检查点
     */
    protected void saveCheckpoint(String suffix) {
        try {
            String filename = String.format("%s/%s_%s.model",
                checkpointDir, model.getName(), suffix);
            model.saveModel(filename);
            System.out.printf("检查点已保存: %s%n", filename);
        } catch (Exception e) {
            System.err.println("保存检查点失败: " + e.getMessage());
        }
    }
    
    /**
     * 创建检查点目录
     */
    protected void createCheckpointDir() {
        File dir = new File(checkpointDir);
        if (!dir.exists()) {
            dir.mkdirs();
        }
    }
    
    // ========== 数据集访问抽象方法（由子类实现）==========
    
    /**
     * 准备数据集
     */
    protected abstract void prepareDataset(boolean shuffle);
    
    /**
     * 检查是否有下一批
     */
    protected abstract boolean hasNextBatch();
    
    /**
     * 获取下一批数据
     */
    protected abstract Batch nextBatch();
    
    /**
     * 获取每个Epoch的步数
     */
    protected abstract int getStepsPerEpoch();
    
    // ========== Batch 数据封装 ==========
    
    /**
     * Batch数据封装类
     */
    public static class Batch {
        private final NdArray inputIds;
        private final NdArray targetIds;
        
        public Batch(NdArray inputIds, NdArray targetIds) {
            this.inputIds = inputIds;
            this.targetIds = targetIds;
        }
        
        public NdArray getInputIds() { return inputIds; }
        public NdArray getTargetIds() { return targetIds; }
    }
    
    // ========== Getter方法 ==========
    
    public M getModel() { return model; }
    public List<Float> getLossHistory() { return lossHistory; }
    public int getCurrentEpoch() { return currentEpoch; }
    public int getGlobalStep() { return globalStep; }
}
