package io.leavesfly.tinyai.gpt1.training;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.gpt1.GPT1Config;
import io.leavesfly.tinyai.gpt1.GPT1Model;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ml.optimize.Optimizer;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * GPT-1预训练器
 * 
 * 实现因果语言建模(Causal Language Modeling)预训练
 * 支持以下功能:
 * - 学习率warmup和cosine衰减
 * - 梯度裁剪
 * - 检查点保存和恢复
 * - 训练指标记录
 * 
 * @author TinyAI
 * @since 2024
 */
public class GPT1Pretrain {
    
    private final GPT1Model model;
    private final GPT1Config config;
    private final GPT1Dataset dataset;
    private final SoftmaxCrossEntropy lossFunction;
    private final Optimizer optimizer;
    
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
     * @param model GPT-1模型
     * @param dataset 训练数据集
     */
    public GPT1Pretrain(GPT1Model model, GPT1Dataset dataset) {
        this.model = model;
        this.config = model.getConfig();
        this.dataset = dataset;
        this.lossFunction = new SoftmaxCrossEntropy();
        
        // 默认超参数(遵循GPT-1论文)
        this.maxEpochs = 10;
        this.initialLearningRate = 2.5e-4f;  // GPT-1默认学习率
        this.minLearningRate = 1e-5f;
        this.warmupSteps = 2000;
        this.maxGradNorm = 1.0f;
        this.logInterval = 100;
        this.saveInterval = 5000;
        this.checkpointDir = "./checkpoints/gpt1_pretrain";
        
        // 使用Adam优化器，收敛速度比SGD快3-5倍，对稀疏梯度处理更好
        this.optimizer = new Adam(model, initialLearningRate, 0.9f, 0.999f, 1e-8f);
        
        // 初始化状态
        this.currentEpoch = 0;
        this.globalStep = 0;
        this.currentLearningRate = 0.0f;
        this.lossHistory = new ArrayList<>();
    }
    
    /**
     * 配置训练参数
     * 
     * @param maxEpochs 最大训练轮次
     * @param learningRate 初始学习率
     * @param warmupSteps warmup步数
     * @param maxGradNorm 梯度裁剪阈值
     * @return this
     */
    public GPT1Pretrain configure(int maxEpochs, float learningRate, 
                                   int warmupSteps, float maxGradNorm) {
        this.maxEpochs = maxEpochs;
        this.initialLearningRate = learningRate;
        this.warmupSteps = warmupSteps;
        this.maxGradNorm = maxGradNorm;
        // 更新优化器学习率
        this.optimizer.setLearningRate(learningRate);
        return this;
    }
    
    /**
     * 设置检查点配置
     * 
     * @param checkpointDir 检查点目录
     * @param saveInterval 保存间隔(Epoch数), 每隔 saveInterval 个 Epoch 保存一次
     * @return this
     */
    public GPT1Pretrain setCheckpoint(String checkpointDir, int saveInterval) {
        this.checkpointDir = checkpointDir;
        this.saveInterval = saveInterval;
        return this;
    }
    
    /**
     * 开始训练
     */
    public void train() {
        // 先准备一次数据以获取正确的批次数量（用于打印和学习率调度）
        dataset.prepare(true);

        System.out.println("=" .repeat(60));
        System.out.println("GPT-1 预训练");
        System.out.println("=".repeat(60));
        System.out.println("模型参数:");
        System.out.println("  - 隐藏维度: " + config.getNEmbd());
        System.out.println("  - 层数: " + config.getNLayer());
        System.out.println("  - 注意力头: " + config.getNHead());
        System.out.println("  - 序列长度: " + config.getNPositions());
        System.out.println("训练配置:");
        System.out.println("  - 训练样本: " + dataset.getSampleCount());
        System.out.println("  - 批次数量: " + dataset.getBatchCount());
        System.out.println("  - 最大轮次: " + maxEpochs);
        System.out.println("  - 初始学习率: " + initialLearningRate);
        System.out.println("  - Warmup步数: " + warmupSteps);
        System.out.println("=".repeat(60));
        
        // 创建检查点目录
        createCheckpointDir();
        
        // 训练循环
        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();
            
            // 按 Epoch 间隔保存检查点
            if ((currentEpoch + 1) % saveInterval == 0) {
                saveCheckpoint("epoch_" + (currentEpoch + 1));
            }
        }
        
        // 保存最终模型
        saveCheckpoint("final");
        
        System.out.println("\n训练完成!");
        System.out.println("最终损失: " + getAverageLoss(100));
    }
    
    /**
     * 训练一个epoch
     */
    private void trainOneEpoch() {
        dataset.prepare(true);  // 打乱数据
        
        double epochLoss = 0.0;
        int batchCount = 0;
        
        long epochStartTime = System.currentTimeMillis();
        
        while (dataset.hasNext()) {
            GPT1Dataset.Batch batch = dataset.nextBatch();
            
            // 训练一步
            float stepLoss = trainStep(batch);
            
            epochLoss += stepLoss;
            batchCount++;
            globalStep++;
            
            // 记录损失
            lossHistory.add(stepLoss);
            
            // 打印日志
            if (globalStep % logInterval == 0) {
                float avgLoss = getAverageLoss(logInterval);
                System.out.printf("Epoch %d/%d | Step %d | Loss: %.4f | LR: %.6f%n",
                    currentEpoch + 1, maxEpochs, globalStep, avgLoss, currentLearningRate);
            }
        }
        
        long epochEndTime = System.currentTimeMillis();
        double avgEpochLoss = batchCount > 0 ? epochLoss / batchCount : 0.0;
        
        System.out.printf("Epoch %d 完成 | 平均损失: %.4f | 耗时: %d ms%n",
            currentEpoch + 1, avgEpochLoss, epochEndTime - epochStartTime);
        
        dataset.reset();
    }
    
    /**
     * 训练单步
     * 
     * @param batch 批次数据
     * @return 损失值
     */
    private float trainStep(GPT1Dataset.Batch batch) {
        // 更新学习率
        updateLearningRate();
        
        // 准备输入数据
        NdArray inputIds = batch.getInputIds();
        NdArray targetIds = batch.getTargetIds();
        
        Variable inputVar = new Variable(inputIds);
        
        // 前向传播
        Variable logits = model.predict(inputVar);
        
        // 重塑为2D以计算损失
        // logits: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
        // targets: (batch_size, seq_len) -> (batch_size * seq_len, 1)
        NdArray logitsArray = logits.getValue();
        int batchSize = logitsArray.getShape().getDimension(0);
        int seqLen = logitsArray.getShape().getDimension(1);
        int vocabSize = logitsArray.getShape().getDimension(2);
        
        Variable reshapedLogits = logits.reshape(io.leavesfly.tinyai.ndarr.Shape.of(batchSize * seqLen, vocabSize));
        Variable reshapedTargets = new Variable(targetIds.reshape(io.leavesfly.tinyai.ndarr.Shape.of(batchSize * seqLen, 1)));
        
        // 计算损失
        Variable loss = lossFunction.loss(reshapedTargets, reshapedLogits);
        
        float lossValue = loss.getValue().getNumber().floatValue();
        
        // 清空梯度
        model.clearGrads();
        
        // 反向传播
        loss.backward();
        
        // 梯度裁剪
        clipGradients();
        
        // 参数更新
        optimizer.update();
        
        // 断开计算图
        loss.unChainBackward();
        
        return lossValue;
    }
    
    /**
     * 更新学习率(带warmup的余弦退火)
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
        
        // 更新优化器学习率
        optimizer.setLearningRate(currentLearningRate);
    }
    
    /**
     * 梯度裁剪
     */
    private void clipGradients() {
        double totalNorm = 0.0;
        
        // 计算梯度范数
        Map<String, Parameter> params = model.getAllParams();
        for (Parameter param : params.values()) {
            if (param.getGrad() != null) {
                NdArray grad = param.getGrad();
                double norm = grad.mul(grad).sum().getNumber().doubleValue();
                totalNorm += norm;
            }
        }
        
        totalNorm = Math.sqrt(totalNorm);
        
        // 如果梯度范数超过阈值,进行裁剪
        if (totalNorm > maxGradNorm) {
            float scale = (float) (maxGradNorm / totalNorm);
            for (Parameter param : params.values()) {
                if (param.getGrad() != null) {
                    NdArray clippedGrad = param.getGrad().mulNum(scale);
                    param.setGrad(clippedGrad);
                }
            }
        }
    }
    
    /**
     * 保存检查点
     * 
     * @param suffix 文件名后缀
     */
    private void saveCheckpoint(String suffix) {
        try {
            String filename = String.format("gpt1_pretrain_%s.model", suffix);
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
     * 
     * @param n 步数
     * @return 平均损失
     */
    private float getAverageLoss(int n) {
        if (lossHistory.isEmpty()) {
            return 0.0f;
        }
        
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
                epoch, step, learningRate, currentLoss, avgLoss
            );
        }
    }
}
