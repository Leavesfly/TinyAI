package io.leavesfly.tinyai.minimind.training;

import io.leavesfly.tinyai.func.Variable;
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
    
    // 训练状态
    private float currentLearningRate;
    
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
        
        // 创建优化器(AdamW)
        this.optimizer = new Adam(model, initialLearningRate, 0.9f, 0.999f, 1e-8f);
        
        // 初始化状态
        this.currentLearningRate = 0.0f;
    }
    
    /**
     * 配置训练参数
     * 
     * @param maxEpochs 最大训练轮次
     * @param learningRate 学习率
     * @param warmupSteps 预热步数
     * @param maxGradNorm 梯度裁剪阈值
     * @return this
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
     * 设置检查点配置
     * 
     * @param checkpointDir 检查点目录
     * @param saveInterval 保存间隔(步数)
     * @return this
     */
    public PretrainTrainer setCheckpoint(String checkpointDir, int saveInterval) {
        this.checkpointDir = checkpointDir;
        this.saveInterval = saveInterval;
        return this;
    }
    
    // ==================== 实现抽象方法 ====================
    
    @Override
    protected float trainStep(Object batch) {
        // 更新学习率
        updateLearningRate();
        
        PretrainDataset.Batch pretrainBatch = (PretrainDataset.Batch) batch;
        
        // 获取输入和目标
        NdArray inputArray = pretrainBatch.getInput();
        NdArray targetArray = pretrainBatch.getTarget();
        
        Variable input = new Variable(inputArray);
        Variable target = new Variable(targetArray);
        
        // 前向传播
        Variable logits = model.predict(input);
        
        // SoftmaxCE 需要 2D 输入，将 [batch, seqLen, vocabSize] reshape 为 [batch*seqLen, vocabSize]
        int[] logitsShape = logits.getValue().getShape().getShapeDims();
        int totalTokens = logitsShape[0] * logitsShape[1];
        int vocabSize = logitsShape[2];
        
        Variable logitsReshaped = logits.reshape(Shape.of(totalTokens, vocabSize));
        Variable targetReshaped = target.reshape(Shape.of(totalTokens, 1));
        
        // 计算损失
        Variable loss = lossFunction.loss(targetReshaped, logitsReshaped);
        float lossValue = loss.getValue().getNumber().floatValue();
        
        // 清除梯度
        model.clearGrads();
        
        // 反向传播
        loss.backward();
        
        // 梯度裁剪（继承自BaseTrainer）
        clipGradients();
        
        // 更新参数
        optimizer.update();
        
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
     * 更新学习率(带预热的余弦退火)
     */
    private void updateLearningRate() {
        if (currentStep < warmupSteps) {
            // 线性预热
            currentLearningRate = initialLearningRate * ((float) currentStep / warmupSteps);
        } else {
            // 余弦退火
            int totalSteps = maxEpochs * dataset.getBatchCount();
            int decaySteps = totalSteps - warmupSteps;
            int currentDecayStep = currentStep - warmupSteps;
            
            double cosineDecay = 0.5 * (1 + Math.cos(Math.PI * currentDecayStep / decaySteps));
            currentLearningRate = initialLearningRate * (float) cosineDecay;
        }
        
        // 更新优化器学习率
        optimizer.setLearningRate(currentLearningRate);
    }
    
    /**
     * 设置日志间隔
     * 
     * @param logInterval 日志打印间隔
     */
    public void setLogInterval(int logInterval) {
        this.logInterval = logInterval;
    }
}