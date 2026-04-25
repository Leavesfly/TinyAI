package io.leavesfly.tinyai.minimind.training;

import io.leavesfly.tinyai.func.Variable;
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
    
    private final SFTDataset dataset;
    private final SoftmaxCrossEntropy lossFunction;
    private final Adam optimizer;
    
    private float learningRate;
    private int accumulationSteps = 1;  // 梯度累积步数
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
    
    // ==================== 实现抽象方法 ====================
    
    @Override
    protected float trainStep(Object batch) {
        // 更新学习率（对标 Python get_lr: cosine with 10% floor）
        updateLearningRate();
        
        SFTDataset.Batch sftBatch = (SFTDataset.Batch) batch;
        
        NdArray inputArray = sftBatch.getInput();
        NdArray labelArray = sftBatch.getLabels();
        
        Variable input = new Variable(inputArray);
        Variable labels = new Variable(labelArray);
        
        Variable logits = model.predict(input);
        
        int[] logitsShape = logits.getValue().getShape().getShapeDims();
        int totalTokens = logitsShape[0] * logitsShape[1];
        int vocabSize = logitsShape[2];
        
        Variable logitsReshaped = logits.reshape(Shape.of(totalTokens, vocabSize));
        Variable labelsReshaped = labels.reshape(Shape.of(totalTokens, 1));
        
        Variable loss = lossFunction.loss(labelsReshaped, logitsReshaped);
        float lossValue = loss.getValue().getNumber().floatValue();
        
        if (Float.isNaN(lossValue) || Float.isInfinite(lossValue)) {
            System.err.println("警告: 损失值异常 (" + lossValue + "), 跳过此batch");
            return Float.NaN;
        }
        
        // 梯度累积
        if (accumulationSteps > 1) {
            Variable scaleVar = new Variable(1.0f / accumulationSteps);
            scaleVar.setRequireGrad(false);
            loss = loss.mul(scaleVar);
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
        double cosineDecay = 0.1 + 0.45 * (1 + Math.cos(Math.PI * currentStep / Math.max(totalSteps, 1)));
        currentLearningRate = learningRate * (float) cosineDecay;
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
    
    /**
     * 设置检查点目录
     */
    public void setCheckpointDir(String checkpointDir) {
        this.checkpointDir = checkpointDir;
    }
}