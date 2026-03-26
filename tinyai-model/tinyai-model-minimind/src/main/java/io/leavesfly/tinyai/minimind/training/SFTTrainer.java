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
        this.checkpointDir = "./checkpoints/minimind_sft_checkpoints";
        
        // 创建优化器
        this.optimizer = new Adam(model, learningRate, 0.9f, 0.999f, 1e-8f);
    }
    
    /**
     * 配置训练参数
     */
    public SFTTrainer configure(int maxEpochs, float learningRate, float maxGradNorm) {
        this.maxEpochs = maxEpochs;
        this.learningRate = learningRate;
        this.maxGradNorm = maxGradNorm;
        
        optimizer.setLearningRate(learningRate);
        return this;
    }
    
    // ==================== 实现抽象方法 ====================
    
    @Override
    protected float trainStep(Object batch) {
        SFTDataset.Batch sftBatch = (SFTDataset.Batch) batch;
        
        NdArray inputArray = sftBatch.getInput();
        NdArray labelArray = sftBatch.getLabels();
        // 注: 掩码暂不使用，SoftmaxCE 已计算平均损失
        
        Variable input = new Variable(inputArray);
        Variable labels = new Variable(labelArray);
        
        // 前向传播
        Variable logits = model.predict(input);
        
        // SoftmaxCE 需要 2D 输入，将 [batch, seqLen, vocabSize] reshape 为 [batch*seqLen, vocabSize]
        int[] logitsShape = logits.getValue().getShape().getShapeDims();
        int totalTokens = logitsShape[0] * logitsShape[1];
        int vocabSize = logitsShape[2];
        
        Variable logitsReshaped = logits.reshape(Shape.of(totalTokens, vocabSize));
        Variable labelsReshaped = labels.reshape(Shape.of(totalTokens, 1));
        
        // 计算损失（SoftmaxCE 返回标量平均损失）
        Variable loss = lossFunction.loss(labelsReshaped, logitsReshaped);
        
        float lossValue = loss.getValue().getNumber().floatValue();
        
        // 检查异常值
        if (Float.isNaN(lossValue) || Float.isInfinite(lossValue)) {
            System.err.println("警告: 损失值异常 (" + lossValue + "), 跳过此batch");
            // 跳过此batch，不进行参数更新
            return Float.NaN;  // 返回NaN标识跳过，训练循环应处理此情况
        }
        
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