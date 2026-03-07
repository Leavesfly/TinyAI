package io.leavesfly.tinyai.banana.training;

import io.leavesfly.tinyai.banana.block.BananaBlock;
import io.leavesfly.tinyai.banana.config.BananaConfig;
import io.leavesfly.tinyai.banana.config.TaskType;
import io.leavesfly.tinyai.banana.model.BananaModel;
import io.leavesfly.tinyai.banana.training.dataset.BananaDataset;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.loss.MeanSquaredLoss;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Banana多模态预训练器
 * 
 * 实现以下预训练任务:
 * 1. 图像重建 (Image Reconstruction)
 * 2. 文本-图像对比学习 (Contrastive Learning)
 * 3. 掩码图像建模 (Masked Image Modeling)
 * 
 * 支持学习率调度、梯度裁剪、检查点保存等功能
 * 
 * @author TinyAI
 * @since 2024
 */
public class PretrainTrainer {
    
    private final BananaModel model;
    private final BananaBlock bananaBlock;
    private final BananaConfig config;
    private final BananaDataset dataset;
    private final MeanSquaredLoss lossFunction;  // 使用MSE作为重建损失
    private final Adam optimizer;
    
    // 训练配置
    private int maxEpochs;
    private float initialLearningRate;
    private float maxGradNorm;       // 梯度裁剪阈值
    private int warmupSteps;         // 学习率预热步数
    private int logInterval;         // 日志打印间隔
    private int saveInterval;        // 检查点保存间隔
    private String checkpointDir;    // 检查点目录
    
    // 训练状态
    private int currentEpoch;
    private int currentStep;
    private float currentLearningRate;
    private List<Float> lossHistory;
    
    /**
     * 构造函数
     * 
     * @param model 模型
     * @param dataset 预训练数据集
     */
    public PretrainTrainer(BananaModel model, BananaDataset dataset) {
        this.model = model;
        this.bananaBlock = model.getBananaBlock();
        this.config = model.getConfig();
        this.dataset = dataset;
        this.lossFunction = new MeanSquaredLoss();
        
        // 默认配置
        this.maxEpochs = 10;
        this.initialLearningRate = 1e-4f;
        this.maxGradNorm = 1.0f;
        this.warmupSteps = 1000;
        this.logInterval = 100;
        this.saveInterval = 1000;
        this.checkpointDir = "./checkpoints/banana_pretrain";
        
        // 创建优化器(AdamW)
        this.optimizer = new Adam(model, initialLearningRate, 0.9f, 0.999f, 1e-8f);
        
        // 初始化状态
        this.currentEpoch = 0;
        this.currentStep = 0;
        this.currentLearningRate = 0.0f;
        this.lossHistory = new ArrayList<>();
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
    
    /**
     * 开始训练
     */
    public void train() {
        System.out.println("=".repeat(60));
        System.out.println("开始Banana多模态预训练");
        System.out.println("=".repeat(60));
        System.out.println("模型配置: Banana " + config.getClass().getSimpleName());
        long totalParams = calculateTotalParams();
        System.out.println("总参数量: " + totalParams + " (" + String.format("%.2fM", totalParams / 1_000_000.0) + ")");
        System.out.println("训练样本数: " + dataset.getSampleCount());
        System.out.println("批次数量: " + dataset.getBatchCount());
        System.out.println("最大轮次: " + maxEpochs);
        System.out.println("初始学习率: " + initialLearningRate);
        System.out.println("图像大小: " + config.getImageSize() + "x" + config.getImageSize());
        System.out.println("=".repeat(60));
        
        // 创建检查点目录
        createCheckpointDir();
        
        // 训练循环
        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();
        }
        
        System.out.println("训练完成!");
    }
    
    /**
     * 训练一个epoch
     */
    private void trainOneEpoch() {
        dataset.prepare(true);  // 打乱数据
        
        // 设置训练模式(通过Config.train控制)
        boolean prevTrain = io.leavesfly.tinyai.util.Config.train;
        io.leavesfly.tinyai.util.Config.train = true;
        
        double epochLoss = 0.0;
        int batchCount = 0;
        
        long epochStartTime = System.currentTimeMillis();
        
        while (dataset.hasNextBatch()) {
            BananaDataset.Batch batch = dataset.getNextBatch();
            
            // 训练一步
            float stepLoss = trainStep(batch);
            
            epochLoss += stepLoss;
            batchCount++;
            currentStep++;
            
            // 记录损失
            lossHistory.add(stepLoss);
            
            // 打印日志
            if (currentStep % logInterval == 0) {
                double avgLoss = lossHistory.stream()
                    .skip(Math.max(0, lossHistory.size() - logInterval))
                    .mapToDouble(Float::doubleValue)
                    .average()
                    .orElse(0.0);
                
                System.out.printf("Epoch %d/%d | Step %d | Loss: %.4f | LR: %.6f%n",
                    currentEpoch + 1, maxEpochs, currentStep, avgLoss, currentLearningRate);
            }
            
            // 保存检查点
            if (currentStep % saveInterval == 0) {
                saveCheckpoint();
            }
        }
        
        long epochEndTime = System.currentTimeMillis();
        double avgEpochLoss = batchCount > 0 ? epochLoss / batchCount : 0.0;
        
        System.out.println(String.format(
            "Epoch %d 完成 | 平均损失: %.4f | 耗时: %d ms",
            currentEpoch + 1, avgEpochLoss, epochEndTime - epochStartTime
        ));
        
        dataset.reset();
    }
    
    /**
     * 训练一步
     * 
     * @param batch 批次数据
     * @return 损失值
     */
    private float trainStep(BananaDataset.Batch batch) {
        // 更新学习率
        updateLearningRate();
        
        // 获取输入数据
        NdArray textInput = batch.getTextInput();
        NdArray imageInput = batch.getImageInput();
        
        Variable textVar = new Variable(textInput);
        Variable imageVar = new Variable(imageInput);
        
        // 前向传播
        // 1. 编码文本和图像
        Variable textFeatures = model.encodeText(textVar);
        Variable imageFeatures = model.encodeImage(imageVar);
        
        // 2. 多模态融合 - 使用双向融合获取融合后的图像特征
        Variable[] fusedResults = bananaBlock.getFusionLayer()
            .forwardBoth(textFeatures, imageFeatures);
        Variable fusedTextFeatures = fusedResults[0];  // [batch, text_len, hidden_size]
        Variable fusedImageFeatures = fusedResults[1]; // [batch, num_patches, hidden_size]
        
        // 3. 计算重建损失
        // 使用融合后的图像特征与原始图像特征计算MSE损失
        // 这鼓励模型通过跨模态注意力保留图像信息
        Variable loss = computeReconstructionLoss(fusedImageFeatures, imageFeatures);
        
        float lossValue = loss.getValue().getNumber().floatValue();
        
        // 清除梯度
        model.clearGrads();
        
        // 反向传播
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
     * 计算重建损失
     * 
     * 使用 meanSquaredError 保持梯度计算图连通，
     * 确保 loss.backward() 能正确回传梯度到模型参数。
     * 
     * @param reconstructed 重建特征
     * @param original 原始特征
     * @return 损失値 (Variable)
     */
    private Variable computeReconstructionLoss(Variable reconstructed, Variable original) {
        // 使用内置 MSE 算子，保持计算图完整
        return reconstructed.meanSquaredError(original);
    }
    
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
     * 梯度裁剪
     */
    private void clipGradients() {
        // 计算梯度范数
        double totalNorm = 0.0;
        
        for (var param : model.getAllParams().values()) {
            if (param.getGrad() != null) {
                NdArray grad = param.getGrad();
                float[] gradData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) grad).buffer;
                
                for (float g : gradData) {
                    totalNorm += g * g;
                }
            }
        }
        
        totalNorm = Math.sqrt(totalNorm);
        
        // 如果超过阈值,进行裁剪
        if (totalNorm > maxGradNorm) {
            float clipCoef = maxGradNorm / (float) totalNorm;
            
            for (var param : model.getAllParams().values()) {
                if (param.getGrad() != null) {
                    NdArray grad = param.getGrad();
                    float[] gradData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) grad).buffer;
                    
                    for (int i = 0; i < gradData.length; i++) {
                        gradData[i] *= clipCoef;
                    }
                }
            }
        }
    }
    
    /**
     * 保存检查点
     */
    private void saveCheckpoint() {
        String filename = String.format("checkpoint_epoch%d_step%d.model", 
            currentEpoch, currentStep);
        String filepath = Paths.get(checkpointDir, filename).toString();
        
        try {
            model.save(new File(filepath));
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
            Path path = Paths.get(checkpointDir);
            if (!Files.exists(path)) {
                Files.createDirectories(path);
            }
        } catch (IOException e) {
            System.err.println("创建检查点目录失败: " + e.getMessage());
        }
    }
    
    /**
     * 计算总参数量
     */
    private long calculateTotalParams() {
        long totalParams = 0;
        for (var param : model.getAllParams().values()) {
            int[] dims = param.getValue().getShape().getShapeDims();
            long size = 1;
            for (int d : dims) size *= d;
            totalParams += size;
        }
        return totalParams;
    }
    
    /**
     * 获取损失历史
     * 
     * @return 损失历史列表
     */
    public List<Float> getLossHistory() {
        return new ArrayList<>(lossHistory);
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
