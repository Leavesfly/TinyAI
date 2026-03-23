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

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Banana多模态微调器
 * 
 * 用于在预训练模型基础上进行任务特定的微调训练
 * 支持以下微调任务:
 * 1. 文本到图像生成 (Text-to-Image Generation)
 * 2. 图像编辑 (Image Editing)
 * 3. 图像描述生成 (Image Captioning)
 * 
 * 支持以下特性:
 * - 较小的学习率(相比预训练)
 * - 早停机制
 * - 验证集评估
 * - 最佳模型保存
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
    private final MeanSquaredLoss lossFunction;
    private final Adam optimizer;
    
    // 微调超参数(与预训练不同)
    private int maxEpochs;
    private float learningRate;           // 微调学习率通常更小
    private float maxGradNorm;
    private int logInterval;
    private int evalInterval;             // 验证评估间隔
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
     * @param model 预训练的Banana模型
     * @param trainDataset 训练数据集
     * @param valDataset 验证数据集
     */
    public FinetuneTrainer(BananaModel model, BananaDataset trainDataset, BananaDataset valDataset) {
        this.model = model;
        this.bananaBlock = model.getBananaBlock();
        this.config = model.getConfig();
        this.trainDataset = trainDataset;
        this.valDataset = valDataset;
        this.lossFunction = new MeanSquaredLoss();
        
        // 默认配置(微调学习率更小)
        this.maxEpochs = 5;
        this.learningRate = 1e-5f;     // 比预训练小10倍
        this.maxGradNorm = 1.0f;
        this.logInterval = 50;
        this.evalInterval = 200;
        this.patience = 3;
        this.checkpointDir = "./checkpoints/banana_finetune";
        
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
     * @param maxEpochs 最大训练轮次
     * @param learningRate 学习率
     * @param patience 早停耐心值
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
            io.leavesfly.tinyai.util.Config.train = prevTrain;
        }
        
        trainDataset.reset();
    }
    
    /**
     * 训练一步
     * 
     * @param batch 批次数据
     * @return 损失值
     */
    private float trainStep(BananaDataset.Batch batch) {
        // 获取输入数据
        NdArray textInput = batch.getTextInput();
        NdArray imageInput = batch.getImageInput();
        
        Variable textVar = new Variable(textInput);
        Variable imageVar = new Variable(imageInput);
        
        // 前向传播
        Variable textFeatures = model.encodeText(textVar);
        Variable imageFeatures = model.encodeImage(imageVar);
        
        // 多模态融合
        Variable fusedResult = bananaBlock.forwardMultiModal(
            textFeatures, imageFeatures, TaskType.TEXT_TO_IMAGE
        );
        
        // 对齐形状：对序列维度进行平均池化
        // fusedResult: [batch, text_len, hidden_size] -> [batch, hidden_size]
        // imageFeatures: [batch, num_patches, hidden_size] -> [batch, hidden_size]
        Variable fusedPooled = fusedResult.mean(1, false);
        Variable imagePooled = imageFeatures.mean(1, false);
        
        // 计算损失
        Variable loss = computeLoss(fusedPooled, imagePooled);
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
     * 评估验证集
     * 
     * @return 验证损失
     */
    private float evaluate() {
        valDataset.prepare(false);  // 不打乱
        
        // 设置推理模式
        boolean prevTrain = io.leavesfly.tinyai.util.Config.train;
        io.leavesfly.tinyai.util.Config.train = false;
        
        try {
            double totalLoss = 0.0;
            int batchCount = 0;
            
            while (valDataset.hasNextBatch()) {
                BananaDataset.Batch batch = valDataset.getNextBatch();
                
                // 获取输入数据
                NdArray textInput = batch.getTextInput();
                NdArray imageInput = batch.getImageInput();
                
                Variable textVar = new Variable(textInput);
                Variable imageVar = new Variable(imageInput);
                
                // 前向传播(不计算梯度)
                Variable textFeatures = model.encodeText(textVar);
                Variable imageFeatures = model.encodeImage(imageVar);
                Variable fusedResult = bananaBlock.forwardMultiModal(
                    textFeatures, imageFeatures, TaskType.TEXT_TO_IMAGE
                );
                
                // 对齐形状：对序列维度进行平均池化
                Variable fusedPooled = fusedResult.mean(1, false);
                Variable imagePooled = imageFeatures.mean(1, false);
                
                // 计算损失
                Variable loss = computeLoss(fusedPooled, imagePooled);
                totalLoss += loss.getValue().getNumber().floatValue();
                batchCount++;
            }
            
            return batchCount > 0 ? (float) (totalLoss / batchCount) : 0.0f;
        } finally {
            io.leavesfly.tinyai.util.Config.train = prevTrain;
        }
    }
    
    /**
     * 计算损失
     * 
     * 使用 meanSquaredError 保持梯度计算图连通，
     * 确保 loss.backward() 能正确回传梯度到模型参数。
     */
    private Variable computeLoss(Variable reconstructed, Variable original) {
        // 使用内置 MSE 算子，保持计算图完整
        return reconstructed.meanSquaredError(original);
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